"""
This training script runs on a single GPU.

To run on a single GPU, example:
$ python train_adamw.py --batch_size=32 --compile=False
"""

import os
import time
import math
import pickle
import glob
import argparse
import inspect
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
import triton
import triton.language as tl

from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on FineWeb
# I/O
out_dir = 'out'
eval_interval = 100
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = True # disabled by default
wandb_project = 'halfbam2_nanogpt_sweep'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# data
train_files = "/scratch/gilbreth/mangla/fineweb10B/fineweb_train_*.bin" # input .bin to train on
val_files = "/scratch/gilbreth/mangla/fineweb10B/fineweb_val_*.bin" # input .bin to eval validation loss on
data_path = os.environ.get("DATA_PATH", "data") # base path for data files
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
val_tokens = 10485760 # how many tokens of validation data to use
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# optimizer
learning_rate = 6e-4 # learning rate for non-2D parameters (AdamW)
use_halfbam = True # if True, use HalfBAM optimizer for 2D parameters and AdamW for others
halfbam_lr = 0.01 # learning rate for HalfBAM optimizer (CLI-tunable)
halfbam_momentum = 0.95 # momentum for HalfBAM optimizer
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# training parameters (will be calculated from effective_batch_size and token budget)
effective_batch_size = None # effective batch size in tokens (CLI-tunable, required)
stable_fraction = 0.45 # fraction of training to operate with flat LR (CLI-tunable)
token_budget = 700000000 # fixed token budget: 700M tokens
gradient_accumulation_steps = None # will be calculated from effective_batch_size
max_iters = None # will be calculated from effective_batch_size and token_budget
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
# Parse CLI arguments
parser = argparse.ArgumentParser(description='Train GPT model with dual optimizers')
parser.add_argument('--effective_batch_size', type=int, default=None, help='Effective batch size in tokens (required)')
parser.add_argument('--stable_fraction', type=float, default=0.45, help='Fraction of training to operate with flat LR')
parser.add_argument('--halfbam_lr', type=float, default=None, help='Learning rate for HalfBAM optimizer (2D hidden parameters)')
parser.add_argument('--halfbam_momentum', type=float, default=None, help='Momentum for HalfBAM optimizer')
parser.add_argument('--configurator', type=str, default='configurator.py', help='Path to configurator.py file')
cmd_args = parser.parse_args()

# Override with CLI arguments
if cmd_args.effective_batch_size is not None:
    effective_batch_size = cmd_args.effective_batch_size
if cmd_args.stable_fraction is not None:
    stable_fraction = cmd_args.stable_fraction
if cmd_args.halfbam_lr is not None:
    halfbam_lr = cmd_args.halfbam_lr
if cmd_args.halfbam_momentum is not None:
    halfbam_momentum = cmd_args.halfbam_momentum

# Load configurator if it exists
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
if os.path.exists(cmd_args.configurator):
    exec(open(cmd_args.configurator).read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
os.makedirs(out_dir, exist_ok=True)

# Validate and calculate training parameters from effective_batch_size
if effective_batch_size is None:
    raise ValueError("--effective_batch_size must be provided (in tokens)")

# Calculate max_iters from token budget
max_iters = token_budget // effective_batch_size
print(f"Token budget: {token_budget:,} tokens")
print(f"Effective batch size: {effective_batch_size:,} tokens")
print(f"Max iterations: {max_iters:,}")

# Calculate gradient accumulation steps
tokens_per_micro_batch = batch_size * block_size
gradient_accumulation_steps = effective_batch_size // tokens_per_micro_batch

# Validate that gradient_accumulation_steps is an integer
if effective_batch_size % tokens_per_micro_batch != 0:
    raise ValueError(f"effective_batch_size ({effective_batch_size}) must be divisible by tokens_per_micro_batch ({tokens_per_micro_batch})")

print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
print(f"Tokens per micro-batch: {tokens_per_micro_batch:,}")
tokens_per_iter = gradient_accumulation_steps * batch_size * block_size
print(f"Tokens per iteration: {tokens_per_iter:,}")

# Calculate eval_iters from val_tokens
eval_iters = val_tokens // (batch_size * block_size)
print(f"Eval iterations: {eval_iters} (using {val_tokens:,} validation tokens)")

# logging
if wandb_log:
    import wandb
    # Generate run name from sweep parameters
    run_name_parts = [
        f"bs{effective_batch_size}",
        f"sf{stable_fraction:.2f}",
        f"halfbam_lr{halfbam_lr:.0e}",
        f"halfbam_momentum{halfbam_momentum:.2f}",
    ]
    if wandb_run_name == 'gpt2':  # only use default if not overridden
        wandb_run_name = "_".join(run_name_parts)
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

torch.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# FineWeb data loader
def _load_data_shard(file: Path):
    """Load a FineWeb data shard file with header format"""
    header = torch.from_file(str(file), False, 256, dtype=torch.int32) # header is 256 int32
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2]) # number of tokens (claimed)
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True)
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy())
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens

def fineweb_data_generator(filename_pattern: str, seq_len: int):
    """Generator for FineWeb data - simplified for single GPU"""
    files = [Path(f) for f in sorted(glob.glob(os.path.join(data_path, filename_pattern)))]
    if len(files) == 0:
        raise FileNotFoundError(f"No files found matching pattern: {os.path.join(data_path, filename_pattern)}")
    file_iter = iter(files)
    tokens, pos = _load_data_shard(next(file_iter)), 0
    
    while True:
        # Extract batch_size sequences
        batch_inputs = []
        batch_targets = []
        for _ in range(batch_size):
            # Check if we need to load next file
            if pos + seq_len + 1 >= len(tokens):
                try:
                    tokens = _load_data_shard(next(file_iter))
                    pos = 0
                except StopIteration:
                    file_iter = iter(files)  # restart from beginning
                    tokens = _load_data_shard(next(file_iter))
                    pos = 0
            
            buf = tokens[pos:pos + seq_len + 1]
            inputs = buf[:-1].to(device=device, dtype=torch.int64, non_blocking=True)
            targets = buf[1:].to(device=device, dtype=torch.int64, non_blocking=True)
            batch_inputs.append(inputs)
            batch_targets.append(targets)
            pos += seq_len
        
        # Stack into batch
        inputs = torch.stack(batch_inputs)  # (batch_size, seq_len)
        targets = torch.stack(batch_targets)  # (batch_size, seq_len)
        yield inputs, targets

# Initialize data generators
train_data_gen = None
val_data_gen = None

def get_batch(split):
    """Get a batch - uses FineWeb data generator"""
    global train_data_gen, val_data_gen
    
    if split == 'train':
        if train_data_gen is None:
            train_data_gen = fineweb_data_generator(train_files, block_size)
        return next(train_data_gen)
    else:
        if val_data_gen is None:
            val_data_gen = fineweb_data_generator(val_files, block_size)
        return next(val_data_gen)

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# FineWeb uses GPT-2 tokenizer, so vocab_size is 50257
# We'll use 50304 (rounded up to nearest multiple of 64) for efficiency
meta_vocab_size = 50304
print(f"Using vocab_size = {meta_vocab_size} for FineWeb (GPT-2 tokenizer)")

# -----------------------------------------------------------------------------
# Triton kernel for symmetric matrix multiplication by @byronxu99
# -----------------------------------------------------------------------------

def _get_autotune_configs():
    return [
        triton.Config(
            {
                "BLOCK_SIZE_M": bm,
                "BLOCK_SIZE_N": bn,
                "BLOCK_SIZE_K": bk,
                "GROUP_SIZE_M": 8,
                "LOWER_UPPER": 1,
            },
            num_stages=stages,
            num_warps=warps,
        )
        for bm in [64, 128]
        for bn in [64, 128, 256]
        for bk in [64, 128]
        for stages, warps in [(3, 4), (3, 8), (4, 4)]
        if bm // bn <= 2 and bn // bm <= 2
    ]

@triton.jit
def _pid_to_block(
    pid,
    M,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # Split output matrix into blocks of size (BLOCK_SIZE_M, BLOCK_SIZE_N)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(M, BLOCK_SIZE_N)

    # Map PID to a single matrix in batch
    batch_idx = pid // (num_pid_m * num_pid_n)
    pid = pid % (num_pid_m * num_pid_n)

    # Map PID to 2D grid of blocks
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    pid_m, pid_n = tl.swizzle2d(pid_m, pid_n, num_pid_m, num_pid_n, GROUP_SIZE_M)

    m_idx = pid_m * BLOCK_SIZE_M
    n_idx = pid_n * BLOCK_SIZE_N
    return batch_idx, m_idx, n_idx

@triton.autotune(
    configs=_get_autotune_configs(),
    key=["M", "K", "a_stride_r", "a_stride_c", "c_stride_r", "c_stride_c"],
)
@triton.jit
def ns_line_1_kernel(
    A_ptr, C_ptr,
    M, K,
    a_stride_b, a_stride_r, a_stride_c,
    c_stride_b, c_stride_r, c_stride_c,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    LOWER_UPPER: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    batch_idx, m_idx, n_idx = _pid_to_block(
        pid, M, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M
    )

    # Skip blocks that don't need to be computed
    skip_block_below_diag = (LOWER_UPPER == 0) and (n_idx + BLOCK_SIZE_N <= m_idx)
    skip_block_above_diag = (LOWER_UPPER != 0) and (m_idx + BLOCK_SIZE_M <= n_idx)
    if skip_block_below_diag or skip_block_above_diag:
        return

    # Index into one matrix of batch
    A_ptr += batch_idx * a_stride_b
    C_ptr += batch_idx * c_stride_b

    # Create pointer arrays for A and A.T
    offs_m = (m_idx + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (n_idx + tl.arange(0, BLOCK_SIZE_N)) % M
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A_ptr + (offs_m[:, None] * a_stride_r + offs_k[None, :] * a_stride_c)
    at_ptrs = A_ptr + (offs_k[:, None] * a_stride_c + offs_n[None, :] * a_stride_r)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Accumulate over blocks of K
    for k in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        at = tl.load(at_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, at, accumulator)
        a_ptrs += BLOCK_SIZE_K * a_stride_c
        at_ptrs += BLOCK_SIZE_K * a_stride_c

    out_dtype = C_ptr.dtype.element_ty
    output = accumulator.to(out_dtype)

    # Store block of C
    offs_cm = m_idx + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = n_idx + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C_ptr + (offs_cm[:, None] * c_stride_r + offs_cn[None, :] * c_stride_c)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < M)
    tl.store(c_ptrs, output, mask=c_mask)

    # Store block of C mirrored across the diagonal
    c_ptrs_t = C_ptr + (offs_cn[:, None] * c_stride_r + offs_cm[None, :] * c_stride_c)
    c_mask_t = (offs_cn[:, None] < M) & (offs_cm[None, :] < M)
    tl.store(c_ptrs_t, output.T, mask=c_mask_t)

def ns_line_1(A: torch.Tensor, out: torch.Tensor):
    """
    Launch Triton kernel to compute C = A @ A.T
    """
    assert A.ndim == 2 or A.ndim == 3
    M, K = A.shape[-2:]
    assert out.size(-2) == M, "Output matrix has incorrect shape"
    assert out.size(-1) == M, "Output matrix has incorrect shape"

    batch_size = A.size(0) if A.ndim == 3 else 1
    input_batch_stride = A.stride(0) if A.ndim == 3 else 0
    output_batch_stride = out.stride(0) if out.ndim == 3 else 0

    grid = lambda meta: (
        batch_size * triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(M, meta["BLOCK_SIZE_N"]),
    )
    ns_line_1_kernel[grid](
        A_ptr=A,
        C_ptr=out,
        M=M,
        K=K,
        a_stride_b=input_batch_stride,
        a_stride_r=A.stride(-2),
        a_stride_c=A.stride(-1),
        c_stride_b=output_batch_stride,
        c_stride_r=out.stride(-2),
        c_stride_c=out.stride(-1),
    )
    return out

@triton.autotune(
    configs=_get_autotune_configs(),
    key=["M", "a_stride_r", "a_stride_c", "c_stride_r", "c_stride_c"],
)
@triton.jit
def ns_line_2_kernel(
    A_ptr, C_ptr,
    M,
    a_stride_b, a_stride_r, a_stride_c,
    c_stride_b, c_stride_r, c_stride_c,
    alpha, beta,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    LOWER_UPPER: tl.constexpr,
):
    # This is mostly duplicated from ns_line_1_kernel, but also loads and adds a block of A
    # Performance is slightly slower than ns_line_1_kernel, so we use two separate kernels
    pid = tl.program_id(axis=0)
    batch_idx, m_idx, n_idx = _pid_to_block(
        pid, M, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M
    )

    # Skip blocks that don't need to be computed
    skip_block_below_diag = (LOWER_UPPER == 0) and (n_idx + BLOCK_SIZE_N <= m_idx)
    skip_block_above_diag = (LOWER_UPPER != 0) and (m_idx + BLOCK_SIZE_M <= n_idx)
    if skip_block_below_diag or skip_block_above_diag:
        return

    # Index into one matrix of batch
    A_ptr += batch_idx * a_stride_b
    C_ptr += batch_idx * c_stride_b

    # Create pointer arrays for A and A.T
    offs_m = (m_idx + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (n_idx + tl.arange(0, BLOCK_SIZE_N)) % M
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A_ptr + (offs_m[:, None] * a_stride_r + offs_k[None, :] * a_stride_c)
    at_ptrs = A_ptr + (offs_k[:, None] * a_stride_c + offs_n[None, :] * a_stride_r)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Accumulate over blocks of K
    for k in tl.range(0, tl.cdiv(M, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < M - k * BLOCK_SIZE_K, other=0.0)
        at = tl.load(at_ptrs, mask=offs_k[:, None] < M - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, at, accumulator)
        a_ptrs += BLOCK_SIZE_K * a_stride_c
        at_ptrs += BLOCK_SIZE_K * a_stride_c

    # Load block of A to add (corresponds to the current block of C)
    offs_am = m_idx + tl.arange(0, BLOCK_SIZE_M)
    offs_an = n_idx + tl.arange(0, BLOCK_SIZE_N)
    a_add_ptrs = A_ptr + (offs_am[:, None] * a_stride_r + offs_an[None, :] * a_stride_c)
    a_add_mask = (offs_am[:, None] < M) & (offs_an[None, :] < M)
    a_add = tl.load(a_add_ptrs, mask=a_add_mask, other=0.0).to(tl.float32)

    # Apply alpha and beta
    accumulator *= alpha
    accumulator += a_add * beta

    out_dtype = C_ptr.dtype.element_ty
    output = accumulator.to(out_dtype)

    # Store block of C
    offs_cm = m_idx + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = n_idx + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C_ptr + (offs_cm[:, None] * c_stride_r + offs_cn[None, :] * c_stride_c)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < M)
    tl.store(c_ptrs, output, mask=c_mask)

    # Store block of C mirrored across the diagonal
    c_ptrs_t = C_ptr + (offs_cn[:, None] * c_stride_r + offs_cm[None, :] * c_stride_c)
    c_mask_t = (offs_cn[:, None] < M) & (offs_cm[None, :] < M)
    tl.store(c_ptrs_t, output.T, mask=c_mask_t)

def ns_line_2(A: torch.Tensor, alpha: float, beta: float, out: torch.Tensor):
    """
    Launch Triton kernel to compute C = alpha * A @ A.T + beta * A
    """
    assert A.ndim == 2 or A.ndim == 3
    M, K = A.shape[-2:]
    assert M == K, "Input matrix must be square"
    assert out.size(-2) == M
    assert out.size(-1) == M

    batch_size = A.size(0) if A.ndim == 3 else 1
    input_batch_stride = A.stride(0) if A.ndim == 3 else 0
    output_batch_stride = out.stride(0) if out.ndim == 3 else 0

    grid = lambda meta: (
        batch_size * triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(M, meta["BLOCK_SIZE_N"]),
    )
    ns_line_2_kernel[grid](
        A_ptr=A,
        C_ptr=out,
        M=M,
        a_stride_b=input_batch_stride,
        a_stride_r=A.stride(-2),
        a_stride_c=A.stride(-1),
        c_stride_b=output_batch_stride,
        c_stride_r=out.stride(-2),
        c_stride_c=out.stride(-1),
        alpha=alpha,
        beta=beta,
    )
    return out

@torch.compile(dynamic=False, fullgraph=True) # Must use dynamic=False or else it's much slower
def newton_schulz_triton(G: torch.Tensor):
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)

    # Allocate buffers
    X = X.contiguous()
    A = torch.empty((*X.shape[:-1], X.size(-2)), device=X.device, dtype=X.dtype)
    B = torch.empty_like(A)
    C = torch.empty_like(X)

    ns_line_3 = torch.baddbmm if X.ndim > 2 else torch.addmm

    # Perform the NS iterations
    for _ in range(5):
        ns_line_1(X, out=A)  # A = X @ X.mT
        ns_line_2(A, alpha=c, beta=b, out=B)  # B = b * A + c * A @ A
        ns_line_3(X, B, X, beta=a, out=C)  # C = a * X + B @ X
        X, C = C, X  # Swap references to avoid unnecessary copies

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


@torch.compile
def half_sink_normed(G, steps: int):
    assert G.ndim >= 2

    for _ in range(steps):
        if G.shape[-2] > G.shape[-1]:
            G_norm_row = torch.linalg.vector_norm(G, ord=2, dim=-2, keepdim=True) + 1e-7
            G = G / G_norm_row
        else:
            G_norm_col = torch.linalg.vector_norm(G, ord=2, dim=-1, keepdim=True) + 1e-7
            G = G / G_norm_col
            
    return G

def halfbam_update(grad, momentum, beta=0.95, nesterov=True):
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim == 4: # for the case of conv filters
        update = update.reshape(len(update), -1)
    update = half_sink_normed(update, steps=1)
    update *= max(1, grad.size(-2) / grad.size(-1))**0.5
    return update

class HalfBAM(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, weight_decay=0, momentum=0.95, track_singulars=False):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
        super().__init__(params, defaults)
        self.last_svs_by_name = {}
        self.track_singulars = track_singulars
        self.step_counter = 0
        self.param_id_to_name = {}

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.step_counter += 1
        should_track = self.track_singulars and (self.step_counter % 50 == 0)
        if should_track:
            print(f"[HalfBAM] Tracking singulars at step {self.step_counter}, param_id_to_name has {len(self.param_id_to_name)} entries")
        svs_by_name = {} if should_track else None

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    # continue
                    p.grad = torch.zeros_like(p)  # Force synchronization
                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)
                update = halfbam_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                if should_track and p.ndim >= 2:
                    W = p.detach().reshape(p.shape[0], -1) if p.ndim > 2 else p.detach()
                    U = update.detach().reshape(update.shape[0], -1) if update.ndim > 2 else update.detach()
                    try:
                        s_w = torch.linalg.svdvals(W.float()).cpu()
                        s_u = torch.linalg.svdvals(U.float()).cpu()
                        name = self.param_id_to_name.get(id(p))
                        if name is not None:
                            svs_by_name[name] = {'weight': s_w, 'update': s_u}
                        else:
                            print(f"[HalfBAM] Warning: Parameter with id {id(p)} and shape {p.shape} not found in param_id_to_name")
                    except RuntimeError as e:
                        print(f"[HalfBAM] Error computing SVD: {e}")
                        pass
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update.reshape(p.shape), alpha=-group["lr"])

        if svs_by_name is not None:
            self.last_svs_by_name = svs_by_name
        else:
            self.last_svs_by_name = {}

        return loss

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    model_args['vocab_size'] = meta_vocab_size
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer configuration
if use_halfbam:
    # Separate parameters: 2D+ for HalfBAM, others for AdamW
    # Embeddings and final layer use AdamW, hidden matrices use HalfBAM
    param_dict = {pn: p for pn, p in model.named_parameters()}
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    
    hidden_matrix_params = []
    embed_params = []
    scalar_params = []
    head_params = []
    
    for n, p in param_dict.items():
        if 'wte' in n or 'wpe' in n:  # token and position embeddings
            embed_params.append(p)
        elif 'lm_head' in n:  # final output layer
            head_params.append(p)
        elif p.ndim >= 2:
            hidden_matrix_params.append(p)
        else:
            scalar_params.append(p)
    
    num_halfbam_params = sum(p.numel() for p in hidden_matrix_params)
    num_adam_params = sum(p.numel() for p in embed_params + scalar_params + head_params)
    print(f"num HalfBAM parameter tensors: {len(hidden_matrix_params)}, with {num_halfbam_params:,} parameters")
    print(f"num AdamW parameter tensors: {len(embed_params + scalar_params + head_params)}, with {num_adam_params:,} parameters")
    
    # Create two optimizers
    # Optimizer 1: AdamW for embeddings, scalars, head
    decay_params_other = [p for p in embed_params + head_params if p.dim() >= 2]
    nodecay_params_other = [p for p in embed_params + scalar_params + head_params if p.dim() < 2]
    optim_groups_other = [
        {'params': decay_params_other, 'weight_decay': weight_decay},
        {'params': nodecay_params_other, 'weight_decay': 0.0}
    ]
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == 'cuda'
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer1 = torch.optim.AdamW(optim_groups_other, lr=learning_rate, betas=(beta1, beta2), **extra_args)
    print(f"Optimizer 1 (AdamW for non-2D): LR={learning_rate}, betas=({beta1}, {beta2}), weight_decay={weight_decay}, fused={use_fused}")
    
    # Optimizer 2: HalfBAM for 2D hidden parameters
    print(f"Optimizer 2 (HalfBAM for 2D): LR={halfbam_lr}, momentum={halfbam_momentum}, weight_decay={weight_decay}")
    optimizer2 = HalfBAM(hidden_matrix_params, lr=halfbam_lr, momentum=halfbam_momentum, weight_decay=weight_decay)
    
    optimizer = [optimizer1, optimizer2]
else:
    # Single AdamW optimizer (fallback)
    optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

# store initial learning rates for each param group
if isinstance(optimizer, list):
    for opt in optimizer:
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]
else:
    for group in optimizer.param_groups:
        group["initial_lr"] = group["lr"]
if init_from == 'resume':
    if isinstance(optimizer, list):
        # Handle list of optimizers
        if 'optimizers' in checkpoint:
            for opt, opt_state in zip(optimizer, checkpoint['optimizers']):
                opt.load_state_dict(opt_state)
        else:
            # Fallback: try to load as single optimizer (for backward compatibility)
            print("WARNING: Checkpoint has single optimizer but model uses multiple optimizers. Skipping optimizer load.")
    else:
        optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
raw_model = model  # keep reference to unoptimized model for checkpoint saving
if compile:
    print("compiling the model... (takes a ~minute)")
    model = torch.compile(model) # requires PyTorch 2.0

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# learning rate schedule: warmup/stable/cooldown
def get_lr(it, max_iters, stable_fraction):
    """
    LR schedule with three phases:
    - Warmup: First 10% of steps - linear from 0 to 1.0
    - Stable: Next stable_fraction of steps - flat at 1.0
    - Cooldown: Remaining steps - cosine decay from 1.0 to 0.1
    """
    warmup_steps = int(0.1 * max_iters)
    stable_steps = int(stable_fraction * max_iters)
    
    if it < warmup_steps:
        # Warmup: linear from 0 to 1.0
        return (it + 1) / warmup_steps
    elif it < warmup_steps + stable_steps:
        # Stable: flat at 1.0
        return 1.0
    else:
        # Cooldown: cosine decay from 1.0 to 0.1
        cooldown_start = warmup_steps + stable_steps
        cooldown_steps = max_iters - cooldown_start
        if cooldown_steps == 0:
            return 0.1
        progress = (it - cooldown_start) / cooldown_steps
        assert 0 <= progress <= 1
        # Cosine decay: 0.1 + 0.9 * (1 + cos(pi * progress)) / 2
        return 0.1 + 0.9 * (1.0 + math.cos(math.pi * progress)) / 2.0

# training loop
X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
running_mfu = -1.0
while True:

    # determine and set the learning rate for this iteration
    if isinstance(optimizer, list):
        # Apply same schedule to both optimizers
        lr_scale = get_lr(iter_num, max_iters, stable_fraction)
        for opt in optimizer:
            for param_group in opt.param_groups:
                param_group['lr'] = param_group['initial_lr'] * lr_scale
        # Apply HalfBAM momentum warmup
        if use_halfbam and len(optimizer) > 1:
            frac = min(iter_num / 300, 1)  # momentum warmup for halfbam
            optimizer[1].param_groups[0]['momentum'] = (1 - frac) * 0.85 + frac * halfbam_momentum
        lr = optimizer[0].param_groups[0]['lr']  # for logging purposes
    else:
        lr_scale = get_lr(iter_num, max_iters, stable_fraction)
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['initial_lr'] * lr_scale
        lr = optimizer.param_groups[0]['lr']

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            log_dict = {
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "mfu": running_mfu*100, # convert to percentage
            }
            # Track learning rates
            if isinstance(optimizer, list):
                log_dict["lr/adamw"] = optimizer[0].param_groups[0]['lr']
                log_dict["lr/halfbam"] = optimizer[1].param_groups[0]['lr']
                log_dict["momentum/halfbam"] = optimizer[1].param_groups[0]['momentum']
            else:
                log_dict["lr"] = lr
            wandb.log(log_dict)
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                if isinstance(optimizer, list):
                    optimizer_states = [opt.state_dict() for opt in optimizer]
                    checkpoint = {
                        'model': raw_model.state_dict(),
                        'optimizers': optimizer_states,
                        'model_args': model_args,
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                        'config': config,
                    }
                else:
                    checkpoint = {
                        'model': raw_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'model_args': model_args,
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                        'config': config,
                    }
                # print(f"saving checkpoint to {out_dir}")
                # torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        if isinstance(optimizer, list):
            for opt in optimizer:
                scaler.unscale_(opt)
        else:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    if isinstance(optimizer, list):
        for opt in optimizer:
            scaler.step(opt)
    else:
        scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    if isinstance(optimizer, list):
        for opt in optimizer:
            opt.zero_grad(set_to_none=True)
    else:
        optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
        if wandb_log:
            log_dict = {
                "iter": iter_num,
                "train/loss": lossf,
                "mfu": running_mfu*100, # convert to percentage
            }
            # Track learning rates
            if isinstance(optimizer, list):
                log_dict["lr/adamw"] = optimizer[0].param_groups[0]['lr']
                log_dict["lr/halfbam"] = optimizer[1].param_groups[0]['lr']
                log_dict["momentum/halfbam"] = optimizer[1].param_groups[0]['momentum']
            else:
                log_dict["lr"] = lr
            wandb.log(log_dict)
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break
