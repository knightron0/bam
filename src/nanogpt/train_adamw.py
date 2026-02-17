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
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

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
wandb_project = 'adamw_nanogpt_sweep'
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
# adamw optimizer
learning_rate = 6e-4 # learning rate for non-2D parameters
use_separate_2d = True # if True, use separate optimizer for 2D hidden parameters
lr_2d = 6e-4 # learning rate for 2D hidden parameters (CLI-tunable)
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# training parameters (will be calculated from effective_batch_size and token budget)
effective_batch_size = None # effective batch size in tokens (CLI-tunable, required)
stable_fraction = 0.45 # fraction of training to operate with flat LR (CLI-tunable)
token_budget = 700000000 # fixed token budget: 700 tokens
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
parser.add_argument('--lr_2d', type=float, default=None, help='Learning rate for 2D hidden parameters')
parser.add_argument('--beta1', type=float, default=None, help='Beta1 (momentum) for AdamW optimizer')
parser.add_argument('--configurator', type=str, default='configurator.py', help='Path to configurator.py file')
cmd_args = parser.parse_args()

# Override with CLI arguments
if cmd_args.effective_batch_size is not None:
    effective_batch_size = cmd_args.effective_batch_size
if cmd_args.stable_fraction is not None:
    stable_fraction = cmd_args.stable_fraction
if cmd_args.lr_2d is not None:
    lr_2d = cmd_args.lr_2d
if cmd_args.beta1 is not None:
    beta1 = cmd_args.beta1

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
        f"lr{learning_rate:.0e}",
        f"beta1{beta1:.2f}",
    ]
    if use_separate_2d:
        run_name_parts.append(f"lr2d{lr_2d:.0e}")
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

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type, 
                                       use_separate_2d=use_separate_2d, lr_2d=lr_2d)
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
                log_dict["lr/non_2d"] = optimizer[0].param_groups[0]['lr']
                log_dict["lr/2d"] = optimizer[1].param_groups[0]['lr']
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
                log_dict["lr/non_2d"] = optimizer[0].param_groups[0]['lr']
                log_dict["lr/2d"] = optimizer[1].param_groups[0]['lr']
            else:
                log_dict["lr"] = lr
            wandb.log(log_dict)
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break
