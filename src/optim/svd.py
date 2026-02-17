import torch
import math

def _orthogonality_gap(mat: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    rows, cols = mat.shape
    if rows >= cols:
        col_norms = torch.linalg.vector_norm(mat, ord=2, dim=0, keepdim=True)
        mat = mat / (col_norms + eps)
        gram = mat.T @ mat
        eye = torch.eye(cols, device=mat.device, dtype=mat.dtype)
        denom = math.sqrt(max(cols * (cols - 1), 1))
    else:
        row_norms = torch.linalg.vector_norm(mat, ord=2, dim=1, keepdim=True)
        mat = mat / (row_norms + eps)
        gram = mat @ mat.T
        eye = torch.eye(rows, device=mat.device, dtype=mat.dtype)
        denom = math.sqrt(max(rows * (rows - 1), 1))
    diff = gram - eye
    return torch.linalg.norm(diff, ord='fro') / denom


def explicit_svd(W):
    U, _, V = torch.linalg.svd(W, full_matrices=False)
    return U @ V

def svd_update(grad, momentum, beta=0.95, nesterov=True):
    # grad = sink_normed(grad, steps=3)
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim == 4: # for the case of conv filters
        update = update.reshape(len(update), -1)
    gap_before = _orthogonality_gap(update)
    update = explicit_svd(update)
    gap_after = _orthogonality_gap(update)
    assert gap_after < gap_before, "Orthogonality gap increased"
    update *= max(1, grad.size(-2) / grad.size(-1))**0.5
    return update


class SVD(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, weight_decay=0, momentum=0.95, track_singulars=False):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
        super().__init__(params, defaults)
        self.last_svs_by_name = {}
        self.track_singulars = track_singulars

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        svs_by_name = {} if self.track_singulars else None

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    # continue
                    p.grad = torch.zeros_like(p)  # Force synchronization
                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)
                update = svd_update(p.grad, state["momentum_buffer"], beta=group["momentum"], nesterov=False)

                if self.track_singulars and p.ndim >= 2:
                    W = p.detach().reshape(p.shape[0], -1) if p.ndim > 2 else p.detach()
                    U = update.detach().reshape(update.shape[0], -1) if update.ndim > 2 else update.detach()
                    try:
                        s_w = torch.linalg.svdvals(W).cpu()
                        s_u = torch.linalg.svdvals(U).cpu()
                        name = getattr(self, 'param_id_to_name', {}).get(id(p))
                        if name is not None:
                            svs_by_name[name] = {'weight': s_w, 'update': s_u}
                    except RuntimeError:
                        pass
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update.reshape(p.shape), alpha=-group["lr"])

        self.last_svs_by_name = svs_by_name or {}
        return loss