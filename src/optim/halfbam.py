import torch

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

def half_bam_update(grad, momentum, beta=0.95, nesterov=True):
    # grad = sink_normed(grad, steps=3)
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim == 4: # for the case of conv filters
        update = update.reshape(len(update), -1)
    update = half_sink_normed(update, steps=1) # default to 3 iterations
    update *= max(1, grad.size(-2) / grad.size(-1))**0.5
    return update


class HalfBAM(torch.optim.Optimizer):
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
                update = half_bam_update(p.grad, state["momentum_buffer"], beta=group["momentum"], nesterov=False)

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