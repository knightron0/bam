import torch

@torch.compile
def zeropower_via_newtonschulz5(G, steps: int):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert G.ndim >= 2 # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X
    
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

def muon_update(grad, momentum, beta=0.95, ns_steps=5, nesterov=True):
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim == 4: # for the case of conv filters
        update = update.reshape(len(update), -1)
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    update *= max(1, grad.size(-2) / grad.size(-1))**0.5
    return update


class Muon(torch.optim.Optimizer):
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
                update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])

                if self.track_singulars and p.ndim >= 2:
                    W = p.detach().reshape(p.shape[0], -1) if p.ndim > 2 else p.detach()
                    U = update.detach().reshape(update.shape[0], -1) if update.ndim > 2 else update.detach()
                    try:
                        s_w = torch.linalg.svdvals(W.float()).cpu()
                        s_u = torch.linalg.svdvals(U.float()).cpu()
                        name = getattr(self, 'param_id_to_name', {}).get(id(p))
                        if name is not None:
                            svs_by_name[name] = {'weight': s_w, 'update': s_u}
                    except RuntimeError:
                        pass

                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update.reshape(p.shape), alpha=-group["lr"])

        self.last_svs_by_name = svs_by_name or {}

        return loss