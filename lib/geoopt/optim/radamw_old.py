import torch.optim

from .mixin import OptimMixin
from ..tensor import ManifoldParameter, ManifoldTensor
from lib.geoopt.manifolds.lorentz.math import expmap0, logmap0, expmap, project_u, project, parallel_transport0back


__all__ = ["RiemannianAdamW"]


class RiemannianAdamW(OptimMixin, torch.optim.AdamW):
    r"""
    Riemannian Adam with the same API as :class:`torch.optim.Adam`.

    Parameters
    ----------
    params : iterable
        iterable of parameters to optimize or dicts defining
        parameter groups
    lr : float (optional)
        learning rate (default: 1e-3)
    betas : Tuple[float, float] (optional)
        coefficients used for computing
        running averages of gradient and its square (default: (0.9, 0.999))
    eps : float (optional)
        term added to the denominator to improve
        numerical stability (default: 1e-8)
    weight_decay : float (optional)
        weight decay (L2 penalty) (default: 0)
    amsgrad : bool (optional)
        whether to use the AMSGrad variant of this
        algorithm from the paper `On the Convergence of Adam and Beyond`_
        (default: False)

    Other Parameters
    ----------------
    stabilize : int
        Stabilize parameters if they are off-manifold due to numerical
        reasons every ``stabilize`` steps (default: ``None`` -- no stabilize)


    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ

    """

    def step(self, closure=None):
        loss = None
        self.man_params = []

        if closure is not None:
            loss = closure()
        with torch.no_grad():
            for group in self.param_groups:

                if group["name"] == "k_group":
                    continue

                betas = group["betas"]
                weight_decay = group["weight_decay"]
                eps = group["eps"]
                learning_rate = group["lr"]
                amsgrad = group["amsgrad"]
                stablilize = False
                for point in group["params"]:
                    grad = point.grad
                    if grad is None:
                        continue
                    if isinstance(point, (ManifoldParameter, ManifoldTensor)):
                        manifold = point.manifold
                        self.man_params.append([point, manifold.k.clone()])
                    else:
                        manifold = self._default_manifold

                    if grad.is_sparse:
                        raise RuntimeError(
                            "RiemannianAdam does not support sparse gradients, use SparseRiemannianAdam instead"
                        )

                    state = self.state[point]

                    # State initialization
                    if len(state) == 0:
                        state["step"] = 0
                        # Exponential moving average of gradient values
                        state["exp_avg"] = torch.zeros_like(point)
                        # Exponential moving average of squared gradient values
                        state["exp_avg_sq"] = torch.zeros_like(point)
                        if amsgrad:
                            # Maintains max of all exp. moving avg. of sq. grad. values
                            state["max_exp_avg_sq"] = torch.zeros_like(point)
                    state["step"] += 1
                    # make local variables for easy access
                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]
                    # actual step
                    #grad.add_(point, alpha=weight_decay)
                    if isinstance(point, (ManifoldParameter, ManifoldTensor)):
                        #param.mul_(1 - lr * weight_decay)
                        means = torch.concat((point.unsqueeze(-2), point.manifold.origin(point.shape).unsqueeze(-2).to(point.device)),
                                             dim=-2)
                        point = point.manifold.centroid(means, w=torch.tensor(((1-learning_rate*weight_decay), learning_rate*weight_decay),dtype=means.dtype, device=means.device))

                    else:
                        point.mul_(1 - learning_rate * weight_decay)

                    grad = manifold.egrad2rgrad(point, grad)
                    exp_avg.mul_(betas[0]).add_(grad, alpha=1 - betas[0])
                    exp_avg_sq.mul_(betas[1]).add_(
                        manifold.component_inner(point, grad), alpha=1 - betas[1]
                    )
                    bias_correction1 = 1 - betas[0] ** state["step"]
                    bias_correction2 = 1 - betas[1] ** state["step"]
                    if amsgrad:
                        max_exp_avg_sq = state["max_exp_avg_sq"]
                        # Maintains the maximum of all 2nd moment running avg. till now
                        torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                        # Use the max. for normalizing running avg. of gradient
                        denom = max_exp_avg_sq.div(bias_correction2).sqrt_()
                    else:
                        denom = exp_avg_sq.div(bias_correction2).sqrt_()
                    # copy the state, we need it for retraction
                    # get the direction for ascend
                    direction = exp_avg.div(bias_correction1) / denom.add_(eps)
                    # transport the exponential averaging to the new point
                    new_point, exp_avg_new = manifold.retr_transp(
                        point, -learning_rate * direction, exp_avg
                    )
                    # use copy only for user facing point
                    point.copy_(new_point)
                    exp_avg.copy_(exp_avg_new)

                    if (
                            group["stabilize"] is not None
                            and state["step"] % group["stabilize"] == 0
                    ):
                        stablilize = True
                if stablilize:
                    self.stabilize_group(group)
            for group in self.param_groups:

                if group["name"] != "k_group":
                    continue

                betas = group["betas"]
                weight_decay = group["weight_decay"]
                eps = group["eps"]
                learning_rate = group["lr"]
                amsgrad = group["amsgrad"]
                stablilize = False
                for point in group["params"]:
                    grad = point.grad
                    if grad is None:
                        continue
                    if isinstance(point, (ManifoldParameter, ManifoldTensor)):
                        manifold = point.manifold
                    else:
                        manifold = self._default_manifold

                    if grad.is_sparse:
                        raise RuntimeError(
                            "RiemannianAdam does not support sparse gradients, use SparseRiemannianAdam instead"
                        )

                    state = self.state[point]

                    # State initialization
                    if len(state) == 0:
                        state["step"] = 0
                        # Exponential moving average of gradient values
                        state["exp_avg"] = torch.zeros_like(point)
                        # Exponential moving average of squared gradient values
                        state["exp_avg_sq"] = torch.zeros_like(point)
                        if amsgrad:
                            # Maintains max of all exp. moving avg. of sq. grad. values
                            state["max_exp_avg_sq"] = torch.zeros_like(point)
                    state["step"] += 1
                    # make local variables for easy access
                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]
                    # actual step
                    #grad.add_(point, alpha=weight_decay)
                    if isinstance(point, (ManifoldParameter, ManifoldTensor)):
                        #param.mul_(1 - lr * weight_decay)
                        means = torch.concat((point.unsqueeze(-2), point.manifold.origin(point.shape).unsqueeze(-2).to(point.device)),
                                             dim=-2)
                        point = point.manifold.centroid(means, w=torch.tensor(((1-learning_rate*weight_decay), learning_rate*weight_decay),dtype=means.dtype, device=means.device))

                    else:
                        point.mul_(1 - learning_rate * weight_decay)

                    grad = manifold.egrad2rgrad(point, grad)
                    exp_avg.mul_(betas[0]).add_(grad, alpha=1 - betas[0])
                    exp_avg_sq.mul_(betas[1]).add_(
                        manifold.component_inner(point, grad), alpha=1 - betas[1]
                    )
                    bias_correction1 = 1 - betas[0] ** state["step"]
                    bias_correction2 = 1 - betas[1] ** state["step"]
                    if amsgrad:
                        max_exp_avg_sq = state["max_exp_avg_sq"]
                        # Maintains the maximum of all 2nd moment running avg. till now
                        torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                        # Use the max. for normalizing running avg. of gradient
                        denom = max_exp_avg_sq.div(bias_correction2).sqrt_()
                    else:
                        denom = exp_avg_sq.div(bias_correction2).sqrt_()
                    # copy the state, we need it for retraction
                    # get the direction for ascend
                    direction = exp_avg.div(bias_correction1) / denom.add_(eps)
                    # transport the exponential averaging to the new point
                    new_point, exp_avg_new = manifold.retr_transp(
                        point, -learning_rate * direction, exp_avg
                    )
                    # use copy only for user facing point
                    point.copy_(new_point.clamp(0.4, 9))
                    exp_avg.copy_(exp_avg_new)

                    if (
                            group["stabilize"] is not None
                            and state["step"] % group["stabilize"] == 0
                    ):
                        stablilize = True
                if stablilize:
                    self.stabilize_group(group)
            self.move_parameters()

        return loss

    @torch.no_grad()
    def move_parameters(self):
        for point in self.man_params:
            if point[1] != point[0].manifold.k:
                if point[0].manifold.k == 0:
                    print("Warning: Manifold")

                state = self.state[point[0]]
                exp_avg = state["exp_avg"]

                # stabalize
                point[0].copy_(project(point[0], k=point[1]))

                exp_avg.copy_(project_u(point[0], exp_avg, k=point[1]))
                exp_avg = parallel_transport0back(point[0], exp_avg, k=point[1])

                tangent = logmap0(point[0], k=point[1], dim=-1)
                # tangent = point[0].manifold.rescale_to_max(tangent).squeeze()

                # tangent = rescale_to_max(tangent, point[0].manifold).squeeze()

                point[0].copy_(point[0].manifold.expmap0(tangent, dim=-1))

                exp_avg.copy_(point[0].manifold.transp0(point[0], exp_avg))

    @torch.no_grad()
    def stabilize_group(self, group):
        for p in group["params"]:
            if not isinstance(p, (ManifoldParameter, ManifoldTensor)):
                continue
            state = self.state[p]
            if not state:  # due to None grads
                continue
            manifold = p.manifold
            exp_avg = state["exp_avg"]
            p.copy_(manifold.projx(p))
            exp_avg.copy_(manifold.proju(p, exp_avg))
