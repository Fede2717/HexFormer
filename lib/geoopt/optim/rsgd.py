import torch.optim.optimizer
from ..tensor import ManifoldParameter, ManifoldTensor
from .mixin import OptimMixin
from lib.geoopt.manifolds.lorentz.math import expmap0, logmap0, expmap, project_u, project, parallel_transport0back, math_check_point_on_manifold, math_check_vector_on_tangent

from lib.lorentz.manifold import CustomLorentz, Lorentz

from lib.utils.utils import rescale_to_max

__all__ = ["RiemannianSGD"]


class RiemannianSGD(OptimMixin, torch.optim.Optimizer):
    r"""
    Riemannian Stochastic Gradient Descent with the same API as :class:`torch.optim.SGD`.

    Parameters
    ----------
    params : iterable
        iterable of parameters to optimize or dicts defining
        parameter groups
    lr : float
        learning rate
    momentum : float (optional)
        momentum factor (default: 0)
    weight_decay : float (optional)
        weight decay (L2 penalty) (default: 0)
    dampening : float (optional)
        dampening for momentum (default: 0)
    nesterov : bool (optional)
        enables Nesterov momentum (default: False)

    Other Parameters
    ----------------
    stabilize : int
        Stabilize parameters if they are off-manifold due to numerical
        reasons every ``stabilize`` steps (default: ``None`` -- no stabilize)
    """

    def __init__(
        self,
        params,
        lr,
        momentum=0,
        dampening=0,
        weight_decay=0,
        nesterov=False,
        stabilize=None,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )

        self.man_params = []

        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super().__init__(params, defaults, stabilize=stabilize)

    def euclid_step(self, point, lr, weight_decay, momentum, dampening, nesterov):

        if point.grad is None:
            return point
        grad = point.grad
        if weight_decay != 0:
            grad = grad.add(point, alpha=weight_decay)
        if momentum != 0:
            param_state = self.state[point]
            if 'momentum_buffer' not in param_state:
                buf = param_state['momentum_buffer'] = torch.clone(grad).detach()
            else:
                buf = param_state['momentum_buffer']
                buf.mul_(momentum).add_(grad, alpha=1 - dampening)
            if nesterov:
                grad = grad.add(buf, alpha=momentum)
            else:
                grad = buf

        point.add_(grad, alpha=-lr)

        return point

    def hyperbolic_step(self, point, lr, weight_decay, momentum, dampening, nesterov):

        grad = point.grad
        if grad is None:
            return point

        if grad.is_sparse:
            raise RuntimeError(
                "RiemannianSGD does not support sparse gradients, use SparseRiemannianSGD instead"
            )
        state = self.state[point]

        # State initialization
        if len(state) == 0:
            if momentum > 0:
                state["momentum_buffer"] = grad.clone()

        manifold = point.manifold

        if isinstance(manifold, (CustomLorentz, Lorentz)):
            self.man_params.append([point, manifold.k.clone()])

        grad.add_(point, alpha=weight_decay)
        grad = manifold.egrad2rgrad(point, grad)

        if momentum > 0:
            momentum_buffer = state["momentum_buffer"]
            momentum_buffer.mul_(momentum).add_(grad, alpha=1 - dampening)
            if nesterov:
                grad = grad.add_(momentum_buffer, alpha=momentum)
            else:
                grad = momentum_buffer

            # we have all the things projected
            new_point, new_momentum_buffer = manifold.retr_transp(
                point, -lr * grad, momentum_buffer
            )
            momentum_buffer.copy_(new_momentum_buffer)
            # use copy only for user facing point
            point.copy_(new_point)
        else:
            new_point = manifold.retr(point, -lr * grad)
            point.copy_(new_point)

        return point

    def step(self, closure=None):
        loss = None
        self.man_params = []
        if closure is not None:
            loss = closure()

        k_group = None

        with torch.no_grad():
            for group in self.param_groups:

                if group["name"] == "k_group":
                    k_group = group
                    continue

                if "step" not in group:
                    group["step"] = 0

                weight_decay = group["weight_decay"]
                momentum = group["momentum"]
                dampening = group["dampening"]
                nesterov = group["nesterov"]
                learning_rate = group["lr"]
                group["step"] += 1

                for point in group["params"]:
                    grad = point.grad
                    if grad is None:
                        continue
                    if grad.is_sparse:
                        raise RuntimeError(
                            "RiemannianSGD does not support sparse gradients, use SparseRiemannianSGD instead"
                        )
                    state = self.state[point]

                    # State initialization
                    if len(state) == 0:
                        if momentum > 0:
                            state["momentum_buffer"] = torch.clone(grad).detach()
                    if isinstance(point, (ManifoldParameter, ManifoldTensor)):
                        point.copy_(self.hyperbolic_step(point, learning_rate, weight_decay, momentum, dampening, nesterov))
                    else:
                        point.copy_(self.euclid_step(point, learning_rate, weight_decay, momentum, dampening, nesterov))

            if k_group is not None:
                if "step" not in k_group:
                    k_group["step"] = 0

                weight_decay = k_group["weight_decay"]
                momentum = k_group["momentum"]
                dampening = k_group["dampening"]
                nesterov = k_group["nesterov"]
                learning_rate = k_group["lr"]
                k_group["step"] += 1

                for point in k_group["params"]:
                    new_point = self.euclid_step(point, learning_rate, weight_decay, momentum, dampening, nesterov)
                    point.copy_(new_point.clamp(0.05, 9))

            self.move_parameters()
            #self.stabilize_group(self.param_groups[1])

        return loss

    @torch.no_grad()
    def move_parameters(self):
        for point in self.man_params:
            if point[1] != point[0].manifold.k:
                if point[0].manifold.k == 0:
                    print("Warning: Manifold")

                # stabalize
                momentum = self.param_groups[1]["momentum"]

                point[0].copy_(project(point[0], k=point[1]))
                if momentum > 0:
                    param_state = self.state[point[0]]
                    if not param_state:  # due to None grads
                        continue
                    if "momentum_buffer" in param_state:
                        buf = param_state["momentum_buffer"]
                        buf.copy_(project_u(point[0], buf, k=point[1]))
                        buf = parallel_transport0back(point[0], buf, k=point[1])

                tangent = logmap0(point[0], k=point[1], dim=-1)
                # tangent = point[0].manifold.rescale_to_max(tangent)

                #tangent = rescale_to_max(tangent, point[0].manifold).squeeze()

                point[0].copy_(point[0].manifold.expmap0(tangent, dim=-1))

                if momentum > 0:
                    param_state = self.state[point[0]]
                    if not param_state:  # due to None grads
                        continue
                    if "momentum_buffer" in param_state:
                        # buf = point[0].manifold.rescale_to_max(buf)
                        buf.copy_(point[0].manifold.transp0(point[0], buf))

                self.stabilize_param(point, momentum)

    @torch.no_grad()
    def move_parameters_old(self):
        for point in self.man_params:
            if point[1] != point[0].manifold.k:
                if point[0].manifold.k == 0:
                    print("Warning: Manifold")

                # stabalize
                momentum = self.param_groups[1]["momentum"]

                point[0].copy_(project(point[0], k=point[1]))
                if momentum > 0:
                    param_state = self.state[point[0]]
                    if not param_state:  # due to None grads
                        continue
                    if "momentum_buffer" in param_state:
                        buf = param_state["momentum_buffer"]
                        buf.copy_(project_u(point[0], buf, k=point[1]))
                        buf = parallel_transport0back(point[0], buf, k=point[1])

                tangent = logmap0(point[0], k=point[1], dim=-1)
                point[0].copy_(point[0].manifold.expmap0(tangent, dim=-1))

                if momentum > 0:
                    param_state = self.state[point[0]]
                    if not param_state:  # due to None grads
                        continue
                    if "momentum_buffer" in param_state:
                        buf.copy_(point[0].manifold.transp0(point[0], buf))

    @torch.no_grad()
    def stabilize_group(self, group):
        for p in group["params"]:
            if not isinstance(p, (ManifoldParameter, ManifoldTensor)):
                continue
            manifold = p.manifold
            momentum = group["momentum"]
            p.copy_(manifold.projx(p))
            if momentum > 0:
                param_state = self.state[p]
                if not param_state:  # due to None grads
                    continue
                if "momentum_buffer" in param_state:
                    buf = param_state["momentum_buffer"]
                    buf.copy_(manifold.proju(p, buf))

    @torch.no_grad()
    def stabilize_param(self, p, momentum):
        if not isinstance(p, (ManifoldParameter, ManifoldTensor)):
            return
        manifold = p.manifold
        p.copy_(manifold.projx(p))
        if momentum > 0:
            param_state = self.state[p]
            if not param_state:  # due to None grads
                return
            if "momentum_buffer" in param_state:
                buf = param_state["momentum_buffer"]
                buf.copy_(manifold.proju(p, buf))
