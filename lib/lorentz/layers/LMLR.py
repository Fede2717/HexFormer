import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from lib.lorentz.manifold import CustomLorentz


class LorentzMLROld(nn.Module):
    """ Multinomial logistic regression (MLR) in the Lorentz model
    """

    def __init__(
            self,
            manifold: CustomLorentz,
            num_features: int,
            num_classes: int
    ):
        super(LorentzMLROld, self).__init__()

        self.manifold = manifold

        self.a = torch.nn.Parameter(torch.normal(0, 0.5, (1, num_classes)))
        z = torch.normal(0, 0.5, (num_features, num_classes))
        z[..., 0] = 0.1
        self.z = torch.nn.Parameter(z)  #  z should not be (0,0)

        self.in_dim = num_features

        self.alpha = self.beta = None

        self.init_weights()

    def forward(self, x, return_distance=False):

        z_norm = self.z.norm(dim=0, keepdim=True)
        w0 = z_norm * torch.sinh(self.a / self.manifold.k.sqrt())
        wr = torch.cosh(self.a / self.manifold.k.sqrt()) * self.z
        W = torch.cat([w0, wr], dim=0)

        numerator = - x.narrow(-1, 0, 1) @ W[[0]] + x.narrow(-1, 1, self.in_dim) @ W[1:]
        denom = z_norm * self.manifold.k.sqrt() + 1e-8
        distance = torch.arcsinh(numerator / denom) * z_norm
        # logits = torch.sign(numerator) * denom * self.manifold.k.sqrt() * torch.abs(distance)

        if return_distance:
            return distance, distance

        return distance

    def init_weights(self):
        stdv = 1. / math.sqrt(self.z.size(1))
        nn.init.uniform_(self.z, -stdv, stdv)
        nn.init.uniform_(self.a, -stdv, stdv)


class LorentzMLR(nn.Module):
    """ Multinomial logistic regression (MLR) in the Lorentz model
    """

    def __init__(
            self,
            manifold: CustomLorentz,
            num_features: int,
            num_classes: int
    ):
        super(LorentzMLR, self).__init__()

        self.manifold = manifold

        #self.a = torch.nn.Parameter(torch.zeros(num_classes, ))
        self.a = torch.nn.Parameter(torch.normal(0, 0.5, (num_classes, )))
        self.z = torch.nn.Parameter(
            F.pad(torch.normal(0, 0.5, (num_classes, num_features - 1)), pad=(1, 0), value=1))  # z should not be (0,0)

        self.alpha = self.beta = None

        self.init_weights()

    def forward(self, x, return_distance=False):
        # Hyperplane
        sqrt_mK = 1 / self.manifold.k.sqrt()
        norm_z = torch.norm(self.z, dim=-1)
        w_t = (torch.sinh(sqrt_mK * self.a) * norm_z)
        w_s = torch.cosh(sqrt_mK * self.a.view(-1, 1)) * self.z
        self.beta = torch.sqrt(-w_t ** 2 + torch.norm(w_s, dim=-1) ** 2)
        self.alpha = -w_t * x.narrow(-1, 0, 1) + (
                    torch.cosh(sqrt_mK * self.a) * torch.inner(x.narrow(-1, 1, x.shape[-1] - 1), self.z))

        d = self.manifold.k.sqrt() * torch.abs(torch.asinh(sqrt_mK * self.alpha / self.beta))  # Distance to hyperplane
        logits = torch.sign(self.alpha) * self.beta * d

        if return_distance:
            return logits, d

        return logits

    def init_weights(self):
        stdv = 1. / math.sqrt(self.z.size(1))
        nn.init.normal_(self.z, 0, stdv)
        nn.init.normal_(self.a, 0, stdv)

