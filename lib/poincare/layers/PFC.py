import torch
import torch.nn as nn

from lib.poincare.manifold import CustomPoincare
from lib.poincare.layers.PMLR import unidirectional_poincare_mlr

class PoincareFullyConnected(nn.Module):
    """ FC-layer in the Poincare ball by Shimizu et al. (2020)
    
        - Modified from: https://github.com/mil-tokyo/hyperbolic_nn_plusplus
    """
    def __init__(
            self,
            manifold: CustomPoincare,
            in_features,
            out_features,
            bias=True
        ):
        super(PoincareFullyConnected, self).__init__()
        gain = 1.

        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features

        weight = torch.empty(in_features, out_features).normal_( 
            mean=0, std=(2 * self.in_features * self.out_features) ** -0.5 * gain)
        self.weight_g = nn.Parameter(weight.norm(dim=0))
        self.weight_v = nn.Parameter(weight)

        self.bias = nn.Parameter(torch.empty(out_features), requires_grad=bias)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.bias)

    def forward(self, x):
        rc = self.manifold.c.sqrt()
        x = unidirectional_poincare_mlr(x, self.weight_g, self.weight_v / self.weight_v.norm(dim=0).clamp_min(1e-15), self.bias, c=self.manifold.c)
        x = (rc * x).sinh() / rc
        x = x/(1+torch.sqrt(1+self.manifold.c*x.pow(2).sum(dim=-1, keepdim=True)))
        return self.manifold.projx(x)