import torch

from lib.geoopt import PoincareBallExact

class CustomPoincare(PoincareBallExact):
    def __init__(self, c, learnable=False):
        super(CustomPoincare, self).__init__(c=c, learnable=learnable)

    def beta_concat(self, x, beta_ni, beta_n):
        x = self.logmap0(x) * beta_n/beta_ni
        x = torch.flatten(x, -2, -1)
        return self.expmap0(x)