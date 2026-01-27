import torch
import torch.nn as nn

from lib.poincare.manifold import CustomPoincare

class PoincareTangentReLU(nn.Module):
    """ Implementation of Poincare ReLU Activation in tangent space. 
    """
    def __init__(self, manifold: CustomPoincare):
        super(PoincareTangentReLU, self).__init__()
        self.manifold = manifold

    def forward(self, x):
        return self.manifold.expmap0(torch.relu(self.manifold.logmap0(x)))
    

class PoincareGlobalAvgPool2d(nn.Module):
    """ Implementation of a Poincare Global Average Pooling based on Poincare midpoint defintion. 
    """
    def __init__(self, manifold: CustomPoincare, keep_dim=False):
        super(PoincareGlobalAvgPool2d, self).__init__()

        self.manifold = manifold
        self.keep_dim = keep_dim

    def forward(self, x):
        """ x has to be in channel-last representation -> Shape = bs x H x W x C """
        bs, h, w, c = x.shape
        x = x.view(bs, -1, c)
        x = self.manifold.weighted_midpoint(x, dim=-1, reducedim=(1,))
        if self.keep_dim:
            x = x.view(bs, 1, 1, c)

        return x