import torch
import torch.nn as nn

from lib.lorentz.manifold import CustomLorentz
from lib.lorentz.layers.FF_betas import LorentzProjection


class ManifoldSwapper1D(nn.Module):

    def __init__(self, manifold=None, manifold_2=None, to_euclidean=False, from_euclidean=False, space_only=False):
        super(ManifoldSwapper1D, self).__init__()

        self.manifold = manifold
        self.manifold_2 = manifold_2
        self.to_euclidean = to_euclidean
        self.from_euclidean = from_euclidean
        self.space_only = space_only

    def forward(self, x):

        if self.to_euclidean:
            if self.space_only:
                return x[..., 1:]
            return self.manifold.logmap0(x)[..., 1:]

        if self.from_euclidean:
            x = self.manifold_2.rescale_to_max(x)
            return self.manifold_2.add_time(x)

        if self.space_only:
            return self.manifold_2.projx(x)

        x = self.manifold.logmap0(x)
        x = self.manifold_2.rescale_to_max(x)

        return self.manifold_2.expmap0(x)


class LorentzAct(nn.Module):
    """ Implementation of a general Lorentz Activation on space components. 
    """
    def __init__(self, activation, manifold: CustomLorentz):
        super(LorentzAct, self).__init__()
        self.manifold = manifold
        self.activation = activation  # e.g. torch.relu

    def forward(self, x):
        if type(x) == tuple:
            return self.manifold.lorentz_activation(x[0], self.activation), x[1]
        else:
            return self.manifold.lorentz_activation(x, self.activation)

class LorentzLearnedNorm(nn.Module):
    """ Implementation of a general Lorentz Activation on space components.
    """
    def __init__(self, manifold: CustomLorentz):
        super(LorentzLearnedNorm, self).__init__()
        self.manifold = manifold
        self.scale = nn.Parameter(torch.ones(1))
    def forward(self, x):
        sq_norm = torch.abs(self.minkowski_dot(x, x, keepdim=False)).clamp(min=1e-2)
        real_norm = torch.sqrt(torch.abs(sq_norm))
        projected_point = torch.einsum("...i,...->...i", x, self.k * self.scale * real_norm)
        return projected_point
    

class LorentzReLU(nn.Module):
    """ Implementation of Lorentz ReLU Activation on space components. 
    """
    def __init__(self, manifold: CustomLorentz):
        super(LorentzReLU, self).__init__()
        self.manifold = manifold

    def forward(self, x):
        return self.manifold.lorentz_relu(x)


class LorentzGlobalAvgPool2d(nn.Module):
    """ Implementation of a Lorentz Global Average Pooling based on Lorentz centroid defintion. 
    """
    def __init__(self, manifold: CustomLorentz, w=None, keep_dim=False, last_dim=None):
        super(LorentzGlobalAvgPool2d, self).__init__()

        self.manifold = manifold
        self.keep_dim = keep_dim
        self.w = nn.Parameter(torch.ones(w)) if w is not None else None

        self.lin = torch.nn.Linear(last_dim, 1) if last_dim is not None else None

    def forward(self, x):
        """ x has to be in channel-last representation -> Shape = bs x H x W x C """
        bs, h, w, c = x.shape
        x = x.reshape(bs, -1, c)

        if self.lin is not None:
            self.w = torch.nn.functional.softmax(self.lin(x[..., 1:]).squeeze(), dim=-1)

        if self.w is not None:
            x = self.manifold.centroid(x, self.w.unsqueeze(-2)).squeeze()

        x = self.manifold.centroid(x).squeeze()

        if self.keep_dim:
            x = x.view(bs, 1, 1, c)

        return x
