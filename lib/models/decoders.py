import torch
import torch.nn as nn

from lib.lorentz.manifold import CustomLorentz
from lib.lorentz.layers.LFC import LorentzFullyConnected
from lib.lorentz.layers.LModules import LorentzReLU
from lib.lorentz.layers import LorentzMLR, LorentzProjection, LorentzBatchNorm1d


class LorentzLinearDecoder(nn.Module):
    """ Multinomial logistic regression (MLR) in the Lorentz model
    """

    def __init__(
            self,
            manifold: CustomLorentz,
            num_features: int,
            num_output: int,
            regularize: bool = False,
            mlp: bool = False
    ):
        super(LorentzLinearDecoder, self).__init__()

        self.manifold = manifold

        self.regularize = regularize

        self.l1 = LorentzFullyConnected(manifold, num_features+1, num_output+1)

        self.rest = nn.Sequential()

        if mlp:
            self.rest = nn.Sequential(LorentzReLU(manifold),
                                      LorentzFullyConnected(manifold, num_output, num_output))

    def forward(self, x):
        x = self.l1(x)
        x = self.rest(x)

        return self.manifold.logmap0(x)[..., 1:]


class LorentzPureLinearDecoder(nn.Module):
    """ Multinomial logistic regression (MLR) in the Lorentz model
    """

    def __init__(
            self,
            manifold: CustomLorentz,
            num_features: int,
            num_output: int,
            regularize: bool = False,
            mlp: bool = False
    ):
        super(LorentzPureLinearDecoder, self).__init__()

        self.manifold = manifold

        self.regularize = regularize

        self.l1 = LorentzProjection(manifold, num_features+1, num_output+1)

        self.rest = nn.Sequential()

        if mlp:
            self.rest = nn.Sequential(LorentzReLU(manifold),
                                      LorentzProjection(manifold, num_output+1, num_output+1))

    def forward(self, x):
        x = self.l1(x)
        x = self.rest(x)

        return self.manifold.logmap0(x)[..., 1:]


class LorentzMLRBlockDecoder(nn.Module):
    """ Multinomial logistic regression (MLR) in the Lorentz model
    """

    def __init__(
            self,
            manifold: CustomLorentz,
            num_features: int,
            num_output: int,
            regularize: bool = False,
            inner_dim: int = 1024,
    ):
        super(LorentzMLRBlockDecoder, self).__init__()

        self.manifold = manifold

        self.regularize = regularize

        self.l1 = LorentzProjection(manifold, num_features+1, inner_dim+1)
        #self.bn = LorentzBatchNorm1d(manifold, inner_dim+1)
        self.decoder = LorentzMLR(self.manifold, inner_dim, num_output)

    def forward(self, x):
        x = self.l1(x)
        #x = self.bn(x)
        if self.regularize:
            x = self.manifold.regularize(x)

        return self.decoder(x)
