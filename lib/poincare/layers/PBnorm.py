import torch
import torch.nn as nn

from lib.geoopt import ManifoldParameter
from lib.poincare.manifold import CustomPoincare


class PoincareBatchNorm(nn.Module):
    """
    """
    def __init__(self, manifold: CustomPoincare, num_features: int):
        super(PoincareBatchNorm, self).__init__()
        self.manifold = manifold
        
        self.beta = ManifoldParameter(self.manifold.origin(num_features), manifold=self.manifold)
        self.gamma = torch.nn.Parameter(torch.ones((1,)))

        self.eps = 1e-5

        # running statistics
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones((1,)))

    def forward(self, x, momentum=0.1):
        assert (len(x.shape)==2) or (len(x.shape)==3), "Wrong input shape in Poincare batch normalization."

        if self.training:
            # Compute batch mean and variance
            mean = self.manifold.weighted_midpoint(x, dim=-1, reducedim=(1,)) 
            if len(x.shape) == 3:
                mean = self.manifold.weighted_midpoint(mean, dim=-1, reducedim=(0,))

            # Transport batch to origin (center batch)
            x_T = self.manifold.logmap(mean, x)
            x_T = self.manifold.transp0back(mean, x_T)

            # Compute Fréchet variance
            if len(x.shape) == 3:
                var = torch.mean(2*torch.norm(x_T, dim=-1), dim=(0,1))
            else:
                var = torch.mean(2*torch.norm(x_T, dim=-1), dim=0)

            # Rescale batch
            x_T = 0.5*x_T*(self.gamma/(var+self.eps))

            # Transport batch to learned mean
            x_T = self.manifold.transp0(self.beta, x_T)
            output = self.manifold.expmap(self.beta, x_T)

            # Save running parameters
            with torch.no_grad():
                running_mean = self.manifold.expmap0(self.running_mean)
                means = torch.concat((running_mean.unsqueeze(0), mean.detach().unsqueeze(0)), dim=0)
                self.running_mean.copy_(self.manifold.logmap0(self.manifold.weighted_midpoint(means, weights=torch.tensor(((1-momentum), momentum), device=means.device))))
                self.running_var.copy_((1 - momentum)*self.running_var + momentum*var.detach())

        else:
            # Transport batch to origin (center batch)
            running_mean = self.manifold.expmap0(self.running_mean)
            x_T = self.manifold.logmap(running_mean, x)
            x_T = self.manifold.transp0back(running_mean, x_T)

            # Rescale batch
            x_T = 0.5*x_T*(self.gamma/(self.running_var+self.eps))

            # Transport batch to learned mean
            x_T = self.manifold.transp0(self.beta, x_T)
            output = self.manifold.expmap(self.beta, x_T)

        return output
    

class PoincareBatchNorm1d(PoincareBatchNorm):
    """ 1D Lorentz Batch Normalization with Centroid and Fréchet variance
    """
    def __init__(self, manifold: CustomPoincare, num_features: int):
        super(PoincareBatchNorm1d, self).__init__(manifold, num_features)

    def forward(self, x, momentum=0.1):
        return super(PoincareBatchNorm1d, self).forward(x, momentum)

class PoincareBatchNorm2d(PoincareBatchNorm):
    """ 2D Lorentz Batch Normalization with Centroid and Fréchet variance
    """
    def __init__(self, manifold: CustomPoincare, num_channels: int):
        super(PoincareBatchNorm2d, self).__init__(manifold, num_channels)

    def forward(self, x, momentum=0.1):
        """ x has to be in channel last representation -> Shape = bs x H x W x C """
        bs, h, w, c = x.shape
        x = x.view(bs, -1, c)
        x = super(PoincareBatchNorm2d, self).forward(x, momentum)
        x = x.reshape(bs, h, w, c)

        return x