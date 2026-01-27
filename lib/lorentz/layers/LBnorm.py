import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from lib.geoopt import ManifoldParameter
from lib.lorentz.manifold import CustomLorentz
from torch.distributions import uniform


class LorentzBatchNorm(nn.Module):
    """ Lorentz Batch Normalization with Centroid and Fréchet variance
    """
    def __init__(self, manifold: CustomLorentz, num_features: int, norm_moment: float):
        super(LorentzBatchNorm, self).__init__()
        self.manifold = manifold


        self.beta = ManifoldParameter(self.manifold.origin(num_features), manifold=self.manifold)
        self.gamma = torch.nn.Parameter(torch.ones((1,)))
        #self.beta = torch.nn.Parameter(self.manifold.projx(self.manifold.origin(num_features)))
        #self.gamma = torch.nn.Parameter(torch.ones((1,)))

        #U = uniform.Uniform(torch.tensor([0.0]), torch.tensor([1.0]))
        #self.beta = ManifoldParameter(self.manifold.add_time(U.sample(torch.Size([num_features-1])).squeeze()), manifold=self.manifold)
        #self.gamma = torch.nn.Parameter(torch.zeros((1,)))

        self.eps = 1e-5

        self.norm_moment = norm_moment

        # running statistics
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones((1,)))

    def forward(self, x, momentum=None):

        if not momentum:
            momentum = self.norm_moment

        assert (len(x.shape)==2) or (len(x.shape)==3), "Wrong input shape in Lorentz batch normalization."

        beta = self.manifold.projx(self.beta)

        if self.training:
            # Compute batch mean
            mean = self.manifold.centroid(x)
            if len(x.shape) == 3:
                mean = self.manifold.centroid(mean)

            # if len(x.shape) == 3:
            #     mean = self.manifold.centroid(x, dim=(0,1))
            # else:
            #     mean = self.manifold.centroid(x)

            # Transport batch to origin (center batch)
            x_T = self.manifold.logmap(mean, x)
            #x_T = self.manifold.transp0back(mean, x_T)

            # Compute Fréchet variance
            if len(x.shape) == 3:
                var = torch.mean(torch.norm(x_T, dim=-1), dim=(0, 1))#.sqrt()
            else:
                var = torch.mean(torch.norm(x_T, dim=-1), dim=0)#.sqrt()

            # Save running parameters
            with torch.no_grad():
                running_mean = self.manifold.expmap0(self.running_mean)
                means = torch.concat((running_mean.unsqueeze(0), mean.detach().unsqueeze(0)), dim=0)
                self.running_mean.copy_(self.manifold.logmap0(self.manifold.centroid(means, w=torch.tensor(((1-momentum), momentum), device=means.device))))
                self.running_var.copy_((1 - momentum)*self.running_var + momentum*var.detach())

        else:
            # Transport batch to origin (center batch)
            mean = self.manifold.expmap0(self.running_mean)
            x_T = self.manifold.logmap(mean, x)
            # x_T = self.manifold.transp0back(mean, x_T)

            var = self.running_var

        x_T = x_T*(self.gamma/(var+self.eps))
        x_T = self.manifold.rescale_to_max(x_T)

        # Transport batch to learned mean
        #x_T = self.manifold.transp0(beta, x_T)
        x_T = self.manifold.transp(mean, beta, x_T)
        output = self.manifold.expmap(beta, x_T)

        return output


class LorentzBatchNorm1d(LorentzBatchNorm):
    """ 1D Lorentz Batch Normalization with Centroid and Fréchet variance
    """
    def __init__(self, manifold: CustomLorentz, num_features: int, norm_moment=0.1):
        super(LorentzBatchNorm1d, self).__init__(manifold, num_features, norm_moment=norm_moment)


    def forward(self, x):
        return super(LorentzBatchNorm1d, self).forward(x)


class LorentzBatchNorm2d(LorentzBatchNorm):
    """ 2D Lorentz Batch Normalization with Centroid and Fréchet variance
    """
    def __init__(self, manifold: CustomLorentz, num_channels: int, norm_moment=0.1):
        super(LorentzBatchNorm2d, self).__init__(manifold, num_channels, norm_moment=norm_moment)

    def forward(self, x):
        """ x has to be in channel last representation -> Shape = bs x H x W x C """
        bs, h, w, c = x.shape
        x = x.view(bs, -1, c)
        x = super(LorentzBatchNorm2d, self).forward(x)
        x = x.reshape(bs, h, w, c)

        return x


class LorentzBatchNormLVar(nn.Module):
    """ Lorentz Batch Normalization with Centroid and Fréchet variance
    """
    def __init__(self, manifold: CustomLorentz, num_features: int, norm_moment=0.1):
        super(LorentzBatchNormLVar, self).__init__()
        self.manifold = manifold

        self.momentum = norm_moment

        self.beta = ManifoldParameter(self.manifold.origin(num_features), manifold=self.manifold)
        self.gamma = torch.nn.Parameter(torch.ones((1,)))

        self.eps = 1e-5

        # running statistics
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones((1,)))

    #@torch.compile
    def forward(self, x):
        assert (len(x.shape)==2) or (len(x.shape)==3), "Wrong input shape in Lorentz batch normalization."

        beta = self.beta

        if self.training:
            # Compute batch mean
            mean = self.manifold.centroid(x)
            if len(x.shape) == 3:
                mean = self.manifold.centroid(mean)

            # Transport batch to origin (center batch)
            x_T = self.manifold.logmap(mean, x)
            x_T = self.manifold.transp0back(mean, x_T)

            # Compute Fréchet variance
            var = torch.var(x, dim=-1).sqrt() #self.manifold.sqdist(x, mean).mean()

            # Rescale batch
            x_T = x_T*(self.gamma/var+self.eps)
            x_T = self.manifold.rescale_to_max(x_T)

            # Transport batch to learned mean
            x_T = self.manifold.transp0(beta, x_T)
            output = self.manifold.expmap(beta, x_T)

            # Save running parameters
            with torch.no_grad():
                means = torch.concat((self.running_mean.unsqueeze(0), mean.detach().unsqueeze(0)), dim=0)
                self.running_mean.copy_(self.manifold.centroid(means, w=torch.tensor(((1-self.momentum), self.momentum), device=means.device)))
                self.running_var.copy_((1 - self.momentum)*self.running_var + self.momentum*var.detach())

        else:
            # Transport batch to origin (center batch)
            running_mean = self.running_mean
            x_T = self.manifold.logmap(running_mean, x)
            x_T = self.manifold.transp0back(running_mean, x_T)

            # Rescale batch
            x_T = x_T*(self.gamma/self.running_var+self.eps)
            x_T = self.manifold.rescale_to_max(x_T)

            # Transport batch to learned mean
            x_T = self.manifold.transp0(beta, x_T)
            output = self.manifold.expmap(beta, x_T)

        return output


class LorentzBatchNorm2dLVar(LorentzBatchNormLVar):
    """ 2D Lorentz Batch Normalization with Centroid and Fréchet variance
    """
    def __init__(self, manifold: CustomLorentz, num_channels: int, norm_moment=0.1):
        super(LorentzBatchNorm2dLVar, self).__init__(manifold, num_channels, norm_moment=norm_moment)

    def forward(self, x):
        """ x has to be in channel last representation -> Shape = bs x H x W x C """
        bs, h, w, c = x.shape
        x = x.view(bs, -1, c)
        x = super(LorentzBatchNorm2dLVar, self).forward(x)
        x = x.reshape(bs, h, w, c)

        return x



class LorentzLayerNorm(nn.Module):
    """ Lorentz Layer Normalization with Centroid and Fréchet variance
    """
    def __init__(self, manifold: CustomLorentz, num_features: int):
        super(LorentzLayerNorm, self).__init__()
        self.manifold = manifold

        self.beta = ManifoldParameter(self.manifold.origin(num_features), manifold=self.manifold)
        self.gamma = torch.nn.Parameter(torch.ones((1,)))

        self.eps = 1e-5


    def forward(self, x):
        # Compute feature mean
        mean = self.manifold.centroid(x)
        if len(x.shape)==3:
            mean = mean.unsqueeze(1)

        # Transport batch to origin (center batch)
        x_T = self.manifold.logmap(mean, x)
        x_T = self.manifold.transp0back(mean, x_T)

        # Compute Fréchet variance
        var = torch.mean(torch.norm(x_T, dim=-1, keepdim=True), dim=1, keepdim=True)

        # Rescale batch
        x_T = x_T*(self.gamma/(var+self.eps))

        # Transport batch to learned mean
        x_T = self.manifold.transp0(self.beta, x_T)
        output = self.manifold.expmap(self.beta, x_T)


        return output


class LorentzLayerNorm_test(nn.Module):
    """ Lorentz Layer Normalization with Centroid and Fréchet variance
    """
    def __init__(self, manifold: CustomLorentz, num_channels: int, norm_moment=0.1):
        super(LorentzLayerNorm_test, self).__init__()
        self.manifold = manifold

        self.momentum = norm_moment

        self.beta = ManifoldParameter(self.manifold.origin(num_channels), manifold=self.manifold)
        self.gamma = torch.nn.Parameter(torch.ones((num_channels,)))

        self.eps = 1e-5


    def forward(self, x):
        # Compute feature mean
        bs, h, w, c = x.shape
        x = x.view(bs, -1, c)

        mean = self.manifold.centroid(x)
        if len(x.shape)==3:
            mean = mean.unsqueeze(1)

        # Transport batch to origin (center batch)
        x_T = self.manifold.logmap(mean, x)
        x_T = self.manifold.transp0back(mean, x_T)

        # Compute Fréchet variance
        var = torch.mean(torch.norm(x_T, dim=-1, keepdim=True), dim=1, keepdim=True)

        # Rescale batch
        x_T = x_T*(self.gamma/(var+self.eps))

        # output = self.manifold.expmap0(x_T)
        # Transport batch to learned mean
        x_T = self.manifold.transp0(self.beta, x_T)
        output = self.manifold.expmap(self.beta, x_T)

        output = output.reshape(bs, h, w, c)

        return output


class LorentzNotGroupNorm(nn.Module):
    """ Lorentz Group Normalization with Centroid and Fréchet variance
    """

    def __init__(self, manifold: CustomLorentz, num_channels: int, groups:int = 8, affine: bool = True, norm_moment=0.1):
        super(LorentzNotGroupNorm, self).__init__()
        self.manifold = manifold

        self.groups = groups
        self.eps = 1e-5

        self.momentum = norm_moment

        self.beta = ManifoldParameter(self.manifold.origin(num_channels), manifold=self.manifold)
        self.gamma = torch.nn.Parameter(torch.ones((num_channels,)))

        self.eps = 1e-5

    def forward(self, x):
        # Compute feature mean
        bs, h, w, c = x.shape
        assert h*w % self.groups == 0
        x = x.view(bs, -1, self.groups, c)

        mean = self.manifold.centroid(x)
        # if len(x.shape) == 3:
        #     mean = mean.unsqueeze(1)

        # Transport batch to origin (center batch)
        x_T = self.manifold.logmap(mean, x)
        x_T = self.manifold.transp0back(mean, x_T)

        # Compute Fréchet variance
        var = torch.mean(torch.norm(x_T, dim=-1, keepdim=True), dim=1, keepdim=True)

        # Rescale batch
        x_T = x_T * (self.gamma / (var + self.eps))

        # output = self.manifold.expmap0(x_T)
        # Transport batch to learned mean
        x_T = self.manifold.transp0(self.beta, x_T)
        output = self.manifold.expmap(self.beta, x_T)

        output = output.reshape(bs, h, w, c)

        return output
