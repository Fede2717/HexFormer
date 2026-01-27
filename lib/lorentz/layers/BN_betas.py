import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from lib.geoopt import ManifoldParameter
from lib.lorentz.manifold import CustomLorentz
from lib.utils.utils import rescale_to_max

BN_TYPE = {"multi_mean"}


class LorentzBatchNorm_beta(nn.Module):
    """ Lorentz Batch Normalization with Centroid and Fréchet variance
    """
    def __init__(self, manifold: CustomLorentz, num_features: int, norm_moment=0.1):
        super(LorentzBatchNorm_beta, self).__init__()
        self.manifold = manifold

        self.momentum=norm_moment

        self.beta = ManifoldParameter(self.manifold.origin(num_features), manifold=self.manifold)
        self.gamma = torch.nn.Parameter(torch.ones((1,)))

        self.eps = 1e-5

        # running statistics
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones((num_features)))

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
            if len(x.shape) == 3:
                var = (x_T.var(dim=[1,0], unbiased=False) +self.eps).sqrt()
            else:
                var = (x_T.var(dim=[0], unbiased=False) + self.eps).sqrt()

            # Rescale batch
            x_T = x_T*(self.gamma/var)

            # Transport batch to learned mean
            x_T = self.manifold.transp0(beta, x_T)
            output = self.manifold.expmap(beta, x_T)

            # Save running parameters
            with torch.no_grad():
                running_mean = self.manifold.expmap0(self.running_mean)
                means = torch.concat((running_mean.unsqueeze(0), mean.detach().unsqueeze(0)), dim=0)
                self.running_mean.copy_(self.manifold.logmap0(self.manifold.centroid(means, w=torch.tensor(((1-self.momentum), self.momentum), device=means.device))))
                self.running_var.copy_((1 - self.momentum)*self.running_var + self.momentum*var.detach())

        else:
            # Transport batch to origin (center batch)
            running_mean = self.manifold.expmap0(self.running_mean)
            x_T = self.manifold.logmap(running_mean, x)
            x_T = self.manifold.transp0back(running_mean, x_T)

            # Rescale batch
            x_T = x_T*(self.gamma/self.running_var)

            # Transport batch to learned mean
            x_T = self.manifold.transp0(beta, x_T)
            output = self.manifold.expmap(beta, x_T)

        return output


class LorentzBatchNorm2d_beta(LorentzBatchNorm_beta):
    """ 2D Lorentz Batch Normalization with Centroid and Fréchet variance
    """

    def __init__(self, manifold: CustomLorentz, num_channels: int, norm_moment=0.1):
        super(LorentzBatchNorm2d_beta, self).__init__(manifold, num_channels, norm_moment=norm_moment)

    def forward(self, x):
        """ x has to be in channel last representation -> Shape = bs x H x W x C """
        bs, h, w, c = x.shape
        x = x.view(bs, -1, c)
        x = super(LorentzBatchNorm2d_beta, self).forward(x)
        x = x.reshape(bs, h, w, c)

        return x



class LorentzBatchNorm_allvar(nn.Module):
    """ Lorentz Batch Normalization with Centroid and Fréchet variance
    """
    def __init__(self, manifold: CustomLorentz, num_features: int, norm_moment=0.1):
        super(LorentzBatchNorm_allvar, self).__init__()
        self.manifold = manifold

        self.beta = ManifoldParameter(self.manifold.origin(num_features), manifold=self.manifold)
        self.gamma = torch.nn.Parameter(torch.ones((num_features,)))
        # U = torch.distributions.uniform.Uniform(torch.tensor([0.0]), torch.tensor([1.0]))
        # self.beta = ManifoldParameter(self.manifold.projx(U.sample(torch.Size([num_features])).squeeze()), manifold=self.manifold)
        # self.gamma = torch.nn.Parameter(torch.ones((num_features,)))

        self.eps = 1e-5

        self.momentum=norm_moment

        # running statistics
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones((num_features,)))

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
            if len(x.shape) == 3:
                var = (x_T.var(dim=[1, 0], unbiased=False) + self.eps).sqrt()
            else:
                var = (x_T.var(dim=[0], unbiased=False) + self.eps).sqrt()

            # var = torch.clamp(var, max=10)

            # Rescale batch
            x_T = x_T*(self.gamma/(var+self.eps))

            #x_T = self.manifold.rescale_to_max(x_T)
            # Transport batch to learned mean
            x_T = self.manifold.transp0(beta, x_T)

            output = self.manifold.expmap(beta, x_T)

            # Save running parameters
            with torch.no_grad():
                running_mean = self.manifold.expmap0(self.running_mean)
                means = torch.concat((running_mean.unsqueeze(0), mean.detach().unsqueeze(0)), dim=0)
                self.running_mean.copy_(self.manifold.logmap0(self.manifold.centroid(means, w=torch.tensor(((1-self.momentum), self.momentum), device=means.device))))
                self.running_var.copy_((1 - self.momentum)*self.running_var + self.momentum*var.detach())

        else:
            #print("yes")
            #running_mean =self.manifold.rescale_to_max(self.running_mean)
            running_mean = self.manifold.expmap0(self.running_mean)
            x_T = self.manifold.logmap(running_mean, x)
            x_T = self.manifold.transp0back(running_mean, x_T)

            # Rescale batch
            x_T = x_T*(self.gamma/(self.running_var+self.eps))

            #x_T = self.manifold.rescale_to_max(x_T)
            # Transport batch to learned mean
            x_T = self.manifold.transp0(beta, x_T)
            output = self.manifold.expmap(beta, x_T)

        return output


class LorentzBatchNorm2d_allvar(LorentzBatchNorm_allvar):
    """ 2D Lorentz Batch Normalization with Centroid and Fréchet variance
    """
    def __init__(self, manifold: CustomLorentz, num_channels: int, norm_moment=0.1):
        super(LorentzBatchNorm2d_allvar, self).__init__(manifold, num_channels, norm_moment)

    def forward(self, x):
        """ x has to be in channel last representation -> Shape = bs x H x W x C """
        bs, h, w, c = x.shape
        x = x.view(bs, -1, c)
        x = super(LorentzBatchNorm2d_allvar, self).forward(x)
        x = x.reshape(bs, h, w, c)

        return x


class LorentzBatchNormCenterOffset2d(nn.Module):
    """ 2D Lorentz Batch Normalization with Centroid and Fréchet variance
    """
    def __init__(self, manifold: CustomLorentz, num_channels: int, norm_moment=0.1):
        super(LorentzBatchNormCenterOffset2d, self).__init__()
        self.manifold = manifold

        self.beta = ManifoldParameter(self.manifold.origin(num_channels), manifold=self.manifold)
        self.gamma = torch.nn.Parameter(torch.ones((1,)))
        self.origin = self.manifold.origin(num_channels)
        self.origin.requires_grad_(False)

        self.eps = 1e-5

        self.momentum=norm_moment

        # running statistics
        self.register_buffer('running_mean', torch.zeros(num_channels))
        self.register_buffer('running_var', torch.ones((1,)))

    def forward(self, x):
        """ x has to be in channel last representation -> Shape = bs x H x W x C """
        bs, h, w, c = x.shape
        x = x.view(bs, -1, c)
        x = self.apply_norm(x)
        x = x.reshape(bs, h, w, c)

        return x

    def apply_norm(self, x):
        beta = self.manifold.centroid(torch.concat((self.beta.unsqueeze(0), self.origin.to(self.beta.device).unsqueeze(0)), dim=0),
                                      w=torch.tensor(((1 - 0.5), 0.5)).to(self.beta.device))

        if self.training:
            # Compute batch mean
            mean = self.manifold.centroid(x)
            if len(x.shape) == 3:
                mean = self.manifold.centroid(mean)

            # Transport batch to origin (center batch)
            x_T = self.manifold.logmap(mean, x)
            x_T = self.manifold.transp0back(mean, x_T)

            # Compute Fréchet variance
            if len(x.shape) == 3:
                var = torch.mean(torch.norm(x_T, dim=-1), dim=(0, 1))
            else:
                var = torch.mean(torch.norm(x_T, dim=-1), dim=0)

            # Rescale batch
            x_T = x_T * (self.gamma / (var + self.eps))

            # Transport batch to learned mean
            x_T = self.manifold.transp0(beta, x_T)
            output = self.manifold.expmap(beta, x_T)

            # Save running parameters
            with torch.no_grad():
                means = torch.concat((self.running_mean.unsqueeze(0), mean.detach().unsqueeze(0)), dim=0)
                self.running_mean.copy_(
                    self.manifold.centroid(means, w=torch.tensor(((1 - self.momentum), self.momentum), device=means.device)))
                self.running_var.copy_((1 - self.momentum) * self.running_var + self.momentum * var.detach())

        else:
            # Transport batch to origin (center batch)

            running_mean =  self.manifold.centroid(torch.concat((self.running_mean.unsqueeze(0), self.origin.to(self.running_mean.device).unsqueeze(0)), dim=0),
                                      w=torch.tensor(((1 - 0.5), 0.5)).to(self.running_mean.device))

            x_T = self.manifold.logmap(running_mean, x)
            x_T = self.manifold.transp0back(running_mean, x_T)

            # Rescale batch
            x_T = x_T * (self.gamma / (self.running_var + self.eps))

            # Transport batch to learned mean
            x_T = self.manifold.transp0(beta, x_T)
            output = self.manifold.expmap(beta, x_T)

        return output


class LorentzBatchNorm_DirectVar(nn.Module):
    """ Lorentz Batch Normalization with Centroid and Fréchet variance
    """
    def __init__(self, manifold: CustomLorentz, num_features: int, norm_moment=0.1):
        super(LorentzBatchNorm_DirectVar, self).__init__()
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

            # Compute Fréchet variance
            if len(x.shape) == 3:
                var = torch.mean(self.manifold.norm(x_T, dim=-1), dim=(0,1))
            else:
                var = torch.mean(self.manifold.norm(x_T, dim=-1), dim=0)

            # Rescale batch
            x_T = x_T*(self.gamma/(var+self.eps))

            # Transport batch to learned mean
            x_T = self.manifold.transp(mean, beta, x_T)

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

            # Rescale batch
            x_T = x_T*(self.gamma/(self.running_var+self.eps))

            # Transport batch to learned mean
            x_T = self.manifold.transp(running_mean, beta, x_T)
            output = self.manifold.expmap(beta, x_T)

        return output


class LorentzBatchNorm2d_DirectVar(LorentzBatchNorm_DirectVar):
    """ 2D Lorentz Batch Normalization with Centroid and Fréchet variance
    """
    def __init__(self, manifold: CustomLorentz, num_channels: int, norm_moment=0.1):
        super(LorentzBatchNorm2d_DirectVar, self).__init__(manifold, num_channels, norm_moment=norm_moment)

    def forward(self, x):
        """ x has to be in channel last representation -> Shape = bs x H x W x C """
        bs, h, w, c = x.shape
        x = x.view(bs, -1, c)
        x = super(LorentzBatchNorm2d_DirectVar, self).forward(x)
        x = x.reshape(bs, h, w, c)

        return x


class LorentzBatchNorm_DistVar(nn.Module):
    """ Lorentz Batch Normalization with Centroid and Fréchet variance
    """
    def __init__(self, manifold: CustomLorentz, num_features: int, norm_moment=0.1):
        super(LorentzBatchNorm_DistVar, self).__init__()
        self.manifold = manifold

        self.momentum = norm_moment

        #U = torch.distributions.uniform.Uniform(torch.tensor([0.0]), torch.tensor([1.0]))
        #self.beta = ManifoldParameter(self.manifold.add_time(U.sample(torch.Size([num_features - 1])).squeeze()),
        #                              manifold=self.manifold)
        #self.gamma = torch.nn.Parameter(torch.zeros((1,)))
        self.beta = ManifoldParameter(self.manifold.origin(num_features), manifold=self.manifold)
        self.gamma = torch.nn.Parameter(torch.ones((1,)))

        self.eps = 1e-5

        # running statistics
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones((1,)))

    #@torch.compile
    def forward(self, x, momentum=None):
        assert (len(x.shape)==2) or (len(x.shape)==3), "Wrong input shape in Lorentz batch normalization."

        if not momentum:
            momentum = self.momentum

        beta = self.beta

        if self.training:
            # Compute batch mean
            mean = self.manifold.centroid(x)
            if len(x.shape) == 3:
                mean = self.manifold.centroid(mean)

            distances = self.manifold.sqdist(x, mean).clamp(min=1e-8)

            # Transport batch to origin (center batch)
            x_T = self.manifold.logmap(mean, x)

            # Compute Fréchet variance
            if len(x.shape) == 3:
                var = torch.mean(distances).sqrt()
            else:
                var = torch.mean(distances).sqrt()

            scale = (self.gamma/(var+self.eps))

            # Rescale batch
            x_T = x_T*scale

            x_T = rescale_to_max(x_T, self.manifold)

            # Transport batch to learned mean
            x_T = self.manifold.transp(mean, beta, x_T)

            output = self.manifold.expmap(beta, x_T)

            # Save running parameters
            with torch.no_grad():
                running_mean = self.manifold.expmap0(self.running_mean)
                means = torch.concat((running_mean.unsqueeze(0), mean.detach().unsqueeze(0)), dim=0)
                self.running_mean.copy_(self.manifold.logmap0(
                    self.manifold.centroid(means, w=torch.tensor(((1 - momentum), momentum), device=means.device))))
                self.running_var.copy_((1 - momentum) * self.running_var + momentum * var.detach())

        else:
            # Transport batch to origin (center batch)
            running_mean = self.manifold.expmap0(self.running_mean)

            x_T = self.manifold.logmap(running_mean, x)

            scale = (self.gamma/(self.running_var+self.eps))

            # Rescale batch w.r.t center
            x_T = x_T*scale

            x_T = rescale_to_max(x_T, self.manifold)

            # Transport batch to learned mean
            x_T = self.manifold.transp(running_mean, beta, x_T)
            output = self.manifold.expmap(beta, x_T)

        return output


class LorentzBatchNorm2d_DistVar(LorentzBatchNorm_DistVar):
    """ 2D Lorentz Batch Normalization with Centroid and Fréchet variance
    """
    def __init__(self, manifold: CustomLorentz, num_channels: int, norm_moment=0.1):
        super(LorentzBatchNorm2d_DistVar, self).__init__(manifold, num_channels, norm_moment=norm_moment)

    def forward(self, x):
        """ x has to be in channel last representation -> Shape = bs x H x W x C """
        bs, h, w, c = x.shape
        x = x.view(bs, -1, c)
        x = super(LorentzBatchNorm2d_DistVar, self).forward(x)
        x = x.reshape(bs, h, w, c)

        return x
