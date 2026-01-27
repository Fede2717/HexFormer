import torch
import torch.nn as nn


from lib.geoopt.manifolds.lorentz import Lorentz
from lib.geoopt.manifolds.lorentz import math

MIN_NORM = 1e-8
MAX_DISTANCE = 8.5


class CustomLorentz(Lorentz):
    def __init__(self, k=1.0, learnable=False):
        super(CustomLorentz, self).__init__(k=k, learnable=learnable)

        # s should not be less than 2.6 ish otherwise the initial part of the rescale function actually increases norms
        # self.s = nn.Parameter(torch.tensor(3), requires_grad=False)
        # self.eps = {torch.float32: 1e-6, torch.float64: 1e-12}
        self.max_dist = torch.sqrt(self.k) * torch.arccosh(2e3 / self.k).to(device=self.k.device)
        self.tanh_factor = torch.atanh(torch.tensor(0.99, device=self.k.device))/(self.max_dist*2)
        # self.max_dist = nn.Parameter(torch.sqrt(self.k) * torch.arccosh(2e3 / self.k), requires_grad=False)
        # self.tanh_factor = nn.Parameter(torch.atanh(torch.tensor(0.99))/(self.max_dist*self.s), requires_grad=False)
        self.learnable_factor = torch.nn.Parameter(torch.randn(1, device=self.k.device))

    @torch.no_grad()
    def update_limits(self):
        self.max_dist = torch.sqrt(self.k) * torch.arccosh(2e3 / self.k).to(device=self.k.device)
        self.tanh_factor = torch.atanh(torch.tensor(0.99, device=self.k.device))/(self.max_dist*2)

    def scale_hyperbolic_vector(self, vector, factor=None, norm=None):
        if norm is None:
            norm = self.dist0(vector, dim=-1, keepdim=True)
        norm = norm/torch.sqrt(self.k)

        if factor is None:
            factor = self.learnable_factor  # Use the learnable factor

        out = vector * (torch.exp(factor * norm) - torch.exp(-factor * norm)) / (torch.exp(norm) - torch.exp(-norm) + MIN_NORM)

        if torch.isnan(out).sum()>0:
            print("break")
        return self.projx(out)
    
    # old
    # def rescale_to_max(self, euclid_vector):
    #     x_norm = torch.norm(euclid_vector, dim=-1, keepdim=True)
    #     x_norm_out = x_norm.clone()

    #     new_norms = self.max_dist * torch.tanh(self.tanh_factor.to(device=self.k.device) * x_norm)
    #     return new_norms * euclid_vector / (x_norm_out+1e-5)
    
    def calculate_max_norm(self, vector, norm=None):
        if norm is None:
            norm = self.dist0(vector, dim=-1, keepdim=True)
        return self.max_dist * torch.tanh(self.tanh_factor.to(device=self.k.device) * norm)

    def rescale_to_max(self, hyperbolic_vector):
        x_norm = self.dist0(hyperbolic_vector, dim=-1, keepdim=True)

        new_norms = self.calculate_max_norm(hyperbolic_vector, x_norm)
        factor = new_norms / (x_norm+1e-6)
        out = self.scale_hyperbolic_vector(hyperbolic_vector, factor, x_norm)

        if torch.isnan(out).sum() > 0:
            print("break")

        return out

    def sqdist(self, x, y, dim=-1):
        """ Squared Lorentzian distance, as defined in the paper 'Lorentzian Distance Learning for Hyperbolic Representation'"""
        return (-2*self.k - 2 * math.inner(x, y, keepdim=False, dim=dim)).clamp_min(0)

    def get_time_from_distance(self, x, dist0=None):

        if dist0 is None:
            dist0 = self.dist0(x)

        return (self.k/torch.sqrt(self.k)) * torch.cosh(dist0/torch.sqrt(self.k))

    def add_time(self, space):
        """ Concatenates time component to given space component. """
        time = self.calc_time(space)
        return torch.cat([time, space], dim=-1)

    def calc_time(self, space):
        """ Calculates time component from given space component. """
        return torch.sqrt(torch.norm(space, dim=-1, keepdim=True)**2+self.k)

    def centroid(self, x, w=None, eps=1e-8, dim=-2):
        """ Centroid implementation. Adapted the code from Chen et al. (2022) """
        if w is not None:
            avg = w.matmul(x)
        else:
            avg = x.mean(dim=dim)

        denom = (-self.inner(avg, avg, keepdim=True))
        denom = denom.abs().clamp_min(eps).sqrt()

        centroid = torch.sqrt(self.k) * avg / denom

        return centroid

    def switch_man(self, x, manifold_in: Lorentz):
        """ Projection between Lorentz manifolds (e.g. change curvature) """
        x = manifold_in.logmap0(x)
        return self.expmap0(x)

    def pt_addition(self, x, y):
        """ Parallel transport addition proposed by Chami et al. (2019) """
        z = self.logmap0(y)
        z = self.transp0(x, z)

        return self.expmap(x, z)
    
    # def angular_distance(self, x, y):
    #     dot_product = -self.minkowski_dot(x, y, keepdim=False)
    #     norm_x = self.minkowski_norm(x, keepdim=False)
    #     norm_y = self.minkowski_norm(y, keepdim=False)
    #     cos_theta = dot_product / (norm_x * norm_y).clamp(min=1e-6)  # Avoid division by zero
    #     return torch.arccos(cos_theta.clamp(-1 + 1e-6, 1 - 1e-6))  # Clamp to valid range for arccos
    
    # def combined_distance(self, x, y, alpha=0.5):
    #     lorentz_dist = self.sqdist(x, y)
    #     angular_dist = self.angular_distance(x, y)
    #     return alpha * lorentz_dist + (1 - alpha) * angular_dist
    
    # def adaptive_curvature(self, metric):
    #     """Update curvature based on a task-specific metric."""
    #     #self.k = nn.Parameter(self.k + torch.sigmoid(metric) * 0.01, requires_grad=True)
    #     """Update curvature based on a task-specific metric."""
    #     # Aggregate the metric to a scalar value, e.g., via mean or sum
    #     metric_signal = torch.mean(metric)  # Reduce metric to a scalar
        
    #     # Update curvature as a scalar
    #     updated_k = self.k + torch.sigmoid(metric_signal) * 0.01
        
    #     # Ensure curvature remains a scalar parameter
    #     self.k = nn.Parameter(updated_k, requires_grad=True)

    def csqdist(self, x, y):
        """ Squared Lorentzian distance between all points. """
        return -2*self.k - 2 * self.cinner(x, y)

    def cinner(self, x, y):
        """ Lorentzian inner product between all points. """
        x = x.clone()
        x.narrow(-1, 0, 1).mul_(-1)
        return x @ y.transpose(-1, -2)

    def scale(self, x, factor, origin=None):
        if origin is None:
            tangent_x = self.logmap0(x)
        else:
            tangent_x = self.logmap(origin, x)

        tangent_x = tangent_x*factor

        # norm = torch.norm(tangent_x, dim=-1)
        # new_norm_factor = torch.nn.functional.sigmoid(norm)
        # tangent_x = tangent_x * ((9*new_norm_factor)/norm).unsqueeze(-1)

        if origin is None:
            x = self.expmap0(tangent_x)
        else:
            x = self.expmap(origin, tangent_x)

        return x

    # def boost_scale_origin(self, x, s):
    #     x_narrow = x.narrow(-1, 1, x.shape[-1] - 2)
    #
    #     x_0 = torch.sqrt(s**2 + 1.0) * x_narrow.narrow(-1, 0, 1) + s * x_narrow.narrow(-1, x_narrow.shape[-1] - 1, 1)
    #     x_n = s * x_narrow.narrow(-1, 0, 1) + torch.sqrt(s**2 + 1.0) * x_narrow.narrow(-1, x_narrow.shape[-1] - 1, 1)
    #     x = torch.cat([x_0, x_narrow, x_n], dim=-1)
    #
    #     return x

    #################################################
    #       Testing operations from paper (Nested spaces)
    #################################################

    def minkowski_dot(self, x, y, keepdim=True):
        res = torch.sum(x * y, dim=-1) - 2 * x[..., 0] * y[..., 0]
        if keepdim:
            res = res.view(res.shape + (1,))
        return res

    def minkowski_norm(self, u, keepdim=True):
        dot = self.minkowski_dot(u, u, keepdim=keepdim)
        return torch.sqrt(torch.clamp(dot, min=self.eps[u.dtype]))

    def regularize(self, x):
        sq_norm = torch.abs(self.minkowski_dot(x, x, keepdim=False)).clamp(min=1e-2)
        real_norm = torch.sqrt(torch.abs(sq_norm))
        projected_point = torch.einsum("...i,...->...i", x, self.k*1.0 / real_norm)

        return projected_point

    # def normalize_to_manifold(self, x, max_val=9):
    #     sq_norm = torch.abs(self.minkowski_dot(x, x, keepdim=False)).clamp(min=1e-2)
    #     real_norm = torch.sqrt(torch.abs(sq_norm))
    #     projected_point = torch.einsum("...i,...->...i", x, self.k*max_val / real_norm)
    #
    #     return projected_point

    def mobius_add(self, x, y):
        u = self.logmap0(y)
        v = self.transp0(x, u)
        return self.expmap(v, x)

    def mobius_matvec(self, m, x):
        u = self.logmap0(x)
        mu = u @ m.transpose(-1, -2)
        return self.expmap0(mu)

    def from_poincare(self, x, ideal=False):
        """Convert from Poincare ball model to hyperboloid model
        Args:
            x: torch.tensor of shape (..., dim)
            ideal: boolean. Should be True if the input vectors are ideal points, False otherwise

        Returns:
            torch.tensor of shape (..., dim+1)

        To do:
            Add some capping to make things numerically stable. This is only needed in the case ideal == False
        """
        if ideal:
            t = torch.ones(x.shape[:-1], device=x.device).unsqueeze(-1)
            return torch.cat((t, x), dim=-1)
        else:
            eucl_squared_norm = (x * x).sum(dim=-1, keepdim=True)
            return torch.cat((1 + eucl_squared_norm, 2 * x), dim=-1) / (1 - eucl_squared_norm).clamp_min(MIN_NORM)

    def to_poincare(self, x, ideal=False):
        """Convert from hyperboloid model to Poincare ball model
        Args:
            x: torch.tensor of shape (..., Minkowski_dim), where Minkowski_dim >= 3
            ideal: boolean. Should be True if the input vectors are ideal points, False otherwise

        Returns:
            torch.tensor of shape (..., Minkowski_dim - 1)
        """
        if ideal:
            return x[..., 1:] / (x[..., 0].unsqueeze(-1)).clamp_min(MIN_NORM)
        else:
            return x[..., 1:] / (1 + x[..., 0].unsqueeze(-1)).clamp_min(MIN_NORM)

    #################################################
    #       Reshaping operations
    #################################################
    def lorentz_flatten(self, x: torch.Tensor) -> torch.Tensor:
        """ Implements flattening operation directly on the manifold. Based on Lorentz Direct Concatenation (Qu et al., 2022) """
        bs,h,w,c = x.shape
        # bs x H x W x C
        time = x.narrow(-1, 0, 1).view(-1, h*w)
        space = x.narrow(-1, 1, x.shape[-1] - 1).flatten(start_dim=1) # concatenate all x_s

        #time_rescaled = torch.sqrt(torch.sum(time**2, dim=-1, keepdim=True)+(((h*w)-1)/-self.k))


        x = self.add_time(space)

        return x

    def lorentz_reshape_img(self, x: torch.Tensor, img_dim) -> torch.Tensor:
        """ Implements reshaping a flat tensor to an image directly on the manifold. Based on Lorentz Direct Split (Qu et al., 2022) """
        space = x.narrow(-1, 1, x.shape[-1] - 1)
        space = space.view((-1, img_dim[0], img_dim[1], img_dim[2]-1))
        img = self.add_time(space)

        return img


    #################################################
    #       Activation functions
    #################################################
    def lorentz_relu(self, x: torch.Tensor, add_time: bool=True) -> torch.Tensor:
        """ Implements ReLU activation directly on the manifold. """
        return self.lorentz_activation(x, torch.relu, add_time)


    # def lorentz_activation(self, x: torch.Tensor, activation, add_time: bool=True) -> torch.Tensor:
    #     """ Implements activation directly on the manifold. """
    # # torch.autograd.set_detect_anomaly(True)
    #     x_t = x.narrow(-1, 0, 1)
    #     x_s = x.narrow(-1, 1, x.shape[-1] - 1)
    #
    #     norm_old = x_s.clone().norm(dim=-1, keepdim=True)
    #
    #     out_s = activation(x_s.clone())
    #     norm_new = out_s.norm(dim=-1, keepdim=True)
    #     mask = norm_new == 0
    #
    #     norm_new = norm_new.masked_fill(mask, 1)
    #
    #     #negs = (x_s<0).sum(dim=-1)/x_s.shape[-1]
    #
    #     out_new = out_s * (0.5*norm_old + 0.5*norm_new)/norm_new
    #     # out_new = out_s * ((1-negs)*norm_old + negs*norm_new)/ norm_new
    #
    #     out = torch.concat((x_t.clone().masked_fill(mask, 1), out_new), dim=-1)
    #
    #     return self.normalize_to_manifold(out, 8)
#


    # def lorentz_activation(self, x: torch.Tensor, activation, add_time: bool=True) -> torch.Tensor:
    #     """ Implements activation directly on the manifold. """
    #     x = activation(x.narrow(-1, 1, x.shape[-1] - 1))
    #     if add_time:
    #         x = self.add_time(x)
    #     return x
    def lorentz_activation(self, x: torch.Tensor, activation, add_time: bool=True) -> torch.Tensor:
        """ Implements activation directly on the manifold. """
        x = activation(x.narrow(-1, 1, x.shape[-1] - 1))
        if add_time:
            x = self.add_time(x)
        return x


    def tangent_relu(self, x: torch.Tensor) -> torch.Tensor:
        """ Implements ReLU activation in tangent space. """
        return self.expmap0(torch.relu(self.logmap0(x)))
