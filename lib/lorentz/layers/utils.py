import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.geoopt.tensor import ManifoldParameter


def cayley_map(X: torch.Tensor) -> torch.Tensor:
    n, k = X.size(-2), X.size(-1)
    transposed = n < k
    if transposed:
        X = X.mT
        n, k = k, n
    # Here n > k and X is a tall matrix

    # We just need n x k - k(k-1)/2 parameters
    X = X.tril()
    if n != k:
        # Embed into a square matrix
        X = torch.cat([X, X.new_zeros(n, n - k).expand(*X.shape[:-2], -1, -1)], dim=-1)
    A = X - X.mH

    # Computes the Cayley retraction (I+A/2)(I-A/2)^{-1}
    Id = torch.eye(n, dtype=A.dtype, device=A.device)
    Q = torch.linalg.solve(torch.add(Id, A, alpha=-0.5), torch.add(Id, A, alpha=0.5))
    # Q is now orthogonal (or unitary) of size (..., n, n)
    if n != k:
        Q = Q[..., :k]
    # Q is now the size of the X (albeit perhaps transposed)
    if transposed:
        Q = Q.mT
    return Q  # type: ignore[possibly-undefined]



class CustomWeightConv2d(nn.Conv2d):
    @staticmethod
    def __new__(cls,
                weight_manifold,
                weight_shape,
                *args,
                **kwargs
                ):
        return super().__new__(cls)

    def __init__(self, weight_manifold, weight_shape, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_manifold = weight_manifold
        self.weight_shape = weight_shape

        #temp_weight = self.weight_manifold.projx(torch.rand(self.weight_shape).reshape(self.weight_shape[0], -1) * 10)
        #self.weight = ManifoldParameter(temp_weight, manifold=self.weight_manifold)

    def forward(self, input):

        weight = self.weight.reshape(self.weight_shape)

        return F.conv2d(input,
                        weight,
                        self.bias,
                        self.stride,
                        self.padding,
                        self.dilation,
                        self.groups)


class HyperboleIt(nn.Module):
    def forward(self, w):
        d_out, d_in, k1, k2 = w.shape

        if d_out < d_in*k1*k2:
            return w

        w = w.reshape(d_out, -1)
        try:
            w = cayley_map(w)
        except:
            print("break")
        return w.reshape(d_out, k1, k2, d_in)


class ReshapeHyperbolicWeight(nn.Module):
    def __init__(self, w_size):
        super(ReshapeHyperbolicWeight, self).__init__()

        self.w_size = w_size

    def forward(self, w):
        d_out, d_in, k1, k2 = self.w_size

        if d_out < d_in * k1 * k2:
            return w

        return w.reshape(d_out, k1, k2, d_in)
