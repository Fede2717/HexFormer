"""A5 regression test — symmetric Z validity mask.

Verifies that ``haa_diagnostics.compute_Z`` masks Q-K pairs that are extremely
close on the manifold (``norm_sq_QK ≤ 1e-6``), preventing the ~1e-5 denominator
explosion in the close-pair regime, while leaving well-separated pairs intact.
"""

import math
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from classification_vit.haa_diagnostics import compute_Z


def _lorentz_point(c_tilde: float, K: float,
                   spatial_axis: int = 1, dim: int = 4) -> torch.Tensor:
    """Build an exact Lorentz point [x_0, ..., x_d] at depth c̃ on H^{d}_{K}.

    Lies on the upper sheet: x_0 = sqrt(K)·cosh(c̃),
    spatial component on a single axis with magnitude sqrt(K)·sinh(c̃).
    Returns shape ``[1, 1, 1, dim]`` to match the (b, h, n, d) layout.
    """
    sqrt_K = math.sqrt(K)
    x = torch.zeros(1, 1, 1, dim, dtype=torch.float64)
    x[..., 0] = sqrt_K * math.cosh(c_tilde)
    x[..., spatial_axis] = sqrt_K * math.sinh(c_tilde)
    return x


def test_close_pair_is_masked():
    K = 1.0
    q = _lorentz_point(c_tilde=0.5, K=K)
    # Build k as a near-identical copy of q with a 1e-7 spatial perturbation,
    # which drives norm_sq_QK toward zero and would otherwise explode the
    # denominator (sqrt(1e-10 + 1e-12) ≈ 1e-5).
    k = q.clone()
    k[..., 1] += 1e-7
    Z = compute_Z(q, k, K)
    assert torch.isfinite(Z).all(), "Z must be finite (no NaN, no Inf)"
    assert Z.abs().max().item() == 0.0, (
        f"close pair must be masked to exactly 0, got {Z.flatten().tolist()}")


def test_normal_pair_is_finite_and_in_range():
    K = 1.0
    q = _lorentz_point(c_tilde=0.4, K=K, spatial_axis=1)
    k = _lorentz_point(c_tilde=1.2, K=K, spatial_axis=2)
    Z = compute_Z(q, k, K)
    assert torch.isfinite(Z).all(), "Z must be finite for a well-separated pair"
    assert (Z >= -1.0).all() and (Z <= 1.0).all(), (
        f"Z must lie in [-1, 1], got {Z.flatten().tolist()}")
    assert Z.abs().max().item() > 0.0, (
        "well-separated pair must not be masked to zero")


if __name__ == "__main__":
    test_close_pair_is_masked()
    test_normal_pair_is_finite_and_in_range()
    print("A5 unit tests OK")
