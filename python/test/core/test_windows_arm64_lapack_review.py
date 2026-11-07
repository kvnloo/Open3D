# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2026 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

"""CPU regression cases for the native Windows ARM64 LAPACK fallback.

These checks run on all CPU builds to compare the Eigen fallback against
the existing LAPACKE implementation without requiring optional GPU modules.
"""

import numpy as np
import open3d.core as o3c
import pytest


@pytest.mark.parametrize("dtype,atol", [
    (o3c.float32, 1e-5),
    (o3c.float64, 1e-12),
])
def test_arm64_pivoted_inverse_cpu(dtype, atol):
    """Check inversion of a matrix requiring a nontrivial row pivot."""
    a = o3c.Tensor([[0.0, 2.0], [1.0, 3.0]],
                   dtype=dtype,
                   device=o3c.Device("CPU:0"))

    # det(A) = -2; the exact inverse is [-3/2, 1; 1/2, 0].
    expected = [[-1.5, 1.0], [0.5, 0.0]]
    np.testing.assert_allclose(o3c.inv(a).cpu().numpy(),
                               expected,
                               rtol=0,
                               atol=atol)


@pytest.mark.parametrize("dtype,atol", [
    (o3c.float32, 1e-5),
    (o3c.float64, 1e-12),
])
def test_arm64_multiple_rhs_solve_cpu(dtype, atol):
    """Check simultaneous solutions of two independent right-hand sides."""
    a = o3c.Tensor([[3.0, 1.0], [1.0, 2.0]],
                   dtype=dtype,
                   device=o3c.Device("CPU:0"))
    b = o3c.Tensor([[9.0, 18.0], [8.0, 16.0]],
                   dtype=dtype,
                   device=o3c.Device("CPU:0"))

    # A * [2, 3] = [9, 8]; A * [4, 6] = [18, 16].
    expected = [[2.0, 4.0], [3.0, 6.0]]
    np.testing.assert_allclose(o3c.solve(a, b).cpu().numpy(),
                               expected,
                               rtol=0,
                               atol=atol)


@pytest.mark.parametrize("dtype,atol", [
    (o3c.float32, 1e-5),
    (o3c.float64, 1e-12),
])
def test_arm64_overdetermined_least_squares_cpu(dtype, atol):
    """Check a full-rank least-squares fit with a nonzero residual."""
    a = o3c.Tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
                   dtype=dtype,
                   device=o3c.Device("CPU:0"))
    b = o3c.Tensor([1.0, 2.0, 4.0],
                   dtype=dtype,
                   device=o3c.Device("CPU:0"))

    # (A.T A)x = A.T b gives [2 1; 1 2]x = [5, 6].
    expected = [4.0 / 3.0, 7.0 / 3.0]
    np.testing.assert_allclose(o3c.lstsq(a, b).cpu().numpy(),
                               expected,
                               rtol=0,
                               atol=atol)
