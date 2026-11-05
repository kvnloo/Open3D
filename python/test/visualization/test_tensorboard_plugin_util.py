# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2026 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------
"""CPU regressions for TensorBoard visualization dtype normalization."""

import numpy as np
import open3d as o3d
import pytest

from open3d.visualization.tensorboard_plugin.util import _normalize


@pytest.mark.parametrize("dtype", [o3d.core.uint8, o3d.core.int32])
def test_normalize_integer_open3d_tensor(dtype):
    """Custom RGB attributes with integer dtypes must not raise TypeError."""
    colors = [[0, 128, 255], [20, 40, 60]]
    tensor = o3d.core.Tensor(colors,
                             dtype=dtype,
                             device=o3d.core.Device("CPU:0"))

    normalized, minimum, maximum = _normalize(tensor)

    assert normalized is tensor
    assert (minimum, maximum) == (0, 1)
    np.testing.assert_array_equal(normalized.numpy(), colors)


@pytest.mark.parametrize("dtype", [o3d.core.float32, o3d.core.float64])
def test_normalize_float_open3d_tensor(dtype):
    tensor = o3d.core.Tensor([2.0, 4.0, 6.0],
                             dtype=dtype,
                             device=o3d.core.Device("CPU:0"))

    normalized, minimum, maximum = _normalize(tensor)

    assert (minimum, maximum) == (2.0, 6.0)
    np.testing.assert_allclose(normalized.numpy(), [0.0, 0.5, 1.0],
                               rtol=0,
                               atol=1e-6)


def test_normalize_integer_numpy_array():
    colors = np.array([[0, 128, 255]], dtype=np.uint8)

    normalized, minimum, maximum = _normalize(colors)

    assert normalized is colors
    assert (minimum, maximum) == (0, 1)
    np.testing.assert_array_equal(normalized, colors)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_normalize_float_numpy_array(dtype):
    values = np.array([2.0, 4.0, 6.0], dtype=dtype)

    normalized, minimum, maximum = _normalize(values)

    assert (minimum, maximum) == (2.0, 6.0)
    np.testing.assert_allclose(normalized, [0.0, 0.5, 1.0],
                               rtol=0,
                               atol=1e-6)
