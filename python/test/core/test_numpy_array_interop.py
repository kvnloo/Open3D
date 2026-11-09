# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2026 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import numpy as np
import open3d.core as o3c
import pytest


def test_numpy_asarray_cpu_tensor_returns_numeric_view():
    source = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    tensor = o3c.Tensor(source, device=o3c.Device("CPU:0"))

    result = np.asarray(tensor)

    assert result.shape == (2, 3)
    assert result.dtype == np.float32
    np.testing.assert_array_equal(result, source)
    result[1, 2] = 7.5
    assert tensor[1, 2].item() == 7.5


def test_numpy_asarray_cpu_tensor_cast_does_not_alias():
    source = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    tensor = o3c.Tensor(source, device=o3c.Device("CPU:0"))

    result = np.asarray(tensor, dtype=np.float64)

    assert result.dtype == np.float64
    np.testing.assert_array_equal(result, source)
    result[0, 0] = 11.5
    assert tensor[0, 0].item() == 1.0


def test_numpy_array_cpu_tensor_copy_is_independent():
    source = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    tensor = o3c.Tensor(source, device=o3c.Device("CPU:0"))

    result = np.array(tensor, copy=True)

    assert result.dtype == np.float32
    np.testing.assert_array_equal(result, source)
    result[0, 1] = 99.0
    assert tensor[0, 1].item() == 2.0


def test_numpy_array_cpu_tensor_rejects_nocopy_dtype_change():
    tensor = o3c.Tensor([1.0, 2.0], dtype=o3c.float32,
                        device=o3c.Device("CPU:0"))

    with pytest.raises(ValueError, match="copy=False"):
        tensor.__array__(dtype=np.float64, copy=False)
