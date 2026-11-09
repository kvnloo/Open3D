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

def test_numpy_array_protocol_accepts_positional_dtype():
    tensor = o3c.Tensor([1.0, 2.0, 3.0], dtype=o3c.float32,
                        device=o3c.Device("CPU:0"))

    # NumPy 1.x passes dtype positionally, without a copy keyword.
    result = tensor.__array__(np.float64)

    assert result.dtype == np.float64
    np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])
    result[0] = 9.0
    assert tensor[0].item() == 1.0


def test_numpy_array_protocol_accepts_copy_keyword():
    tensor = o3c.Tensor([1.0, 2.0, 3.0], dtype=o3c.float32,
                        device=o3c.Device("CPU:0"))

    # NumPy 2.x supplies copy; these direct calls also work with NumPy 1.x.
    view = tensor.__array__(dtype=np.float32, copy=False)
    view[1] = 8.0
    assert tensor[1].item() == 8.0

    copied = tensor.__array__(copy=True)
    copied[2] = 42.0
    assert tensor[2].item() == 3.0

    if int(np.__version__.split(".")[0]) >= 2:
        numpy_view = np.asarray(tensor, copy=False)
        numpy_view[0] = 7.0
        assert tensor[0].item() == 7.0


def test_numpy_asarray_cpu_tensor_strided_slice_preserves_values():
    source = np.arange(24, dtype=np.float32).reshape(4, 6)
    tensor = o3c.Tensor(source, device=o3c.Device("CPU:0"))
    sliced = tensor[::2, 1::2]

    result = np.asarray(sliced)

    assert result.shape == (2, 3)
    assert result.strides == (48, 8)
    np.testing.assert_array_equal(result, [[1, 3, 5], [13, 15, 17]])
    result[1, 1] = 123.0
    assert tensor[2, 3].item() == 123.0


def test_numpy_asarray_cpu_tensor_transpose_preserves_values():
    source = np.arange(12, dtype=np.float64).reshape(3, 4)
    tensor = o3c.Tensor.from_numpy(source.T)

    result = np.asarray(tensor)

    assert result.shape == (4, 3)
    assert result.strides == (8, 32)
    np.testing.assert_array_equal(result, source.T)
    result[2, 1] = -9.5
    assert source[1, 2] == -9.5
    assert tensor[2, 1].item() == -9.5
