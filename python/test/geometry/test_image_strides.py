# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2026 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import numpy as np
import open3d as o3d
import pytest


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32])
def test_image_strided_grayscale_buffer(dtype):
    data = np.arange(24, dtype=dtype).reshape(4, 6)
    views = (
        data[::-1],
        data[:, ::-1],
        data[:, ::2],
        data.T,
        np.asfortranarray(data),
    )
    for view in views:
        assert not view.flags.c_contiguous
        image = o3d.geometry.Image(view)
        np.testing.assert_array_equal(np.asarray(image), view)


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32])
def test_image_strided_multichannel_buffer(dtype):
    data = np.arange(72, dtype=dtype).reshape(4, 6, 3)
    views = (
        data[::-1],
        data[:, ::-1],
        data[:, :, ::-1],
        data.transpose(1, 0, 2),
        np.asfortranarray(data),
        data[::-1, ::-2, ::-1],
    )
    for view in views:
        assert not view.flags.c_contiguous
        image = o3d.geometry.Image(view)
        np.testing.assert_array_equal(np.asarray(image), view)


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32])
def test_image_contiguous_buffer_unchanged(dtype):
    for shape in ((4, 6), (4, 6, 3)):
        data = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
        assert data.flags.c_contiguous
        image = o3d.geometry.Image(data)
        np.testing.assert_array_equal(np.asarray(image), data)
