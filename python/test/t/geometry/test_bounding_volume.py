# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2026 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import numpy as np
import open3d as o3d
import open3d.core as o3c
import pytest


@pytest.mark.parametrize("dtype", [o3c.float32, o3c.float64])
def test_axis_aligned_bounding_box_copy_with_mixed_sign_bounds(dtype):
    """Copy and clone preserve valid bounds even when max x is negative."""
    device = o3c.Device("CPU:0")
    min_bound = o3c.Tensor([-2.0, 1.0, -1.0], dtype, device)
    max_bound = o3c.Tensor([-0.5, 2.0, 0.5], dtype, device)
    color = o3c.Tensor([0.25, 0.5, 0.75], dtype, device)
    box = o3d.t.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    box.set_color(color)

    for copied in (box.to(device, copy=True), box.clone()):
        assert copied.dtype == dtype
        np.testing.assert_array_equal(copied.min_bound.cpu().numpy(),
                                      min_bound.numpy())
        np.testing.assert_array_equal(copied.max_bound.cpu().numpy(),
                                      max_bound.numpy())
        np.testing.assert_array_equal(copied.color.cpu().numpy(),
                                      color.numpy())
        assert copied.volume() == pytest.approx(2.25)
