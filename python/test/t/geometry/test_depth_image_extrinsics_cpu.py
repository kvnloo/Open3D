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


@pytest.mark.parametrize("depth_dtype", [np.uint16, np.float32])
@pytest.mark.parametrize("extrinsics_dtype", [np.float32, np.float64])
def test_depth_image_with_normals_translated_extrinsics_cpu(
        depth_dtype, extrinsics_dtype):
    """Non-identity extrinsics must transform Float32 vertices without a dtype error."""
    # Uniform depth of 1000 / 1000 = 1 meter gives valid interior normals.
    depth = o3d.t.geometry.Image(
        o3c.Tensor(np.full((8, 8), 1000, dtype=depth_dtype)))
    intrinsics = o3c.Tensor(np.eye(3, dtype=np.float32))

    identity = np.eye(4, dtype=extrinsics_dtype)
    translated = identity.copy()
    translated[0, 3] = 1.0  # World-to-camera, so camera-to-world subtracts 1.

    reference = o3d.t.geometry.PointCloud.create_from_depth_image(
        depth,
        intrinsics,
        o3c.Tensor(identity),
        depth_scale=1000.0,
        depth_max=3.0,
        with_normals=True)
    result = o3d.t.geometry.PointCloud.create_from_depth_image(
        depth,
        intrinsics,
        o3c.Tensor(translated),
        depth_scale=1000.0,
        depth_max=3.0,
        with_normals=True)

    ref_positions = reference.point.positions.numpy()
    result_positions = result.point.positions.numpy()
    assert len(ref_positions) > 0
    assert result_positions.shape == ref_positions.shape
    np.testing.assert_allclose(ref_positions[:, 2], 1.0, atol=1e-6)
    np.testing.assert_allclose(
        result_positions,
        ref_positions + np.array([-1.0, 0.0, 0.0], dtype=np.float32),
        rtol=1e-6,
        atol=1e-5)
    np.testing.assert_allclose(
        result.point.normals.numpy(),
        reference.point.normals.numpy(),
        rtol=1e-5,
        atol=1e-5)
