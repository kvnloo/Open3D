# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2026 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

from pathlib import Path
import runpy

import numpy as np
import pytest


def test_pose_estimation_rejects_missing_depth():
    pytest.importorskip("cv2")
    pytest.importorskip("matplotlib")
    path = Path(__file__).resolve().parents[
        2] / "examples/python/reconstruction_system/opencv_pose_estimation.py"
    pose = runpy.run_path(str(path))
    import open3d as o3d
    rng = np.random.default_rng(24)
    gray = rng.random((144, 192)).astype(np.float32)
    frame = o3d.geometry.RGBDImage()
    frame.color = o3d.geometry.Image(gray)
    frame.depth = o3d.geometry.Image(np.zeros_like(gray))
    intrinsic = o3d.camera.PinholeCameraIntrinsic(192, 144, 130., 130., 96.,
                                                  72.)
    success, transformation = pose['pose_estimation'](frame, frame, intrinsic,
                                                      False)
    assert not success
    np.testing.assert_array_equal(transformation, np.eye(4))

@pytest.fixture(scope="module")
def xyz_from_pts():
    """Load the real CPU depth projection helper from the reconstruction example."""
    path = (Path(__file__).resolve().parents[2] /
            "examples/python/reconstruction_system/opencv_pose_estimation.py")
    return runpy.run_path(str(path))["get_xyz_from_pts"]


@pytest.mark.parametrize("invalid_depth", [
    pytest.param(0.0, id="zero"),
    pytest.param(-2.0, id="negative"),
    pytest.param(float("nan"), id="nan"),
    pytest.param(float("inf"), id="infinite"),
])
def test_bilinear_depth_rejects_invalid_neighbor(xyz_from_pts, invalid_depth):
    """A missing depth neighbor must not yield a usable 3D correspondence."""
    depth = np.full((4, 4), 2.0, dtype=np.float32)
    depth[1, 1] = invalid_depth

    xyz = xyz_from_pts([1.5, 1.5], depth, 1.0, 1.0, 2.0)

    # Pose estimation accepts a projected point only with finite positive Z.
    assert not (np.isfinite(xyz[2]) and xyz[2] > 0.0)


def test_bilinear_depth_preserves_valid_gradient(xyz_from_pts):
    """Four valid neighbors still use ordinary bilinear interpolation."""
    depth = np.full((4, 4), 2.0, dtype=np.float32)
    depth[1, 2] = 4.0
    depth[2, 1] = 6.0
    depth[2, 2] = 8.0

    xyz = xyz_from_pts([1.5, 1.5], depth, 1.0, 1.0, 2.0)

    np.testing.assert_allclose(xyz, [1.25, 1.25, 5.0], rtol=0, atol=0)
