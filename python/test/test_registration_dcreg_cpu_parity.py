# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2026 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

"""CPU-only parity checks for DCReg's target/world degeneracy frame."""

import numpy as np
import open3d as o3d


def _cloud(points, normals):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.normals = o3d.utility.Vector3dVector(normals)
    return cloud


def test_dcreg_cylinder_axis_permutation_parity():
    # Point-to-plane normals cannot constrain translation or rotation along
    # the cylinder axis, even when that axis is not world z.
    points, normals = [], []
    for angle in np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False):
        x, y = np.cos(angle), np.sin(angle)
        for z in np.linspace(-0.6, 0.6, 7):
            points.append([x, y, z])
            normals.append([x, y, 0.0])
    points = np.asarray(points, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    shift = np.array([-0.12, 0.05, -0.35])
    permutation = np.array([[0.0, 0.0, 1.0],
                            [1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0]])
    correspondences = o3d.utility.Vector2iVector(
        np.asarray([[i, i] for i in range(len(points))], dtype=np.int32))
    registration = o3d.pipelines.registration

    results = []
    for rotation in (np.eye(3), permutation):
        target = _cloud(points @ rotation.T, normals @ rotation.T)
        source = _cloud((points + shift) @ rotation.T, normals @ rotation.T)
        delta = registration.TransformationEstimationPointToPlaneDCReg(
        ).compute_transformation(source, target, correspondences)
        analysis = registration.compute_dcreg_degeneracy_analysis(
            source, target, correspondences)
        assert np.isfinite(delta).all()
        assert analysis.is_rank_deficient
        assert "target/world" in analysis.coordinate_frame
        results.append((delta, analysis))

    reference, z_analysis = results[0]
    rotated, x_analysis = results[1]
    np.testing.assert_allclose(reference[:3, :3], np.eye(3), atol=1e-8)
    np.testing.assert_allclose(reference[:3, 3], [0.12, -0.05, 0.0],
                               atol=1e-8)
    np.testing.assert_allclose(rotated[:3, :3],
                               permutation @ reference[:3, :3] @
                               permutation.T, atol=1e-8)
    np.testing.assert_allclose(rotated[:3, 3],
                               permutation @ reference[:3, 3], atol=1e-8)
    np.testing.assert_allclose(rotated[:3, 3], [0.0, 0.12, -0.05],
                               atol=1e-8)
    np.testing.assert_array_equal(z_analysis.weak_rotation_axes, [0, 0, 1])
    np.testing.assert_array_equal(z_analysis.weak_translation_axes, [0, 0, 1])
    np.testing.assert_array_equal(x_analysis.weak_rotation_axes, [1, 0, 0])
    np.testing.assert_array_equal(x_analysis.weak_translation_axes, [1, 0, 0])
