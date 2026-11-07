# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2026 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

"""CPU regression cases for real-point voxel reductions and Python bindings."""

import numpy as np
import open3d as o3d


def _cloud(points, colors):
    """Construct point/color pairs without requiring any external fixtures."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(
        np.asarray(points, dtype=np.float64))
    pcd.colors = o3d.utility.Vector3dVector(
        np.asarray(colors, dtype=np.float64))
    return pcd


def _triad():
    """Use asymmetric positions so center and centroid choose different points."""
    return _cloud(
        [[0.5] * 3, [0.6] * 3, [0.9] * 3],
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    )


def _sorted_points_and_colors(pcd):
    """Sort by the distinct x values because voxel output order is unspecified."""
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    order = np.argsort(points[:, 0])
    return points[order], colors[order]


def test_voxel_default_averaging_is_unchanged():
    """The positional-only call must retain the pre-feature mean behavior."""
    result = _triad().voxel_down_sample(1.0)
    np.testing.assert_allclose(
        np.asarray(result.points), [[2.0 / 3.0] * 3],
        rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        np.asarray(result.colors), [[1.0 / 3.0] * 3],
        rtol=0, atol=1e-12)


def test_voxel_nearest_modes_disagree_and_copy_real_attributes():
    """Center is the first point; centroid-nearest is the middle point."""
    reduction = o3d.geometry.PointCloud.VoxelReduction
    for mode, point, color in (
        (reduction.NearestToCentroid, [0.6] * 3, [0, 1, 0]),
        (reduction.NearestToCenter, [0.5] * 3, [1, 0, 0]),
    ):
        result = _triad().voxel_down_sample(1.0, mode)
        np.testing.assert_allclose(
            np.asarray(result.points), [point], rtol=0, atol=1e-12)
        np.testing.assert_allclose(
            np.asarray(result.colors), [color], rtol=0, atol=1e-12)


def test_voxel_nearest_winners_are_independent_per_voxel():
    """Each occupied x-voxel must select its own input point and color."""
    pcd = _cloud(
        [[x, 0.5, 0.5] for x in (0.5, 0.7, 0.9, 1.5, 1.7, 1.9)],
        [[1, 0, 0], [0, 1, 0], [0, 0, 1],
         [1, 1, 0], [1, 0, 1], [0, 1, 1]],
    )
    reduction = o3d.geometry.PointCloud.VoxelReduction
    for mode, expected_x, expected_colors in (
        (reduction.NearestToCentroid, [0.7, 1.7],
         [[0, 1, 0], [1, 0, 1]]),
        (reduction.NearestToCenter, [0.5, 1.5],
         [[1, 0, 0], [1, 1, 0]]),
    ):
        points, colors = _sorted_points_and_colors(
            pcd.voxel_down_sample(1.0, mode))
        np.testing.assert_allclose(
            points, [[x, 0.5, 0.5] for x in expected_x],
            rtol=0, atol=1e-12)
        np.testing.assert_allclose(
            colors, expected_colors, rtol=0, atol=1e-12)


def test_voxel_singleton_preserves_point_and_color_in_all_modes():
    """No mode may deform or recolor a voxel containing one observation."""
    pcd = _cloud([[5, 5, 5]], [[0.25, 0.5, 0.75]])
    reduction = o3d.geometry.PointCloud.VoxelReduction
    for mode in (reduction.Centroid,
                 reduction.NearestToCentroid,
                 reduction.NearestToCenter):
        result = pcd.voxel_down_sample(1.0, mode)
        np.testing.assert_allclose(
            np.asarray(result.points), [[5, 5, 5]],
            rtol=0, atol=1e-12)
        np.testing.assert_allclose(
            np.asarray(result.colors), [[0.25, 0.5, 0.75]],
            rtol=0, atol=1e-12)
