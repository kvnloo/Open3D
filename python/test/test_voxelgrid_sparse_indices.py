# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import numpy as np
import open3d as o3d
import pytest


def _sparse_grid(indices, colors, origin, voxel_size):
    return o3d.geometry.VoxelGrid.create_from_sparse_indices(
        o3d.utility.Vector3iVector(np.asarray(indices, dtype=np.int32)),
        o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64)),
        np.asarray(origin, dtype=np.float64),
        voxel_size,
    )


def test_sparse_indices_preserve_centers_and_colors():
    grid = _sparse_grid([[0, 0, 0], [2, 1, 3]],
                        [[1, 0, 0], [0, 1, 0]], [1, 2, -1], 0.5)
    voxels = {tuple(v.grid_index): np.asarray(v.color)
              for v in grid.get_voxels()}
    assert set(voxels) == {(0, 0, 0), (2, 1, 3)}
    np.testing.assert_allclose(voxels[(0, 0, 0)], [1, 0, 0])
    np.testing.assert_allclose(voxels[(2, 1, 3)], [0, 1, 0])
    np.testing.assert_allclose(
        grid.get_voxel_center_coordinate(np.array([0, 0, 0])),
        [1.25, 2.25, -0.75])
    np.testing.assert_allclose(
        grid.get_voxel_center_coordinate(np.array([2, 1, 3])),
        [2.25, 2.75, 0.75])


def test_sparse_indices_duplicate_colors_match_average_pooling():
    indices = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0]])
    colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0]])
    sparse = _sparse_grid(indices, colors, [0, 0, 0], 1.0)
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(indices.astype(float) + 0.5)
    cloud.colors = o3d.utility.Vector3dVector(colors.astype(float))
    pooled = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(
        cloud, 1.0, np.zeros(3), np.full(3, 3.0),
        o3d.geometry.VoxelGrid.VoxelPoolingMode.AVG)

    def colors_by_index(grid):
        return {tuple(voxel.grid_index): np.asarray(voxel.color)
                for voxel in grid.get_voxels()}

    sparse_colors = colors_by_index(sparse)
    pooled_colors = colors_by_index(pooled)
    assert set(sparse_colors) == set(pooled_colors) == {
        (1, 0, 0), (0, 1, 0)
    }
    np.testing.assert_allclose(sparse_colors[(1, 0, 0)], [1 / 3] * 3)
    np.testing.assert_allclose(sparse_colors[(0, 1, 0)], [0, 1, 0])
    for index in sparse_colors:
        np.testing.assert_allclose(sparse_colors[index], pooled_colors[index])

    reversed_grid = _sparse_grid(indices[::-1].copy(),
                                 colors[::-1].copy(), [0, 0, 0], 1.0)
    for index, color in colors_by_index(reversed_grid).items():
        np.testing.assert_allclose(color, sparse_colors[index])
    np.testing.assert_allclose(
        sparse.get_voxel_center_coordinate(np.array([1, 0, 0])),
        [1.5, 0.5, 0.5])


@pytest.mark.parametrize("voxel_size", [0.0, -1.0])
def test_sparse_indices_reject_nonpositive_voxel_size(voxel_size):
    with pytest.raises(RuntimeError, match="voxel_size"):
        _sparse_grid([[0, 0, 0]], [[1, 0, 0]], [0, 0, 0], voxel_size)
