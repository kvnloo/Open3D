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


def test_sparse_indices_duplicate_last_color_wins():
    grid = _sparse_grid([[1, 0, 0], [1, 0, 0]],
                        [[1, 0, 0], [0, 0, 1]], [0, 0, 0], 1.0)
    voxels = grid.get_voxels()
    assert len(voxels) == 1
    np.testing.assert_array_equal(voxels[0].grid_index, [1, 0, 0])
    np.testing.assert_allclose(voxels[0].color, [0, 0, 1])
    np.testing.assert_allclose(
        grid.get_voxel_center_coordinate(np.array([1, 0, 0])),
        [1.5, 0.5, 0.5])


@pytest.mark.parametrize("voxel_size", [0.0, -1.0])
def test_sparse_indices_reject_nonpositive_voxel_size(voxel_size):
    with pytest.raises(RuntimeError, match="voxel_size"):
        _sparse_grid([[0, 0, 0]], [[1, 0, 0]], [0, 0, 0], voxel_size)
