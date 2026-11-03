# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2026 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import numpy as np
import open3d as o3d
import pytest


@pytest.mark.parametrize(("triangle_mask", "expected_ids"), [
    ([False, True, False], [2, 1]),
    ([True, False, True], [0]),
    ([True, True, True], []),
])
def test_remove_triangles_by_mask_preserves_material_ids(
        triangle_mask, expected_ids):
    """Keep material IDs aligned with the triangles left by a mask."""
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])
    mesh.triangles = o3d.utility.Vector3iVector(
        [[0, 1, 2], [1, 3, 2], [0, 2, 3]])
    mesh.triangle_material_ids = o3d.utility.IntVector([2, 0, 1])

    assert mesh.has_triangle_material_ids()
    mesh.remove_triangles_by_mask(triangle_mask)

    assert len(mesh.triangles) == len(expected_ids)
    assert np.asarray(mesh.triangle_material_ids).tolist() == expected_ids
    assert mesh.has_triangle_material_ids() == bool(expected_ids)
