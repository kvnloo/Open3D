# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2026 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import numpy as np
import open3d as o3d
import pytest


@pytest.mark.parametrize("method", ["subdivide_midpoint", "subdivide_loop"])
@pytest.mark.parametrize("iterations, child_count", [(0, 1), (1, 4), (2, 16)])
def test_subdivision_preserves_triangle_material_ids(method, iterations,
                                                     child_count):
    """Each subdivided triangle inherits its original material ID."""
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])
    mesh.triangles = o3d.utility.Vector3iVector([[0, 1, 2], [1, 3, 2]])
    mesh.triangle_material_ids = o3d.utility.IntVector([2, 5])

    assert mesh.has_triangle_material_ids()
    result = getattr(mesh, method)(iterations)
    expected = np.repeat([2, 5], child_count)

    assert len(result.triangles) == 2 * child_count
    assert result.has_triangle_material_ids()
    np.testing.assert_array_equal(np.asarray(result.triangle_material_ids),
                                  expected)
    np.testing.assert_array_equal(np.asarray(mesh.triangle_material_ids),
                                  [2, 5])


@pytest.mark.parametrize("method", ["subdivide_midpoint", "subdivide_loop"])
def test_subdivision_of_empty_triangle_mesh_has_no_material_ids(method):
    """Empty input retains an empty triangle/material schema."""
    mesh = o3d.geometry.TriangleMesh()
    result = getattr(mesh, method)(1)

    assert len(result.triangles) == 0
    assert len(result.triangle_material_ids) == 0
    assert not result.has_triangle_material_ids()
