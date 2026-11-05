# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2026 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import numpy as np
import open3d as o3d
import pytest


def _labeled_mesh(triangles, material_ids):
    """Build a legacy mesh with distinct source-face material IDs."""
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0],
         [1, 1, 0], [2, 0, 0], [9, 9, 9]])
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.triangle_material_ids = o3d.utility.IntVector(material_ids)
    assert mesh.has_triangle_material_ids()
    return mesh


@pytest.mark.parametrize("method", [
    "filter_smooth_simple",
    "filter_smooth_laplacian",
    "filter_smooth_taubin",
])
@pytest.mark.parametrize("iterations", [0, 1, 2])
def test_smoothing_preserves_triangle_material_ids(method, iterations):
    """Smoothing only moves vertex data; face IDs must retain their order."""
    mesh = _labeled_mesh([[0, 1, 2], [1, 3, 2]], [11, 19])
    original_faces = np.asarray(mesh.triangles).copy()

    result = getattr(mesh, method)(iterations)

    assert result.has_triangle_material_ids()
    np.testing.assert_array_equal(np.asarray(result.triangles), original_faces)
    np.testing.assert_array_equal(np.asarray(result.triangle_material_ids),
                                  [11, 19])
    np.testing.assert_array_equal(np.asarray(mesh.triangle_material_ids),
                                  [11, 19])


@pytest.mark.parametrize("method", [
    "filter_smooth_simple",
    "filter_smooth_laplacian",
    "filter_smooth_taubin",
])
def test_smoothing_does_not_create_material_ids(method):
    """An unlabeled input stays unlabeled, including an empty mesh."""
    result = getattr(o3d.geometry.TriangleMesh(), method)(1)
    assert len(result.triangles) == 0
    assert len(result.triangle_material_ids) == 0
    assert not result.has_triangle_material_ids()


def test_duplicate_cleanup_keeps_first_face_material_id():
    """Cyclic duplicates lose their IDs; opposite winding remains distinct."""
    mesh = _labeled_mesh([[0, 1, 2], [1, 2, 0], [0, 2, 3], [0, 2, 1]],
                         [3, 99, 7, 13])
    mesh.remove_duplicated_triangles()

    assert mesh.has_triangle_material_ids()
    np.testing.assert_array_equal(np.asarray(mesh.triangles),
                                  [[0, 1, 2], [0, 2, 3], [0, 2, 1]])
    np.testing.assert_array_equal(np.asarray(mesh.triangle_material_ids),
                                  [3, 7, 13])


def test_degenerate_cleanup_keeps_surviving_material_ids():
    """Compaction must move the ID with each surviving source triangle."""
    mesh = _labeled_mesh([[0, 1, 2], [1, 1, 3], [0, 2, 3], [2, 2, 2]],
                         [31, 41, 59, 26])
    mesh.remove_degenerate_triangles()

    assert mesh.has_triangle_material_ids()
    np.testing.assert_array_equal(np.asarray(mesh.triangles),
                                  [[0, 1, 2], [0, 2, 3]])
    np.testing.assert_array_equal(np.asarray(mesh.triangle_material_ids),
                                  [31, 59])


def test_unreferenced_vertex_cleanup_leaves_face_ids_unchanged():
    """Vertex reindexing never reorders or removes triangles."""
    mesh = _labeled_mesh([[0, 1, 2], [0, 2, 3]], [11, 19])
    mesh.remove_unreferenced_vertices()

    assert len(mesh.vertices) == 4
    assert mesh.has_triangle_material_ids()
    np.testing.assert_array_equal(np.asarray(mesh.triangle_material_ids),
                                  [11, 19])


def test_chained_triangle_cleanup_keeps_aligned_ids():
    """Deduplication then degenerate cleanup retains original survivor IDs."""
    mesh = _labeled_mesh([[0, 1, 2], [1, 2, 0], [1, 1, 3], [0, 2, 3]],
                         [5, 55, 66, 9])
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()

    assert len(mesh.triangles) == 2
    assert mesh.has_triangle_material_ids()
    np.testing.assert_array_equal(np.asarray(mesh.triangle_material_ids),
                                  [5, 9])


def test_removing_all_degenerate_triangles_clears_ids():
    """No surviving faces means no surviving material IDs."""
    mesh = _labeled_mesh([[0, 0, 1], [2, 2, 2]], [4, 8])
    mesh.remove_degenerate_triangles()

    assert len(mesh.triangles) == 0
    assert len(mesh.triangle_material_ids) == 0
    assert not mesh.has_triangle_material_ids()
