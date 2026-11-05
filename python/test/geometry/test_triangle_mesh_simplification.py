# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2026 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import numpy as np
import open3d as o3d
import pytest


def _labeled_mesh(vertices, triangles, material_ids):
    """Create a legacy mesh with one material ID per source triangle."""
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.triangle_material_ids = o3d.utility.IntVector(material_ids)
    assert mesh.has_triangle_material_ids()
    return mesh


@pytest.mark.parametrize("contraction", [
    o3d.geometry.SimplificationContraction.Average,
    o3d.geometry.SimplificationContraction.Quadric,
])
def test_vertex_clustering_preserves_source_material_ids(contraction):
    """Duplicate clustered faces use the earliest source material ID."""
    mesh = _labeled_mesh(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0],
         [0.02, 0, 0], [1.02, 0, 0], [0, 1.02, 0],
         [10, 0, 0], [11, 0, 0], [10, 1, 0]],
        [[0, 1, 2], [3, 4, 5], [0, 3, 2], [6, 7, 8]],
        [11, 99, 777, 41])

    result = mesh.simplify_vertex_clustering(0.1, contraction=contraction)
    assert len(result.triangles) == 2
    assert result.has_triangle_material_ids()

    triangles = np.asarray(result.triangles)
    vertices = np.asarray(result.vertices)
    ids = np.asarray(result.triangle_material_ids)
    # Input faces 0 and 1 merge (first ID 11); face 2 degenerates.
    # Face 3 remains separate (ID 41). Output face order is unspecified.
    x_centers = vertices[triangles, 0].mean(axis=1)
    order = np.argsort(x_centers)
    assert x_centers[order[0]] < 5.0 < x_centers[order[1]]
    np.testing.assert_array_equal(ids[order], [11, 41])
    np.testing.assert_array_equal(
        np.asarray(mesh.triangle_material_ids), [11, 99, 777, 41])


@pytest.mark.parametrize(("target", "expected_ids"), [
    (2, [7, 29]),
    (1, [29]),
])
def test_quadric_decimation_preserves_surviving_material_ids(
        target, expected_ids):
    """Decimation keeps labels aligned when a smaller face is removed."""
    mesh = _labeled_mesh(
        [[0, 0, 0], [0.05, 0, 0], [0, 0.05, 0],
         [100, 0, 0], [105, 0, 0], [100, 5, 0]],
        [[0, 1, 2], [3, 4, 5]], [7, 29])

    result = mesh.simplify_quadric_decimation(target)
    assert len(result.triangles) == len(expected_ids)
    assert result.has_triangle_material_ids()
    np.testing.assert_array_equal(
        np.asarray(result.triangle_material_ids), expected_ids)
    np.testing.assert_array_equal(
        np.asarray(mesh.triangle_material_ids), [7, 29])


@pytest.mark.parametrize("method, kwargs", [
    ("simplify_vertex_clustering", {"voxel_size": 0.1}),
    ("simplify_vertex_clustering", {
        "voxel_size": 0.1,
        "contraction": o3d.geometry.SimplificationContraction.Quadric
    }),
    ("simplify_quadric_decimation", {"target_number_of_triangles": 0}),
])
def test_simplifying_empty_mesh_keeps_material_ids_empty(method, kwargs):
    """An empty input cannot acquire material IDs."""
    result = getattr(o3d.geometry.TriangleMesh(), method)(**kwargs)
    assert len(result.triangles) == 0
    assert len(result.triangle_material_ids) == 0
    assert not result.has_triangle_material_ids()
