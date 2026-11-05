# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2026 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import numpy as np
import open3d as o3d
import pytest


def _two_triangle_mesh(labeled=True):
    """Build two distinct faces with a stable material-ID order."""
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [2, 0, 0], [2, 1, 0]])
    mesh.triangles = o3d.utility.Vector3iVector([[0, 1, 2], [1, 3, 4]])
    if labeled:
        mesh.triangle_material_ids = o3d.utility.IntVector([11, 19])
        assert mesh.has_triangle_material_ids()
    return mesh


def _assert_face_ids_unchanged(source, result, expected):
    """Verify unchanged face order and one material ID per source face."""
    np.testing.assert_array_equal(np.asarray(result.triangles),
                                  np.asarray(source.triangles))
    assert result.has_triangle_material_ids()
    np.testing.assert_array_equal(np.asarray(result.triangle_material_ids),
                                  expected)


@pytest.mark.parametrize("iterations", [0, 1, 2])
def test_sharpen_preserves_triangle_material_ids(iterations):
    """Sharpening must not detach IDs from its unchanged two faces."""
    mesh = _two_triangle_mesh()
    result = mesh.filter_sharpen(number_of_iterations=iterations,
                                 strength=0.25)

    _assert_face_ids_unchanged(mesh, result, [11, 19])
    np.testing.assert_array_equal(np.asarray(mesh.triangle_material_ids),
                                  [11, 19])
    if iterations == 1:
        # Vertex 0 has neighbors (1, 0, 0), (0, 1, 0).
        np.testing.assert_allclose(np.asarray(result.vertices)[0],
                                   [-0.25, -0.25, 0],
                                   rtol=0,
                                   atol=1e-12)


@pytest.mark.parametrize("iterations", [0, 1])
def test_sharpen_does_not_invent_material_ids(iterations):
    """Unlabeled triangles stay unlabeled, including at zero iterations."""
    mesh = _two_triangle_mesh(labeled=False)
    result = mesh.filter_sharpen(number_of_iterations=iterations,
                                 strength=0.25)

    np.testing.assert_array_equal(np.asarray(result.triangles),
                                  np.asarray(mesh.triangles))
    assert len(result.triangle_material_ids) == 0
    assert not result.has_triangle_material_ids()


def test_sharpen_empty_mesh_keeps_empty_material_ids():
    """An empty mesh stays empty after sharpening."""
    result = o3d.geometry.TriangleMesh().filter_sharpen(
        number_of_iterations=1, strength=0.25)
    assert len(result.triangles) == 0
    assert len(result.triangle_material_ids) == 0
    assert not result.has_triangle_material_ids()


@pytest.mark.parametrize("iterations", [0, 1])
def test_arap_preserves_triangle_material_ids(iterations):
    """ARAP vertex movement must preserve all four tetrahedron face IDs."""
    mesh = o3d.geometry.TriangleMesh.create_tetrahedron()
    expected_ids = [2, 5, 7, 11]
    assert len(mesh.triangles) == 4
    mesh.triangle_material_ids = o3d.utility.IntVector(expected_ids)
    original = np.asarray(mesh.vertices).copy()
    translation = np.array([0.25, -0.5, 1.0])
    constrained = original + translation

    result = mesh.deform_as_rigid_as_possible(
        o3d.utility.IntVector(list(range(len(original)))),
        o3d.utility.Vector3dVector(constrained),
        max_iter=iterations)

    _assert_face_ids_unchanged(mesh, result, expected_ids)
    expected_vertices = original if iterations == 0 else constrained
    np.testing.assert_allclose(np.asarray(result.vertices),
                               expected_vertices,
                               rtol=0,
                               atol=1e-8)


def test_arap_does_not_invent_material_ids():
    """ARAP does not create face labels on an unlabeled tetrahedron."""
    mesh = o3d.geometry.TriangleMesh.create_tetrahedron()
    vertices = np.asarray(mesh.vertices)
    result = mesh.deform_as_rigid_as_possible(
        o3d.utility.IntVector(list(range(len(vertices)))),
        o3d.utility.Vector3dVector(vertices.copy()),
        max_iter=0)

    np.testing.assert_array_equal(np.asarray(result.triangles),
                                  np.asarray(mesh.triangles))
    assert len(result.triangle_material_ids) == 0
    assert not result.has_triangle_material_ids()


def test_selection_crop_and_normal_recompute_already_keep_material_ids():
    """Existing same-topology paths need no additional material-ID fix."""
    mesh = _two_triangle_mesh()
    mesh.compute_triangle_normals()
    np.testing.assert_array_equal(np.asarray(mesh.triangle_material_ids),
                                  [11, 19])

    selected = mesh.select_by_index([0, 1, 2, 3, 4], cleanup=False)
    cropped = mesh.crop(
        o3d.geometry.AxisAlignedBoundingBox([-1, -1, -1], [3, 2, 1]))
    _assert_face_ids_unchanged(mesh, selected, [11, 19])
    _assert_face_ids_unchanged(mesh, cropped, [11, 19])
