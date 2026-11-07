# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2026 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import numpy as np
import open3d as o3d


def _colored_triangle():
    """A CPU TriangleMesh with a face and all common vertex attributes."""
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                     dtype=np.float64)),
        o3d.utility.Vector3iVector(np.array([[0, 1, 2]], dtype=np.int32)))
    mesh.vertex_normals = o3d.utility.Vector3dVector(
        np.array([[0.0, 0.0, 1.0]] * 3, dtype=np.float64))
    mesh.vertex_colors = o3d.utility.Vector3dVector(
        np.eye(3, dtype=np.float64))
    mesh.triangle_normals = o3d.utility.Vector3dVector(
        np.array([[0.0, 0.0, 1.0]], dtype=np.float64))
    mesh.triangle_uvs = o3d.utility.Vector2dVector(
        np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
                 dtype=np.float64))
    return mesh


def test_triangle_mesh_copy_preserves_attributes():
    original = _colored_triangle()
    copied = o3d.geometry.TriangleMesh(original)

    assert copied.has_vertex_normals()
    assert copied.has_vertex_colors()
    assert copied.has_triangle_normals()
    assert copied.has_triangle_uvs()
    for name in ("vertices", "vertex_normals", "vertex_colors", "triangles",
                 "triangle_normals", "triangle_uvs"):
        np.testing.assert_array_equal(
            np.asarray(getattr(copied, name)),
            np.asarray(getattr(original, name)))

    # Reassigning the source data must not alter the copied colors.
    original.vertex_colors = o3d.utility.Vector3dVector(
        np.zeros((3, 3), dtype=np.float64))
    np.testing.assert_array_equal(
        np.asarray(copied.vertex_colors), np.eye(3, dtype=np.float64))


def test_triangle_mesh_equality_compares_oriented_geometry():
    original = _colored_triangle()
    same = _colored_triangle()
    assert original == same

    # Attributes do not participate in the reviewed PR's equality operator.
    same.vertex_colors = o3d.utility.Vector3dVector(
        np.zeros((3, 3), dtype=np.float64))
    assert original == same

    same.triangles = o3d.utility.Vector3iVector(
        np.array([[0, 2, 1]], dtype=np.int32))
    assert not (original == same)

    same.triangles = o3d.utility.Vector3iVector(
        np.array([[0, 1, 2]], dtype=np.int32))
    same.vertices = o3d.utility.Vector3dVector(
        np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                 dtype=np.float64))
    assert not (original == same)


def _mesh(vertices, triangles):
    """Construct legacy CPU geometry without external data or a renderer."""
    return o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.array(vertices, dtype=np.float64)),
        o3d.utility.Vector3iVector(np.array(triangles, dtype=np.int32)))


def test_triangle_mesh_equality_reindexed_vertices_and_faces():
    mesh = _mesh([[0, 0, 0], [2, 0, 0], [0, 3, 0], [0, 0, 4]],
                 [[0, 1, 2], [0, 2, 3]])
    reordered = _mesh([[0, 3, 0], [0, 0, 4], [2, 0, 0], [0, 0, 0]],
                      [[3, 0, 1], [2, 0, 3]])
    assert mesh == reordered
    assert reordered == mesh


def test_triangle_mesh_equality_duplicate_vertices_and_winding():
    mesh = _mesh([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 0]],
                 [[0, 1, 2], [3, 1, 2]])
    reordered = _mesh([[0, 0, 0], [0, 1, 0], [1, 0, 0], [0, 0, 0]],
                      [[3, 2, 1], [0, 2, 1]])
    assert mesh == reordered

    # Coincident vertex IDs are not individually distinguished.
    reordered.triangles = o3d.utility.Vector3iVector(
        np.array([[0, 2, 1], [0, 2, 1]], dtype=np.int32))
    assert mesh == reordered
    reordered.triangles = o3d.utility.Vector3iVector(
        np.array([[0, 2, 1], [0, 1, 2]], dtype=np.int32))
    assert not (mesh == reordered)

    missing_duplicate = _mesh([[0, 0, 0], [0, 1, 0], [1, 0, 0]],
                              [[0, 2, 1], [0, 2, 1]])
    assert not (mesh == missing_duplicate)


def test_triangle_mesh_equality_position_tolerance():
    mesh = _colored_triangle()
    near = _colored_triangle()
    vertices = np.asarray(near.vertices).copy()
    vertices[1, 0] = 1.0 + 1e-13
    near.vertices = o3d.utility.Vector3dVector(vertices)
    assert mesh == near

    vertices[1, 0] = 1.0 + 1e-6
    near.vertices = o3d.utility.Vector3dVector(vertices)
    assert not (mesh == near)

    vertices[1, 0] = 1.0
    vertices[0, 2] = 1e-13
    near.vertices = o3d.utility.Vector3dVector(vertices)
    assert not (mesh == near)


def test_triangle_mesh_equality_ignores_appearance_attributes():
    mesh = _colored_triangle()
    appearance = _colored_triangle()
    appearance.vertex_colors = o3d.utility.Vector3dVector(
        np.zeros((3, 3), dtype=np.float64))
    appearance.vertex_normals = o3d.utility.Vector3dVector(
        np.zeros((3, 3), dtype=np.float64))
    appearance.triangle_normals = o3d.utility.Vector3dVector(
        np.zeros((1, 3), dtype=np.float64))
    appearance.triangle_uvs = o3d.utility.Vector2dVector(
        np.zeros((3, 2), dtype=np.float64))
    assert mesh == appearance
    assert appearance == mesh


def test_triangle_mesh_equality_rotates_coincident_face_corners():
    mesh = _mesh([[0, 0, 0], [0, 0, 0], [1, 0, 0]], [[0, 1, 2]])
    rotated = _mesh([[0, 0, 0], [0, 0, 0], [1, 0, 0]], [[1, 2, 0]])
    assert mesh == rotated
    assert rotated == mesh
