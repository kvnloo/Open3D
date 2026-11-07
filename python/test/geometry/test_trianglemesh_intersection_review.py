# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2026 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

"""CPU regression controls for triangle intersection issue #5117."""

import numpy as np
import open3d as o3d


P = np.array([[0.0, 0.13918686, 1.0],
              [0.0, 0.0, 1.1270161],
              [1.0, 0.0, 1.0284119]])
Q = np.array([[1.0, 1.1269569, 0.0],
              [1.0, 0.03113556, 1.0],
              [2.0, 1.0189056, 0.0]])


def make_mesh(vertices):
    triangles = np.array([[0, 1, 2], [3, 4, 5]])
    return o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices),
        o3d.utility.Vector3iVector(triangles))


def test_issue5117_separated():
    mesh = make_mesh(np.vstack([P, Q]))
    assert not mesh.is_self_intersecting()
    assert len(mesh.get_self_intersecting_triangles()) == 0


def test_issue5117_rotated():
    mesh = make_mesh(np.vstack([P, Q]))
    c = np.sqrt(0.5)
    rotation = np.array([[1.0, 0.0, 0.0],
                         [0.0, c, c],
                         [0.0, -c, c]])
    mesh.rotate(rotation, center=(0.0, 0.0, 0.0))
    assert not mesh.is_self_intersecting()
    assert len(mesh.get_self_intersecting_triangles()) == 0


def test_coplanar_overlap_control():
    triangle = np.array([[0.0, 0.0, 0.0],
                         [1.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0]])
    mesh = make_mesh(np.vstack([triangle, triangle]))
    assert mesh.is_self_intersecting()
    assert len(mesh.get_self_intersecting_triangles()) >= 1
