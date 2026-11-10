# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2026 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import numpy as np
import open3d as o3d


def _oriented_sphere():
    """Construct a deterministic CPU point cloud with valid surface normals."""
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0,
                                                     resolution=12)
    sphere.compute_vertex_normals()
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = sphere.vertices
    point_cloud.normals = sphere.vertex_normals
    return point_cloud


def _reconstruct(point_cloud, depth, width):
    # A single Poisson worker keeps repeated reconstructions deterministic.
    return o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        point_cloud,
        depth=depth,
        width=width,
        scale=1.1,
        full_depth=3,
        n_threads=1)


def test_poisson_explicit_depth_ignores_width():
    """Width must not change reconstruction when tree depth is explicit."""
    point_cloud = _oriented_sphere()
    expected, expected_densities = _reconstruct(point_cloud, 6, 0.0)
    actual, actual_densities = _reconstruct(point_cloud, 6, 0.5)

    assert len(expected.vertices) > 0
    assert len(expected.triangles) > 0
    np.testing.assert_array_equal(np.asarray(actual.vertices),
                                  np.asarray(expected.vertices))
    np.testing.assert_array_equal(np.asarray(actual.triangles),
                                  np.asarray(expected.triangles))
    np.testing.assert_array_equal(np.asarray(actual_densities),
                                  np.asarray(expected_densities))


def test_poisson_zero_depth_keeps_width_selection():
    """Zero depth still permits selecting resolution through cell width."""
    point_cloud = _oriented_sphere()
    mesh, densities = _reconstruct(point_cloud, 0, 0.2)

    assert len(mesh.vertices) > 0
    assert len(mesh.triangles) > 0
    assert len(densities) == len(mesh.vertices)
