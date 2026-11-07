# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2026 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import importlib.util
from pathlib import Path

import numpy as np
import open3d as o3d
import pytest


@pytest.fixture
def comparison():
    """Load the example directly without registration, GUI or network data."""
    path = (Path(__file__).resolve().parents[3] /
            "examples/python/geometry/point_cloud_comparison.py")
    spec = importlib.util.spec_from_file_location("point_cloud_comparison",
                                                  path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cloud(points):
    """Construct deterministic CPU legacy point clouds."""
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(
        np.asarray(points, dtype=np.float64))
    return cloud


def test_identical_point_clouds_have_zero_distance_blue_heatmap(comparison):
    """Identical geometry should yield finite low-end colors, not NaNs."""
    points = [(0., 0., 0.), (1., 0., 0.), (0., 1., 0.)]
    cloud = _cloud(points)
    distances = comparison.compute_deviation_distances(cloud, cloud)
    np.testing.assert_allclose(distances, [0., 0., 0.], rtol=0., atol=1e-7)

    colors = comparison.compute_heatmap_colors(distances)
    assert np.all(np.isfinite(colors))
    np.testing.assert_allclose(colors, [[0., 0., 0.5]] * 3,
                               rtol=0., atol=1e-7)


def test_nonzero_point_cloud_distance_retains_jet_dynamic_range(comparison):
    """Cloud targets continue using nearest-neighbor distance coloring."""
    target = _cloud([(0., 0., 0.), (1., 0., 0.), (0., 1., 0.)])
    source = _cloud([(0., 0., 0.), (1., 0., 0.), (0., 1., 1.)])
    distances = comparison.compute_deviation_distances(source, target)
    np.testing.assert_allclose(distances, [0., 0., 1.], rtol=0., atol=1e-7)

    colors = comparison.compute_heatmap_colors(distances)
    np.testing.assert_allclose(colors, [[0., 0., 0.5],
                                        [0., 0., 0.5],
                                        [0.5, 0., 0.]],
                               rtol=0., atol=1e-7)


def test_mesh_surface_distances_ignore_registration_sample(comparison):
    """Interior mesh points have zero error even far from mesh vertices."""
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(
        [[0., 0., 0.], [2., 0., 0.], [0., 2., 0.]])
    mesh.triangles = o3d.utility.Vector3iVector([[0, 1, 2]])

    source = _cloud([(0.5, 0.5, 0.),
                     (0.5, 0.5, 1.),
                     (1., 0., 0.)])
    vertex_sample = _cloud(np.asarray(mesh.vertices))

    # The old point-to-sample proxy gives nonzero errors on the surface.
    sampled_distances = np.asarray(
        source.compute_point_cloud_distance(vertex_sample))
    np.testing.assert_allclose(sampled_distances,
                               [np.sqrt(0.5), np.sqrt(1.5), 1.],
                               rtol=0., atol=1e-6)

    distances = comparison.compute_deviation_distances(
        source, vertex_sample, target_mesh=mesh)
    np.testing.assert_allclose(distances, [0., 1., 0.],
                               rtol=0., atol=1e-6)
    colors = comparison.compute_heatmap_colors(distances)
    np.testing.assert_allclose(colors, [[0., 0., 0.5],
                                        [0.5, 0., 0.],
                                        [0., 0., 0.5]],
                               rtol=0., atol=1e-6)
