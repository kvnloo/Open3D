# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2026 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import numpy as np
import open3d as o3d
import open3d.core as o3c
import pytest


def _cloud(xs, tensor_api):
    """Construct matching legacy or tensor point clouds on the CPU."""
    points = np.asarray([(x, 0.0, 0.0) for x in xs], dtype=np.float64)
    if tensor_api:
        device = o3c.Device("CPU:0")
        cloud = o3d.t.geometry.PointCloud(device)
        cloud.point.positions = o3c.Tensor(points, o3c.float64, device)
        return cloud
    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))


@pytest.mark.parametrize("tensor_api", [False, True], ids=["legacy", "tensor"])
@pytest.mark.parametrize(
    "source_x,target_x,expected",
    [
        ((0.0, 1.0), (0.0, 1.0, 10.0, 11.0), 1.0),
        ((0.0, 1.0, 10.0, 11.0), (0.0, 1.0), 0.5),
        ((0.0, 2.0, 4.0), (0.0, 10.0, 12.0, 14.0), 1.0 / 3.0),
        ((0.0, 1.0), (10.0, 11.0), 0.0),
    ],
    ids=["more-targets", "more-sources", "one-inlier", "no-inliers"],
)
def test_registration_fitness_divides_by_source_cpu(
        tensor_api, source_x, target_x, expected):
    """Both registration APIs normalize matched points by source count."""
    source = _cloud(source_x, tensor_api)
    target = _cloud(target_x, tensor_api)
    if tensor_api:
        transform = o3c.Tensor.eye(4, o3c.float64, o3c.Device("CPU:0"))
        result = o3d.t.pipelines.registration.evaluate_registration(
            source, target, 0.05, transform)
    else:
        result = o3d.pipelines.registration.evaluate_registration(
            source, target, 0.05, np.eye(4))
    assert result.fitness == pytest.approx(expected, abs=1e-12, rel=0)


def test_odometry_fitness_binding_docs_name_source_vertex_map_cpu():
    """The bound native docstring must describe the source pixel denominator."""
    descriptor_doc = (
        o3d.t.pipelines.odometry.OdometryResult.fitness.__doc__ or "")
    assert "source vertex map" in descriptor_doc
    assert "target" not in descriptor_doc.lower()
