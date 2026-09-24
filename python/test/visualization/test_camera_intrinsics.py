# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2026 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------
"""Pixel-center calibration through the actual Filament camera and renderer."""

import json
import multiprocessing
import os
import platform

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    not (platform.system() == "Linux" and platform.machine() == "x86_64") or
    os.getenv("OPEN3D_CPU_RENDERING", "") != "true",
    reason="Requires the Linux x86_64 offscreen CPU-rendering test environment")

CALIBRATIONS = [
    (640, 480, 525.0, 525.0, 319.5, 239.5),
    (641, 481, 520.0, 610.0, 320.0, 240.0),
    (320, 240, 275.0, 330.0, 143.25, 127.75),
    (321, 241, 280.0, 335.0, 173.75, 110.25),
]


def _plane(o3d, sx, sy):
    xy = np.array([[-4.0, -4.0], [4.0, -4.0], [4.0, 4.0], [-4.0, 4.0]])
    vertices = np.column_stack((xy, 2.0 + sx * xy[:, 0] + sy * xy[:, 1]))
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector([[0, 2, 1], [0, 3, 2]])
    mesh.compute_vertex_normals()
    return mesh


def _projection_result(calibration, moved, output):
    import open3d as o3d
    import open3d.visualization.rendering as rendering

    width, height, fx, fy, cx, cy = calibration
    intrinsic = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    extrinsic = np.eye(4)
    if moved:
        extrinsic = np.array([[0.0, -1.0, 0.0, 0.25], [1.0, 0.0, 0.0, -0.5],
                              [0.0, 0.0, 1.0, 0.125], [0.0, 0.0, 0.0, 1.0]])
    camera_points = np.array([[-0.4, -0.2, 1.0], [0.0, 0.0,
                                                  1.0], [0.2, 0.3, 1.0],
                              [-0.5, 0.2, 2.0], [0.0, 0.0, 2.0],
                              [0.4, -0.3, 2.0], [-0.8, -0.5, 4.0],
                              [0.0, 0.0, 4.0], [0.7, 0.6, 4.0]])
    world = (np.linalg.inv(extrinsic) @ np.column_stack(
        (camera_points, np.ones(len(camera_points)))).T).T
    projected = (intrinsic @ camera_points.T).T
    expected = projected[:, :2] / projected[:, 2:3]

    render = rendering.OffscreenRenderer(width, height)
    mesh = _plane(o3d, 0.0, 0.0)
    mesh.transform(np.linalg.inv(extrinsic))
    render.scene.add_geometry("bounds", mesh, rendering.MaterialRecord())
    render.setup_camera(intrinsic, extrinsic, width, height)
    camera = render.scene.camera
    projection = np.asarray(camera.get_projection_matrix(), dtype=np.float64)
    view = np.asarray(camera.get_view_matrix(), dtype=np.float64)
    clip = (projection @ view @ world.T).T
    ndc = clip[:, :2] / clip[:, 3:4]
    actual = np.column_stack(((ndc[:, 0] + 1.0) * width / 2.0 - 0.5,
                              (1.0 - ndc[:, 1]) * height / 2.0 - 0.5))
    camera.set_projection(intrinsic, camera.get_near(), camera.get_far(), width,
                          height)
    repeated = np.asarray(camera.get_projection_matrix()).copy()
    camera.copy_from(camera)
    copied = np.asarray(camera.get_projection_matrix()).copy()
    with open(output, "w") as stream:
        json.dump(
            {
                "pixel_error": (actual - expected).tolist(),
                "reapply_error": float(np.max(np.abs(repeated - projection))),
                "copy_error": float(np.max(np.abs(copied - projection)))
            }, stream)


def _depth_result(sx, sy, output):
    import open3d as o3d
    import open3d.visualization.rendering as rendering

    width, height = 64, 48
    fx, fy, cx, cy = 50.0, 55.0, 29.25, 25.75
    intrinsic = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    render = rendering.OffscreenRenderer(width, height)
    material = rendering.MaterialRecord()
    material.shader = "defaultUnlit"
    render.scene.add_geometry("plane", _plane(o3d, sx, sy), material)
    render.setup_camera(intrinsic, np.eye(4), width, height)
    actual = np.asarray(render.render_to_depth_image(z_in_view_space=True))
    v, u = np.mgrid[:height, :width]
    expected = 2.0 / (1.0 - sx * (u - cx) / fx - sy * (v - cy) / fy)
    actual, expected = actual[3:-3, 3:-3], expected[3:-3, 3:-3]
    complete = bool(np.all(np.isfinite(actual)) and np.all(actual > 0.0))
    with open(output, "w") as stream:
        json.dump(
            {
                "complete": complete,
                "max_depth_error": float(np.max(np.abs(actual - expected)))
            }, stream)


def _run_probe(target, args, tmp_path):
    output = tmp_path / "result.json"
    proc = multiprocessing.get_context("spawn").Process(target=target,
                                                        args=(*args,
                                                              str(output)))
    proc.start()
    proc.join(timeout=60)
    if proc.exitcode is None:
        proc.kill()
        proc.join()
        pytest.fail("Renderer subprocess timed out")
    assert proc.exitcode == 0, "Renderer subprocess failed before measurement"
    with output.open() as stream:
        return json.load(stream)


@pytest.mark.parametrize(
    "calibration",
    CALIBRATIONS,
    ids=["even-centered", "odd-centered", "even-offset", "odd-offset"])
@pytest.mark.parametrize("moved", [False, True], ids=["identity", "moved"])
def test_intrinsics_project_pixel_centers(calibration, moved, tmp_path):
    result = _run_probe(_projection_result, (calibration, moved), tmp_path)
    np.testing.assert_allclose(result["pixel_error"],
                               0.0,
                               rtol=0.0,
                               atol=1e-4,
                               err_msg="CALIBRATION_PIXEL_MISMATCH")
    assert result["reapply_error"] == 0.0
    assert result["copy_error"] == 0.0


@pytest.mark.parametrize(
    "sx,sy", [(0.0, 0.0), (0.4, 0.0), (-0.4, 0.0), (0.0, 0.4), (0.0, -0.4)],
    ids=["flat", "tilt-x", "tilt-minus-x", "tilt-y", "tilt-minus-y"])
def test_intrinsics_render_analytic_plane(sx, sy, tmp_path):
    result = _run_probe(_depth_result, (sx, sy), tmp_path)
    assert result["complete"], "Plane did not cover the measured image region"
    assert result["max_depth_error"] < 2e-4, (
        f"CALIBRATION_DEPTH_MISMATCH: {result['max_depth_error']}")
