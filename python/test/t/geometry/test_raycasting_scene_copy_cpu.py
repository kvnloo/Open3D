"""CPU-only correctness checks for RaycastingScene copy semantics."""

import copy

import numpy as np
import open3d as o3d


CPU = o3d.core.Device("CPU:0")


def _add_triangle(scene, z):
    """Attach a triangle at height z and return its Embree geometry ID."""
    vertices = o3d.core.Tensor(
        [[0.0, 0.0, z], [1.0, 0.0, z], [0.0, 1.0, z]],
        dtype=o3d.core.float32,
        device=CPU,
    )
    triangles = o3d.core.Tensor(
        [[0, 1, 2]], dtype=o3d.core.uint32, device=CPU
    )
    return scene.add_triangles(vertices, triangles)


def _cast_down(scene):
    """Cast a ray from z=1: a triangle at z gives distance 1-z."""
    rays = o3d.core.Tensor(
        [[0.25, 0.25, 1.0, 0.0, 0.0, -1.0]],
        dtype=o3d.core.float32,
        device=CPU,
    )
    result = scene.cast_rays(rays)
    return (
        float(result["t_hit"][0].item()),
        int(result["geometry_ids"][0].item()),
    )


def test_copy_empty_scene_cpu_misses():
    source = o3d.t.geometry.RaycastingScene(device=CPU)
    clone = o3d.t.geometry.RaycastingScene(source)
    distance, geom_id = _cast_down(clone)
    assert np.isinf(distance)
    assert geom_id == o3d.t.geometry.RaycastingScene.INVALID_ID


def test_copy_cpu_preserves_hit_distance_and_id():
    source = o3d.t.geometry.RaycastingScene(device=CPU, nthreads=2)
    source_id = _add_triangle(source, 0.0)
    clone = o3d.t.geometry.RaycastingScene(source)
    original_distance, original_id = _cast_down(source)
    copied_distance, copied_id = _cast_down(clone)
    np.testing.assert_allclose(
        [original_distance, copied_distance],
        [1.0, 1.0],
        rtol=0.0,
        atol=1e-6,
    )
    assert (original_id, copied_id) == (source_id, source_id)


def test_deepcopy_cpu_does_not_share_geometry():
    source = o3d.t.geometry.RaycastingScene(device=CPU)
    first_id = _add_triangle(source, 0.0)
    clone = copy.deepcopy(source)
    second_id = _add_triangle(clone, 0.5)
    assert second_id != first_id
    original_distance, original_id = _cast_down(source)
    cloned_distance, cloned_id = _cast_down(clone)
    np.testing.assert_allclose(
        [original_distance, cloned_distance],
        [1.0, 0.5],
        rtol=0.0,
        atol=1e-6,
    )
    assert (original_id, cloned_id) == (first_id, second_id)


def test_copy_protocol_cpu_produces_independent_scene():
    source = o3d.t.geometry.RaycastingScene(device=CPU)
    _add_triangle(source, 0.0)
    clone = copy.copy(source)
    _add_triangle(clone, 0.75)
    assert np.isclose(_cast_down(source)[0], 1.0, rtol=0, atol=1e-6)
    assert np.isclose(_cast_down(clone)[0], 0.25, rtol=0, atol=1e-6)
