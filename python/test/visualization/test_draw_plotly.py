# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2026 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------
"""CPU-only regression tests for Plotly triangle mesh colors."""

import numpy as np
import open3d as o3d

from open3d.visualization.draw_plotly import get_mesh_object, get_plotly_fig


def _triangle_mesh():
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector([[0, 0, 0], [1, 0, 0],
                                                 [0, 1, 0]])
    mesh.triangles = o3d.utility.Vector3iVector([[0, 1, 2]])
    return mesh


def test_plotly_mesh_preserves_vertex_colors_in_mixed_scene():
    """The mesh colors survive alongside other geometries in a figure."""
    mesh = _triangle_mesh()
    mesh.vertex_colors = o3d.utility.Vector3dVector([[1, 0, 0], [0, 1, 0],
                                                      [0, 0, 1]])

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector([[0, 0, 1]])
    cloud.paint_uniform_color((1.0, 1.0, 0.0))

    figure = get_plotly_fig([mesh, cloud])

    assert len(figure.data) == 2
    assert figure.data[0].type == 'mesh3d'
    assert list(figure.data[0].vertexcolor) == [
        'rgb(255,0,0)', 'rgb(0,255,0)', 'rgb(0,0,255)'
    ]
    assert figure.data[0].intensity is None
    assert figure.data[0].colorscale is None
    assert figure.data[1].type == 'scatter3d'


def test_plotly_mesh_without_vertex_colors_keeps_grayscale():
    """Uncolored meshes retain the original X-coordinate intensity."""
    mesh = _triangle_mesh()

    trace = get_mesh_object(mesh)

    assert trace.vertexcolor is None
    np.testing.assert_array_equal(trace.intensity, [0, 1, 0])
    assert trace.colorscale is not None
