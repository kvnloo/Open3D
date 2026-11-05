# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2026 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import open3d as o3d
import numpy as np


def test_in_memory_xyz():
    # Reading/Writing bytes from bytes object
    pcb0 = b"1.0000000000 2.0000000000 3.0000000000\n4.0000000000 5.0000000000 6.0000000000\n7.0000000000 8.0000000000 9.0000000000\n"
    pc0 = o3d.io.read_point_cloud_from_bytes(pcb0, "mem::xyz")
    assert len(pc0.points) == 3
    pcb1 = o3d.io.write_point_cloud_to_bytes(pc0, "mem::xyz")
    assert len(pcb1) == len(pcb0)
    pc1 = o3d.io.read_point_cloud_from_bytes(pcb1, "mem::xyz")
    assert len(pc1.points) == 3
    # Reading/Writing bytes from PointCloud
    pc2 = o3d.geometry.PointCloud()
    pc2_points = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
    pc2.points = o3d.utility.Vector3dVector(pc2_points)
    pcb2 = o3d.io.write_point_cloud_to_bytes(pc2, "mem::xyz")
    assert len(pcb2) == len(pcb0)
    pc3 = o3d.io.read_point_cloud_from_bytes(pcb2, "mem::xyz")
    assert len(pc3.points) == 3
    np.testing.assert_allclose(np.asarray(pc3.points), pc2_points)


def test_obj_reader_preserves_duplicate_vertices_without_post_processing(tmp_path):
    # Two used OBJ vertex indices have identical coordinates. Disabling
    # post-processing should preserve both, rather than weld their indices.
    obj = ("v 0 0 0\n"
           "v 1 0 0\n"
           "v 0 1 0\n"
           "v 0 0 0\n"
           "v 1 1 0\n"
           "f 1 2 3\n"
           "f 4 5 3\n")
    filename = tmp_path / "duplicate_vertices.obj"
    filename.write_text(obj, encoding="ascii")

    raw = o3d.io.read_triangle_mesh(str(filename),
                                    enable_post_processing=False)
    welded = o3d.io.read_triangle_mesh(str(filename),
                                       enable_post_processing=True)

    assert len(raw.vertices) == 5
    assert len(raw.triangles) == 2
    assert np.count_nonzero(
        np.all(np.asarray(raw.vertices) == [0, 0, 0], axis=1)) == 2
    assert len(welded.vertices) == 4
    assert len(welded.triangles) == 2
