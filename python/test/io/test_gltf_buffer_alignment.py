# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2026 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import json
import struct

import numpy as np
import open3d as o3d
import pytest


def _read_gltf_metadata(path):
    """Read the glTF JSON metadata from an ASCII or binary export."""
    data = path.read_bytes()
    if path.suffix == ".gltf":
        return json.loads(data)

    assert data[:4] == b"glTF"
    assert struct.unpack_from("<I", data, 4)[0] == 2
    assert struct.unpack_from("<I", data, 8)[0] == len(data)
    json_length = struct.unpack_from("<I", data, 12)[0]
    assert data[16:20] == b"JSON"
    return json.loads(data[20:20 + json_length])


@pytest.mark.parametrize("suffix", [".gltf", ".glb"])
@pytest.mark.parametrize("triangle_count", [1, 2])
def test_gltf_positions_are_four_byte_aligned(tmp_path, suffix, triangle_count):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(
        np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]],
                 dtype=np.float64))
    triangles = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
    mesh.triangles = o3d.utility.Vector3iVector(triangles[:triangle_count])

    output_path = tmp_path / ("mesh" + suffix)
    assert o3d.io.write_triangle_mesh(str(output_path), mesh)
    metadata = _read_gltf_metadata(output_path)

    primitive = metadata["meshes"][0]["primitives"][0]
    indices = metadata["accessors"][primitive["indices"]]
    positions = metadata["accessors"][primitive["attributes"]["POSITION"]]
    index_view = metadata["bufferViews"][indices["bufferView"]]
    position_view = metadata["bufferViews"][positions["bufferView"]]

    index_bytes = triangle_count * 3 * 2  # uint16 indices
    aligned_offset = ((index_bytes + 3) // 4) * 4
    assert indices["componentType"] == 5123
    assert index_view.get("byteOffset", 0) == 0
    assert index_view["byteLength"] == index_bytes
    assert position_view["byteOffset"] == aligned_offset
    assert position_view["byteOffset"] % 4 == 0
    assert position_view["byteLength"] == 4 * 3 * 4  # four float32 vertices
