# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2026 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

"""Regression coverage for reserved padding fields in tensor PCD input."""

import struct

import numpy as np
import open3d as o3d
import pytest


def _pcd_data(data_format):
    """Make two points with differently sized repeated "_" padding fields."""
    records = [
        struct.pack("<f3Bf2Hff", 1, 7, 8, 9, 2, 0x1234, 0x5678, 3, 0.25),
        struct.pack("<f3Bf2Hff", 4, 11, 12, 13, 5, 0xABCD, 0xFFFF, 6, 0.75),
    ]
    if data_format == "ascii":
        return (b"1 7 8 9 2 4660 22136 3 0.25\n"
                b"4 11 12 13 5 43981 65535 6 0.75\n")
    if data_format == "binary":
        return b"".join(records)

    # PCD binary_compressed stores contiguous columns. The LZF stream
    # contains literal runs only, avoiding a test-only compression dependency.
    columns = []
    offset = 0
    for size in (4, 3, 4, 4, 4, 4):
        columns.append(b"".join(row[offset:offset + size] for row in records))
        offset += size
    uncompressed = b"".join(columns)
    compressed = b"".join(
        bytes([len(uncompressed[i:i + 32]) - 1]) + uncompressed[i:i + 32]
        for i in range(0, len(uncompressed), 32))
    return struct.pack("<II", len(compressed), len(uncompressed)) + compressed


@pytest.mark.parametrize("data_format", ["ascii", "binary", "binary_compressed"])
def test_pcd_ignores_repeated_padding_fields(tmp_path, data_format):
    """PCD padding has storage width but must never become a tensor attribute."""
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        "FIELDS x _ y _ z intensity\n"
        "SIZE 4 1 4 2 4 4\n"
        "TYPE F U F U F F\n"
        "COUNT 1 3 1 2 1 1\n"
        "WIDTH 2\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        "POINTS 2\n"
        f"DATA {data_format}\n"
    )
    path = tmp_path / "padded.pcd"
    path.write_bytes(header.encode("ascii") + _pcd_data(data_format))

    cloud = o3d.t.io.read_point_cloud(str(path))

    assert "_" not in cloud.point
    assert set(cloud.point.keys()) == {"positions", "intensity"}
    np.testing.assert_array_equal(
        cloud.point.positions.numpy(),
        np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32))
    np.testing.assert_array_equal(
        cloud.point.intensity.numpy(),
        np.array([[0.25], [0.75]], dtype=np.float32))
