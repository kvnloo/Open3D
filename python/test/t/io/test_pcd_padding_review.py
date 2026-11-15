# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2026 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

"""CPU regressions for PCD reserved padding fields."""

import struct

import numpy as np
import open3d as o3d
import pytest


@pytest.mark.parametrize(
    ("encoding", "padded"),
    [("ascii", True), ("binary", True), ("ascii", False)],
    ids=["ascii-padding", "binary-padding", "ascii-control"],
)
def test_tensor_pcd_padding_is_not_an_attribute(tmp_path, encoding, padded):
    """Padding consumes file bytes but never becomes a tensor attribute."""
    records = [
        ((1.0, 2.0, 3.0), (10, 11, 12), 7),
        ((2.0, 4.0, 5.0), (20, 21, 22), 9),
    ]
    fields = "x y z _ labels" if padded else "x y z labels"
    sizes = "4 4 4 1 4" if padded else "4 4 4 4"
    types = "F F F U U" if padded else "F F F U"
    counts = "1 1 1 3 1" if padded else "1 1 1 1"
    header = (
        "VERSION 0.7\n"
        f"FIELDS {fields}\n"
        f"SIZE {sizes}\n"
        f"TYPE {types}\n"
        f"COUNT {counts}\n"
        "WIDTH 2\n"
        "HEIGHT 1\n"
        "POINTS 2\n"
        f"DATA {encoding}\n"
    ).encode("ascii")

    if encoding == "binary":
        payload = b"".join(
            struct.pack("<fff3BI", *xyz, *padding, label)
            for xyz, padding, label in records
        )
        assert len(payload) == 2 * (12 + 3 + 4) == 38
    elif padded:
        payload = b"1 2 3 10 11 12 7\n2 4 5 20 21 22 9\n"
    else:
        payload = b"1 2 3 7\n2 4 5 9\n"

    path = tmp_path / "padding_case.pcd"
    path.write_bytes(header + payload)
    cloud = o3d.t.io.read_point_cloud(str(path))

    np.testing.assert_array_equal(
        cloud.point.positions.numpy(),
        np.array([[1, 2, 3], [2, 4, 5]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        cloud.point["labels"].numpy(),
        np.array([[7], [9]], dtype=np.uint32),
    )
    assert cloud.point.positions.dtype == o3d.core.float32
    assert cloud.point["labels"].dtype == o3d.core.uint32
    assert "_" not in cloud.point
