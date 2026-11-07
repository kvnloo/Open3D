# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2026 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import pickle
import struct

import numpy as np
import open3d as o3d
import pytest


@pytest.mark.parametrize(
    "positions",
    [
        [],
        [[1.5, -2.0, 3.25]],
        [[1.5, -2.0, 3.25], [0.0, 4.5, -6.0]],
    ],
)
def test_legacy_pointcloud_pickle_round_trip_cpu(positions):
    points = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    n = len(points)
    normals = np.repeat([[0.0, 0.0, 1.0]], n, axis=0)
    colors = np.repeat([[0.25, 0.5, 0.75]], n, axis=0)
    covariances = np.asarray(
        [np.diag([i + 1.0, i + 2.0, i + 3.0]) for i in range(n)],
        dtype=np.float64,
    ).reshape(n, 3, 3)

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.normals = o3d.utility.Vector3dVector(normals)
    cloud.colors = o3d.utility.Vector3dVector(colors)
    cloud.covariances = o3d.utility.Matrix3dVector(covariances)

    restored = pickle.loads(pickle.dumps(cloud, protocol=pickle.HIGHEST_PROTOCOL))
    for attribute, expected in (
        ("points", points),
        ("normals", normals),
        ("colors", colors),
        ("covariances", covariances),
    ):
        np.testing.assert_array_equal(
            np.asarray(getattr(restored, attribute)), expected
        )


@pytest.mark.parametrize(
    "constructor, values",
    [
        (
            o3d.utility.Vector3dVector,
            np.array([[1.5, -2.0, 3.25], [0.0, 4.5, -6.0]], dtype=np.float64),
        ),
        (
            o3d.utility.Matrix3dVector,
            np.array([np.diag([1.0, 2.0, 3.0])], dtype=np.float64),
        ),
    ],
)
def test_eigen_container_pickle_round_trip_cpu(constructor, values):
    restored = pickle.loads(
        pickle.dumps(constructor(values), protocol=pickle.HIGHEST_PROTOCOL)
    )
    np.testing.assert_array_equal(np.asarray(restored), values)


class _MalformedEigenPickle:
    """Request a specific __setstate__ payload during Python unpickling."""

    def __init__(self, constructor, state):
        self.constructor = constructor
        self.state = state

    def __reduce__(self):
        return self.constructor, (), self.state


@pytest.mark.parametrize(
    "constructor, element_bytes",
    [
        (o3d.utility.Vector3dVector, 3 * 8),
        (o3d.utility.Matrix3dVector, 3 * 3 * 8),
    ],
)
@pytest.mark.parametrize(
    "malformation",
    ["nonempty-zero", "short", "long", "overflow", "tuple-arity"],
)
def test_eigen_pickle_rejects_malformed_state_cpu(
    constructor, element_bytes, malformation
):
    if malformation == "nonempty-zero":
        state = (b"\x00", 0)
    elif malformation == "short":
        state = (b"\x00" * (element_bytes - 1), 1)
    elif malformation == "long":
        state = (b"\x00" * (element_bytes + 1), 1)
    elif malformation == "overflow":
        # This count fits size_t but count * element_bytes does not.
        size_t_bits = struct.calcsize("P") * 8
        state = (b"", ((1 << size_t_bits) // element_bytes) + 1)
    else:
        state = (b"",)

    encoded = pickle.dumps(
        _MalformedEigenPickle(constructor, state),
        protocol=pickle.HIGHEST_PROTOCOL,
    )
    with pytest.raises(ValueError, match="Eigen pickle"):
        pickle.loads(encoded)
