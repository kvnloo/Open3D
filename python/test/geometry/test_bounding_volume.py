# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2026 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import numpy as np
import open3d as o3d
import pytest

_CENTER = np.array([1.0, 2.0, 3.0])
_EXTENT = np.array([2.0, 4.0, 6.0])


def _rotation():
    """Rotation built from the 30, 45 and 60 degree Euler angles."""
    return o3d.geometry.get_rotation_matrix_from_xyz(
        (np.pi / 6.0, np.pi / 4.0, np.pi / 3.0))


def _transform(linear, translation):
    transformation = np.eye(4)
    transformation[:3, :3] = linear
    transformation[:3, 3] = translation
    return transformation


def _box():
    return o3d.geometry.OrientedBoundingBox(_CENTER, _rotation(), _EXTENT)


def test_oriented_bounding_box_transform_matches_points():
    """The box corners follow the transform the way point coordinates do."""
    transformation = _transform(2.0 * _rotation(), [-4.0, 5.0, 0.5])
    corners = np.asarray(_box().get_box_points())
    expected = corners @ transformation[:3, :3].T + transformation[:3, 3]

    box = _box()
    box.transform(transformation)

    np.testing.assert_allclose(np.asarray(box.get_box_points()),
                               expected,
                               atol=1e-9)
    np.testing.assert_allclose(box.extent, 2.0 * _EXTENT, atol=1e-9)


def test_oriented_bounding_box_transform_matches_point_cloud():
    """A box and a point cloud of its corners transform consistently."""
    transformation = _transform(_rotation(), [-4.0, 5.0, 0.5])
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(
        np.asarray(_box().get_box_points()))
    cloud.transform(transformation)

    box = _box()
    box.transform(transformation)

    np.testing.assert_allclose(np.asarray(box.get_box_points()),
                               np.asarray(cloud.points),
                               atol=1e-9)


@pytest.mark.parametrize("linear", [
    np.diag([1.0, 2.0, 3.0]),
    np.array([[1.0, 0.5, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
    -np.eye(3),
])
def test_oriented_bounding_box_transform_rejects_non_similarity(linear):
    box = _box()
    with pytest.raises(RuntimeError):
        box.transform(_transform(linear, [0.0, 0.0, 0.0]))


def test_oriented_bounding_ellipsoid_transform():
    transformation = _transform(2.0 * _rotation(), [-4.0, 5.0, 0.5])
    ellipsoid = o3d.geometry.OrientedBoundingEllipsoid(_CENTER, np.eye(3),
                                                       [1.0, 2.0, 3.0])
    ellipsoid.transform(transformation)

    np.testing.assert_allclose(ellipsoid.center,
                               transformation[:3, :3] @ _CENTER +
                               transformation[:3, 3],
                               atol=1e-9)
    np.testing.assert_allclose(ellipsoid.R, _rotation(), atol=1e-9)
    np.testing.assert_allclose(ellipsoid.radii, [2.0, 4.0, 6.0], atol=1e-9)


def test_oriented_bounding_volumes_reject_tiny_projective_coefficient():
    """A tiny nonzero perspective term is unsafe for distant geometry."""
    transformation = np.eye(4)
    transformation[3, 0] = 1e-13
    center = np.array([6_000_000.0, 0.0, 0.0])
    projected_x = center[0] / (1.0 + transformation[3, 0] * center[0])
    assert center[0] - projected_x > 3.5

    volumes = (
        o3d.geometry.OrientedBoundingBox(center, np.eye(3), [2, 2, 2]),
        o3d.geometry.OrientedBoundingEllipsoid(center, np.eye(3), [1, 2, 3]),
    )
    for volume in volumes:
        with pytest.raises(RuntimeError, match="projective"):
            volume.transform(transformation)

_VOLUME_CASES = [
    pytest.param(o3d.geometry.OrientedBoundingBox, "extent", [2.0, 4.0, 6.0],
                 id="box"),
    pytest.param(o3d.geometry.OrientedBoundingEllipsoid, "radii",
                 [1.0, 2.0, 3.0], id="ellipsoid"),
]


@pytest.mark.parametrize("volume_type, size_attribute, sizes", _VOLUME_CASES)
def test_oriented_bounding_volumes_accept_near_singular_uniform_scale(
        volume_type, size_attribute, sizes):
    """A small positive uniform scale remains a valid affine similarity."""
    center = np.array([6_000_000.0, -2_000_000.0, 3_000_000.0])
    transformation = _transform(1e-6 * np.eye(3), [0.5, -2.0, 7.0])
    transformation[3, 0] = -0.0
    volume = volume_type(center, np.eye(3), sizes)

    volume.transform(transformation)

    # (6, -2, 3) + (0.5, -2, 7) after the 1e-6 uniform scale.
    np.testing.assert_allclose(volume.center, [6.5, -4.0, 10.0],
                               rtol=0, atol=1e-10)
    np.testing.assert_allclose(volume.R, np.eye(3), rtol=0, atol=1e-10)
    np.testing.assert_allclose(getattr(volume, size_attribute),
                               1e-6 * np.asarray(sizes), rtol=1e-9,
                               atol=1e-15)


@pytest.mark.parametrize("volume_type, size_attribute, sizes", _VOLUME_CASES)
@pytest.mark.parametrize("linear, error", [
    pytest.param(np.diag([1.0, 1.0, 1e-12]), "general",
                 id="near-singular-anisotropic"),
    pytest.param(np.diag([1.0, 1.0, 0.0]), "collapses", id="singular"),
    pytest.param(np.diag([-1.0, 1.0, 1.0]), "mirrors", id="reflected"),
    pytest.param(np.diag([2.0, 1.0, 1.0]), "general",
                 id="non-uniform-scale"),
])
def test_oriented_bounding_volumes_reject_invalid_linear_parts(
        volume_type, size_attribute, sizes, linear, error):
    """Collapse, reflection, and anisotropy cannot be represented."""
    center = np.array([1.0, 2.0, 3.0])
    volume = volume_type(center, np.eye(3), sizes)

    with pytest.raises(RuntimeError, match=error):
        volume.transform(_transform(linear, [4.0, -2.0, 1.0]))

    np.testing.assert_array_equal(volume.center, center)
    np.testing.assert_array_equal(volume.R, np.eye(3))
    np.testing.assert_array_equal(getattr(volume, size_attribute), sizes)


@pytest.mark.parametrize("volume_type, size_attribute, sizes", _VOLUME_CASES)
@pytest.mark.parametrize("column, coefficient", [
    pytest.param(0, 1e-16, id="tiny-x-perspective"),
    pytest.param(1, -1e-16, id="tiny-y-perspective"),
    pytest.param(2, 1e-16, id="tiny-z-perspective"),
    pytest.param(3, np.nextafter(1.0, np.inf), id="nonunit-w"),
])
def test_oriented_bounding_volumes_require_exact_affine_last_row(
        volume_type, size_attribute, sizes, column, coefficient):
    """Even sub-tolerance homogeneous-row changes must be rejected."""
    center = np.array([6_000_000.0, -2_000_000.0, 3_000_000.0])
    transformation = np.eye(4)
    transformation[3, column] = coefficient
    volume = volume_type(center, np.eye(3), sizes)

    if column == 0:
        projected_x = center[0] / (1.0 + coefficient * center[0])
        assert 0.003 < center[0] - projected_x < 0.004

    with pytest.raises(RuntimeError, match="projective"):
        volume.transform(transformation)

    np.testing.assert_array_equal(volume.center, center)
    np.testing.assert_array_equal(getattr(volume, size_attribute), sizes)
