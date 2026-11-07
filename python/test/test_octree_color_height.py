# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2026 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import importlib.util
from pathlib import Path

import numpy as np
import open3d as o3d
import pytest


EXAMPLE = (Path(__file__).resolve().parents[2] /
           "examples/python/geometry/color_octree_by_height.py")


def _load_example():
    """Import the real example without starting its visual demonstration."""
    spec = importlib.util.spec_from_file_location("color_octree_by_height",
                                                   EXAMPLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_octree_example_import_does_not_start_demo():
    """Importing the helper must not construct a cloud, octree, or GUI."""
    example = _load_example()
    assert callable(example.apply_color)
    assert callable(example.main)
    assert not hasattr(example, "pcd")
    assert not hasattr(example, "octree")


@pytest.mark.parametrize(
    "xs, rgb, expected_left, expected_right",
    [
        ([0.25, 1.75], [[1, 0, 0], [0, 1, 0]],
         [1, 0, 0], [0, 1, 0]),
        ([0.25, 1.00], [[1, 0, 0], [0, 0, 1]],
         [1, 0, 0], [0, 0, 1]),
        ([0.25, 1.75, 2.00],
         [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
         [1, 0, 0], [0, 1, 0]),
    ],
    ids=["interior-control", "child-split", "root-upper-bound"],
)
def test_octree_example_recolor_respects_leaf_membership(
        xs, rgb, expected_left, expected_right):
    """Recolor only the exact indices stored by each CPU octree leaf."""
    xyz = np.asarray([[x, 0.25, 0.25] for x in xs], dtype=np.float64)
    colors = np.asarray(rgb, dtype=np.float64)
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(xyz)
    cloud.colors = o3d.utility.Vector3dVector(colors)

    tree = o3d.geometry.Octree(1, np.zeros(3), 2.0)
    init_leaf = o3d.geometry.OctreePointColorLeafNode.get_init_function()
    for index, (point, color) in enumerate(zip(xyz, colors)):
        update_leaf = (
            o3d.geometry.OctreePointColorLeafNode.get_update_function(
                index, color))
        tree.insert_point(point, init_leaf, update_leaf)

    returned_cloud, returned_tree = _load_example().apply_color(
        cloud, tree, "rgb", rgb_colors=colors)
    assert returned_cloud is cloud
    assert returned_tree is tree

    left, _ = tree.locate_leaf_node(np.asarray([0.25, 0.25, 0.25]))
    right, _ = tree.locate_leaf_node(np.asarray([1.25, 0.25, 0.25]))
    assert isinstance(left, o3d.geometry.OctreePointColorLeafNode)
    assert isinstance(right, o3d.geometry.OctreePointColorLeafNode)
    assert list(left.indices) == [0]
    assert list(right.indices) == [1]
    np.testing.assert_allclose(left.color, expected_left, rtol=0, atol=1e-12)
    np.testing.assert_allclose(right.color, expected_right, rtol=0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(cloud.colors), colors,
                               rtol=0, atol=1e-12)
