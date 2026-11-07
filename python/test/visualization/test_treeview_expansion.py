# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2026 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------
"""Tests TreeView expansion state through Python bindings without a GUI window."""

from open3d.visualization import gui


def test_treeview_expansion_default_and_roundtrip():
    """New items start expanded and retain each explicit state change."""
    tree = gui.TreeView()
    parent = tree.add_text_item(tree.get_root_item(), "parent")
    tree.add_text_item(parent, "child")

    assert tree.is_item_expanded(parent) is True
    tree.set_item_expanded(parent, False)
    assert tree.is_item_expanded(parent) is False
    tree.set_item_expanded(parent, True)
    assert tree.is_item_expanded(parent) is True


def test_treeview_expansion_is_independent_per_item():
    """Changing one item's expansion must not modify other items."""
    tree = gui.TreeView()
    root = tree.get_root_item()
    left = tree.add_text_item(root, "left")
    right = tree.add_text_item(root, "right")
    child = tree.add_text_item(left, "child")

    tree.set_item_expanded(left, False)
    assert tree.is_item_expanded(left) is False
    assert tree.is_item_expanded(right) is True
    assert tree.is_item_expanded(child) is True


def test_treeview_unknown_and_removed_items_are_not_expanded():
    """Unknown and recursively removed items must never report expanded."""
    tree = gui.TreeView()
    root = tree.get_root_item()
    parent = tree.add_text_item(root, "parent")
    child = tree.add_text_item(parent, "child")
    missing = max(root, parent, child) + 1000

    assert tree.is_item_expanded(missing) is False
    tree.set_item_expanded(missing, True)
    assert tree.is_item_expanded(missing) is False

    tree.remove_item(parent)
    assert tree.is_item_expanded(parent) is False
    assert tree.is_item_expanded(child) is False
