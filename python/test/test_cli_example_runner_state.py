# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2026 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import argparse
import sys

import pytest

from open3d.tools import cli


def _setup_example(tmp_path, monkeypatch, source, argv, show=False):
    """Discover a temporary example while exercising the real CLI runner."""
    folder = tmp_path / "geometry"
    folder.mkdir()
    (folder / "example.py").write_text(source, encoding="utf-8")
    monkeypatch.setattr(cli, "_get_examples_dir", lambda: tmp_path)
    monkeypatch.setattr(cli, "_get_runnable_examples_dict",
                        lambda: {"geometry": ["example"]})
    monkeypatch.setattr(sys, "argv", argv)
    args = argparse.Namespace(category_example="geometry/example",
                              list=False,
                              show=show)
    return args, folder


def test_sibling_import_preserves_forwarded_arguments(tmp_path, monkeypatch):
    """Resolve a sibling helper and restore the caller after success."""
    output = tmp_path / "out.txt"
    argv = ["open3d", "example", "geometry/example", str(output), "token"]
    args, folder = _setup_example(
        tmp_path, monkeypatch,
        "from o3d_7618_sibling import value\n"
        "import sys\n"
        "from pathlib import Path\n"
        "Path(sys.argv[1]).write_text(f'{value}:{sys.argv[2]}')\n",
        argv)
    (folder / "o3d_7618_sibling.py").write_text("value = 42\n",
                                               encoding="utf-8")
    original_argv = sys.argv
    original_path = sys.path
    before_argv = sys.argv.copy()
    before_path = sys.path.copy()
    try:
        assert cli._example(argparse.ArgumentParser(), args) == 0
        assert output.read_text(encoding="utf-8") == "42:token"
        assert sys.argv == before_argv
        assert sys.path == before_path
        assert sys.argv is original_argv
        assert sys.path is original_path
    finally:
        sys.modules.pop("o3d_7618_sibling", None)


def test_example_exception_restores_caller_state(tmp_path, monkeypatch):
    """An example's exception must not corrupt CLI argument or path state."""
    argv = ["open3d", "example", "geometry/example", "arg"]
    args, _ = _setup_example(tmp_path, monkeypatch,
                             "raise RuntimeError('sentinel')\n", argv)
    original_argv = sys.argv
    original_path = sys.path
    before_argv = sys.argv.copy()
    before_path = sys.path.copy()

    with pytest.raises(RuntimeError, match="sentinel"):
        cli._example(argparse.ArgumentParser(), args)

    assert sys.argv == before_argv
    assert sys.path == before_path
    assert sys.argv is original_argv
    assert sys.path is original_path


def test_show_does_not_modify_search_path(tmp_path, monkeypatch, capsys):
    """Showing source must not mutate process-wide import search paths."""
    argv = ["open3d", "example", "geometry/example"]
    args, _ = _setup_example(tmp_path, monkeypatch,
                             "print('DISPLAY ONLY')\n", argv, show=True)
    original_argv = sys.argv
    original_path = sys.path
    before_argv = sys.argv.copy()
    before_path = sys.path.copy()

    assert cli._example(argparse.ArgumentParser(), args) == 0
    assert "print('DISPLAY ONLY')" in capsys.readouterr().out
    assert sys.argv == before_argv
    assert sys.path == before_path
    assert sys.argv is original_argv
    assert sys.path is original_path
