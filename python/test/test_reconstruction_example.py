# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2026 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import importlib.util
from pathlib import Path

import numpy as np


def test_reconstruction_registration_pool_initializer(monkeypatch, tmp_path):
    """The reconstruction example must pass its thread limiter to workers."""
    source = (Path(__file__).resolve().parents[2] / "examples" / "python" /
              "reconstruction_system" / "register_fragments.py")
    spec = importlib.util.spec_from_file_location("register_fragments", source)
    register_fragments = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(register_fragments)
    from open3d_example import limit_open3d_threads

    observed = {}

    class DummyPool:

        def __init__(self, processes, initializer):
            observed["processes"] = processes
            observed["initializer"] = initializer

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def starmap(self, func, args):
            observed["pairs"] = len(args)
            return [(False, np.eye(4), np.zeros((6, 6))) for _ in args]

    class DummyContext:

        def Pool(self, processes, initializer):
            return DummyPool(processes, initializer)

    monkeypatch.setattr(register_fragments.multiprocessing, "cpu_count",
                        lambda: 4)
    monkeypatch.setattr(register_fragments.multiprocessing, "get_context",
                        lambda method: DummyContext())
    monkeypatch.setattr(
        register_fragments.o3d.io, "write_pose_graph",
        lambda filename, graph: observed.update(
            filename=filename, nodes=len(graph.nodes), edges=len(graph.edges)))

    config = {
        "python_multi_threading": True,
        "path_dataset": str(tmp_path),
        "template_global_posegraph": "scene.json",
    }
    register_fragments.make_posegraph_for_scene(["first.ply", "second.ply"],
                                                config)

    assert observed["initializer"] is limit_open3d_threads
    assert observed["processes"] == 1
    assert observed["pairs"] == 1
    assert observed["nodes"] == 1
    assert observed["edges"] == 0
    assert observed["filename"] == str(tmp_path / "scene.json")
