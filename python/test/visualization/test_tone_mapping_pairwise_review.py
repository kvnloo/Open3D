# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2026 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------
"""Check pairwise color-grading tone mappers using CPU offscreen rendering.

The comparison uses the same white scene for all three tone-mapper pairs,
so it detects a mapper that is silently ignored by the Filament backend.
"""

import os
import platform
from multiprocessing import Process

import pytest


def _compare_tone_mappers(first, second, minimum):
    """Render each mode in the same software-renderer process.

    Compare only a uniform interior crop to avoid edge and platform noise.
    """
    import numpy as np
    import open3d.visualization.rendering as rendering

    render = rendering.OffscreenRenderer(160, 90)
    scene = render.scene
    scene.show_skybox(False)
    scene.view.set_post_processing(True)
    grading = rendering.ColorGrading

    def mean_for(mode_name):
        """Capture the interior mean after selecting a public tone mapper."""
        mode = getattr(grading.ToneMapping, mode_name)
        scene.view.set_color_grading(grading(grading.Quality.ULTRA, mode))
        scene.set_background([1.0, 1.0, 1.0, 1.0])
        pixels = np.asarray(render.render_to_image())
        return float(pixels[15:60, 15:60, :3].mean())

    first_mean = mean_for(first)
    second_mean = mean_for(second)
    delta = abs(first_mean - second_mean)
    assert delta > minimum, (
        f"{first}/{second}: mean RGB {first_mean:.4f}/{second_mean:.4f}, "
        f"delta {delta:.4f} must exceed {minimum:.1f}")


@pytest.mark.parametrize("first,second,minimum", [
    ("LINEAR", "ACES", 20.0),
    ("LINEAR", "FILMIC", 10.0),
    ("FILMIC", "ACES", 10.0),
])
@pytest.mark.skipif(
    not (platform.system() == "Linux" and platform.machine() == "x86_64") or
    os.getenv("OPEN3D_CPU_RENDERING", "") != "true",
    reason="Offscreen CPU rendering is only supported on x86_64 Linux")
def test_tone_mapping_pair_cpu(first, second, minimum):
    """Isolate Filament rendering failures in a bounded child process."""
    proc = Process(target=_compare_tone_mappers,
                   args=(first, second, minimum))
    proc.start()
    proc.join(timeout=15)
    if proc.exitcode is None:
        proc.kill()
        proc.join()
        pytest.fail("CPU tone-mapping pair did not complete in 15 seconds")
    assert proc.exitcode == 0
