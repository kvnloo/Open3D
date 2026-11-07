"""CPU regression for SLAM accuracy with late GT association."""

from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from slam_demo import accuracy_kpis  # noqa: E402


def _pose_x(x_m):
    """Create a camera-to-world pose translated by x meters."""
    pose = np.eye(4, dtype=np.float64)
    pose[0, 3] = float(x_m)
    return pose


@pytest.mark.parametrize("first_gt_frame", [0, 1, 2])
def test_ate_aligns_to_first_associated_gt_frame(first_gt_frame):
    """Partial GT must not shift ATE for identical relative trajectories."""
    estimated = [_pose_x(i) for i in range(4)]
    # load_records normalizes GT to the first associated sample, not frame zero.
    gt = {i: _pose_x(i - first_gt_frame)
          for i in range(first_gt_frame, 4)}
    metrics = accuracy_kpis(estimated, gt)

    assert metrics["associated_frames"] == 4 - first_gt_frame
    assert metrics["rpe_translation_rmse_m"] == pytest.approx(0.0)
    assert metrics["rpe_rotation_rmse_deg"] == pytest.approx(0.0)
    assert metrics["ate_rmse_m"] == pytest.approx(0.0)
    assert metrics["ate_median_m"] == pytest.approx(0.0)
    assert metrics["ate_max_m"] == pytest.approx(0.0)
    assert metrics["final_position_error_m"] == pytest.approx(0.0)
    # KPI evaluation must leave estimation and reconstruction poses untouched.
    assert [pose[0, 3] for pose in estimated] == [0.0, 1.0, 2.0, 3.0]
