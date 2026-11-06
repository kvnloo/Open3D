import numpy as np
import open3d as o3d
import open3d.core as o3c
import pytest

CPU = o3c.Device("CPU:0")
EXPECTED = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])


def populated_cloud():
    cloud = o3d.t.geometry.PointCloud(
        o3c.Tensor(EXPECTED.tolist(), o3c.float64, CPU))
    cloud.point.labels = o3c.Tensor([7, 9], o3c.int64, CPU)
    return cloud


@pytest.mark.parametrize("empty_first", [True, False])
@pytest.mark.parametrize("operation", ["append", "add"])
def test_schema_less_identity_adopts_dtype_schema_and_copies(
        empty_first, operation):
    populated = populated_cloud()
    empty = o3d.t.geometry.PointCloud(CPU)
    left, right = ((empty, populated) if empty_first else
                   (populated, empty))
    result = (left.append(right) if operation == "append" else left + right)

    np.testing.assert_array_equal(result.point.positions.numpy(), EXPECTED)
    np.testing.assert_array_equal(result.point.labels.numpy(), [7, 9])
    assert result.point.positions.dtype == o3c.float64
    assert result.point.labels.dtype == o3c.int64
    assert result.device == CPU
    result.point.positions[0] = o3c.Tensor([8., 8., 8.], o3c.float64, CPU)
    np.testing.assert_array_equal(populated.point.positions.numpy(),
                                  EXPECTED)


def test_two_schema_less_clouds_keep_no_schema():
    empty = o3d.t.geometry.PointCloud(CPU)
    result = empty.append(o3d.t.geometry.PointCloud(CPU))
    assert "positions" not in result.point
    assert result.device == CPU


def test_typed_zero_length_still_checks_dtype_and_projects_left_schema():
    populated = populated_cloud()
    typed = o3d.t.geometry.PointCloud(CPU)
    typed.point.positions = o3c.Tensor.ones((0, 3), o3c.float64, CPU)
    result = typed.append(populated)
    np.testing.assert_array_equal(result.point.positions.numpy(), EXPECTED)
    assert len(result.point.positions) == 2
    assert "labels" not in result.point

    wrong_dtype = o3d.t.geometry.PointCloud(CPU)
    wrong_dtype.point.positions = o3c.Tensor.ones((0, 3),
                                                   o3c.float32, CPU)
    with pytest.raises(RuntimeError):
        wrong_dtype.append(populated)
