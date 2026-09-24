import json
import tempfile
from itertools import product
from pathlib import Path

import numpy as np
import open3d as o3d
import open3d.core as o3c


DEVICE = o3c.Device("CPU:0")
BLOCK_RESOLUTION = 2


def make_keys(values):
    return np.asarray(list(product(values, repeat=3)), dtype=np.int32)


CASES = {
    "positive": make_keys([0, 1, 2, 3]),
    "negative": make_keys([-4, -3, -2, -1]),
    "mixed": make_keys([-2, -1, 0, 1]),
}


def sorted_rows(array):
    array = np.asarray(array)
    order = np.lexsort((array[:, 2], array[:, 1], array[:, 0]))
    return array[order]


def run_case(name, keys_np, directory):
    count = len(keys_np)
    grid = o3d.t.geometry.VoxelBlockGrid(
        attr_names=("tsdf", "weight"),
        attr_dtypes=(o3c.float32, o3c.float32),
        attr_channels=((1,), (1,)),
        voxel_size=0.01,
        block_resolution=BLOCK_RESOLUTION,
        block_count=count,
        device=DEVICE,
    )

    shape = (count, BLOCK_RESOLUTION, BLOCK_RESOLUTION, BLOCK_RESOLUTION, 1)
    tsdf_np = np.empty(shape, dtype=np.float32)
    weight_np = np.empty(shape, dtype=np.float32)
    for i in range(count):
        tsdf_np[i].fill((i - count / 2) / 100.0)
        weight_np[i].fill(i + 1)

    keys = o3c.Tensor(keys_np, dtype=o3c.int32, device=DEVICE)
    tsdf = o3c.Tensor(tsdf_np, dtype=o3c.float32, device=DEVICE)
    weight = o3c.Tensor(weight_np, dtype=o3c.float32, device=DEVICE)

    inserted, insert_masks = grid.hashmap().insert(keys, [tsdf, weight])
    inserted_np = inserted.cpu().numpy().reshape(-1)
    insert_masks_np = insert_masks.cpu().numpy().reshape(-1)
    assert insert_masks_np.all(), (
        f"{name}: initial insert rejected keys: "
        f"{np.where(~insert_masks_np)[0].tolist()}"
    )
    assert np.all(inserted_np >= 0), (
        f"{name}: initial insert returned negative buffer index "
        f"{inserted_np.min()}"
    )
    assert grid.hashmap().size() == count

    path = directory / f"{name}.npz"
    grid.save(str(path))
    loaded = o3d.t.geometry.VoxelBlockGrid.load(str(path))

    found, found_masks = loaded.hashmap().find(keys)
    found_np = found.cpu().numpy().reshape(-1)
    found_masks_np = found_masks.cpu().numpy().reshape(-1)

    assert loaded.hashmap().size() == count, (
        f"{name}: loaded size {loaded.hashmap().size()} != {count}"
    )
    assert found_masks_np.all(), (
        f"{name}: loaded map cannot find keys at query indices "
        f"{np.where(~found_masks_np)[0].tolist()}"
    )
    assert np.all(found_np >= 0), (
        f"{name}: loaded find returned negative buffer index {found_np.min()}"
    )

    active = loaded.hashmap().active_buf_indices().cpu().numpy().reshape(-1)
    assert len(active) == count
    assert np.all(active >= 0)

    loaded_key_buffer = loaded.hashmap().key_tensor().cpu().numpy()
    loaded_active_keys = loaded_key_buffer[active]
    np.testing.assert_array_equal(sorted_rows(loaded_active_keys),
                                  sorted_rows(keys_np))

    loaded_tsdf = loaded.attribute("tsdf").cpu().numpy()[found_np]
    loaded_weight = loaded.attribute("weight").cpu().numpy()[found_np]
    np.testing.assert_array_equal(loaded_tsdf, tsdf_np)
    np.testing.assert_array_equal(loaded_weight, weight_np)

    # A second round trip protects against a file that only happens to load once.
    path2 = directory / f"{name}-second.npz"
    loaded.save(str(path2))
    loaded2 = o3d.t.geometry.VoxelBlockGrid.load(str(path2))
    found2, masks2 = loaded2.hashmap().find(keys)
    found2_np = found2.cpu().numpy().reshape(-1)
    masks2_np = masks2.cpu().numpy().reshape(-1)
    assert masks2_np.all()
    assert np.all(found2_np >= 0)
    np.testing.assert_array_equal(
        loaded2.attribute("tsdf").cpu().numpy()[found2_np], tsdf_np)
    np.testing.assert_array_equal(
        loaded2.attribute("weight").cpu().numpy()[found2_np], weight_np)

    return {
        "key_count": count,
        "key_min": keys_np.min(axis=0).tolist(),
        "key_max": keys_np.max(axis=0).tolist(),
        "loaded_size": int(loaded.hashmap().size()),
        "loaded_capacity": int(loaded.hashmap().capacity()),
        "min_found_buffer_index": int(found_np.min()),
        "max_found_buffer_index": int(found_np.max()),
        "roundtrips": 2,
    }


def main():
    results = {
        "open3d_version": o3d.__version__,
        "build_config": dict(o3d._build_config),
        "device": str(DEVICE),
        "cases": {},
    }
    failures = {}
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        for name, keys in CASES.items():
            try:
                results["cases"][name] = run_case(name, keys, directory)
            except Exception as exc:
                failures[name] = f"{type(exc).__name__}: {exc}"

    results["failures"] = failures
    Path("vbg-6795-results.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(results, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(f"Round-trip failures: {failures}")


if __name__ == "__main__":
    main()
