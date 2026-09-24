"""Fixed-input surface measurements; no assertion that changed counts are correct."""
import argparse
import gc
import hashlib
import json
import platform
from pathlib import Path
import sys

import numpy as np


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def generate(folder):
    folder.mkdir(parents=True, exist_ok=True)
    w, h, fx, fy, cx, cy = 96, 72, 85., 92., 47.5, 35.5
    v, u = np.mgrid[:h, :w]
    rx, ry = (u - cx) / fx, (v - cy) / fy
    k = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.]], np.float64)
    planes = [('flat', 0., 0.), ('x+', .4, 0.), ('x-', -.4, 0.),
              ('y+', 0., .4), ('y-', 0., -.4), ('xy+', .3, .2),
              ('xy-', -.3, -.2)]
    manifest = []
    for name, sx, sy in planes:
        for views in [1, 3]:
            centers = [(0., 0.)] if views == 1 else [(-.035, -.02), (0., 0.), (.035, .02)]
            depths, poses = [], []
            for dx, dy in centers:
                depth = (1.25 + sx * dx + sy * dy) / (1 - sx * rx - sy * ry)
                assert np.all(np.isfinite(depth)) and depth.min() > 0 and depth.max() < 3
                residual = depth - 1.25 - sx * (dx + rx * depth) - sy * (dy + ry * depth)
                assert np.max(np.abs(residual)) < 1e-12
                pose = np.eye(4, dtype=np.float64)
                pose[:2, 3] = [-dx, -dy]
                depths.append(depth.astype(np.float32)[..., None])
                poses.append(pose)
            path = folder / f'{name}-{views}.npz'
            np.savez(path, depths=np.stack(depths), poses=np.stack(poses), k=k,
                     plane=np.array([sx, sy, 1.25]), wh=np.array([w, h]))
            manifest.append({'file': path.name, 'sha256': sha(path)})
    (folder / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(f'Generated and geometrically checked {len(manifest)} fixed input files')


def measure(points, plane, k, wh):
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
        raise AssertionError('Invalid extracted positions')
    z = points[:, 2]
    good = z > 0
    uv = (k @ points[good].T).T
    uv = uv[:, :2] / uv[:, 2:3]
    w, h = wh
    roi = ((uv[:, 0] >= .25*w) & (uv[:, 0] <= .75*w) &
           (uv[:, 1] >= .25*h) & (uv[:, 1] <= .75*h))
    p = points[good][roi]
    if len(p) < 100:
        raise AssertionError(f'Insufficient central surface: {len(p)}')
    sx, sy, intercept = plane
    error = (p[:, 2] - intercept - sx*p[:, 0] - sy*p[:, 1]) / np.sqrt(1+sx*sx+sy*sy)
    return {'total_vertices': len(points), 'roi_vertices': len(p),
            'signed_mean_mm': float(1000*error.mean()),
            'rms_mm': float(1000*np.sqrt(np.mean(error**2))),
            'median_absolute_mm': float(1000*np.median(np.abs(error))),
            'p95_absolute_mm': float(1000*np.quantile(np.abs(error), .95)),
            'max_absolute_mm': float(1000*np.max(np.abs(error)))}


def probe(folder, label, version, output):
    import open3d as o3d
    c = o3d.core
    if o3d.__version__ != version:
        raise AssertionError((o3d.__version__, version))
    manifest = json.loads((folder / 'manifest.json').read_text())
    root = Path(o3d.__file__).resolve().parent
    report = {'label': label, 'version': o3d.__version__, 'python': sys.version,
              'platform': platform.platform(), 'open3d_path': str(root),
              'build_config': o3d._build_config, 'inputs': manifest,
              'modules': {str(p.relative_to(root)): sha(p) for p in root.rglob('pybind*.so')},
              'cases': []}
    if not report['modules']:
        raise AssertionError('No compiled Open3D module found')
    output.parent.mkdir(parents=True, exist_ok=True)
    for item in manifest:
        path = folder / item['file']
        assert sha(path) == item['sha256'], 'Input changed'
        with np.load(path) as data:
            for voxel in [.01, .02]:
                grid = o3d.t.geometry.VoxelBlockGrid(
                    attr_names=('tsdf', 'weight'), attr_dtypes=(c.float32, c.float32),
                    attr_channels=((1,), (1,)), voxel_size=voxel,
                    block_resolution=8, block_count=3000, device=c.Device('CPU:0'))
                intrinsic = c.Tensor(data['k'], dtype=c.float64)
                frames = []
                for depth, pose in zip(data['depths'], data['poses']):
                    image = o3d.t.geometry.Image(c.Tensor(depth))
                    extrinsic = c.Tensor(pose, dtype=c.float64)
                    blocks = grid.compute_unique_block_coordinates(image, intrinsic, extrinsic, 1., 3., 4.)
                    frames.append((blocks.clone(), image, extrinsic))
                keys = np.unique(np.concatenate([x[0].numpy() for x in frames]), axis=0)
                grid.hashmap().activate(c.Tensor(keys, dtype=c.int32))
                grid.attribute('weight')[:] = 0.
                grid.attribute('tsdf')[:] = 0.
                for blocks, image, extrinsic in frames:
                    grid.integrate(blocks, image, intrinsic, extrinsic, 1., 3., 4.)
                cloud = grid.extract_point_cloud(weight_threshold=.5)
                mesh = grid.extract_triangle_mesh(weight_threshold=.5)
                result = {'input': item['file'], 'voxel_m': voxel,
                          'active_blocks': len(keys),
                          'block_keys_sha256': hashlib.sha256(keys.tobytes()).hexdigest(),
                          'triangles': int(mesh.triangle.indices.shape[0]),
                          'pointcloud': measure(cloud.point.positions.numpy(), data['plane'], data['k'], data['wh']),
                          'mesh': measure(mesh.vertex.positions.numpy(), data['plane'], data['k'], data['wh'])}
                report['cases'].append(result)
                output.write_text(json.dumps(report, indent=2))
                print(label, item['file'], voxel, json.dumps(result['mesh']), flush=True)
                del grid, cloud, mesh, frames
                gc.collect()
    assert len(report['cases']) == 28


def compare(control, candidate, output):
    a, b = json.loads(control.read_text()), json.loads(candidate.read_text())
    assert a['inputs'] == b['inputs'] and len(a['cases']) == len(b['cases']) == 28
    lines = ['# Fixed-input TSDF surface measurements', '',
             'Completed measurements are not an all-green reconstruction verdict.', '',
             '| Plane / views | Voxel | Mean signed error, mm (old -> new) | RMS, mm (old -> new) | Triangles (old -> new) |',
             '| --- | --- | --- | --- | --- |']
    rows = []
    for x, y in zip(a['cases'], b['cases']):
        for key in ['input', 'voxel_m', 'active_blocks', 'block_keys_sha256']:
            assert x[key] == y[key], f'Uncontrolled difference in {key}'
        p, q = x['mesh'], y['mesh']
        row = {'input': x['input'], 'voxel_m': x['voxel_m'],
               'rms_improved': q['rms_mm'] <= p['rms_mm'],
               'absolute_mean_bias_improved': abs(q['signed_mean_mm']) <= abs(p['signed_mean_mm'])}
        rows.append(row)
        lines.append(f"| {x['input']} | {x['voxel_m']} | {p['signed_mean_mm']:.5f} -> {q['signed_mean_mm']:.5f} | {p['rms_mm']:.5f} -> {q['rms_mm']:.5f} | {x['triangles']} -> {y['triangles']} |")
    lines += ['', 'No reference counts or tolerances were updated. Both input hashes and active block keys matched.',
              'CPU ARM64 only. Other backends, occlusions, curved surfaces, and the original Redwood count failures remain separate validation.']
    output.write_text('\n'.join(lines) + '\n')
    output.with_suffix('.json').write_text(json.dumps(rows, indent=2))
    print('\n'.join(lines))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['generate', 'probe', 'compare'])
    p.add_argument('--inputs', type=Path, default=Path('results/inputs'))
    p.add_argument('--label')
    p.add_argument('--version')
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    if args.mode == 'generate': generate(args.inputs)
    elif args.mode == 'probe': probe(args.inputs, args.label, args.version, args.output)
    else: compare(Path('results/control.json'), Path('results/candidate.json'), args.output)
