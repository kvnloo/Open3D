"""Compare reconstructed meshes at identical rays, independent of vertex counts."""
import hashlib
import numpy as np


def measure_rays(mesh, plane, k, wh):
    import open3d as o3d
    w, h = (int(x) for x in wh)
    v, u = np.mgrid[h//4:3*h//4, w//4:3*w//4]
    directions = np.column_stack(((u.ravel()-k[0, 2])/k[0, 0],
                                  (v.ravel()-k[1, 2])/k[1, 1],
                                  np.ones(u.size))).astype(np.float32)
    rays = np.column_stack((np.zeros_like(directions), directions))
    scene = o3d.t.geometry.RaycastingScene(nthreads=1)
    scene.add_triangles(mesh)
    hit = scene.cast_rays(o3d.core.Tensor(rays))['t_hit'].numpy()
    covered = np.isfinite(hit) & (hit > 0)
    if not np.all(covered):
        raise AssertionError(f'Incomplete fixed-ray coverage: {covered.sum()}/{len(hit)}')
    sx, sy, intercept = plane
    # Ray t is axial depth because direction.z == 1; directions are not unit vectors.
    # Compare hit points to the analytic world plane, not a second generated mesh.
    p = directions.astype(np.float64) * hit[:, None]
    signed_mm = 1000 * (p[:, 2]-intercept-sx*p[:, 0]-sy*p[:, 1]) / np.sqrt(1+sx*sx+sy*sy)
    return {'rays': len(hit), 'covered_rays': int(covered.sum()),
            'rays_sha256': hashlib.sha256(rays.tobytes()).hexdigest(),
            'signed_mean_mm': float(signed_mm.mean()),
            'rms_mm': float(np.sqrt(np.mean(signed_mm**2))),
            'median_absolute_mm': float(np.median(np.abs(signed_mm))),
            'p95_absolute_mm': float(np.quantile(np.abs(signed_mm), .95)),
            'max_absolute_mm': float(np.max(np.abs(signed_mm)))}
