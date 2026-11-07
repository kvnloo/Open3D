# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2026 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------
# Based on the comparison example contributed by shevi-kukis:
# https://github.com/isl-org/Open3D/pull/7416

import argparse

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


def preprocess_point_cloud(pcd, voxel_size):
    """Downsample a point cloud and compute FPFH registration features."""
    pcd_down = pcd.voxel_down_sample(voxel_size)
    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature,
                                             max_nn=100))
    return pcd_down, pcd_fpfh


def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    """Use feature matching and RANSAC to initialize source alignment."""
    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result


def compute_deviation_distances(source, target, target_mesh=None):
    """Compute nearest geometry distances after source alignment.

    Mesh targets use the exact triangle surface on the CPU rather than the
    sampled points used for registration.
    """
    if target_mesh is None:
        return np.asarray(source.compute_point_cloud_distance(target))

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(target_mesh))
    query_points = o3d.core.Tensor(
        np.asarray(source.points, dtype=np.float32))
    return scene.compute_distance(query_points).numpy()


def compute_heatmap_colors(distances):
    """Apply the jet heatmap, keeping zero deviation at the blue low end.

    When every distance is zero, avoid division by zero and NaN colors.
    """
    values = np.asarray(distances, dtype=np.float64)
    mean_distance = values.mean() if values.size else 0.0
    scale = 2.0 * mean_distance
    normalized = values / scale if scale > 0 else np.zeros_like(values)
    return plt.get_cmap("jet")(normalized)[:, :3]


def run_comparison(source_path, target_path, is_target_mesh=False):
    """Load, register, color and show the source against target geometry."""
    print(":: Loading datasets...")
    mesh = None
    if is_target_mesh:
        mesh = o3d.io.read_triangle_mesh(target_path)
        mesh.compute_vertex_normals()
        # Sample only for registration: final deviations use mesh triangles.
        target = mesh.sample_points_poisson_disk(number_of_points=50000)
    else:
        target = o3d.io.read_point_cloud(target_path)

    source = o3d.io.read_point_cloud(source_path)

    voxel_size = 0.05
    print(":: Performing Global Registration (RANSAC)...")
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)

    global_result = execute_global_registration(source_down, target_down,
                                                 source_fpfh, target_fpfh,
                                                 voxel_size)
    source.transform(global_result.transformation)

    print(":: Performing Fine Registration (ICP)...")
    target.estimate_normals()
    source.estimate_normals()
    icp_result = o3d.pipelines.registration.registration_icp(
        source, target, voxel_size, np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    source.transform(icp_result.transformation)

    # Use the actual CAD triangles after registration, not the sample proxy.
    distances = compute_deviation_distances(source, target, mesh)
    source.colors = o3d.utility.Vector3dVector(
        compute_heatmap_colors(distances))

    print(f":: Final Mean Distance: {np.mean(distances):.6f}")
    o3d.visualization.draw_geometries([source],
                                      window_name="Aligned Heatmap Result")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Point cloud to mesh/cloud comparison with heatmap")
    parser.add_argument("--source", type=str, help="Path to source file")
    parser.add_argument("--target", type=str, help="Path to target file")
    parser.add_argument("--is_mesh", action="store_true",
                        help="Flag if target is a mesh")
    args = parser.parse_args()

    if args.source and args.target:
        run_comparison(args.source, args.target, args.is_mesh)
    else:
        print(":: No paths provided. Running official Demo data...")
        knot = o3d.data.KnotMesh()
        demo_pcd = o3d.data.DemoICPPointClouds()
        run_comparison(demo_pcd.paths[0], knot.path, is_target_mesh=True)
