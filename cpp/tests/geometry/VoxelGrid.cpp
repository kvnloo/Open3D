// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <stdexcept>
#include <vector>

#include "open3d/geometry/VoxelGrid.h"

#include "open3d/geometry/LineSet.h"
#include "open3d/geometry/PointCloud.h"
#include "open3d/geometry/TriangleMesh.h"
#include "open3d/visualization/utility/DrawGeometry.h"
#include "tests/Tests.h"

namespace open3d {
namespace tests {

TEST(VoxelGrid, Bounds) {
    auto voxel_grid = std::make_shared<geometry::VoxelGrid>();
    voxel_grid->origin_ = Eigen::Vector3d(0, 0, 0);
    voxel_grid->voxel_size_ = 5;
    voxel_grid->AddVoxel(geometry::Voxel(Eigen::Vector3i(1, 0, 0)));
    voxel_grid->AddVoxel(geometry::Voxel(Eigen::Vector3i(0, 2, 0)));
    voxel_grid->AddVoxel(geometry::Voxel(Eigen::Vector3i(0, 0, 3)));
    ExpectEQ(voxel_grid->GetMinBound(), Eigen::Vector3d(0, 0, 0));
    ExpectEQ(voxel_grid->GetMaxBound(), Eigen::Vector3d(10, 15, 20));
}

TEST(VoxelGrid, GetVoxel) {
    auto voxel_grid = std::make_shared<geometry::VoxelGrid>();
    voxel_grid->origin_ = Eigen::Vector3d(0, 0, 0);
    voxel_grid->voxel_size_ = 5;
    ExpectEQ(voxel_grid->GetVoxel(Eigen::Vector3d(0, 0, 0)),
             Eigen::Vector3i(0, 0, 0));
    ExpectEQ(voxel_grid->GetVoxel(Eigen::Vector3d(0, 1, 0)),
             Eigen::Vector3i(0, 0, 0));
    // Test near boundary voxel_size_ == 5
    ExpectEQ(voxel_grid->GetVoxel(Eigen::Vector3d(0, 4.9, 0)),
             Eigen::Vector3i(0, 0, 0));
    ExpectEQ(voxel_grid->GetVoxel(Eigen::Vector3d(0, 5, 0)),
             Eigen::Vector3i(0, 1, 0));
    ExpectEQ(voxel_grid->GetVoxel(Eigen::Vector3d(0, 5.1, 0)),
             Eigen::Vector3i(0, 1, 0));
}

TEST(VoxelGrid, CreateFromSparseIndicesPreservesCentersAndColors) {
    // Explicit indices must map to the same voxel centers as other factories,
    // without passing through a point-cloud voxelization path.
    const std::vector<Eigen::Vector3i> indices = {
            Eigen::Vector3i(0, 0, 0), Eigen::Vector3i(2, 1, 3)};
    const std::vector<Eigen::Vector3d> colors = {
            Eigen::Vector3d(1, 0, 0), Eigen::Vector3d(0, 1, 0)};
    auto grid = geometry::VoxelGrid::CreateFromSparseIndices(
            indices, colors, Eigen::Vector3d(1, 2, -1), 0.5);

    ASSERT_EQ(grid->voxels_.size(), 2u);
    EXPECT_DOUBLE_EQ(grid->voxel_size_, 0.5);
    ExpectEQ(grid->GetVoxelCenterCoordinate(indices[0]),
             Eigen::Vector3d(1.25, 2.25, -0.75));
    ExpectEQ(grid->GetVoxelCenterCoordinate(indices[1]),
             Eigen::Vector3d(2.25, 2.75, 0.75));
    const auto first = grid->voxels_.find(indices[0]);
    const auto second = grid->voxels_.find(indices[1]);
    ASSERT_NE(first, grid->voxels_.end());
    ASSERT_NE(second, grid->voxels_.end());
    ExpectEQ(first->second.color_, colors[0]);
    ExpectEQ(second->second.color_, colors[1]);
}

TEST(VoxelGrid, CreateFromSparseIndicesAveragesDuplicateColors) {
    // Multiple colors for one index match the existing AVG point-cloud path,
    // regardless of the insertion order.
    const std::vector<Eigen::Vector3i> indices = {
            Eigen::Vector3i(1, 0, 0), Eigen::Vector3i(0, 1, 0),
            Eigen::Vector3i(1, 0, 0), Eigen::Vector3i(1, 0, 0)};
    const std::vector<Eigen::Vector3d> colors = {
            Eigen::Vector3d(1, 0, 0), Eigen::Vector3d(0, 1, 0),
            Eigen::Vector3d(0, 0, 1), Eigen::Vector3d(0, 1, 0)};
    const auto sparse = geometry::VoxelGrid::CreateFromSparseIndices(
            indices, colors, Eigen::Vector3d::Zero(), 1.0);

    geometry::PointCloud cloud;
    for (size_t i = 0; i < indices.size(); ++i) {
        cloud.points_.push_back(indices[i].cast<double>() +
                                Eigen::Vector3d(0.5, 0.5, 0.5));
        cloud.colors_.push_back(colors[i]);
    }
    const auto pooled = geometry::VoxelGrid::CreateFromPointCloudWithinBounds(
            cloud, 1.0, Eigen::Vector3d::Zero(), Eigen::Vector3d(3, 3, 3),
            geometry::VoxelGrid::VoxelPoolingMode::AVG);

    ASSERT_EQ(sparse->voxels_.size(), 2u);
    ASSERT_EQ(pooled->voxels_.size(), 2u);
    for (const auto &item : sparse->voxels_) {
        const auto pooled_voxel = pooled->voxels_.find(item.first);
        ASSERT_NE(pooled_voxel, pooled->voxels_.end());
        ExpectEQ(item.second.color_, pooled_voxel->second.color_);
    }
    const auto repeated = sparse->voxels_.find(Eigen::Vector3i(1, 0, 0));
    ASSERT_NE(repeated, sparse->voxels_.end());
    ExpectEQ(repeated->second.color_,
             Eigen::Vector3d(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0));

    const std::vector<Eigen::Vector3i> reversed_indices(indices.rbegin(),
                                                        indices.rend());
    const std::vector<Eigen::Vector3d> reversed_colors(colors.rbegin(),
                                                       colors.rend());
    const auto reversed = geometry::VoxelGrid::CreateFromSparseIndices(
            reversed_indices, reversed_colors, Eigen::Vector3d::Zero(), 1.0);
    ASSERT_EQ(reversed->voxels_.size(), 2u);
    for (const auto &item : sparse->voxels_) {
        ExpectEQ(item.second.color_, reversed->voxels_.at(item.first).color_);
    }
}

TEST(VoxelGrid, CreateFromSparseIndicesRejectsNonpositiveVoxelSize) {
    // Reject the same invalid sizes as CreateFromPointCloudWithinBounds.
    const std::vector<Eigen::Vector3i> indices = {
            Eigen::Vector3i(0, 0, 0)};
    const std::vector<Eigen::Vector3d> colors = {
            Eigen::Vector3d(1, 0, 0)};
    for (double size : {0.0, -1.0}) {
        EXPECT_THROW(geometry::VoxelGrid::CreateFromSparseIndices(
                             indices, colors, Eigen::Vector3d::Zero(), size),
                     std::runtime_error);
    }
}

TEST(VoxelGrid, Visualization) {
    auto voxel_grid = std::make_shared<geometry::VoxelGrid>();
    voxel_grid->origin_ = Eigen::Vector3d(0, 0, 0);
    voxel_grid->voxel_size_ = 5;
    voxel_grid->AddVoxel(geometry::Voxel(Eigen::Vector3i(0, 0, 0),
                                         Eigen::Vector3d(0.9, 0, 0)));
    voxel_grid->AddVoxel(geometry::Voxel(Eigen::Vector3i(0, 1, 0),
                                         Eigen::Vector3d(0.9, 0.9, 0)));

    // Uncomment the line below for visualization test
    // visualization::DrawGeometries({voxel_grid});
}

}  // namespace tests
}  // namespace open3d
