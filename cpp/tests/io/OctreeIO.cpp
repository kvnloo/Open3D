// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2026 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "open3d/io/OctreeIO.h"

#include <json/json.h>

#include <cstdio>
#include <fstream>

#include "open3d/data/Dataset.h"
#include "open3d/geometry/Octree.h"
#include "open3d/geometry/PointCloud.h"
#include "open3d/io/PointCloudIO.h"
#include "open3d/utility/FileSystem.h"
#include "open3d/utility/IJsonConvertible.h"
#include "tests/Tests.h"

namespace open3d {
namespace tests {

void WriteReadAndAssertEqual(const geometry::Octree& src_octree,
                             const std::string& file_name) {
    // Write to file
    std::string full_file_name =
            utility::filesystem::GetTempDirectoryPath() + file_name;
    EXPECT_TRUE(io::WriteOctree(full_file_name, src_octree));

    // Read from file
    geometry::Octree dst_octree;
    EXPECT_TRUE(io::ReadOctree(full_file_name, dst_octree));
    EXPECT_TRUE(src_octree == dst_octree);
}

TEST(OctreeIO, EmptyTree) {
    geometry::Octree octree(10);
    ExpectEQ(octree.origin_, Eigen::Vector3d(0, 0, 0));
    EXPECT_EQ(octree.size_, 0);

    WriteReadAndAssertEqual(octree, "/temp_octree.json");
    WriteReadAndAssertEqual(octree, "/temp_octree.bin");
}

TEST(OctreeIO, ZeroDepth) {
    geometry::Octree octree(0, Eigen::Vector3d(-1, -1, -1), 2);
    Eigen::Vector3d point(0, 0, 0);
    Eigen::Vector3d color(0, 0.1, 0.2);
    octree.InsertPoint(point, geometry::OctreeColorLeafNode::GetInitFunction(),
                       geometry::OctreeColorLeafNode::GetUpdateFunction(color));

    WriteReadAndAssertEqual(octree, "/temp_octree.json");
    WriteReadAndAssertEqual(octree, "/temp_octree.bin");
}

TEST(OctreeIO, FileIOFragment) {
    // Create octree
    geometry::PointCloud pcd;
    data::PLYPointCloud pointcloud_ply;
    io::ReadPointCloud(pointcloud_ply.GetPath(), pcd);
    size_t max_depth = 6;
    geometry::Octree octree(max_depth);
    octree.ConvertFromPointCloud(pcd, 0.01);

    WriteReadAndAssertEqual(octree, "/temp_octree.json");
    WriteReadAndAssertEqual(octree, "/temp_octree.bin");
}

TEST(OctreeIO, FileIOSevenCubes) {
    // Build octree
    std::vector<Eigen::Vector3d> points{
            Eigen::Vector3d(0.5, 0.5, 0.5), Eigen::Vector3d(1.5, 0.5, 0.5),
            Eigen::Vector3d(0.5, 1.5, 0.5), Eigen::Vector3d(1.5, 1.5, 0.5),
            Eigen::Vector3d(0.5, 0.5, 1.5), Eigen::Vector3d(1.5, 0.5, 1.5),
            Eigen::Vector3d(0.5, 1.5, 1.5)};
    std::vector<Eigen::Vector3d> colors{
            Eigen::Vector3d(0.0, 0.0, 0.0),  Eigen::Vector3d(0.25, 0.0, 0.0),
            Eigen::Vector3d(0.0, 0.25, 0.0), Eigen::Vector3d(0.25, 0.25, 0.0),
            Eigen::Vector3d(0.0, 0.0, 0.25), Eigen::Vector3d(0.25, 0.0, 0.25),
            Eigen::Vector3d(0.0, 0.25, 0.25)};
    geometry::Octree octree(1, Eigen::Vector3d(0, 0, 0), 2);
    for (size_t i = 0; i < points.size(); ++i) {
        octree.InsertPoint(
                points[i], geometry::OctreeColorLeafNode::GetInitFunction(),
                geometry::OctreeColorLeafNode::GetUpdateFunction(colors[i]));
    }

    WriteReadAndAssertEqual(octree, "/temp_octree.json");
    WriteReadAndAssertEqual(octree, "/temp_octree.bin");
}

TEST(OctreeIO, RejectMalformedBinaryWithoutMutation) {
    geometry::Octree empty(2, Eigen::Vector3d(0, 0, 0), 2.0);
    std::string header;
    ASSERT_TRUE(empty.SerializeToBinaryStream(header));
    ASSERT_EQ(header.size(), 40u);

    const std::string filename =
            utility::filesystem::GetTempDirectoryPath() +
            "/octree_invalid_root.bin";
    {
        std::ofstream file(filename, std::ios::binary);
        ASSERT_TRUE(file.is_open());
        file.write(header.data(), static_cast<std::streamsize>(header.size()));
        file.put('\xff');
        file.close();
        ASSERT_TRUE(file.good());
    }

    geometry::Octree destination(7, Eigen::Vector3d(3, 4, 5), 6.0);
    EXPECT_FALSE(io::ReadOctree(filename, destination));
    EXPECT_EQ(destination.max_depth_, 7u);
    EXPECT_DOUBLE_EQ(destination.size_, 6.0);
    ExpectEQ(destination.origin_, Eigen::Vector3d(3, 4, 5));
    EXPECT_EQ(destination.root_node_, nullptr);
    std::remove(filename.c_str());
}

}  // namespace tests
}  // namespace open3d
