// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2026 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <utility>

#include "open3d/core/Tensor.h"
#include "open3d/t/geometry/RaycastingScene.h"

namespace open3d {
namespace tests {
namespace {

const core::Device kCPU("CPU:0");

// One triangle fills the unit simplex in the horizontal z plane.
uint32_t AddTriangle(t::geometry::RaycastingScene& scene, float z) {
    auto vertices = core::Tensor::Init<float>(
            {{0.f, 0.f, z}, {1.f, 0.f, z}, {0.f, 1.f, z}}, kCPU);
    auto triangles =
            core::Tensor::Init<uint32_t>({{0u, 1u, 2u}}, kCPU);
    return scene.AddTriangles(vertices, triangles);
}

// The downward unit ray intersects the simplex at parameter t = 1 - z.
std::pair<float, uint32_t> CastDown(
        const t::geometry::RaycastingScene& scene) {
    auto ray = core::Tensor::Init<float>(
            {{0.25f, 0.25f, 1.f, 0.f, 0.f, -1.f}}, kCPU);
    auto result = scene.CastRays(ray);
    return {result.at("t_hit").GetDataPtr<float>()[0],
            result.at("geometry_ids").GetDataPtr<uint32_t>()[0]};
}

}  // namespace

TEST(RaycastingSceneCopyCPU, EmptyCopyMisses) {
    t::geometry::RaycastingScene source(2, kCPU);
    t::geometry::RaycastingScene copy(source);

    const auto hit = CastDown(copy);
    EXPECT_TRUE(std::isinf(hit.first));
    EXPECT_EQ(hit.second, t::geometry::RaycastingScene::INVALID_ID());
}

TEST(RaycastingSceneCopyCPU, DeepCopyPreservesHitsAndIsolation) {
    t::geometry::RaycastingScene source(2, kCPU);
    const uint32_t first_id = AddTriangle(source, 0.f);

    t::geometry::RaycastingScene clone(source);
    const uint32_t second_id = AddTriangle(clone, 0.5f);
    EXPECT_NE(first_id, second_id);

    const auto original_hit = CastDown(source);
    const auto cloned_hit = CastDown(clone);
    EXPECT_NEAR(original_hit.first, 1.f, 1e-6f);
    EXPECT_EQ(original_hit.second, first_id);
    EXPECT_NEAR(cloned_hit.first, 0.5f, 1e-6f);
    EXPECT_EQ(cloned_hit.second, second_id);
}

TEST(RaycastingSceneCopyCPU, AssignmentStressReleasesOldScenes) {
    t::geometry::RaycastingScene source(2, kCPU);
    const uint32_t id = AddTriangle(source, 0.f);

    // On an ASan/LSan build this exercises release of both replaced handles.
    for (int iteration = 0; iteration < 128; ++iteration) {
        t::geometry::RaycastingScene destination(3, kCPU);
        AddTriangle(destination, 0.75f);
        destination = source;

        auto copied_hit = CastDown(destination);
        EXPECT_NEAR(copied_hit.first, 1.f, 1e-6f);
        EXPECT_EQ(copied_hit.second, id);

        t::geometry::RaycastingScene movable(1, kCPU);
        AddTriangle(movable, 0.5f);
        destination = std::move(movable);
        EXPECT_NEAR(CastDown(destination).first, 0.5f, 1e-6f);

        destination = source;
        EXPECT_NEAR(CastDown(destination).first, 1.f, 1e-6f);

        t::geometry::RaycastingScene moved(std::move(destination));
        EXPECT_NEAR(CastDown(moved).first, 1.f, 1e-6f);
    }
}

}  // namespace tests
}  // namespace open3d
