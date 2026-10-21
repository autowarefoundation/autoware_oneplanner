// Copyright 2026 TIER IV, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "autoware/oneplanner/bev_feature/temporal_bev_cache.hpp"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <array>
#include <vector>

namespace autoware::oneplanner
{

namespace
{
constexpr int64_t kSize = 180;

std::vector<float> slot_zero(const float * device_history)
{
  std::vector<float> host(static_cast<size_t>(kSize * kSize));
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  EXPECT_EQ(
    cudaMemcpy(host.data(), device_history, host.size() * sizeof(float), cudaMemcpyDeviceToHost),
    cudaSuccess);
  return host;
}
}  // namespace

// planning_time: the plan starts where the ego is when planning runs, one BEV cell ahead of
// where the newest cloud was recorded. The newest map is then warped like the older ones
// (as in training); planned at the cloud's own pose, it is copied.
TEST(TemporalBevCacheTest, WarpsTheNewestMapIntoALaterPlanningPose)
{
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    GTEST_SKIP() << "no CUDA device";
  }
  const double cell = 2.0 * 122.4 / static_cast<double>(kSize);
  TemporalBevCache::Config config;
  config.frames = 2;
  config.interval_seconds = 0.1;
  TemporalBevCache cache(config, 1, kSize, kSize);

  std::vector<float> map(static_cast<size_t>(kSize * kSize), 0.0f);
  map[100 * kSize + 45] = 1.0f;  // row = x forward, column = y left
  float * device_map = nullptr;
  ASSERT_EQ(cudaMalloc(&device_map, map.size() * sizeof(float)), cudaSuccess);
  ASSERT_EQ(
    cudaMemcpy(device_map, map.data(), map.size() * sizeof(float), cudaMemcpyHostToDevice),
    cudaSuccess);

  const std::array<double, 4> older{1000.0 - cell, 2000.0, 1.0, 0.0};
  const std::array<double, 4> recorded{1000.0, 2000.0, 1.0, 0.0};
  cache.insert(device_map, older, rclcpp::Time(10, 0), nullptr);
  cache.insert(device_map, recorded, rclcpp::Time(10, 100000000), nullptr);
  ASSERT_TRUE(cache.ready());

  EXPECT_EQ(slot_zero(cache.build_history(nullptr, recorded)), map);

  const std::array<double, 4> planning{1000.0 + cell, 2000.0, 1.0, 0.0};
  const auto warped = slot_zero(cache.build_history(nullptr, planning));
  // One cell further on, what the cloud saw at row 100 is one row behind the ego.
  EXPECT_NEAR(warped[99 * kSize + 45], 1.0f, 1e-4f);
  EXPECT_NEAR(warped[100 * kSize + 45], 0.0f, 1e-4f);

  cudaFree(device_map);
}

}  // namespace autoware::oneplanner
