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

// Adapted from autoware_bevfusion's preprocess_kernel.cu (autoware_universe), itself derived
// from NVIDIA's CUDA-BEVFusion (SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION &
// AFFILIATES. SPDX-License-Identifier: Apache-2.0).

#include "autoware/oneplanner/bev_feature/voxelizer.hpp"

#include <spconvlib/spconv/csrc/sparse/all/ops3d/Point2Voxel.h>  // cSpell:ignore spconvlib
#include <tensorview/tensor.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <tuple>

namespace autoware::oneplanner::bev
{

namespace
{

constexpr unsigned int kThreadsPerBlock = 256;

template <bool USE_INTENSITY>
__global__ void generate_points_kernel(
  const InputPoint * __restrict__ input, const std::size_t num_points, const int num_features,
  float * __restrict__ output)
{
  const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= num_points) {
    return;
  }
  const InputPoint & point = input[index];
  float * row = output + index * num_features;
  row[0] = point.x;
  row[1] = point.y;
  row[2] = point.z;
  if (USE_INTENSITY) {
    row[3] = static_cast<float>(point.intensity);
    row[4] = 0.0f;  // time lag of a single sweep
  } else {
    row[3] = 0.0f;
  }
}

}  // namespace

struct Voxelizer::Scratch
{
  tv::Tensor hash_key_value;
  tv::Tensor point_indices_data;
  tv::Tensor points_voxel_id;
};

Voxelizer::Voxelizer(const VoxelizerConfig & config, cudaStream_t stream)
: config_(config), stream_(stream), scratch_(std::make_unique<Scratch>())
{
  scratch_->hash_key_value = tv::empty({config.cloud_capacity * 2}, tv::custom128, 0);
  scratch_->point_indices_data = tv::empty({config.cloud_capacity}, tv::int64, 0);
  scratch_->points_voxel_id = tv::empty({config.cloud_capacity}, tv::int64, 0);
}

Voxelizer::~Voxelizer() = default;

cudaError_t Voxelizer::generate_points(
  const InputPoint * points, const std::size_t num_points, float * output)
{
  if (num_points == 0) {
    return cudaSuccess;
  }
  const dim3 blocks((num_points + kThreadsPerBlock - 1) / kThreadsPerBlock);
  const dim3 threads(kThreadsPerBlock);
  const auto num_features = static_cast<int>(config_.num_point_features());
  if (config_.use_intensity) {
    generate_points_kernel<true>
      <<<blocks, threads, 0, stream_>>>(points, num_points, num_features, output);
  } else {
    generate_points_kernel<false>
      <<<blocks, threads, 0, stream_>>>(points, num_points, num_features, output);
  }
  return cudaGetLastError();
}

std::size_t Voxelizer::generate_voxels(
  const float * points, const unsigned int num_points, float * voxel_features,
  std::int32_t * voxel_coords, std::int32_t * num_points_per_voxel)
{
  using Point2VoxelGPU3D = spconvlib::spconv::csrc::sparse::all::ops3d::Point2Voxel;

  // spconv orders every axis z, y, x.
  const auto & range = config_.point_cloud_range;
  const auto & size = config_.voxel_size;
  const std::array<float, 3> voxel_size_zyx{size[2], size[1], size[0]};
  const std::array<std::int32_t, 3> grid_size_zyx{
    static_cast<std::int32_t>(config_.grid_size(2)),
    static_cast<std::int32_t>(config_.grid_size(1)),
    static_cast<std::int32_t>(config_.grid_size(0))};
  const std::array<std::int64_t, 3> grid_stride{
    config_.grid_size(1) * config_.grid_size(0), config_.grid_size(0), 1};
  const std::array<float, 6> range_zyx{range[2], range[1], range[0], range[5], range[4], range[3]};

  const auto num_features = config_.num_point_features();
  tv::Tensor point_rows =
    tv::from_blob(points, {static_cast<std::int64_t>(num_points), num_features}, tv::float32, 0);
  tv::Tensor voxels = tv::from_blob(
    voxel_features, {config_.max_num_voxels, config_.max_points_per_voxel, num_features},
    tv::float32, 0);
  tv::Tensor coords = tv::from_blob(voxel_coords, {config_.max_num_voxels, 3}, tv::int32, 0);
  tv::Tensor counts = tv::from_blob(num_points_per_voxel, {config_.max_num_voxels}, tv::int32, 0);

  const auto result = Point2VoxelGPU3D::point_to_voxel_hash_static(
    point_rows, voxels, coords, counts, scratch_->hash_key_value, scratch_->point_indices_data,
    scratch_->points_voxel_id, voxel_size_zyx, grid_size_zyx, grid_stride, range_zyx, true, false,
    reinterpret_cast<std::uintptr_t>(stream_));
  return static_cast<std::size_t>(std::get<0>(result).dim(0));
}

}  // namespace autoware::oneplanner::bev
