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

// Adapted from autoware_bevfusion's PreprocessCuda (autoware_universe), itself derived from
// NVIDIA's CUDA-BEVFusion (SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION &
// AFFILIATES. SPDX-License-Identifier: Apache-2.0).

#ifndef AUTOWARE__ONEPLANNER__BEV_FEATURE__VOXELIZER_HPP_
#define AUTOWARE__ONEPLANNER__BEV_FEATURE__VOXELIZER_HPP_

#include <cuda_runtime_api.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace autoware::oneplanner::bev
{

/**
 * @brief One point of an Autoware `PointXYZIRC` cloud, the concatenated cloud's layout.
 *
 * x, y, z as float32 at offsets 0, 4, 8; intensity as uint8 at 12; return type and channel
 * after it. TrtBevFeatureExtractor checks an incoming cloud's fields against this layout.
 */
struct InputPoint
{
  float x;
  float y;
  float z;
  std::uint8_t intensity;
  std::uint8_t return_type;
  std::uint16_t channel;
};
static_assert(sizeof(InputPoint) == 16, "InputPoint must match the 16-byte PointXYZIRC layout");

/**
 * @brief The voxelization geometry of a BEVFusion-L LiDAR branch.
 *
 * The grid is derived exactly as autoware_bevfusion derives it, in float arithmetic, so a
 * range and voxel size from the same configuration give the same grid.
 */
struct VoxelizerConfig
{
  std::int64_t cloud_capacity{0};
  std::int64_t max_points_per_voxel{0};
  std::int64_t max_num_voxels{0};
  bool use_intensity{false};
  //! x min, y min, z min, x max, y max, z max [m].
  std::array<float, 6> point_cloud_range{};
  //! x, y, z [m].
  std::array<float, 3> voxel_size{};

  //! x, y, z, (intensity,) time lag.
  std::int64_t num_point_features() const { return use_intensity ? 5 : 4; }
  std::int64_t grid_size(const int axis) const
  {
    return static_cast<std::int64_t>(
      (point_cloud_range[axis + 3] - point_cloud_range[axis]) / voxel_size[axis]);
  }
};

/**
 * @class Voxelizer
 * @brief Point cloud to padded voxels on the GPU, with spconv's hash voxelization.
 *
 * The same two steps autoware_bevfusion runs before its LiDAR backbone: the points become
 * `[N, F]` feature rows (x, y, z, intensity, time lag), then spconv's `point_to_voxel_hash_static`
 * groups them into at most `max_num_voxels` voxels of at most `max_points_per_voxel` points.
 */
class Voxelizer
{
public:
  Voxelizer(const VoxelizerConfig & config, cudaStream_t stream);
  ~Voxelizer();

  Voxelizer(const Voxelizer &) = delete;
  Voxelizer & operator=(const Voxelizer &) = delete;

  /**
   * @brief Write `[num_points, F]` feature rows for a single sweep already in the target frame
   *        (identity transform, zero time lag). Queued on the stream, not waited for.
   */
  cudaError_t generate_points(const InputPoint * points, std::size_t num_points, float * output);

  /**
   * @brief Voxelize `[num_points, F]` feature rows into the padded voxel buffers.
   * @return The number of voxels produced; waits for the device to know it.
   */
  std::size_t generate_voxels(
    const float * points, unsigned int num_points, float * voxel_features,
    std::int32_t * voxel_coords, std::int32_t * num_points_per_voxel);

private:
  //! spconv's hash-table scratch, kept out of this header so that only the CUDA library
  //! needs the spconv headers.
  struct Scratch;

  VoxelizerConfig config_;
  cudaStream_t stream_;
  std::unique_ptr<Scratch> scratch_;
};

}  // namespace autoware::oneplanner::bev

#endif  // AUTOWARE__ONEPLANNER__BEV_FEATURE__VOXELIZER_HPP_
