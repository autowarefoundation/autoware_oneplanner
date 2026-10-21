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

#ifndef AUTOWARE__ONEPLANNER__BEV_FEATURE__BEV_WARP_KERNEL_HPP_
#define AUTOWARE__ONEPLANNER__BEV_FEATURE__BEV_WARP_KERNEL_HPP_

#include "autoware/oneplanner/bev_feature/bev_warp.hpp"

#include <cuda_runtime.h>

#include <cstdint>

namespace autoware::oneplanner
{

/**
 * @brief Warp a `[C, H, W]` feature map on the GPU (bilinear, zero padding).
 */
cudaError_t launch_se2_warp_kernel(
  const float * d_source, float * d_output, const Se2WarpParams & params, const int32_t channels,
  cudaStream_t stream);

}  // namespace autoware::oneplanner

#endif  // AUTOWARE__ONEPLANNER__BEV_FEATURE__BEV_WARP_KERNEL_HPP_
