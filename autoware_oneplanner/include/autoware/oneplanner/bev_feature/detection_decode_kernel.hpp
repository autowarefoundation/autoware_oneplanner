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

#ifndef AUTOWARE__ONEPLANNER__BEV_FEATURE__DETECTION_DECODE_KERNEL_HPP_
#define AUTOWARE__ONEPLANNER__BEV_FEATURE__DETECTION_DECODE_KERNEL_HPP_

#include "autoware/oneplanner/bev_feature/detection_decode.hpp"

#include <cuda_runtime.h>

#include <cstdint>

namespace autoware::oneplanner
{

/**
 * @brief Decode the head's proposals and cut them, compacting the survivors.
 *
 * One thread per proposal, with `autoware_bevfusion`'s `generateBoxes3D_kernel` arithmetic:
 * the yaw-norm gate, the centre's radial distance picking the first band whose upper limit
 * exceeds it (none: dropped), then the band's threshold for the class. A proposal whose
 * label is outside [0, num_classes) is dropped, where that kernel would read out of bounds.
 * Survivors are appended to `boxes` in no particular order; `count` must be zero on entry.
 */
cudaError_t launch_decode_detections(
  const int64_t * label_pred, const float * bbox_pred, const float * score,
  const float * yaw_norm_thresholds, const float * squared_upper_limits,
  const float * score_thresholds, const DetectionDecodeGeometry & geometry, DecodedBox * boxes,
  int32_t * count, cudaStream_t stream);

}  // namespace autoware::oneplanner

#endif  // AUTOWARE__ONEPLANNER__BEV_FEATURE__DETECTION_DECODE_KERNEL_HPP_
