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

#ifndef AUTOWARE__ONEPLANNER__BEV_FEATURE__DETECTION_DECODE_HPP_
#define AUTOWARE__ONEPLANNER__BEV_FEATURE__DETECTION_DECODE_HPP_

#include <cstddef>
#include <cstdint>
#include <vector>

namespace autoware::oneplanner
{

/**
 * @brief One detected box in the frame of the cloud it came from, in metres (the box type of
 *        autoware_bevfusion).
 */
struct Box3D
{
  int label;
  float score;
  float x;
  float y;
  float z;
  float width;
  float length;
  float height;
  float yaw;
  float vx;
  float vy;
};

/**
 * @brief One decoded proposal that passed the score thresholds.
 *
 * The fields of Box3D, plus the proposal's index in the head's output: the host restores
 * autoware_bevfusion's order (a stable descending sort over the proposals) from it.
 */
struct DecodedBox
{
  int32_t label;
  float score;
  float x;
  float y;
  float z;
  float width;
  float length;
  float height;
  float yaw;
  float vx;
  float vy;
  int32_t proposal;
};

/// The TransFusion bbox coder's geometry, as `autoware_bevfusion` decodes with it.
struct DetectionDecodeGeometry
{
  int32_t num_proposals;
  int32_t num_classes;
  int32_t num_bands;
  float out_size_factor;
  float voxel_size_x;
  float voxel_size_y;
  float min_x_range;
  float min_y_range;
};

/**
 * @brief The score thresholds in the layout the kernel reads.
 *
 * `autoware_bevfusion`'s `detection_score_thresholds`: `distance_bin_upper_limits` [m],
 * strictly ascending, and one threshold per class per band, distance-major
 * (`thresholds[band * num_classes + label]`). Built as that node builds it: the limits are
 * squared in float, and a threshold outside [0, 1) is read as 0.
 */
struct ScoreThresholdTable
{
  std::vector<float> squared_upper_limits;
  std::vector<float> thresholds;
};

/**
 * @throws std::runtime_error when there is no band, the limits are not positive and
 *         strictly ascending, or the thresholds are not one per class per band.
 */
ScoreThresholdTable make_score_threshold_table(
  const std::vector<double> & distance_bin_upper_limits,
  const std::vector<double> & score_thresholds, size_t num_classes);

/// `autoware_bevfusion`'s order: score descending, ties in proposal order (its radix sort
/// is stable).
void sort_like_bevfusion(std::vector<DecodedBox> & boxes);

//! The decoded proposal as a Box3D.
Box3D to_box3d(const DecodedBox & box);

}  // namespace autoware::oneplanner

#endif  // AUTOWARE__ONEPLANNER__BEV_FEATURE__DETECTION_DECODE_HPP_
