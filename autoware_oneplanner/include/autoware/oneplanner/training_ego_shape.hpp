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

#ifndef AUTOWARE__ONEPLANNER__TRAINING_EGO_SHAPE_HPP_
#define AUTOWARE__ONEPLANNER__TRAINING_EGO_SHAPE_HPP_

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace autoware::oneplanner
{

//! (wheel base, length, width) in metres: the model's `ego_shape` input.
struct EgoShape
{
  double wheel_base;
  double length;
  double width;
};

/**
 * @brief The ego shapes of the vehicles a model was trained on, from the flat
 *        `[wheel_base, length, width, ...]` list its ml_package file states.
 * @throws std::runtime_error when the list is not whole triples of positive finite values.
 */
inline std::vector<EgoShape> parse_ego_shapes(const std::vector<double> & flat)
{
  if (flat.size() % 3 != 0) {
    throw std::runtime_error(
      "context.training_ego_shapes must hold [wheel_base, length, width] triples, got " +
      std::to_string(flat.size()) + " values");
  }
  std::vector<EgoShape> shapes;
  for (size_t i = 0; i < flat.size(); i += 3) {
    for (size_t j = i; j < i + 3; ++j) {
      if (!std::isfinite(flat[j]) || flat[j] <= 0.0) {
        throw std::runtime_error("context.training_ego_shapes values must be positive and finite");
      }
    }
    shapes.push_back({flat[i], flat[i + 1], flat[i + 2]});
  }
  return shapes;
}

/**
 * @brief The training shape of the vehicle the node runs on: the one whose wheel base is
 *        closest to this vehicle's.
 *
 * A model trained on several vehicles learned each one's `ego_shape` from the training data's
 * own table, not from this vehicle's `vehicle_info`, so it is fed what it learned.
 *
 * @throws std::runtime_error when no shape lies within `tolerance_m` of the wheel base: a
 *         vehicle the model never trained on.
 */
inline EgoShape closest_ego_shape(
  const std::vector<EgoShape> & shapes, const double wheel_base_m, const double tolerance_m = 0.5)
{
  const EgoShape * best = nullptr;
  for (const auto & shape : shapes) {
    if (
      !best ||
      std::abs(shape.wheel_base - wheel_base_m) < std::abs(best->wheel_base - wheel_base_m)) {
      best = &shape;
    }
  }
  if (!best || !(std::abs(best->wheel_base - wheel_base_m) <= tolerance_m)) {
    std::string known;
    for (const auto & shape : shapes) {
      known += (known.empty() ? "" : ", ") + std::to_string(shape.wheel_base) + " m";
    }
    throw std::runtime_error(
      "vehicle_info wheel base " + std::to_string(wheel_base_m) +
      " m matches no vehicle the model was trained on (" + known + ")");
  }
  return *best;
}

}  // namespace autoware::oneplanner

#endif  // AUTOWARE__ONEPLANNER__TRAINING_EGO_SHAPE_HPP_
