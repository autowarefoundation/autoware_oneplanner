// Copyright 2025-2026 TIER IV, Inc.
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

// Adapted from autoware_diffusion_planner's planning-factor utilities (autoware_universe).

#include "autoware/oneplanner/postprocess/planning_factor.hpp"

#include <rclcpp/duration.hpp>

#include <optional>
#include <vector>

namespace autoware::oneplanner
{

namespace
{

using autoware_planning_msgs::msg::TrajectoryPoint;

std::optional<DetectedStopFactor> detect_stop_factor(
  const std::vector<TrajectoryPoint> & points, const PlanningFactorDetectionConfig & config)
{
  std::optional<size_t> stop_index;
  rclcpp::Duration stop_start_time(0, 0);
  bool holds = true;
  for (size_t i = 0; i < points.size(); ++i) {
    const rclcpp::Duration time(points[i].time_from_start);
    if (!stop_index && points[i].longitudinal_velocity_mps <= config.stop_velocity_threshold) {
      stop_index = i;
      stop_start_time = time;
    }
    if (
      stop_index && (time - stop_start_time).seconds() < config.stop_keep_duration_threshold &&
      points[i].longitudinal_velocity_mps > config.stop_velocity_threshold) {
      holds = false;
    }
  }
  if (!stop_index || !holds) {
    return std::nullopt;
  }
  return DetectedStopFactor{points.front().pose, points[*stop_index].pose};
}

std::optional<DetectedSlowdownFactor> detect_slowdown_factor(
  const std::vector<TrajectoryPoint> & points, const PlanningFactorDetectionConfig & config)
{
  std::optional<size_t> start;
  std::optional<size_t> end;
  for (size_t i = 0; i < points.size(); ++i) {
    if (points[i].acceleration_mps2 < config.slowdown_accel_threshold) {
      if (!start) {
        start = i;
      }
    } else if (start && !end) {
      end = i;
    }
  }
  if (!start) {
    return std::nullopt;
  }
  if (!end) {
    end = points.size() - 1;
  }
  return DetectedSlowdownFactor{
    points.front().pose, points[*start].pose, points[*end].pose,
    points[*start].longitudinal_velocity_mps, points[*end].longitudinal_velocity_mps};
}

}  // namespace

PlanningFactorDetectionResult detect_planning_factors(
  const std::vector<TrajectoryPoint> & points, const PlanningFactorDetectionConfig & config)
{
  PlanningFactorDetectionResult result;
  if (points.empty()) {
    return result;
  }
  result.stop = detect_stop_factor(points, config);
  result.slowdown = detect_slowdown_factor(points, config);
  return result;
}

}  // namespace autoware::oneplanner
