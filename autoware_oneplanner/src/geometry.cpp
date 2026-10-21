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

// Adapted from autoware_diffusion_planner's utilities (autoware_universe).

#include "autoware/oneplanner/geometry.hpp"

#include <Eigen/Geometry>
#include <autoware_utils_geometry/geometry.hpp>

#include <cmath>
#include <cstddef>
#include <deque>
#include <stdexcept>
#include <utility>
#include <vector>

namespace autoware::oneplanner
{

namespace
{

Eigen::Matrix3d quaternion_to_matrix(const geometry_msgs::msg::Quaternion & q)
{
  const double norm = std::sqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);
  if (norm < 1e-6) {
    throw std::runtime_error("Quaternion norm is too small");
  }
  return Eigen::Quaterniond(q.w, q.x, q.y, q.z).toRotationMatrix();
}

}  // namespace

Eigen::Matrix4d pose_to_matrix4d(const geometry_msgs::msg::Pose & pose)
{
  Eigen::Matrix4d matrix = Eigen::Matrix4d::Identity();
  matrix.block<3, 3>(0, 0) = quaternion_to_matrix(pose.orientation);
  matrix.block<3, 1>(0, 3) = Eigen::Vector3d(pose.position.x, pose.position.y, pose.position.z);
  return matrix;
}

Eigen::Matrix4d inverse(const Eigen::Matrix4d & transform)
{
  return Eigen::Isometry3d(transform).inverse().matrix();
}

std::pair<float, float> rotation_matrix_to_cos_sin(const Eigen::Matrix3d & rotation)
{
  const float yaw = std::atan2(rotation(1, 0), rotation(0, 0));
  return {std::cos(yaw), std::sin(yaw)};
}

geometry_msgs::msg::Pose shift_x(const geometry_msgs::msg::Pose & pose, const double shift_length)
{
  const Eigen::Vector3d shift =
    quaternion_to_matrix(pose.orientation) * Eigen::Vector3d(shift_length, 0.0, 0.0);
  geometry_msgs::msg::Pose shifted = pose;
  shifted.position.x += shift.x();
  shifted.position.y += shift.y();
  shifted.position.z += shift.z();
  return shifted;
}

std::vector<float> create_ego_agent_past(
  const std::deque<nav_msgs::msg::Odometry> & odometry, const std::size_t num_timesteps,
  const Eigen::Matrix4d & map_to_ego, const rclcpp::Time & reference_time)
{
  constexpr std::size_t kFeatures = 4;  // x, y, cos, sin
  constexpr double kStepSeconds = 0.1;
  std::vector<float> history(num_timesteps * kFeatures, 0.0f);
  for (std::size_t t = 0; t < num_timesteps; ++t) {
    history[t * kFeatures + 2] = 1.0f;  // identity heading
  }
  if (odometry.empty()) {
    return history;
  }

  const auto store = [&](const std::size_t t, const geometry_msgs::msg::Pose & pose) {
    const Eigen::Matrix4d in_ego = map_to_ego * pose_to_matrix4d(pose);
    const auto [cos_yaw, sin_yaw] = rotation_matrix_to_cos_sin(in_ego.block<3, 3>(0, 0));
    history[t * kFeatures + 0] = static_cast<float>(in_ego(0, 3));
    history[t * kFeatures + 1] = static_cast<float>(in_ego(1, 3));
    history[t * kFeatures + 2] = cos_yaw;
    history[t * kFeatures + 3] = sin_yaw;
  };
  const auto seconds = [](const builtin_interfaces::msg::Time & stamp) {
    return static_cast<double>(stamp.sec) + static_cast<double>(stamp.nanosec) * 1e-9;
  };

  const double reference = reference_time.seconds();
  const double first = seconds(odometry.front().header.stamp);
  const double last = seconds(odometry.back().header.stamp);
  // Both the targets and the odometry stamps increase, so the search carries forward.
  std::size_t lower = 0;
  for (std::size_t t = 0; t < num_timesteps; ++t) {
    const double target = reference - static_cast<double>(num_timesteps - 1 - t) * kStepSeconds;
    if (target <= first) {
      store(t, odometry.front().pose.pose);
      continue;
    }
    if (target >= last) {
      store(t, odometry.back().pose.pose);
      continue;
    }
    while (lower + 1 < odometry.size() && target > seconds(odometry[lower + 1].header.stamp)) {
      ++lower;
    }
    const double t0 = seconds(odometry[lower].header.stamp);
    const double t1 = seconds(odometry[lower + 1].header.stamp);
    const double ratio = t1 > t0 ? (target - t0) / (t1 - t0) : 0.0;
    store(
      t, autoware_utils_geometry::calc_interpolated_pose(
           odometry[lower].pose.pose, odometry[lower + 1].pose.pose, ratio, false));
  }
  return history;
}

}  // namespace autoware::oneplanner
