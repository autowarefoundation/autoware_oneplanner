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

#ifndef AUTOWARE__ONEPLANNER__GEOMETRY_HPP_
#define AUTOWARE__ONEPLANNER__GEOMETRY_HPP_

#include <Eigen/Core>
#include <rclcpp/time.hpp>

#include <geometry_msgs/msg/pose.hpp>
#include <nav_msgs/msg/odometry.hpp>

#include <cstddef>
#include <deque>
#include <utility>
#include <vector>

namespace autoware::oneplanner
{

//! The pose as a 4x4 homogeneous transform. @throws std::runtime_error on a degenerate quaternion.
Eigen::Matrix4d pose_to_matrix4d(const geometry_msgs::msg::Pose & pose);

//! The inverse of a rigid 4x4 transform.
Eigen::Matrix4d inverse(const Eigen::Matrix4d & transform);

//! cos and sin of the yaw of a rotation matrix.
std::pair<float, float> rotation_matrix_to_cos_sin(const Eigen::Matrix3d & rotation);

//! The pose moved `shift_length` along its own x axis.
geometry_msgs::msg::Pose shift_x(const geometry_msgs::msg::Pose & pose, double shift_length);

/**
 * @brief The ego's past poses on a 0.1 s grid ending at `reference_time`, in the ego frame.
 *
 * `[num_timesteps, 4]` row-major, oldest first: x, y, cos(yaw), sin(yaw). Each sample is
 * interpolated between the two odometry messages bracketing it; a sample before the first or
 * after the last message takes that message's pose. Empty odometry gives the identity pose.
 *
 * @param odometry Odometry messages, oldest first.
 * @param map_to_ego Map to ego frame transform.
 */
std::vector<float> create_ego_agent_past(
  const std::deque<nav_msgs::msg::Odometry> & odometry, std::size_t num_timesteps,
  const Eigen::Matrix4d & map_to_ego, const rclcpp::Time & reference_time);

}  // namespace autoware::oneplanner

#endif  // AUTOWARE__ONEPLANNER__GEOMETRY_HPP_
