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

#include "autoware/oneplanner/postprocess/trajectory_postprocessor.hpp"

#include "autoware/oneplanner/geometry.hpp"

#include <Eigen/Geometry>
#include <builtin_interfaces/msg/duration.hpp>

#include <autoware_internal_planning_msgs/msg/candidate_trajectory.hpp>
#include <autoware_internal_planning_msgs/msg/generator_info.hpp>
#include <std_msgs/msg/string.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace autoware::oneplanner
{

namespace
{

//! x, y, cos(yaw), sin(yaw) per step.
constexpr int64_t kPoseDim = 4;
constexpr double kTimeStep = 0.1;

/**
 * @brief The trajectory through `poses` (map frame, one per 0.1 s from t = 0.1 s), with
 *        autoware_diffusion_planner's velocity profile.
 *
 * Velocity is the distance from the previous point (the first from `base`) over 0.1 s,
 * smoothed with a trailing window of `smoothing_window` points; the last points keep the last
 * smoothed value. With `enable_force_stop`, a drop below `stopping_threshold` after a point
 * above it latches a stop: velocity zero and the pose held from then on. Acceleration is the
 * forward difference of the velocity, zero at the last point.
 */
autoware_planning_msgs::msg::Trajectory trajectory_from_poses(
  const std::vector<Eigen::Matrix4d> & poses, const geometry_msgs::msg::Point & base,
  const rclcpp::Time & stamp, const int64_t smoothing_window, const bool enable_force_stop,
  const double stopping_threshold)
{
  autoware_planning_msgs::msg::Trajectory trajectory;
  trajectory.header.stamp = stamp;
  trajectory.header.frame_id = "map";

  double prev_x = base.x;
  double prev_y = base.y;
  double prev_z = base.z;
  for (size_t i = 0; i < poses.size(); ++i) {
    const double time = kTimeStep * static_cast<double>(i + 1);
    autoware_planning_msgs::msg::TrajectoryPoint p;
    p.time_from_start.sec = static_cast<int>(time);
    p.time_from_start.nanosec = static_cast<uint32_t>((time - p.time_from_start.sec) * 1e9);
    p.pose.position.x = poses[i](0, 3);
    p.pose.position.y = poses[i](1, 3);
    p.pose.position.z = poses[i](2, 3);
    const Eigen::Quaterniond orientation(Eigen::Matrix3d(poses[i].block<3, 3>(0, 0)));
    p.pose.orientation.x = orientation.x();
    p.pose.orientation.y = orientation.y();
    p.pose.orientation.z = orientation.z();
    p.pose.orientation.w = orientation.w();
    const double distance = std::hypot(
      p.pose.position.x - prev_x, p.pose.position.y - prev_y, p.pose.position.z - prev_z);
    p.longitudinal_velocity_mps = static_cast<float>(distance / kTimeStep);
    prev_x = p.pose.position.x;
    prev_y = p.pose.position.y;
    prev_z = p.pose.position.z;
    trajectory.points.push_back(p);
  }

  const auto num_points = static_cast<int64_t>(poses.size());
  if (num_points <= smoothing_window) {
    throw std::invalid_argument("velocity_smoothing_window must be smaller than number of points");
  }
  bool force_stop = false;
  const auto threshold = static_cast<float>(stopping_threshold);
  auto & points = trajectory.points;
  for (int64_t i = 0; i + smoothing_window <= num_points; ++i) {
    double sum = 0.0;
    for (int64_t w = 0; w < smoothing_window; ++w) {
      sum += points[i + w].longitudinal_velocity_mps;
    }
    points[i].longitudinal_velocity_mps =
      static_cast<float>(sum / static_cast<double>(smoothing_window));
    if (
      enable_force_stop && i > 0 && std::abs(points[i - 1].longitudinal_velocity_mps) > threshold &&
      std::abs(points[i].longitudinal_velocity_mps) < threshold) {
      force_stop = true;
    }
    if (i > 0 && force_stop) {
      points[i].longitudinal_velocity_mps = 0.0f;
      points[i].pose = points[i - 1].pose;
    }
  }
  const auto last_smoothed = points[num_points - smoothing_window].longitudinal_velocity_mps;
  for (int64_t i = num_points - smoothing_window + 1; i < num_points; ++i) {
    points[i].longitudinal_velocity_mps = last_smoothed;
    if (force_stop) {
      points[i].longitudinal_velocity_mps = 0.0f;
      points[i].pose = points[i - 1].pose;
    }
  }
  for (int64_t i = 0; i + 1 < num_points; ++i) {
    const double v0 = points[i].longitudinal_velocity_mps;
    const double v1 = points[i + 1].longitudinal_velocity_mps;
    points[i].acceleration_mps2 = static_cast<float>((v1 - v0) / kTimeStep);
  }
  points.back().acceleration_mps2 = 0.0f;
  return trajectory;
}

/// Append one candidate trajectory with its generator info.
void append_candidate(
  autoware_internal_planning_msgs::msg::CandidateTrajectories & candidates,
  const autoware_planning_msgs::msg::Trajectory & trajectory, const std::string & generator_name,
  const unique_identifier_msgs::msg::UUID & generator_uuid)
{
  // Assigned field by field, not with build<>(), so that the code also compiles against
  // autoware_internal_planning_msgs versions that add fields, such as turn_indicators_command.
  // A new field keeps its default value.
  autoware_internal_planning_msgs::msg::CandidateTrajectory candidate_trajectory;
  candidate_trajectory.header = trajectory.header;
  candidate_trajectory.generator_id = generator_uuid;
  candidate_trajectory.points = trajectory.points;

  std_msgs::msg::String generator_name_msg;
  generator_name_msg.data = generator_name;

  const auto generator_info =
    autoware_internal_planning_msgs::build<autoware_internal_planning_msgs::msg::GeneratorInfo>()
      .generator_id(generator_uuid)
      .generator_name(generator_name_msg);

  candidates.candidate_trajectories.push_back(candidate_trajectory);
  candidates.generator_info.push_back(generator_info);
}

}  // namespace

TrajectoryPostprocessor::TrajectoryPostprocessor(const PostprocessParams & params) : params_(params)
{
  // The velocity profile and time_from_start are computed on a 0.1 s step. Reject
  // configurations that silently disagree.
  if (std::abs(params_.time_step - kTimeStep) > std::numeric_limits<double>::epsilon()) {
    throw std::runtime_error(
      "postprocess.time_step must be 0.1 s; got " + std::to_string(params_.time_step));
  }
  if (params_.horizon_seconds <= 0.0) {
    throw std::runtime_error("postprocess.horizon_seconds must be positive");
  }
}

void TrajectoryPostprocessor::validate_output_specs(const std::vector<TensorSpec> & output_specs)
{
  const TensorSpec * spec = find_spec(output_specs, params_.prediction_tensor);
  if (!spec) {
    throw std::runtime_error(
      "The model has no output tensor named '" + params_.prediction_tensor +
      "' (set postprocess.prediction_tensor to match the model)");
  }

  const auto & shape = spec->shape;
  if (shape.size() == 4) {
    batch_size_ = shape[0];
    num_agents_ = shape[1];
    num_timesteps_ = shape[2];
  } else if (shape.size() == 3) {
    batch_size_ = shape[0];
    num_agents_ = 1;
    num_timesteps_ = shape[1];
  } else {
    throw std::runtime_error(
      "Model output '" + spec->name + "' has shape " + shape_to_string(shape) +
      "; expected [B, A, T, 4] or [B, T, 4]");
  }
  if (shape.back() != kPoseDim) {
    throw std::runtime_error(
      "Model output '" + spec->name + "' has shape " + shape_to_string(shape) +
      "; the last dimension must be " + std::to_string(kPoseDim) + " (x, y, cos(yaw), sin(yaw))");
  }

  const auto expected_timesteps =
    static_cast<int64_t>(std::llround(params_.horizon_seconds / params_.time_step));
  if (num_timesteps_ != expected_timesteps) {
    throw std::runtime_error(
      "Model output '" + spec->name + "' has " + std::to_string(num_timesteps_) +
      " timesteps, but postprocess.horizon_seconds=" + std::to_string(params_.horizon_seconds) +
      " with a 0.1 s step expects " + std::to_string(expected_timesteps));
  }
  if (num_timesteps_ <= params_.velocity_smoothing_window) {
    throw std::runtime_error(
      "postprocess.velocity_smoothing_window (" +
      std::to_string(params_.velocity_smoothing_window) +
      ") must be smaller than the number of trajectory points (" + std::to_string(num_timesteps_) +
      ")");
  }

  for (const auto & extra_name : params_.extra_trajectory_tensors) {
    const TensorSpec * extra_spec = find_spec(output_specs, extra_name);
    if (!extra_spec) {
      throw std::runtime_error(
        "The model has no output tensor named '" + extra_name +
        "' (listed in postprocess.extra_trajectory_tensors)");
    }
    const std::vector<int64_t> expected_shape = {batch_size_, num_timesteps_, kPoseDim};
    if (extra_spec->shape != expected_shape) {
      throw std::runtime_error(
        "Model output '" + extra_name + "' has shape " + shape_to_string(extra_spec->shape) +
        "; extra trajectory tensors must be ego-only with shape " +
        shape_to_string(expected_shape));
    }
  }
}

std::vector<std::vector<std::vector<Eigen::Matrix4d>>> TrajectoryPostprocessor::parse_predictions(
  const std::vector<float> & prediction, const Eigen::Matrix4d & ego_to_map,
  const int64_t num_agents) const
{
  const size_t required_size = batch_size_ * num_agents * num_timesteps_ * kPoseDim;
  if (prediction.size() < required_size) {
    throw std::runtime_error(
      "Prediction vector size (" + std::to_string(prediction.size()) +
      ") is smaller than required (" + std::to_string(required_size) + ")");
  }

  std::vector<std::vector<std::vector<Eigen::Matrix4d>>> parsed(
    batch_size_,
    std::vector<std::vector<Eigen::Matrix4d>>(
      num_agents, std::vector<Eigen::Matrix4d>(num_timesteps_, Eigen::Matrix4d::Identity())));

  for (int64_t batch_idx = 0; batch_idx < batch_size_; ++batch_idx) {
    for (int64_t agent_idx = 0; agent_idx < num_agents; ++agent_idx) {
      for (int64_t time_idx = 0; time_idx < num_timesteps_; ++time_idx) {
        const int64_t base_idx =
          ((batch_idx * num_agents + agent_idx) * num_timesteps_ + time_idx) * kPoseDim;

        const auto x = static_cast<double>(prediction[base_idx + 0]);
        const auto y = static_cast<double>(prediction[base_idx + 1]);
        const auto cos_yaw = static_cast<double>(prediction[base_idx + 2]);
        const auto sin_yaw = static_cast<double>(prediction[base_idx + 3]);

        Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
        pose(0, 0) = cos_yaw;
        pose(0, 1) = -sin_yaw;
        pose(1, 0) = sin_yaw;
        pose(1, 1) = cos_yaw;
        pose(0, 3) = x;
        pose(1, 3) = y;

        parsed[batch_idx][agent_idx][time_idx] = ego_to_map * pose;
      }
    }
  }
  return parsed;
}

autoware_planning_msgs::msg::Trajectory TrajectoryPostprocessor::create_trajectory(
  const std::vector<std::vector<std::vector<Eigen::Matrix4d>>> & agent_poses, const EgoFrame & ego,
  const rclcpp::Time & stamp, const int64_t batch_idx) const
{
  const bool enable_force_stop =
    ego.odometry.twist.twist.linear.x > std::numeric_limits<double>::epsilon();

  auto trajectory = trajectory_from_poses(
    agent_poses.at(batch_idx).at(0), ego.odometry.pose.pose.position, stamp,
    params_.velocity_smoothing_window, enable_force_stop, params_.stopping_threshold);

  if (params_.base_link_offset != 0.0) {
    for (auto & point : trajectory.points) {
      point.pose = shift_x(point.pose, -params_.base_link_offset);
    }
  }

  // The plan starts at the trajectory stamp: a temporal follower looks its reference up at
  // (now - stamp), which is below the first point's 0.1 s for a plan planned at the newest
  // odometry, and would clamp to that point. Prepend the ego pose at t = 0 (base_link, so
  // after the offset shift). Velocity and acceleration are copied from the first plan point,
  // not the measured twist: the smoothed plan speed is what stop logic reads, and a measured
  // speed of 0 in front of a departing plan would read as a stop or as a large acceleration.
  if (!trajectory.points.empty()) {
    auto start = trajectory.points.front();
    start.time_from_start = builtin_interfaces::msg::Duration();
    start.pose = ego.odometry.pose.pose;
    trajectory.points.insert(trajectory.points.begin(), start);
  }
  return trajectory;
}

TrajectoryPostprocessor::Output TrajectoryPostprocessor::process(
  const TensorMap & outputs, const EgoFrame & ego, const rclcpp::Time & stamp,
  const unique_identifier_msgs::msg::UUID & generator_uuid) const
{
  const auto it = outputs.find(params_.prediction_tensor);
  if (it == outputs.end()) {
    throw std::runtime_error("Inference outputs lack '" + params_.prediction_tensor + "'");
  }

  const auto agent_poses = parse_predictions(it->second.host_data, ego.ego_to_map, num_agents_);

  Output output;
  for (int64_t batch_idx = 0; batch_idx < batch_size_; ++batch_idx) {
    const auto trajectory = create_trajectory(agent_poses, ego, stamp, batch_idx);
    if (batch_idx == 0) {
      output.trajectory = trajectory;
    }
    append_candidate(
      output.candidate_trajectories, trajectory,
      params_.generator_name + "_batch_" + std::to_string(batch_idx), generator_uuid);
  }

  // Extra ego-only trajectory outputs (e.g. a prior before refinement) become candidates.
  for (const auto & extra_name : params_.extra_trajectory_tensors) {
    const auto extra_it = outputs.find(extra_name);
    if (extra_it == outputs.end()) {
      throw std::runtime_error("Inference outputs lack '" + extra_name + "'");
    }
    const auto extra_poses =
      parse_predictions(extra_it->second.host_data, ego.ego_to_map, /*num_agents=*/1);
    for (int64_t batch_idx = 0; batch_idx < batch_size_; ++batch_idx) {
      append_candidate(
        output.candidate_trajectories, create_trajectory(extra_poses, ego, stamp, batch_idx),
        params_.generator_name + "_" + extra_name + "_batch_" + std::to_string(batch_idx),
        generator_uuid);
    }
  }

  return output;
}

}  // namespace autoware::oneplanner
