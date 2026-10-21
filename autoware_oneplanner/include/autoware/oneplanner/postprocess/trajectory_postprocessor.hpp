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

#ifndef AUTOWARE__ONEPLANNER__POSTPROCESS__TRAJECTORY_POSTPROCESSOR_HPP_
#define AUTOWARE__ONEPLANNER__POSTPROCESS__TRAJECTORY_POSTPROCESSOR_HPP_

#include "autoware/oneplanner/types.hpp"

#include <Eigen/Core>
#include <rclcpp/time.hpp>

#include <autoware_internal_planning_msgs/msg/candidate_trajectories.hpp>
#include <autoware_planning_msgs/msg/trajectory.hpp>
#include <unique_identifier_msgs/msg/uuid.hpp>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace autoware::oneplanner
{

/**
 * @brief Deployment parameters of the trajectory postprocessing.
 *
 * The interpretation of the prediction tensor and the velocity profile are those of
 * autoware_diffusion_planner; only the tensor dimensions are read from the engine instead of
 * being compile-time constants.
 */
struct PostprocessParams
{
  std::string prediction_tensor{"prediction"};
  //! Additional ego-only trajectory outputs (`[B, T, 4]`, e.g. ResWorld's `prior_trajectory`),
  //! published as extra candidate trajectories.
  std::vector<std::string> extra_trajectory_tensors{};
  double horizon_seconds{4.0};
  //! Must be 0.1 s: the velocity profile is computed on this step.
  double time_step{0.1};
  int64_t velocity_smoothing_window{8};
  double stopping_threshold{0.3};
  std::string generator_name{"OnePlanner"};
  //! base_link -> vehicle-center offset applied in reverse to output poses when the node runs
  //! with `shift_x: true`; 0 disables the shift.
  double base_link_offset{0.0};
};

/**
 * @class TrajectoryPostprocessor
 * @brief Converts the prediction tensor into trajectory (and optional object) messages.
 *
 * Accepted prediction shapes: `[B, A, T, 4]` (agent 0 = ego) or `[B, T, 4]` (ego only), with
 * `(x, y, cos(yaw), sin(yaw))` per step in the ego frame at `time_step` intervals.
 */
class TrajectoryPostprocessor
{
public:
  struct Output
  {
    autoware_planning_msgs::msg::Trajectory trajectory;
    autoware_internal_planning_msgs::msg::CandidateTrajectories candidate_trajectories;
  };

  explicit TrajectoryPostprocessor(const PostprocessParams & params);

  /**
   * @brief Resolve and validate the prediction tensor against the engine output manifest.
   * @throws std::runtime_error when the tensor is missing, has an unexpected shape, or its
   *         horizon disagrees with `horizon_seconds` / `time_step`.
   */
  void validate_output_specs(const std::vector<TensorSpec> & output_specs);

  /**
   * @brief Create output messages from the inference outputs.
   *
   * @param outputs Inference output tensors.
   * @param ego Ego frame the prediction is relative to.
   *        (nullptr or empty to skip publishing them).
   * @param stamp Message timestamp.
   * @param generator_uuid Candidate trajectory generator id.
   * @throws std::runtime_error on malformed outputs.
   */
  Output process(
    const TensorMap & outputs, const EgoFrame & ego, const rclcpp::Time & stamp,
    const unique_identifier_msgs::msg::UUID & generator_uuid) const;

  int64_t num_timesteps() const { return num_timesteps_; }
  int64_t num_agents() const { return num_agents_; }

private:
  /// Parse a raw prediction into per-agent pose matrices in map coordinates.
  std::vector<std::vector<std::vector<Eigen::Matrix4d>>> parse_predictions(
    const std::vector<float> & prediction, const Eigen::Matrix4d & ego_to_map,
    const int64_t num_agents) const;

  /// Create one ego trajectory for `batch_idx` (shared by the main and extra outputs), led by
  /// the ego pose at t = 0.
  autoware_planning_msgs::msg::Trajectory create_trajectory(
    const std::vector<std::vector<std::vector<Eigen::Matrix4d>>> & agent_poses,
    const EgoFrame & ego, const rclcpp::Time & stamp, const int64_t batch_idx) const;

  PostprocessParams params_;
  int64_t batch_size_{1};
  int64_t num_agents_{1};
  int64_t num_timesteps_{0};
};

}  // namespace autoware::oneplanner

#endif  // AUTOWARE__ONEPLANNER__POSTPROCESS__TRAJECTORY_POSTPROCESSOR_HPP_
