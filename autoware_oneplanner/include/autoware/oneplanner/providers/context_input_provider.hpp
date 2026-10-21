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

#ifndef AUTOWARE__ONEPLANNER__PROVIDERS__CONTEXT_INPUT_PROVIDER_HPP_
#define AUTOWARE__ONEPLANNER__PROVIDERS__CONTEXT_INPUT_PROVIDER_HPP_

#include "autoware/oneplanner/curvature_bias.hpp"
#include "autoware/oneplanner/input_provider.hpp"
#include "autoware/oneplanner/signal_history.hpp"
#include "autoware/oneplanner/training_ego_shape.hpp"
#include "autoware/oneplanner/vector_map/lane_segments.hpp"

#include <autoware/vehicle_info_utils/vehicle_info.hpp>
#include <autoware_utils_rclcpp/polling_subscriber.hpp>
#include <rclcpp/rclcpp.hpp>

#include <autoware_map_msgs/msg/lanelet_map_bin.hpp>
#include <autoware_perception_msgs/msg/traffic_light_group_array.hpp>
#include <autoware_planning_msgs/msg/lanelet_route.hpp>
#include <autoware_vehicle_msgs/msg/turn_indicators_report.hpp>

#include <lanelet2_core/LaneletMap.h>

#include <cstdint>
#include <deque>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace autoware::oneplanner
{

/**
 * @class ContextInputProvider
 * @brief Produces the ego, map, route and traffic-light context tensors, each one optional.
 *
 * Claimable tensors (fixed names, the diffusion-planner tensor layout the OnePlanner models
 * share): `ego_current_state`, `ego_agent_past`, `static_objects`, `lanes`,
 * `lanes_speed_limit`, `lanes_has_speed_limit`, `route_lanes`, `route_lanes_speed_limit`,
 * `route_lanes_has_speed_limit`, `lanes_on_route`, `polygons`, `line_strings`, `goal_pose`,
 * `ego_shape`, `turn_indicators`, `sensor_latency`, `ego_curvature_bias`, `lane_signals`,
 * `route_signals`.
 *
 * Free dimensions (history length, lane/route segment count) are taken from the engine
 * manifest. Every subscription is created in the constructor, whether or not the model turns
 * out to claim the tensor behind it, with autoware_diffusion_planner's topics and QoS. A
 * polling subscription costs nothing to hold; what it buys is that the latched inputs (map,
 * route) are there before the engines finish building.
 */
class ContextInputProvider : public InputProviderInterface
{
public:
  //! @param training_ego_shape The `ego_shape` the model learned for this vehicle; std::nullopt
  //!   feeds the vehicle's own dimensions from `vehicle_info`.
  ContextInputProvider(
    rclcpp::Node & node, const autoware::vehicle_info_utils::VehicleInfo & vehicle_info,
    const std::optional<EgoShape> & training_ego_shape = std::nullopt);

  std::string name() const override { return "context"; }
  std::vector<std::string> claim_inputs(const std::vector<TensorSpec> & engine_inputs) override;
  bool collect(
    const EgoFrame & ego, const rclcpp::Time & now, TensorMap & inputs,
    std::string & error) override;

private:
  using TrafficLightGroupArray = autoware_perception_msgs::msg::TrafficLightGroupArray;
  using TurnIndicatorsReport = autoware_vehicle_msgs::msg::TurnIndicatorsReport;
  using LaneletRoute = autoware_planning_msgs::msg::LaneletRoute;
  using LaneletMapBin = autoware_map_msgs::msg::LaneletMapBin;

  //! The training data's traffic-light rule, into traffic_light_id_map_ (see the .cpp).
  void select_traffic_signals_at(
    const std::vector<TrafficLightGroupArray::ConstSharedPtr> & incoming,
    const rclcpp::Time & frame_stamp);

  void create_subscriptions();
  void on_map(const LaneletMapBin::ConstSharedPtr map_msg);

  bool collect_ego_tensors(const EgoFrame & ego, TensorMap & inputs, std::string & error);
  bool collect_map_tensors(const EgoFrame & ego, TensorMap & inputs, std::string & error);
  bool collect_route_tensors(const EgoFrame & ego, TensorMap & inputs, std::string & error);
  bool collect_turn_indicator_tensor(TensorMap & inputs, std::string & error);

  rclcpp::Node & node_;

  // Vehicle geometry fed as `ego_shape` (see the constructor)
  double wheel_base_;
  double vehicle_length_;
  double vehicle_width_;

  // Deployment parameters
  //! `context.turn_indicators.enabled`. Off for a network whose graph carries the
  //! input only to keep a stable signature and never reads it: nothing is
  //! subscribed and the tensor is a constant built once at claim time.
  bool turn_indicators_enabled_{true};
  double line_string_max_step_m_{5.0};

  // Claims and engine-derived dimensions. A claim is active when its shape is non-empty.
  std::vector<int64_t> ego_current_state_shape_;
  std::vector<int64_t> ego_agent_past_shape_;
  std::vector<int64_t> static_objects_shape_;
  std::vector<int64_t> lanes_shape_;
  std::vector<int64_t> lanes_speed_limit_shape_;
  std::vector<int64_t> lanes_has_speed_limit_shape_;
  std::vector<int64_t> route_lanes_shape_;
  std::vector<int64_t> route_lanes_speed_limit_shape_;
  std::vector<int64_t> route_lanes_has_speed_limit_shape_;
  //! `lanes_on_route` `[1, S, 2]`: per lane slot, whether the mission route runs through it
  //! and how far along the route it lies. Derived from the lane and route selections of the
  //! same tick, so it needs both `lanes` and `route_lanes`.
  std::vector<int64_t> lanes_on_route_shape_;
  //! Segment table indices behind this tick's `lanes` slots, kept for `lanes_on_route`.
  std::vector<int64_t> lane_segment_indices_;
  std::vector<int64_t> polygons_shape_;
  std::vector<int64_t> line_strings_shape_;
  std::vector<int64_t> goal_pose_shape_;
  std::vector<int64_t> ego_shape_shape_;
  std::vector<int64_t> turn_indicators_shape_;
  std::vector<int64_t> sensor_latency_shape_;  //!< Planning-time graphs only.
  //! `ego_curvature_bias` `[1, 1]`, complementary-steer graphs only; its filter ticks once
  //! per collect, on the values that tick's `ego_current_state` carries.
  std::vector<int64_t> curvature_bias_shape_;
  bool curvature_bias_enabled_{false};
  std::optional<CurvatureBiasFilter> curvature_bias_;
  std::optional<rclcpp::Time> curvature_bias_stamp_;
  //! `lane_signals` / `route_signals` `[1, rows, steps, 5]`, signal-history graphs only;
  //! each row followed by its segment over the ticks (cleared when the map is rebuilt).
  std::vector<int64_t> lane_signals_shape_;
  std::vector<int64_t> route_signals_shape_;
  bool signal_history_enabled_{false};
  std::optional<SignalHistory> lane_signal_history_;
  std::optional<SignalHistory> route_signal_history_;
  std::vector<float> turn_indicators_constant_;

  // Subscriptions, created in the constructor (see the class comment). Turn indicators
  // are the one exception: `context.turn_indicators.enabled` false means the deployment
  // has decided the network never reads them, and nothing is subscribed.
  std::unique_ptr<autoware_utils_rclcpp::InterProcessPollingSubscriber<
    TrafficLightGroupArray, autoware_utils_rclcpp::polling_policy::All>>
    sub_traffic_signals_;
  std::unique_ptr<autoware_utils_rclcpp::InterProcessPollingSubscriber<TurnIndicatorsReport>>
    sub_turn_indicators_;
  std::unique_ptr<autoware_utils_rclcpp::InterProcessPollingSubscriber<
    LaneletRoute, autoware_utils_rclcpp::polling_policy::Newest>>
    sub_route_;
  rclcpp::Subscription<LaneletMapBin>::SharedPtr sub_map_;

  // State
  std::deque<TurnIndicatorsReport> turn_indicators_history_;
  vector_map::TrafficSignalMap traffic_light_id_map_;
  //! Every recent TrafficLightGroupArray, by stamp, so a frame can take the latest one at or
  //! before its LiDAR stamp (see select_traffic_signals_at()).
  std::deque<TrafficLightGroupArray::ConstSharedPtr> traffic_light_messages_;
  std::unique_ptr<vector_map::LaneSegmentContext> lane_segment_context_;
  //! Why the latched map could not be turned into lane segments, if it could not. The
  //! tensors stay unavailable either way; this only distinguishes "no map yet" from
  //! "the map that arrived is unusable" in the diagnostic.
  std::string map_error_;
  LaneletRoute::ConstSharedPtr route_ptr_;
};

}  // namespace autoware::oneplanner

#endif  // AUTOWARE__ONEPLANNER__PROVIDERS__CONTEXT_INPUT_PROVIDER_HPP_
