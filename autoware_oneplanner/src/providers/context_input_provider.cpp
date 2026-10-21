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

#include "autoware/oneplanner/providers/context_input_provider.hpp"

#include "autoware/oneplanner/geometry.hpp"
#include "autoware/oneplanner/planning_time.hpp"
#include "autoware/oneplanner/providers/map_tensors.hpp"
#include "autoware/oneplanner/vector_map/dimensions.hpp"

#include <autoware/lanelet2_utils/conversion.hpp>

#include <algorithm>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace autoware::oneplanner
{
namespace
{
//! `lanes_on_route`: for every lane slot, whether the mission route runs through it and how far
//! along the route it lies -- 0 at the route's first visible lanelet, 1 at its last, and 0 on
//! every off-route slot, where the flag tells the two apart. Training derives it by matching
//! route polylines against lane polylines on exact centreline equality; here both tensors are
//! built from the same segment table, so the match is an index lookup with the same result. A
//! route lanelet outside the lane set marks nothing, as in training.
std::vector<float> mark_lanes_on_route(
  const std::vector<int64_t> & lane_segments, const std::vector<int64_t> & route_segments,
  const int64_t num_lane_slots)
{
  std::vector<float> marker(static_cast<size_t>(num_lane_slots) * 2, 0.0f);
  if (route_segments.empty()) {
    return marker;
  }
  std::unordered_map<int64_t, size_t> route_position;
  for (size_t i = 0; i < route_segments.size(); ++i) {
    route_position.emplace(route_segments[i], i);
  }
  const auto denominator = static_cast<float>(std::max<size_t>(1, route_segments.size() - 1));
  const size_t slots = std::min(lane_segments.size(), static_cast<size_t>(num_lane_slots));
  for (size_t slot = 0; slot < slots; ++slot) {
    const auto found = route_position.find(lane_segments[slot]);
    if (found == route_position.end()) {
      continue;
    }
    marker[slot * 2] = 1.0f;
    marker[slot * 2 + 1] = static_cast<float>(found->second) / denominator;
  }
  return marker;
}
}  // namespace

namespace vm = vector_map;

namespace
{

/// Validate a claimed spec against an expectation, with a readable error on mismatch.
void validate_shape(
  const TensorSpec & spec, const std::vector<int64_t> & expected, const std::string & reason)
{
  if (spec.shape != expected) {
    throw std::runtime_error(
      "Model input '" + spec.name + "' has shape " + shape_to_string(spec.shape) + "; expected " +
      shape_to_string(expected) + " (" + reason + ")");
  }
}

}  // namespace

ContextInputProvider::ContextInputProvider(
  rclcpp::Node & node, const autoware::vehicle_info_utils::VehicleInfo & vehicle_info,
  const std::optional<EgoShape> & training_ego_shape)
: node_(node),
  wheel_base_(vehicle_info.wheel_base_m),
  vehicle_length_(
    vehicle_info.front_overhang_m + vehicle_info.wheel_base_m + vehicle_info.rear_overhang_m),
  vehicle_width_(
    vehicle_info.left_overhang_m + vehicle_info.wheel_tread_m + vehicle_info.right_overhang_m)
{
  // A model that learned ego_shape from its training data's per-vehicle table is fed what it
  // learned, not this vehicle's vehicle_info; say so when the two differ.
  if (training_ego_shape) {
    RCLCPP_INFO(
      node_.get_logger(),
      "ego_shape from the training table (%.3f, %.3f, %.3f); vehicle_info gives (%.3f, %.3f, %.3f)",
      training_ego_shape->wheel_base, training_ego_shape->length, training_ego_shape->width,
      wheel_base_, vehicle_length_, vehicle_width_);
    wheel_base_ = training_ego_shape->wheel_base;
    vehicle_length_ = training_ego_shape->length;
    vehicle_width_ = training_ego_shape->width;
  }
  turn_indicators_enabled_ = node_.declare_parameter<bool>("context.turn_indicators.enabled", true);
  line_string_max_step_m_ = node_.declare_parameter<double>("context.line_string_max_step_m", 5.0);
  curvature_bias_enabled_ = node_.declare_parameter<bool>("context.curvature_bias.enabled", false);
  const double bias_tau_s = node_.declare_parameter<double>("context.curvature_bias.tau_s", 2.0);
  const double bias_min_speed =
    node_.declare_parameter<double>("context.curvature_bias.min_speed_mps", 1.0);
  if (curvature_bias_enabled_) {
    curvature_bias_.emplace(bias_tau_s, bias_min_speed);
  }
  signal_history_enabled_ = node_.declare_parameter<bool>("context.signal_history.enabled", false);
  const auto signal_steps = node_.declare_parameter<int64_t>("context.signal_history.steps", 7);
  const double signal_step_s =
    node_.declare_parameter<double>("context.signal_history.step_s", 0.5);
  if (signal_history_enabled_) {
    lane_signal_history_.emplace(static_cast<int>(signal_steps), signal_step_s);
    route_signal_history_.emplace(static_cast<int>(signal_steps), signal_step_s);
  }

  const double interval =
    node_.declare_parameter<double>("context.ego_history_interval_seconds", 0.1);
  if (interval != 0.1) {
    throw std::runtime_error(
      "context.ego_history_interval_seconds is " + std::to_string(interval) +
      ": this node builds the ego history on a 0.1 s grid only");
  }

  create_subscriptions();
}

std::vector<std::string> ContextInputProvider::claim_inputs(
  const std::vector<TensorSpec> & engine_inputs)
{
  std::vector<std::string> claimed;
  const auto claim = [&engine_inputs, &claimed](
                       const std::string & name,
                       std::vector<int64_t> & shape_out) -> const TensorSpec * {
    const TensorSpec * spec = find_spec(engine_inputs, name);
    if (spec) {
      shape_out = spec->shape;
      claimed.push_back(name);
    }
    return spec;
  };

  if (const auto * spec = claim("ego_current_state", ego_current_state_shape_)) {
    validate_shape(*spec, {1, 10}, "ego current state feature");
  }
  if (const auto * spec = claim("ego_agent_past", ego_agent_past_shape_)) {
    if (spec->shape.size() != 3 || spec->shape[0] != 1 || spec->shape[2] != vm::POSE_DIM) {
      throw std::runtime_error(
        "Model input 'ego_agent_past' has shape " + shape_to_string(spec->shape) +
        "; expected [1, T, " + std::to_string(vm::POSE_DIM) + "]");
    }
  }
  claim("static_objects", static_objects_shape_);  // Always zero-filled; any shape is accepted.

  if (const auto * spec = claim("lanes", lanes_shape_)) {
    if (
      spec->shape.size() != 4 || spec->shape[0] != 1 || spec->shape[2] != vm::POINTS_PER_SEGMENT ||
      spec->shape[3] != vm::SEGMENT_POINT_DIM) {
      throw std::runtime_error(
        "Model input 'lanes' has shape " + shape_to_string(spec->shape) + "; expected [1, S, " +
        std::to_string(vm::POINTS_PER_SEGMENT) + ", " + std::to_string(vm::SEGMENT_POINT_DIM) +
        "]");
    }
  }
  if (const auto * spec = claim("lanes_speed_limit", lanes_speed_limit_shape_)) {
    if (lanes_shape_.empty()) {
      throw std::runtime_error("Model takes 'lanes_speed_limit' but not 'lanes'");
    }
    validate_shape(*spec, {1, lanes_shape_[1], 1}, "one speed limit per lane segment");
  }
  if (const auto * spec = claim("lanes_has_speed_limit", lanes_has_speed_limit_shape_)) {
    if (lanes_shape_.empty()) {
      throw std::runtime_error("Model takes 'lanes_has_speed_limit' but not 'lanes'");
    }
    validate_shape(*spec, {1, lanes_shape_[1], 1}, "one speed limit flag per lane segment");
  }

  if (const auto * spec = claim("route_lanes", route_lanes_shape_)) {
    if (
      spec->shape.size() != 4 || spec->shape[0] != 1 || spec->shape[2] != vm::POINTS_PER_SEGMENT ||
      spec->shape[3] != vm::SEGMENT_POINT_DIM) {
      throw std::runtime_error(
        "Model input 'route_lanes' has shape " + shape_to_string(spec->shape) +
        "; expected [1, S, " + std::to_string(vm::POINTS_PER_SEGMENT) + ", " +
        std::to_string(vm::SEGMENT_POINT_DIM) + "]");
    }
  }
  if (const auto * spec = claim("route_lanes_speed_limit", route_lanes_speed_limit_shape_)) {
    if (route_lanes_shape_.empty()) {
      throw std::runtime_error("Model takes 'route_lanes_speed_limit' but not 'route_lanes'");
    }
    validate_shape(*spec, {1, route_lanes_shape_[1], 1}, "one speed limit per route segment");
  }
  if (
    const auto * spec = claim("route_lanes_has_speed_limit", route_lanes_has_speed_limit_shape_)) {
    if (route_lanes_shape_.empty()) {
      throw std::runtime_error("Model takes 'route_lanes_has_speed_limit' but not 'route_lanes'");
    }
    validate_shape(*spec, {1, route_lanes_shape_[1], 1}, "one speed limit flag per route segment");
  }
  if (const auto * spec = claim("lanes_on_route", lanes_on_route_shape_)) {
    if (lanes_shape_.empty() || route_lanes_shape_.empty()) {
      throw std::runtime_error(
        "Model takes 'lanes_on_route' but not both 'lanes' and 'route_lanes'");
    }
    validate_shape(
      *spec, {1, lanes_shape_[1], 2}, "per lane slot: on-route flag, position along the route");
  }

  // The slot counts are the model's; the point counts and the types are the map
  // conversion's.
  if (const auto * spec = claim("polygons", polygons_shape_)) {
    validate_shape(
      *spec,
      {1, spec->shape.size() == 4 ? spec->shape[1] : -1, vm::POINTS_PER_POLYGON,
       2 + vm::POLYGON_TYPE_NUM},
      "slot count from the model; points, x, y and the polygon type one-hot fixed by the map "
      "conversion");
  }
  if (const auto * spec = claim("line_strings", line_strings_shape_)) {
    validate_shape(
      *spec,
      {1, spec->shape.size() == 4 ? spec->shape[1] : -1, vm::POINTS_PER_LINE_STRING,
       2 + vm::LINE_STRING_TYPE_NUM},
      "slot count from the model; points, x, y and the line-string type one-hot fixed by the "
      "map conversion");
  }
  if (const auto * spec = claim("goal_pose", goal_pose_shape_)) {
    validate_shape(*spec, {1, vm::POSE_DIM}, "goal pose as x, y, cos(yaw), sin(yaw)");
  }
  if (const auto * spec = claim("ego_shape", ego_shape_shape_)) {
    validate_shape(*spec, {1, 3}, "wheel base, length, width");
  }
  if (const auto * spec = claim(SENSOR_LATENCY_TENSOR, sensor_latency_shape_)) {
    // Present only in a planning-time graph; the node has already checked that the
    // package plans that way (check_planning_time_inputs).
    validate_shape(*spec, {1, 1}, "seconds from the cloud stamp to planning");
  }
  check_curvature_bias_inputs(
    curvature_bias_enabled_, find_spec(engine_inputs, CURVATURE_BIAS_TENSOR) != nullptr);
  check_signal_history_inputs(
    signal_history_enabled_, find_spec(engine_inputs, LANE_SIGNALS_TENSOR) != nullptr,
    find_spec(engine_inputs, ROUTE_SIGNALS_TENSOR) != nullptr);
  // check_signal_history_inputs() has refused a graph that reads the signals while the history
  // is off, so a claim below always has its history.
  if (const auto * spec = claim(LANE_SIGNALS_TENSOR, lane_signals_shape_)) {
    if (lanes_shape_.empty()) {
      throw std::runtime_error(
        "Model takes 'lane_signals' but not 'lanes', whose rows they follow");
    }
    if (lane_signal_history_) {
      validate_shape(
        *spec, {1, lanes_shape_[1], lane_signal_history_->steps(), SIGNAL_STATES},
        "each lane row's traffic light over time");
    }
  }
  if (const auto * spec = claim(ROUTE_SIGNALS_TENSOR, route_signals_shape_)) {
    if (route_lanes_shape_.empty()) {
      throw std::runtime_error(
        "Model takes 'route_signals' but not 'route_lanes', whose rows they follow");
    }
    if (route_signal_history_) {
      validate_shape(
        *spec, {1, route_lanes_shape_[1], route_signal_history_->steps(), SIGNAL_STATES},
        "each route row's traffic light over time");
    }
  }
  if (const auto * spec = claim(CURVATURE_BIAS_TENSOR, curvature_bias_shape_)) {
    validate_shape(*spec, {1, 1}, "the steering sensor's curvature bias, 1/m");
    if (ego_current_state_shape_.empty()) {
      throw std::runtime_error(
        "Model takes 'ego_curvature_bias' but not 'ego_current_state', whose values it filters");
    }
  }
  if (const auto * spec = claim("turn_indicators", turn_indicators_shape_)) {
    if (spec->shape.size() != 2 || spec->shape[0] != 1) {
      throw std::runtime_error(
        "Model input 'turn_indicators' has shape " + shape_to_string(spec->shape) +
        "; expected [1, T]");
    }
    if (!turn_indicators_enabled_) {
      // The graph wants the tensor; this deployment says the network never reads it.
      // Bind a constant once instead of subscribing and filling it every tick.
      turn_indicators_constant_.assign(
        static_cast<size_t>(spec->shape[1]), static_cast<float>(TurnIndicatorsReport::DISABLE));
      RCLCPP_INFO(
        node_.get_logger(),
        "Model input 'turn_indicators' is disabled by context.turn_indicators.enabled: "
        "no subscription, constant DISABLE");
    }
  }

  return claimed;
}

void ContextInputProvider::create_subscriptions()
{
  // autoware_diffusion_planner's subscriptions, one for one: the same topics, the same QoS,
  // polling for everything but the map, and all of them created at construction. Two of them
  // are latched (transient_local), and rclcpp refuses intra-process delivery for a latched
  // subscription, so the node must not be composed with use_intra_process_comms.
  sub_traffic_signals_ = std::make_unique<autoware_utils_rclcpp::InterProcessPollingSubscriber<
    TrafficLightGroupArray, autoware_utils_rclcpp::polling_policy::All>>(
    &node_, "~/input/traffic_signals", rclcpp::QoS{10});
  if (turn_indicators_enabled_) {
    sub_turn_indicators_ =
      std::make_unique<autoware_utils_rclcpp::InterProcessPollingSubscriber<TurnIndicatorsReport>>(
        &node_, "~/input/turn_indicators");
  }
  sub_route_ = std::make_unique<autoware_utils_rclcpp::InterProcessPollingSubscriber<
    LaneletRoute, autoware_utils_rclcpp::polling_policy::Newest>>(
    &node_, "~/input/route", rclcpp::QoS{1}.transient_local());
  sub_map_ = node_.create_subscription<LaneletMapBin>(
    "~/input/vector_map", rclcpp::QoS{1}.transient_local(),
    std::bind(&ContextInputProvider::on_map, this, std::placeholders::_1));
}

void ContextInputProvider::on_map(const LaneletMapBin::ConstSharedPtr map_msg)
{
  // A subscription callback: parsing throws on a map this build cannot read, and an exception
  // leaving here unwinds through the executor and terminates the process, which may be a
  // container shared with other nodes. Report it and keep waiting instead; the map tensors
  // simply stay unavailable and every tick says so.
  try {
    const auto lanelet_map_ptr =
      autoware::experimental::lanelet2_utils::from_autoware_map_msgs(*map_msg);
    lane_segment_context_ =
      std::make_unique<vm::LaneSegmentContext>(lanelet_map_ptr, line_string_max_step_m_);
    map_error_.clear();
    // Segment indices name other lanelets in a rebuilt table.
    if (lane_signal_history_) lane_signal_history_->clear();
    if (route_signal_history_) route_signal_history_->clear();
  } catch (const std::exception & e) {
    map_error_ = std::string("The vector map could not be loaded: ") + e.what();
    RCLCPP_ERROR_STREAM(node_.get_logger(), map_error_);
  }
}

void ContextInputProvider::select_traffic_signals_at(
  const std::vector<TrafficLightGroupArray::ConstSharedPtr> & incoming,
  const rclcpp::Time & frame_stamp)
{
  // As the training data selected them, per GROUP: the elements from the latest message
  // stamped at or before the frame (the LiDAR stamp, not processing time), at most 500 ms
  // old, in which that group has a recognised colour. Several publishers may share the topic
  // with different group sets and disagree (red / unknown), so neither a whole message nor a
  // group's newest entry is the light's state. Messages are kept by stamp, so one that
  // arrives after this frame is processed still serves the next.
  constexpr int64_t tolerance_ns = 500'000'000;
  const auto stamp_ns = [](const auto & msg) { return rclcpp::Time(msg->stamp).nanoseconds(); };
  for (const auto & msg : incoming) {
    // upper_bound: a stamp tie keeps arrival order, the later-received last.
    const auto at = std::upper_bound(
      traffic_light_messages_.begin(), traffic_light_messages_.end(), msg,
      [&](const auto & a, const auto & b) { return stamp_ns(a) < stamp_ns(b); });
    traffic_light_messages_.insert(at, msg);
  }
  const int64_t frame_ns = frame_stamp.nanoseconds();
  while (!traffic_light_messages_.empty() &&
         stamp_ns(traffic_light_messages_.front()) < frame_ns - 2 * tolerance_ns) {
    traffic_light_messages_.pop_front();
  }
  traffic_light_id_map_.clear();
  for (auto it = traffic_light_messages_.rbegin(); it != traffic_light_messages_.rend(); ++it) {
    const int64_t msg_ns = stamp_ns(*it);
    if (msg_ns > frame_ns) {
      continue;
    }
    if (frame_ns - msg_ns > tolerance_ns) {
      break;
    }
    for (const auto & group : (*it)->traffic_light_groups) {
      if (traffic_light_id_map_.count(group.traffic_light_group_id) > 0) {
        continue;
      }
      const bool recognised =
        std::any_of(group.elements.begin(), group.elements.end(), [](const auto & element) {
          return element.color != autoware_perception_msgs::msg::TrafficLightElement::UNKNOWN;
        });
      if (recognised) {
        auto & entry = traffic_light_id_map_[group.traffic_light_group_id];
        entry.signal = group;
        entry.stamp = (*it)->stamp;
      }
    }
  }
}

bool ContextInputProvider::collect(
  const EgoFrame & ego, const rclcpp::Time & /* now */, TensorMap & inputs, std::string & error)
{
  // Traffic light states are accumulated regardless of individual tensor claims, so the lane
  // tensors always carry the freshest signal encoding. At the cloud stamp even when the plan
  // starts later: training selects the lights at the LiDAR frame in both conventions, and the
  // recognition's own latency is not measured, so the cloud stamp is the one choice exact on
  // both sides.
  if (sub_traffic_signals_) {
    select_traffic_signals_at(sub_traffic_signals_->take_data(), ego.sensor_stamp);
  }
  if (sub_route_) {
    const auto route = sub_route_->take_data();
    route_ptr_ = (!route_ptr_ || route) ? route : route_ptr_;
  }

  return collect_ego_tensors(ego, inputs, error) && collect_map_tensors(ego, inputs, error) &&
         collect_route_tensors(ego, inputs, error) && collect_turn_indicator_tensor(inputs, error);
}

bool ContextInputProvider::collect_ego_tensors(
  const EgoFrame & ego, TensorMap & inputs, std::string & error)
{
  if (!ego_agent_past_shape_.empty()) {
    const auto history_length = static_cast<size_t>(ego_agent_past_shape_[1]);
    inputs["ego_agent_past"] = Tensor::from_host(
      ego_agent_past_shape_,
      create_ego_agent_past(ego.reference_history, history_length, ego.map_to_ego, ego.stamp));
  }

  if (!ego_current_state_shape_.empty()) {
    if (!ego.acceleration) {
      error = "No acceleration received yet (required by 'ego_current_state')";
      return false;
    }
    if (!ego.steering_angle) {
      error = "No steering report received yet (required by 'ego_current_state')";
      return false;
    }
    // As recorded: the EKF velocity, acceleration and yaw rate unmodified at every speed, and
    // the steering tire angle as reported. No low-speed zeroing, no clamp, no offset.
    const auto & twist = ego.odometry.twist.twist;
    const auto & accel = ego.acceleration->accel.accel.linear;
    inputs["ego_current_state"] = Tensor::from_host(
      ego_current_state_shape_,
      {0.0f, 0.0f, 1.0f, 0.0f, static_cast<float>(twist.linear.x),
       static_cast<float>(twist.linear.y), static_cast<float>(accel.x), static_cast<float>(accel.y),
       *ego.steering_angle, static_cast<float>(twist.angular.z)});
  }

  // check_curvature_bias_inputs() has refused a graph that reads the bias while the filter is
  // off, so a claimed bias always has its filter.
  if (!curvature_bias_shape_.empty() && curvature_bias_) {
    // The same values ego_current_state carries this tick.
    if (!ego.steering_angle) {
      error = "No steering report received yet (required by 'ego_curvature_bias')";
      return false;
    }
    const auto & twist = ego.odometry.twist.twist;
    const double dt_s =
      curvature_bias_stamp_ ? (ego.stamp - *curvature_bias_stamp_).seconds() : 0.0;
    if (!curvature_bias_stamp_ || dt_s > 0.0) {
      curvature_bias_->update(
        twist.linear.x, twist.angular.z, *ego.steering_angle, wheel_base_, dt_s);
      curvature_bias_stamp_ = ego.stamp;
    }
    // Raw 1/m; the graph never normalises it.
    inputs[CURVATURE_BIAS_TENSOR] =
      Tensor::from_host(curvature_bias_shape_, {curvature_bias_->value()});
  }

  if (!sensor_latency_shape_.empty()) {
    // Raw seconds; the graph scales it (the observation normaliser would read 0 as padding).
    inputs[SENSOR_LATENCY_TENSOR] = Tensor::from_host(
      sensor_latency_shape_, {static_cast<float>((ego.stamp - ego.sensor_stamp).seconds())});
  }

  if (!ego_shape_shape_.empty()) {
    inputs["ego_shape"] = Tensor::from_host(
      ego_shape_shape_, {static_cast<float>(wheel_base_), static_cast<float>(vehicle_length_),
                         static_cast<float>(vehicle_width_)});
  }

  if (!static_objects_shape_.empty()) {
    inputs["static_objects"] = Tensor::from_host(
      static_objects_shape_,
      std::vector<float>(static_cast<size_t>(shape_num_elements(static_objects_shape_)), 0.0f));
  }

  return true;
}

bool ContextInputProvider::collect_map_tensors(
  const EgoFrame & ego, TensorMap & inputs, std::string & error)
{
  const bool needs_map = !lanes_shape_.empty() || !polygons_shape_.empty() ||
                         !line_strings_shape_.empty() || !route_lanes_shape_.empty();
  if (!needs_map) {
    return true;
  }
  if (!lane_segment_context_) {
    error = map_error_.empty() ? "Vector map not received yet" : map_error_;
    return false;
  }

  // Crop and order at the cloud pose, points in the planning frame (see map_tensors.hpp).
  if (!lanes_shape_.empty()) {
    auto lane_slots =
      map_tensors::build_lanes(*lane_segment_context_, ego, traffic_light_id_map_, lanes_shape_[1]);
    if (!lane_signals_shape_.empty() && lane_signal_history_) {
      // At the cloud stamp, where the lights were selected (select_traffic_signals_at).
      inputs[LANE_SIGNALS_TENSOR] = Tensor::from_host(
        lane_signals_shape_,
        lane_signal_history_->update(
          ego.sensor_stamp.nanoseconds(), lane_slots.indices, lane_slots.data, lanes_shape_[1],
          lanes_shape_[2], lanes_shape_[3], vm::TRAFFIC_LIGHT));
    }
    lane_segment_indices_ = std::move(lane_slots.indices);
    auto lanes_speed_limit = std::move(lane_slots.speed_limit);
    inputs["lanes"] = Tensor::from_host(lanes_shape_, std::move(lane_slots.data));
    if (!lanes_has_speed_limit_shape_.empty()) {
      // Same values as the speed limit tensor; the engine converts to bool by dtype.
      inputs["lanes_has_speed_limit"] =
        Tensor::from_host(lanes_has_speed_limit_shape_, lanes_speed_limit);
    }
    if (!lanes_speed_limit_shape_.empty()) {
      inputs["lanes_speed_limit"] =
        Tensor::from_host(lanes_speed_limit_shape_, std::move(lanes_speed_limit));
    }
  }

  if (!polygons_shape_.empty()) {
    inputs["polygons"] = Tensor::from_host(
      polygons_shape_, map_tensors::build_polygons(
                         *lane_segment_context_, ego, polygons_shape_[1], polygons_shape_[3] - 2));
  }
  if (!line_strings_shape_.empty()) {
    inputs["line_strings"] = Tensor::from_host(
      line_strings_shape_,
      map_tensors::build_line_strings(*lane_segment_context_, ego, line_strings_shape_[1]));
  }

  return true;
}

bool ContextInputProvider::collect_route_tensors(
  const EgoFrame & ego, TensorMap & inputs, std::string & error)
{
  if (route_lanes_shape_.empty() && goal_pose_shape_.empty()) {
    return true;
  }
  if (!route_ptr_) {
    error = "Route not received yet";
    return false;
  }

  if (!route_lanes_shape_.empty()) {
    if (!lane_segment_context_) {
      error = map_error_.empty() ? "Vector map not received yet" : map_error_;
      return false;
    }
    auto route_slots = map_tensors::build_route_lanes(
      *lane_segment_context_, ego, *route_ptr_, traffic_light_id_map_, route_lanes_shape_[1]);
    const std::vector<int64_t> & segment_indices = route_slots.indices;
    if (!route_signals_shape_.empty() && route_signal_history_) {
      inputs[ROUTE_SIGNALS_TENSOR] = Tensor::from_host(
        route_signals_shape_,
        route_signal_history_->update(
          ego.sensor_stamp.nanoseconds(), route_slots.indices, route_slots.data,
          route_lanes_shape_[1], route_lanes_shape_[2], route_lanes_shape_[3], vm::TRAFFIC_LIGHT));
    }
    auto route_speed_limit = std::move(route_slots.speed_limit);
    inputs["route_lanes"] = Tensor::from_host(route_lanes_shape_, std::move(route_slots.data));
    if (!lanes_on_route_shape_.empty()) {
      inputs["lanes_on_route"] = Tensor::from_host(
        lanes_on_route_shape_,
        mark_lanes_on_route(lane_segment_indices_, segment_indices, lanes_on_route_shape_[1]));
    }
    if (!route_lanes_has_speed_limit_shape_.empty()) {
      inputs["route_lanes_has_speed_limit"] =
        Tensor::from_host(route_lanes_has_speed_limit_shape_, route_speed_limit);
    }
    if (!route_lanes_speed_limit_shape_.empty()) {
      inputs["route_lanes_speed_limit"] =
        Tensor::from_host(route_lanes_speed_limit_shape_, std::move(route_speed_limit));
    }
  }

  if (!goal_pose_shape_.empty()) {
    const Eigen::Matrix4d goal_pose_ego = ego.map_to_ego * pose_to_matrix4d(route_ptr_->goal_pose);
    const auto [cos_yaw, sin_yaw] = rotation_matrix_to_cos_sin(goal_pose_ego.block<3, 3>(0, 0));
    inputs["goal_pose"] = Tensor::from_host(
      goal_pose_shape_, {static_cast<float>(goal_pose_ego(0, 3)),
                         static_cast<float>(goal_pose_ego(1, 3)), cos_yaw, sin_yaw});
  }

  return true;
}

bool ContextInputProvider::collect_turn_indicator_tensor(TensorMap & inputs, std::string & error)
{
  if (turn_indicators_shape_.empty()) {
    return true;
  }

  if (!turn_indicators_enabled_) {
    inputs["turn_indicators"] =
      Tensor::from_host(turn_indicators_shape_, turn_indicators_constant_);
    return true;
  }

  const auto report = sub_turn_indicators_->take_data();
  if (!report) {
    error = "No turn indicator report received yet";
    return false;
  }

  const auto history_length = static_cast<size_t>(turn_indicators_shape_[1]);
  turn_indicators_history_.push_back(*report);
  while (turn_indicators_history_.size() > history_length) {
    turn_indicators_history_.pop_front();
  }

  // Copy from back to front; pad with the oldest value, as autoware_diffusion_planner does.
  std::vector<float> data(history_length, 0.0f);
  const auto last_index = static_cast<int64_t>(history_length) - 1;
  for (int64_t t = 0; t <= last_index; ++t) {
    const int64_t index =
      std::max(static_cast<int64_t>(turn_indicators_history_.size()) - 1 - t, int64_t{0});
    data[last_index - t] = static_cast<float>(turn_indicators_history_[index].report);
  }
  inputs["turn_indicators"] = Tensor::from_host(turn_indicators_shape_, std::move(data));
  return true;
}

}  // namespace autoware::oneplanner
