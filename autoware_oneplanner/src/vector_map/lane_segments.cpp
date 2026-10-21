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

// Adapted from autoware_diffusion_planner's LaneSegmentContext (autoware_universe).

#include "autoware/oneplanner/vector_map/lane_segments.hpp"

#include <Eigen/Core>

#include <autoware_perception_msgs/msg/traffic_light_element.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace autoware::oneplanner::vector_map
{

namespace
{

using autoware_perception_msgs::msg::TrafficLightElement;

std::map<lanelet::Id, size_t> index_by_id(const std::vector<LaneSegment> & lane_segments)
{
  std::map<lanelet::Id, size_t> index;
  for (size_t i = 0; i < lane_segments.size(); ++i) {
    index[lane_segments[i].id] = i;
  }
  return index;
}

bool inside_crop(const double x, const double y, const double center_x, const double center_y)
{
  return x > center_x - LANE_MASK_RANGE_M && x < center_x + LANE_MASK_RANGE_M &&
         y > center_y - LANE_MASK_RANGE_M && y < center_y + LANE_MASK_RANGE_M;
}

//! The colour of the element that governs a lane turning `turn_direction`: the only recognised
//! element; else the most confident arrow pointing that way; else the most confident circle;
//! else the most confident element.
uint8_t current_light_color(
  const int64_t turn_direction, const std::vector<TrafficLightElement> & elements)
{
  std::vector<TrafficLightElement> recognised;
  for (const auto & element : elements) {
    if (element.color != TrafficLightElement::UNKNOWN) {
      recognised.push_back(element);
    }
  }
  if (recognised.empty()) {
    return TrafficLightElement::UNKNOWN;
  }
  if (recognised.size() == 1) {
    return recognised.front().color;
  }

  const std::map<int64_t, uint8_t> arrow_of_direction = {
    {LaneSegment::TURN_DIRECTION_NONE, TrafficLightElement::UNKNOWN},
    {LaneSegment::TURN_DIRECTION_STRAIGHT, TrafficLightElement::UP_ARROW},
    {LaneSegment::TURN_DIRECTION_LEFT, TrafficLightElement::LEFT_ARROW},
    {LaneSegment::TURN_DIRECTION_RIGHT, TrafficLightElement::RIGHT_ARROW}};
  const auto arrow = arrow_of_direction.find(turn_direction);
  const uint8_t target_shape =
    arrow != arrow_of_direction.end() ? arrow->second : TrafficLightElement::UNKNOWN;

  const auto most_confident = [](const std::vector<TrafficLightElement> & candidates) {
    return std::max_element(
             candidates.begin(), candidates.end(),
             [](const TrafficLightElement & a, const TrafficLightElement & b) {
               return a.confidence < b.confidence;
             })
      ->color;
  };
  const auto with_shape = [&recognised](const uint8_t shape) {
    std::vector<TrafficLightElement> matching;
    for (const auto & element : recognised) {
      if (element.shape == shape) {
        matching.push_back(element);
      }
    }
    return matching;
  };
  if (const auto arrows = with_shape(target_shape); !arrows.empty()) {
    return most_confident(arrows);
  }
  if (const auto circles = with_shape(TrafficLightElement::CIRCLE); !circles.empty()) {
    return most_confident(circles);
  }
  return most_confident(recognised);
}

}  // namespace

LaneSegmentContext::LaneSegmentContext(
  const std::shared_ptr<const lanelet::LaneletMap> & lanelet_map,
  const double line_string_max_step_m)
: lanelet_map_(convert_lanelet_map(lanelet_map, line_string_max_step_m)),
  lanelet_id_to_index_(index_by_id(lanelet_map_.lane_segments))
{
  if (lanelet_map_.lane_segments.empty()) {
    throw std::runtime_error("No lane segments found in the map");
  }
}

bool LaneSegmentContext::segment_in_range(
  const LaneSegment & segment, const double center_x, const double center_y) const
{
  const auto & c = segment.centerline;
  return inside_crop(segment.mean_point.x(), segment.mean_point.y(), center_x, center_y) ||
         (!c.empty() && (inside_crop(c.front().x(), c.front().y(), center_x, center_y) ||
                         inside_crop(c.back().x(), c.back().y(), center_x, center_y)));
}

std::vector<int64_t> LaneSegmentContext::select_route_segment_indices(
  const autoware_planning_msgs::msg::LaneletRoute & route, const double center_x,
  const double center_y, const int64_t max_segments) const
{
  // The route's segments the map holds, and which of them is closest to the centre.
  std::vector<int64_t> route_indices;
  double closest_distance = std::numeric_limits<double>::max();
  size_t closest = 0;
  for (const auto & route_segment : route.segments) {
    const auto it = lanelet_id_to_index_.find(route_segment.preferred_primitive.id);
    if (it == lanelet_id_to_index_.end()) {
      continue;
    }
    route_indices.push_back(static_cast<int64_t>(it->second));
    double distance = std::numeric_limits<double>::max();
    for (const LanePoint & point : lanelet_map_.lane_segments[it->second].centerline) {
      const double dx = point.x() - center_x;
      const double dy = point.y() - center_y;
      distance = std::min(distance, std::sqrt(dx * dx + dy * dy));
    }
    if (distance < closest_distance) {
      closest_distance = distance;
      closest = route_indices.size() - 1;
    }
  }

  std::vector<int64_t> selected;
  for (size_t i = closest; i < route_indices.size(); ++i) {
    const int64_t index = route_indices[i];
    if (!segment_in_range(lanelet_map_.lane_segments[index], center_x, center_y)) {
      continue;
    }
    selected.push_back(index);
    if (selected.size() >= static_cast<size_t>(max_segments)) {
      break;
    }
  }
  return selected;
}

std::vector<int64_t> LaneSegmentContext::select_lane_segment_indices(
  const Eigen::Matrix4d & transform_matrix, const double center_x, const double center_y,
  const int64_t max_segments) const
{
  const auto distance_of = [&](const LanePoint & point) {
    const Eigen::Vector4d transformed =
      transform_matrix * Eigen::Vector4d(point.x(), point.y(), point.z(), 1.0);
    return std::sqrt(transformed.x() * transformed.x() + transformed.y() * transformed.y());
  };

  struct Candidate
  {
    int64_t index;
    int64_t distance_mm;
    lanelet::Id id;
  };
  std::vector<Candidate> candidates;
  candidates.reserve(lanelet_map_.lane_segments.size());
  for (size_t i = 0; i < lanelet_map_.lane_segments.size(); ++i) {
    const LaneSegment & segment = lanelet_map_.lane_segments[i];
    if (!segment_in_range(segment, center_x, center_y) || segment.centerline.size() < 2) {
      continue;
    }
    // The closer of the first point and the second to last: the last point is the next
    // segment's first, and would give two adjacent segments the same distance.
    const double distance = std::min(
      distance_of(segment.centerline.front()),
      distance_of(segment.centerline[segment.centerline.size() - 2]));
    candidates.push_back({static_cast<int64_t>(i), slot_order_mm(distance), segment.id});
  }
  // Forked lanelets tie exactly, so the map id breaks the tie as the training data did.
  std::sort(candidates.begin(), candidates.end(), [](const Candidate & a, const Candidate & b) {
    return std::make_pair(a.distance_mm, a.id) < std::make_pair(b.distance_mm, b.id);
  });

  std::vector<int64_t> selected;
  for (const auto & candidate : candidates) {
    if (selected.size() >= static_cast<size_t>(max_segments)) {
      break;
    }
    selected.push_back(candidate.index);
  }
  return selected;
}

std::pair<std::vector<float>, std::vector<float>>
LaneSegmentContext::create_tensor_data_from_indices(
  const Eigen::Matrix4d & transform_matrix, const TrafficSignalMap & traffic_lights,
  const std::vector<int64_t> & segment_indices, const int64_t max_segments) const
{
  Eigen::MatrixXd output(SEGMENT_POINT_DIM, max_segments * POINTS_PER_SEGMENT);
  output.setZero();
  std::vector<float> speed_limits(static_cast<size_t>(max_segments), 0.0f);

  const auto homogeneous = [](const LanePoint & point) {
    return Eigen::Vector4d(point.x(), point.y(), point.z(), 1.0);
  };
  const auto line_type_one_hot = [](const int64_t line_type) {
    Eigen::Matrix<double, LINE_TYPE_NUM, 1> one_hot =
      Eigen::Matrix<double, LINE_TYPE_NUM, 1>::Zero();
    if (line_type >= 0 && line_type < LINE_TYPE_NUM) {
      one_hot[line_type] = 1.0;
    }
    return one_hot;
  };

  int64_t row = 0;
  for (const int64_t segment_index : segment_indices) {
    if (row >= max_segments) {
      break;
    }
    const LaneSegment & segment = lanelet_map_.lane_segments[segment_index];
    const auto & centerline = segment.centerline;
    const auto & left = segment.left_boundary;
    const auto & right = segment.right_boundary;
    if (
      centerline.size() != static_cast<size_t>(POINTS_PER_SEGMENT) ||
      left.size() != static_cast<size_t>(POINTS_PER_SEGMENT) ||
      right.size() != static_cast<size_t>(POINTS_PER_SEGMENT)) {
      continue;
    }

    Eigen::Matrix<double, TRAFFIC_LIGHT_ONE_HOT_DIM, 1> light =
      Eigen::Matrix<double, TRAFFIC_LIGHT_ONE_HOT_DIM, 1>::Zero();
    if (segment.traffic_light_id == LaneSegment::TRAFFIC_LIGHT_ID_NONE) {
      light[TRAFFIC_LIGHT_NONE] = 1.0;
    } else if (
      const auto it = traffic_lights.find(segment.traffic_light_id); it == traffic_lights.end()) {
      light[TRAFFIC_LIGHT_UNKNOWN] = 1.0;
    } else {
      switch (current_light_color(segment.turn_direction, it->second.signal.elements)) {
        case TrafficLightElement::GREEN:
          light[TRAFFIC_LIGHT_GREEN] = 1.0;
          break;
        case TrafficLightElement::AMBER:
          light[TRAFFIC_LIGHT_AMBER] = 1.0;
          break;
        case TrafficLightElement::RED:
          light[TRAFFIC_LIGHT_RED] = 1.0;
          break;
        case TrafficLightElement::UNKNOWN:
        case TrafficLightElement::WHITE:  // the training data has no white state
          light[TRAFFIC_LIGHT_UNKNOWN] = 1.0;
          break;
        default:
          break;
      }
    }
    const auto left_type = line_type_one_hot(segment.left_line_type);
    const auto right_type = line_type_one_hot(segment.right_line_type);

    for (int64_t i = 0; i < POINTS_PER_SEGMENT; ++i) {
      const int64_t column = row * POINTS_PER_SEGMENT + i;
      const Eigen::Vector4d center = transform_matrix * homogeneous(centerline[i]);
      output(X, column) = center.x();
      output(Y, column) = center.y();
      if (i > 0) {
        output(DX, column - 1) = output(X, column) - output(X, column - 1);
        output(DY, column - 1) = output(Y, column) - output(Y, column - 1);
      }
      const Eigen::Vector4d left_point = transform_matrix * homogeneous(left[i]);
      output(LB_X, column) = left_point.x() - center.x();
      output(LB_Y, column) = left_point.y() - center.y();
      const Eigen::Vector4d right_point = transform_matrix * homogeneous(right[i]);
      output(RB_X, column) = right_point.x() - center.x();
      output(RB_Y, column) = right_point.y() - center.y();
      output.block<TRAFFIC_LIGHT_ONE_HOT_DIM, 1>(TRAFFIC_LIGHT, column) = light;
      output.block<LINE_TYPE_NUM, 1>(LINE_TYPE_LEFT_START, column) = left_type;
      output.block<LINE_TYPE_NUM, 1>(LINE_TYPE_RIGHT_START, column) = right_type;
    }
    speed_limits[static_cast<size_t>(row)] = segment.speed_limit_mps.value_or(0.0f);
    ++row;
  }

  // Column-major, one point's SEGMENT_POINT_DIM values after another.
  const Eigen::MatrixXf output_f = output.cast<float>();
  return {std::vector<float>(output_f.data(), output_f.data() + output_f.size()), speed_limits};
}

template <typename T>
std::vector<float> LaneSegmentContext::create_line_tensor(
  const std::vector<T> & elements, const Eigen::Matrix4d & transform_matrix, const double center_x,
  const double center_y, const int64_t num_elements, const int64_t num_points,
  const int64_t num_types) const
{
  struct Candidate
  {
    Polyline polyline;
    int64_t type;
    int64_t distance_mm;
    int64_t id;
    int64_t piece;
  };
  std::vector<Candidate> candidates;
  for (const auto & element : elements) {
    const bool in_crop = std::any_of(
      element.points.begin(), element.points.end(),
      [&](const LanePoint & p) { return inside_crop(p.x(), p.y(), center_x, center_y); });
    if (!in_crop) {
      continue;
    }
    Polyline transformed;
    transformed.reserve(element.points.size());
    double min_distance = std::numeric_limits<double>::max();
    for (const auto & point : element.points) {
      const Eigen::Vector4d p =
        transform_matrix * Eigen::Vector4d(point.x(), point.y(), point.z(), 1.0);
      transformed.emplace_back(p.x(), p.y(), p.z());
      min_distance = std::min(min_distance, std::sqrt(p.x() * p.x() + p.y() * p.y()));
    }
    int64_t piece = 0;
    if constexpr (std::is_same_v<T, LineString>) {
      piece = element.piece;
    }
    candidates.push_back(
      {std::move(transformed), static_cast<int64_t>(element.type), slot_order_mm(min_distance),
       element.id, piece});
  }
  std::sort(candidates.begin(), candidates.end(), [](const Candidate & a, const Candidate & b) {
    return std::make_tuple(a.distance_mm, a.id, a.piece) <
           std::make_tuple(b.distance_mm, b.id, b.piece);
  });

  // A type past num_types would write into the next point's x.
  for (const auto & candidate : candidates) {
    if (candidate.type < 0 || candidate.type >= num_types) {
      throw std::runtime_error(
        "map element type " + std::to_string(candidate.type) + " does not fit a " +
        std::to_string(num_types) + "-type tensor");
    }
  }
  const int64_t point_dim = 2 + num_types;
  std::vector<float> tensor(static_cast<size_t>(num_elements * num_points * point_dim), 0.0f);
  const size_t count = std::min(static_cast<size_t>(num_elements), candidates.size());
  for (size_t i = 0; i < count; ++i) {
    const auto & candidate = candidates[i];
    const size_t points = std::min(static_cast<size_t>(num_points), candidate.polyline.size());
    for (size_t j = 0; j < points; ++j) {
      const size_t base =
        (i * static_cast<size_t>(num_points) + j) * static_cast<size_t>(point_dim);
      tensor[base + 0] = static_cast<float>(candidate.polyline[j].x());
      tensor[base + 1] = static_cast<float>(candidate.polyline[j].y());
      tensor[base + 2 + static_cast<size_t>(candidate.type)] = 1.0f;
    }
  }
  return tensor;
}

template std::vector<float> LaneSegmentContext::create_line_tensor<Polygon>(
  const std::vector<Polygon> &, const Eigen::Matrix4d &, double, double, int64_t, int64_t,
  int64_t) const;
template std::vector<float> LaneSegmentContext::create_line_tensor<LineString>(
  const std::vector<LineString> &, const Eigen::Matrix4d &, double, double, int64_t, int64_t,
  int64_t) const;

}  // namespace autoware::oneplanner::vector_map
