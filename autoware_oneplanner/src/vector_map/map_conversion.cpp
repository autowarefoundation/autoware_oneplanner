// Copyright 2024-2026 TIER IV, Inc.
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

// Adapted from autoware_diffusion_planner's lanelet conversion (autoware_universe).

#include "autoware/oneplanner/vector_map/map_conversion.hpp"

#include <lanelet2_core/Forward.h>
#include <lanelet2_core/primitives/BasicRegulatoryElements.h>
#include <lanelet2_core/primitives/Lanelet.h>

#include <algorithm>
#include <cmath>
#include <iterator>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace autoware::oneplanner::vector_map
{

namespace
{

const std::map<std::string, LineType> LINE_TYPES = {
  {"crosswalk", LINE_TYPE_CROSSWALK},     {"curbstone", LINE_TYPE_CURBSTONE},
  {"guard_rail", LINE_TYPE_GUARD_RAIL},   {"line_thick", LINE_TYPE_LINE_THICK},
  {"line_thin", LINE_TYPE_LINE_THIN},     {"pedestrian_marking", LINE_TYPE_PEDESTRIAN_MARKING},
  {"road_border", LINE_TYPE_ROAD_BORDER}, {"road_shoulder", LINE_TYPE_ROAD_SHOULDER},
  {"virtual", LINE_TYPE_VIRTUAL},         {"zebra_marking", LINE_TYPE_ZEBRA_MARKING}};

const std::map<std::string, LineStringType> LINE_STRING_TYPES = {
  {"stop_line", LINE_STRING_TYPE_STOP_LINE}, {"road_border", LINE_STRING_TYPE_ROAD_BORDER}};

const std::map<std::string, PolygonType> POLYGON_TYPES = {
  {"intersection_area", POLYGON_TYPE_INTERSECTION_AREA}};

const std::set<std::string> DRIVABLE_LANE_SUBTYPES = {
  "bicycle_lane", "highway", "road", "road_shoulder"};

const std::map<std::string, int64_t> TURN_DIRECTIONS = {
  {"straight", LaneSegment::TURN_DIRECTION_STRAIGHT},
  {"left", LaneSegment::TURN_DIRECTION_LEFT},
  {"right", LaneSegment::TURN_DIRECTION_RIGHT}};

template <typename T>
Polyline to_polyline(const T & line_string)
{
  Polyline output;
  output.reserve(line_string.size());
  for (const auto & point : line_string) {
    output.emplace_back(point.x(), point.y(), point.z());
  }
  return output;
}

//! `num_points` points evenly spaced in arc length, the first and last exactly the input's.
//! An input of fewer than two points is returned unchanged.
Polyline interpolate_points(const Polyline & input, const size_t num_points)
{
  if (input.size() < 2 || num_points < 2) {
    return input;
  }
  std::vector<double> arc_lengths(input.size(), 0.0);
  for (size_t i = 1; i < input.size(); ++i) {
    arc_lengths[i] = arc_lengths[i - 1] + (input[i] - input[i - 1]).norm();
  }
  const double step = arc_lengths.back() / static_cast<double>(num_points - 1);

  Polyline result;
  result.reserve(num_points);
  result.push_back(input.front());
  size_t segment = 0;
  for (size_t i = 1; i + 1 < num_points; ++i) {
    const double target = static_cast<double>(i) * step;
    while (segment + 1 < arc_lengths.size() && arc_lengths[segment + 1] < target) {
      ++segment;
    }
    segment = std::min(segment, arc_lengths.size() - 2);
    // A zero-length segment would divide by zero.
    const double length = std::max(arc_lengths[segment + 1] - arc_lengths[segment], 1e-6);
    const double t = std::clamp((target - arc_lengths[segment]) / length, 0.0, 1.0);
    result.push_back(input[segment] + t * (input[segment + 1] - input[segment]));
  }
  result.push_back(input.back());
  return result;
}

//! Equal pieces of `num_points` points each, no more than `max_step_m` apart, positions linear
//! in arc length after consecutive duplicates are dropped; the pieces share their end points.
std::vector<Polyline> resample_line_string(
  const Polyline & input, const size_t num_points, const double max_step_m)
{
  Polyline points;
  for (const auto & p : input) {
    if (points.empty() || (p - points.back()).norm() > 0.0) {
      points.push_back(p);
    }
  }
  if (points.size() < 2) {
    points = {input.front(), input.back()};
  }
  std::vector<double> arc(points.size(), 0.0);
  for (size_t i = 1; i < points.size(); ++i) {
    arc[i] = arc[i - 1] + (points[i] - points[i - 1]).norm();
  }
  const double total = arc.back();
  if (total < 1e-6) {
    return {Polyline(num_points, points.front())};
  }
  const auto pieces = static_cast<size_t>(std::max(
    1.0, std::ceil(total / static_cast<double>(num_points - 1) / std::max(max_step_m, 1e-6))));
  const double length = total / static_cast<double>(pieces);
  const auto at = [&](const double s) {
    // As numpy.interp: the first segment whose end is at or past s.
    const auto upper = std::upper_bound(arc.begin(), arc.end(), s);
    const size_t j =
      std::clamp<size_t>(static_cast<size_t>(std::distance(arc.begin(), upper)), 1, arc.size() - 1);
    const double t = std::clamp((s - arc[j - 1]) / (arc[j] - arc[j - 1]), 0.0, 1.0);
    return LanePoint(points[j - 1] + t * (points[j] - points[j - 1]));
  };
  std::vector<Polyline> result(pieces);
  for (size_t k = 0; k < pieces; ++k) {
    result[k].reserve(num_points);
    for (size_t i = 0; i < num_points; ++i) {
      const double s = std::clamp(
        static_cast<double>(k) * length +
          static_cast<double>(i) * (length / static_cast<double>(num_points - 1)),
        0.0, total);
      result[k].push_back(at(s));
    }
  }
  result.front().front() = input.front();
  result.back().back() = input.back();
  return result;
}

LineType line_type_of(const lanelet::ConstLineString3d & bound)
{
  const auto it = LINE_TYPES.find(bound.attributeOr("type", ""));
  return it == LINE_TYPES.end() ? LINE_TYPE_VIRTUAL : it->second;
}

}  // namespace

LaneletMap convert_lanelet_map(
  const lanelet::LaneletMapConstPtr & lanelet_map, const double line_string_max_step_m)
{
  LaneletMap converted;
  converted.lane_segments.reserve(lanelet_map->laneletLayer.size());
  converted.polygons.reserve(lanelet_map->polygonLayer.size());
  converted.line_strings.reserve(lanelet_map->lineStringLayer.size());

  // Appended after the polygon layer: the training data lists intersection areas first, and
  // the tensor's nearest-first order breaks distance ties by map id, so the order here only
  // matters for identical ids.
  std::vector<Polygon> crosswalks;

  for (const auto & lanelet : lanelet_map->laneletLayer) {
    if (!lanelet.hasAttribute("subtype")) {
      continue;
    }
    const auto subtype = lanelet.attribute("subtype").as<std::string>();
    if (!subtype) {
      continue;
    }
    if (*subtype == "crosswalk") {
      Polyline outline = to_polyline(lanelet.leftBound3d());
      const Polyline right = to_polyline(lanelet.rightBound3d());
      outline.insert(outline.end(), right.rbegin(), right.rend());
      crosswalks.push_back(
        Polygon{
          interpolate_points(outline, POINTS_PER_POLYGON), POLYGON_TYPE_CROSSWALK, lanelet.id()});
      continue;
    }
    if (DRIVABLE_LANE_SUBTYPES.count(*subtype) == 0) {
      continue;
    }

    const Polyline left_boundary =
      interpolate_points(to_polyline(lanelet.leftBound3d()), POINTS_PER_SEGMENT);
    const Polyline right_boundary =
      interpolate_points(to_polyline(lanelet.rightBound3d()), POINTS_PER_SEGMENT);
    if (
      left_boundary.size() != static_cast<size_t>(POINTS_PER_SEGMENT) ||
      right_boundary.size() != static_cast<size_t>(POINTS_PER_SEGMENT)) {
      continue;  // a bound too short to resample
    }
    Polyline centerline(left_boundary.size());
    LanePoint mean_point(0.0, 0.0, 0.0);
    for (size_t i = 0; i < centerline.size(); ++i) {
      centerline[i] = 0.5 * (left_boundary[i] + right_boundary[i]);
      mean_point += centerline[i];
    }
    mean_point /= static_cast<double>(centerline.size());

    const lanelet::AttributeMap & attributes = lanelet.attributes();
    std::optional<float> speed_limit_mps;
    if (const auto it = attributes.find("speed_limit"); it != attributes.end()) {
      speed_limit_mps = static_cast<float>(std::stod(it->second.value()) * 0.621371 * 0.44704);
    }

    int64_t turn_direction = LaneSegment::TURN_DIRECTION_NONE;
    if (const auto it = attributes.find("turn_direction"); it != attributes.end()) {
      if (
        const auto direction = TURN_DIRECTIONS.find(it->second.value());
        direction != TURN_DIRECTIONS.end()) {
        turn_direction = direction->second;
      }
    }

    // A lanelet should reference at most one traffic light; older maps sometimes reference
    // several, and the first one is used.
    const auto traffic_lights = lanelet.regulatoryElementsAs<const lanelet::TrafficLight>();
    const int64_t traffic_light_id =
      traffic_lights.empty() ? LaneSegment::TRAFFIC_LIGHT_ID_NONE : traffic_lights.front()->id();

    converted.lane_segments.push_back(
      LaneSegment{
        lanelet.id(), centerline, left_boundary, right_boundary, mean_point,
        line_type_of(lanelet.leftBound()), line_type_of(lanelet.rightBound()), speed_limit_mps,
        turn_direction, traffic_light_id});
  }

  for (const auto & polygon : lanelet_map->polygonLayer) {
    const auto it = POLYGON_TYPES.find(polygon.attributeOr("type", ""));
    if (it == POLYGON_TYPES.end()) {
      continue;
    }
    converted.polygons.push_back(
      Polygon{
        interpolate_points(to_polyline(polygon.basicLineString()), POINTS_PER_POLYGON), it->second,
        polygon.id()});
  }
  converted.polygons.insert(converted.polygons.end(), crosswalks.begin(), crosswalks.end());

  for (const auto & line_string : lanelet_map->lineStringLayer) {
    const auto it = LINE_STRING_TYPES.find(line_string.attributeOr("type", ""));
    if (it == LINE_STRING_TYPES.end()) {
      continue;
    }
    const auto pieces = resample_line_string(
      to_polyline(line_string), POINTS_PER_LINE_STRING, line_string_max_step_m);
    for (size_t k = 0; k < pieces.size(); ++k) {
      converted.line_strings.push_back(
        LineString{pieces[k], it->second, line_string.id(), static_cast<int64_t>(k)});
    }
  }

  return converted;
}

}  // namespace autoware::oneplanner::vector_map
