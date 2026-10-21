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

#ifndef AUTOWARE__ONEPLANNER__VECTOR_MAP__MAP_CONVERSION_HPP_
#define AUTOWARE__ONEPLANNER__VECTOR_MAP__MAP_CONVERSION_HPP_

#include "autoware/oneplanner/vector_map/dimensions.hpp"

#include <Eigen/Core>

#include <lanelet2_core/LaneletMap.h>

#include <cmath>
#include <cstdint>
#include <optional>
#include <vector>

namespace autoware::oneplanner::vector_map
{

using LanePoint = Eigen::Vector3d;
using Polyline = std::vector<LanePoint>;

struct Polygon
{
  Polyline points;
  PolygonType type;
  int64_t id{0};  //!< Map id: the polygon's, or the crosswalk lanelet's.
};

struct LineString
{
  Polyline points;
  LineStringType type;
  int64_t id{0};     //!< The source line string's map id.
  int64_t piece{0};  //!< Which equal piece of it (see convert_lanelet_map()).
};

struct LaneSegment
{
  int64_t id;
  Polyline centerline;
  Polyline left_boundary;
  Polyline right_boundary;
  LanePoint mean_point;
  LineType left_line_type;
  LineType right_line_type;
  std::optional<float> speed_limit_mps;
  int64_t turn_direction;
  int64_t traffic_light_id;

  static constexpr int64_t TURN_DIRECTION_NONE = -1;
  static constexpr int64_t TURN_DIRECTION_STRAIGHT = 0;
  static constexpr int64_t TURN_DIRECTION_LEFT = 1;
  static constexpr int64_t TURN_DIRECTION_RIGHT = 2;

  static constexpr int64_t TRAFFIC_LIGHT_ID_NONE = -1;
};

struct LaneletMap
{
  std::vector<LaneSegment> lane_segments;
  std::vector<Polygon> polygons;
  std::vector<LineString> line_strings;
};

/**
 * @brief Convert a lanelet2 map into the elements the vector-map tensors are built from, the
 *        way the OnePlanner training data converted its maps.
 *
 * - Lanes are the lanelets of subtype `road`, `highway`, `road_shoulder` and `bicycle_lane`.
 *   Both bounds are resampled to POINTS_PER_SEGMENT points evenly spaced in arc length, and the
 *   centerline is their point-wise midpoint (not lanelet2's centerline, which may be a mapped
 *   one). A lanelet whose bounds cannot be resampled is dropped.
 * - The speed limit attribute (km/h) is converted through mph, as the training data stored it
 *   (x 0.621371 x 0.44704), 4e-6 off km/h / 3.6; the model learned that value.
 * - Polygons are the `intersection_area` polygons, followed by every `crosswalk` lanelet
 *   outlined as its left bound and then its right bound reversed, each resampled to
 *   POINTS_PER_POLYGON points.
 * - Line strings are the `stop_line` and `road_border` line strings, cut into equal pieces of
 *   POINTS_PER_LINE_STRING points no more than `line_string_max_step_m` apart, resampled
 *   linearly in arc length after consecutive duplicate points are dropped; the pieces share
 *   their end points.
 */
[[nodiscard]] LaneletMap convert_lanelet_map(
  const lanelet::LaneletMapConstPtr & lanelet_map, double line_string_max_step_m);

//! The slot-order distance of the training data: whole millimetres, floor(x + 0.5).
inline int64_t slot_order_mm(const double distance_m)
{
  return static_cast<int64_t>(std::floor(distance_m * 1000.0 + 0.5));
}

}  // namespace autoware::oneplanner::vector_map

#endif  // AUTOWARE__ONEPLANNER__VECTOR_MAP__MAP_CONVERSION_HPP_
