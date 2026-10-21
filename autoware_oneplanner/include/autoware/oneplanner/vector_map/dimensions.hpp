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

// The vector-map tensor layout of autoware_diffusion_planner (autoware_universe), which the
// OnePlanner models share.

#ifndef AUTOWARE__ONEPLANNER__VECTOR_MAP__DIMENSIONS_HPP_
#define AUTOWARE__ONEPLANNER__VECTOR_MAP__DIMENSIONS_HPP_

#include <cstdint>

namespace autoware::oneplanner::vector_map
{

//! Half side of the square crop around the ego that map elements are selected from [m].
inline constexpr double LANE_MASK_RANGE_M = 100.0;

inline constexpr int64_t POINTS_PER_SEGMENT = 20;
inline constexpr int64_t POINTS_PER_POLYGON = 40;
inline constexpr int64_t POINTS_PER_LINE_STRING = 20;

//! x, y, cos(yaw), sin(yaw).
inline constexpr int64_t POSE_DIM = 4;

// Line types of a lane boundary, one-hot encoded per side.
enum LineType : int64_t {
  LINE_TYPE_CROSSWALK = 0,
  LINE_TYPE_CURBSTONE = 1,
  LINE_TYPE_GUARD_RAIL = 2,
  LINE_TYPE_LINE_THICK = 3,
  LINE_TYPE_LINE_THIN = 4,
  LINE_TYPE_PEDESTRIAN_MARKING = 5,
  LINE_TYPE_ROAD_BORDER = 6,
  LINE_TYPE_ROAD_SHOULDER = 7,
  LINE_TYPE_VIRTUAL = 8,
  LINE_TYPE_ZEBRA_MARKING = 9,
  LINE_TYPE_NUM = 10
};

enum LineStringType : int64_t {
  LINE_STRING_TYPE_STOP_LINE = 0,
  LINE_STRING_TYPE_ROAD_BORDER = 1,
  LINE_STRING_TYPE_NUM = 2
};

enum PolygonType : int64_t {
  POLYGON_TYPE_INTERSECTION_AREA = 0,
  POLYGON_TYPE_CROSSWALK = 1,
  POLYGON_TYPE_NUM = 2
};

// The traffic-light one-hot of a lane point, in this order.
inline constexpr int64_t TRAFFIC_LIGHT_GREEN = 0;
inline constexpr int64_t TRAFFIC_LIGHT_AMBER = 1;
inline constexpr int64_t TRAFFIC_LIGHT_RED = 2;
//! A light the lanelet has but whose state is not recognised (or is white).
inline constexpr int64_t TRAFFIC_LIGHT_UNKNOWN = 3;
//! The lanelet has no traffic light.
inline constexpr int64_t TRAFFIC_LIGHT_NONE = 4;
inline constexpr int64_t TRAFFIC_LIGHT_ONE_HOT_DIM = 5;

// Columns of one lane point: centre x, y; direction to the next point; left and right boundary
// offsets from the centre; the traffic-light one-hot; the left and right line-type one-hots.
inline constexpr int64_t X = 0;
inline constexpr int64_t Y = 1;
inline constexpr int64_t DX = 2;
inline constexpr int64_t DY = 3;
inline constexpr int64_t LB_X = 4;
inline constexpr int64_t LB_Y = 5;
inline constexpr int64_t RB_X = 6;
inline constexpr int64_t RB_Y = 7;
inline constexpr int64_t TRAFFIC_LIGHT = 8;
inline constexpr int64_t LINE_TYPE_LEFT_START = TRAFFIC_LIGHT + TRAFFIC_LIGHT_ONE_HOT_DIM;
inline constexpr int64_t LINE_TYPE_RIGHT_START = LINE_TYPE_LEFT_START + LINE_TYPE_NUM;
inline constexpr int64_t SEGMENT_POINT_DIM = LINE_TYPE_RIGHT_START + LINE_TYPE_NUM;

}  // namespace autoware::oneplanner::vector_map

#endif  // AUTOWARE__ONEPLANNER__VECTOR_MAP__DIMENSIONS_HPP_
