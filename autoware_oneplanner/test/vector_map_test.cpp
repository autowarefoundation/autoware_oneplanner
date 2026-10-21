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

#include "autoware/oneplanner/vector_map/lane_segments.hpp"
#include "autoware/oneplanner/vector_map/map_conversion.hpp"

#include <Eigen/Core>

#include <autoware_perception_msgs/msg/traffic_light_element.hpp>
#include <autoware_planning_msgs/msg/lanelet_route.hpp>

#include <gtest/gtest.h>
#include <lanelet2_core/LaneletMap.h>
#include <lanelet2_core/primitives/BasicRegulatoryElements.h>
#include <lanelet2_core/primitives/Lanelet.h>

#include <cstdint>
#include <memory>
#include <vector>

namespace autoware::oneplanner::vector_map
{
namespace
{

using autoware_perception_msgs::msg::TrafficLightElement;

constexpr lanelet::Id kFirstLane = 100;
constexpr lanelet::Id kSecondLane = 101;
constexpr lanelet::Id kSidewalk = 102;
constexpr lanelet::Id kCrosswalk = 200;
constexpr lanelet::Id kIntersection = 300;
constexpr lanelet::Id kStopLine = 400;
constexpr lanelet::Id kTrafficLight = 500;

lanelet::Id next_id = 10000;

lanelet::LineString3d line(
  const std::vector<std::pair<double, double>> & xy, const char * type,
  const lanelet::Id id = lanelet::InvalId)
{
  lanelet::Points3d points;
  for (const auto & [x, y] : xy) {
    points.emplace_back(next_id++, x, y, 0.0);
  }
  lanelet::LineString3d result(id == lanelet::InvalId ? next_id++ : id, points);
  result.attributes()["type"] = type;
  return result;
}

lanelet::Lanelet lane(const lanelet::Id id, const double x0, const double x1, const char * subtype)
{
  lanelet::Lanelet result(
    id, line({{x0, 1.0}, {x1, 1.0}}, "line_thin"), line({{x0, -1.0}, {x1, -1.0}}, "line_thick"));
  result.attributes()["subtype"] = subtype;
  return result;
}

/**
 * Two straight road lanes along x (0..10 m, 10..20 m), a sidewalk the conversion ignores, a
 * crosswalk, an intersection area and a 30 m stop line. The second lane has a traffic light.
 * Lane bounds are thin / thick lines, so the stop line is the map's only line string the
 * conversion keeps (a road-border bound would be one too).
 */
std::shared_ptr<lanelet::LaneletMap> make_map()
{
  auto map = std::make_shared<lanelet::LaneletMap>();
  auto first = lane(kFirstLane, 0.0, 10.0, "road");
  first.attributes()["speed_limit"] = "36";
  first.attributes()["turn_direction"] = "left";
  auto second = lane(kSecondLane, 10.0, 20.0, "road");
  const auto light_bulbs = line({{20.0, 3.0}, {20.0, 4.0}}, "traffic_light");
  second.addRegulatoryElement(
    lanelet::TrafficLight::make(kTrafficLight, lanelet::AttributeMap(), {light_bulbs}));
  map->add(first);
  map->add(second);
  map->add(lane(kSidewalk, 0.0, 10.0, "walkway"));
  map->add(lane(kCrosswalk, 5.0, 6.0, "crosswalk"));

  lanelet::Polygon3d intersection(
    kIntersection,
    {lanelet::Point3d(next_id++, 20.0, -5.0, 0.0), lanelet::Point3d(next_id++, 30.0, -5.0, 0.0),
     lanelet::Point3d(next_id++, 30.0, 5.0, 0.0)});
  intersection.attributes()["type"] = "intersection_area";
  map->add(intersection);
  map->add(line({{0.0, 3.0}, {30.0, 3.0}}, "stop_line", kStopLine));
  return map;
}

TrafficSignalMap light_with(const uint8_t color)
{
  TrafficSignalMap signals;
  TrafficLightElement element;
  element.color = color;
  element.shape = TrafficLightElement::CIRCLE;
  element.confidence = 1.0f;
  signals[kTrafficLight].signal.traffic_light_group_id = kTrafficLight;
  signals[kTrafficLight].signal.elements.push_back(element);
  return signals;
}

//! The value of `column` at point `point` of lane row `row` in a [rows, 20, 33] tensor.
float at(
  const std::vector<float> & tensor, const int64_t row, const int64_t point, const int64_t column)
{
  return tensor[static_cast<size_t>(
    (row * POINTS_PER_SEGMENT + point) * SEGMENT_POINT_DIM + column)];
}

}  // namespace

TEST(MapConversion, KeepsDrivableLanesWithMidpointCenterlines)
{
  const auto converted = convert_lanelet_map(make_map(), 5.0);
  ASSERT_EQ(converted.lane_segments.size(), 2u);  // the walkway and the crosswalk are not lanes
  for (const auto & segment : converted.lane_segments) {
    ASSERT_EQ(segment.centerline.size(), static_cast<size_t>(POINTS_PER_SEGMENT));
    for (const auto & point : segment.centerline) {
      EXPECT_DOUBLE_EQ(point.y(), 0.0);
    }
    EXPECT_EQ(segment.left_line_type, LINE_TYPE_LINE_THIN);
    EXPECT_EQ(segment.right_line_type, LINE_TYPE_LINE_THICK);
  }
  const auto & first = converted.lane_segments[0].id == kFirstLane ? converted.lane_segments[0]
                                                                   : converted.lane_segments[1];
  ASSERT_TRUE(first.speed_limit_mps);
  // km/h through mph, as the training data stored it.
  EXPECT_FLOAT_EQ(*first.speed_limit_mps, static_cast<float>(36.0 * 0.621371 * 0.44704));
  EXPECT_EQ(first.turn_direction, LaneSegment::TURN_DIRECTION_LEFT);
  EXPECT_EQ(first.traffic_light_id, LaneSegment::TRAFFIC_LIGHT_ID_NONE);
}

TEST(MapConversion, CrosswalksFollowTheIntersectionAreas)
{
  const auto converted = convert_lanelet_map(make_map(), 5.0);
  ASSERT_EQ(converted.polygons.size(), 2u);
  EXPECT_EQ(converted.polygons[0].type, POLYGON_TYPE_INTERSECTION_AREA);
  EXPECT_EQ(converted.polygons[0].id, kIntersection);
  EXPECT_EQ(converted.polygons[1].type, POLYGON_TYPE_CROSSWALK);
  EXPECT_EQ(converted.polygons[1].id, kCrosswalk);
  for (const auto & polygon : converted.polygons) {
    EXPECT_EQ(polygon.points.size(), static_cast<size_t>(POINTS_PER_POLYGON));
  }
}

TEST(MapConversion, CutsLongLineStringsIntoPiecesThatShareTheirEnds)
{
  // 30 m over 19 steps is 1.58 m per step: one piece at a 5 m bound, two at a 1 m bound.
  EXPECT_EQ(convert_lanelet_map(make_map(), 5.0).line_strings.size(), 1u);
  const auto pieces = convert_lanelet_map(make_map(), 1.0).line_strings;
  ASSERT_EQ(pieces.size(), 2u);
  EXPECT_EQ(pieces[0].type, LINE_STRING_TYPE_STOP_LINE);
  EXPECT_EQ(pieces[0].piece, 0);
  EXPECT_EQ(pieces[1].piece, 1);
  EXPECT_DOUBLE_EQ(pieces[0].points.front().x(), 0.0);
  EXPECT_NEAR(pieces[0].points.back().x(), 15.0, 1e-9);
  EXPECT_NEAR(pieces[1].points.front().x(), 15.0, 1e-9);
  EXPECT_DOUBLE_EQ(pieces[1].points.back().x(), 30.0);
}

TEST(LaneSegmentContext, SelectsLanesNearestFirst)
{
  const LaneSegmentContext context(make_map(), 5.0);
  Eigen::Matrix4d map_to_center = Eigen::Matrix4d::Identity();
  map_to_center(0, 3) = -19.0;  // centred at x = 19: the second lane is nearer
  const auto selected = context.select_lane_segment_indices(map_to_center, 19.0, 0.0, 140);
  ASSERT_EQ(selected.size(), 2u);
  EXPECT_EQ(context.lanelet_map().lane_segments[selected[0]].id, kSecondLane);
  EXPECT_EQ(context.lanelet_map().lane_segments[selected[1]].id, kFirstLane);
  EXPECT_EQ(context.select_lane_segment_indices(map_to_center, 19.0, 0.0, 1).size(), 1u);
}

TEST(LaneSegmentContext, StartsTheRouteAtTheClosestLane)
{
  const LaneSegmentContext context(make_map(), 5.0);
  autoware_planning_msgs::msg::LaneletRoute route;
  for (const auto id : {kFirstLane, kSecondLane}) {
    autoware_planning_msgs::msg::LaneletSegment segment;
    segment.preferred_primitive.id = id;
    route.segments.push_back(segment);
  }
  const auto from_start = context.select_route_segment_indices(route, 1.0, 0.0, 25);
  ASSERT_EQ(from_start.size(), 2u);
  const auto from_second = context.select_route_segment_indices(route, 18.0, 0.0, 25);
  ASSERT_EQ(from_second.size(), 1u);
  EXPECT_EQ(context.lanelet_map().lane_segments[from_second[0]].id, kSecondLane);
}

TEST(LaneSegmentContext, WritesLaneRowsWithTheirLightAndLineTypes)
{
  const LaneSegmentContext context(make_map(), 5.0);
  const Eigen::Matrix4d identity = Eigen::Matrix4d::Identity();
  const auto selected = context.select_lane_segment_indices(identity, 0.0, 0.0, 140);
  ASSERT_EQ(selected.size(), 2u);
  const int64_t rows = 3;
  const auto [lanes, speed_limits] = context.create_tensor_data_from_indices(
    identity, light_with(TrafficLightElement::RED), selected, rows);
  ASSERT_EQ(lanes.size(), static_cast<size_t>(rows * POINTS_PER_SEGMENT * SEGMENT_POINT_DIM));
  ASSERT_EQ(speed_limits.size(), static_cast<size_t>(rows));

  // Row 0 is the first lane (nearest the origin): x from 0 to 10 m, boundaries 1 m to the side.
  EXPECT_FLOAT_EQ(at(lanes, 0, 0, X), 0.0f);
  EXPECT_FLOAT_EQ(at(lanes, 0, POINTS_PER_SEGMENT - 1, X), 10.0f);
  EXPECT_NEAR(at(lanes, 0, 0, DX), 10.0f / (POINTS_PER_SEGMENT - 1), 1e-5);
  EXPECT_FLOAT_EQ(at(lanes, 0, 0, LB_Y), 1.0f);
  EXPECT_FLOAT_EQ(at(lanes, 0, 0, RB_Y), -1.0f);
  EXPECT_FLOAT_EQ(at(lanes, 0, 0, TRAFFIC_LIGHT + TRAFFIC_LIGHT_NONE), 1.0f);
  EXPECT_FLOAT_EQ(at(lanes, 0, 0, LINE_TYPE_LEFT_START + LINE_TYPE_LINE_THIN), 1.0f);
  EXPECT_FLOAT_EQ(at(lanes, 0, 0, LINE_TYPE_RIGHT_START + LINE_TYPE_LINE_THICK), 1.0f);
  EXPECT_GT(speed_limits[0], 0.0f);

  // Row 1 is the second lane, whose light is red; row 2 is padding.
  EXPECT_FLOAT_EQ(at(lanes, 1, 5, TRAFFIC_LIGHT + TRAFFIC_LIGHT_RED), 1.0f);
  for (int64_t column = 0; column < SEGMENT_POINT_DIM; ++column) {
    EXPECT_EQ(at(lanes, 2, 0, column), 0.0f);
  }
  EXPECT_EQ(speed_limits[2], 0.0f);
}

TEST(LaneSegmentContext, ReadsAMissingOrWhiteLightAsUnknown)
{
  const LaneSegmentContext context(make_map(), 5.0);
  const Eigen::Matrix4d identity = Eigen::Matrix4d::Identity();
  const auto selected = context.select_lane_segment_indices(identity, 0.0, 0.0, 140);
  for (const auto & signals : {TrafficSignalMap{}, light_with(TrafficLightElement::WHITE)}) {
    const auto lanes =
      context.create_tensor_data_from_indices(identity, signals, selected, 2).first;
    EXPECT_FLOAT_EQ(at(lanes, 1, 0, TRAFFIC_LIGHT + TRAFFIC_LIGHT_UNKNOWN), 1.0f);
  }
}

TEST(LaneSegmentContext, WritesPolygonsWithTheirTypeOneHot)
{
  const LaneSegmentContext context(make_map(), 5.0);
  const Eigen::Matrix4d identity = Eigen::Matrix4d::Identity();
  const int64_t slots = 4;
  const auto polygons = context.create_polygon_tensor(identity, 10.0, 0.0, slots, POLYGON_TYPE_NUM);
  const int64_t point_dim = 2 + POLYGON_TYPE_NUM;
  ASSERT_EQ(polygons.size(), static_cast<size_t>(slots * POINTS_PER_POLYGON * point_dim));
  // The crosswalk (at x = 5..6) is nearer the origin than the intersection area (x >= 20).
  EXPECT_EQ(polygons[2 + POLYGON_TYPE_CROSSWALK], 1.0f);
  EXPECT_EQ(polygons[POINTS_PER_POLYGON * point_dim + 2 + POLYGON_TYPE_INTERSECTION_AREA], 1.0f);
}

}  // namespace autoware::oneplanner::vector_map
