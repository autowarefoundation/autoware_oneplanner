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

#ifndef AUTOWARE__ONEPLANNER__VECTOR_MAP__LANE_SEGMENTS_HPP_
#define AUTOWARE__ONEPLANNER__VECTOR_MAP__LANE_SEGMENTS_HPP_

#include "autoware/oneplanner/vector_map/map_conversion.hpp"

#include <Eigen/Core>
#include <builtin_interfaces/msg/time.hpp>

#include <autoware_perception_msgs/msg/traffic_light_group.hpp>
#include <autoware_planning_msgs/msg/lanelet_route.hpp>

#include <lanelet2_core/LaneletMap.h>

#include <cstdint>
#include <map>
#include <memory>
#include <utility>
#include <vector>

namespace autoware::oneplanner::vector_map
{

//! A traffic light group's state and the stamp of the message it came from.
struct TrafficSignalStamped
{
  builtin_interfaces::msg::Time stamp;
  autoware_perception_msgs::msg::TrafficLightGroup signal;
};

using TrafficSignalMap = std::map<lanelet::Id, TrafficSignalStamped>;

/**
 * @class LaneSegmentContext
 * @brief The converted vector map, and the selection and tensor writing over it.
 *
 * Selection follows the OnePlanner training data: a lane is in the crop around a centre when
 * its mean, first or last centerline point lies strictly inside the square of half side
 * LANE_MASK_RANGE_M; lanes are ordered nearest first by whole millimetres, then by map id.
 */
class LaneSegmentContext
{
public:
  /**
   * @param line_string_max_step_m Largest spacing of a line string's resampled points.
   * @throws std::runtime_error when the map holds no lane.
   */
  LaneSegmentContext(
    const std::shared_ptr<const lanelet::LaneletMap> & lanelet_map, double line_string_max_step_m);

  /**
   * @brief The route's lane segments from the one closest to the centre onwards (closest in 2-D,
   *        over centerline points), skipping those outside the crop, at most `max_segments`.
   */
  std::vector<int64_t> select_route_segment_indices(
    const autoware_planning_msgs::msg::LaneletRoute & route, double center_x, double center_y,
    int64_t max_segments) const;

  /**
   * @brief The lane segments in the crop, nearest first, at most `max_segments`.
   * @param transform_matrix Map to the frame the distances are measured in (the crop centre's).
   */
  std::vector<int64_t> select_lane_segment_indices(
    const Eigen::Matrix4d & transform_matrix, double center_x, double center_y,
    int64_t max_segments) const;

  /**
   * @brief `[max_segments, POINTS_PER_SEGMENT, SEGMENT_POINT_DIM]` lane rows and the speed limit
   *        of each, in the frame of `transform_matrix` (map to that frame).
   *
   * A segment's traffic-light one-hot is its light group's state in `traffic_lights`; a light
   * the lanelet references but that has no state there reads UNKNOWN, as does a white light
   * (the training data has no white state). Rows past the selection are zero.
   */
  std::pair<std::vector<float>, std::vector<float>> create_tensor_data_from_indices(
    const Eigen::Matrix4d & transform_matrix, const TrafficSignalMap & traffic_lights,
    const std::vector<int64_t> & segment_indices, int64_t max_segments) const;

  /**
   * @brief `[num_elements, POINTS_PER_POLYGON, 2 + num_types]`: x, y and the type one-hot of the
   *        polygons with a point in the crop, nearest first.
   */
  std::vector<float> create_polygon_tensor(
    const Eigen::Matrix4d & transform_matrix, double center_x, double center_y,
    int64_t num_elements, int64_t num_types) const
  {
    return create_line_tensor(
      lanelet_map_.polygons, transform_matrix, center_x, center_y, num_elements, POINTS_PER_POLYGON,
      num_types);
  }

  /**
   * @brief `[num_elements, POINTS_PER_LINE_STRING, 2 + LINE_STRING_TYPE_NUM]`, as
   *        create_polygon_tensor() for the line strings.
   */
  std::vector<float> create_line_string_tensor(
    const Eigen::Matrix4d & transform_matrix, double center_x, double center_y,
    int64_t num_elements) const
  {
    return create_line_tensor(
      lanelet_map_.line_strings, transform_matrix, center_x, center_y, num_elements,
      POINTS_PER_LINE_STRING, LINE_STRING_TYPE_NUM);
  }

  const LaneletMap & lanelet_map() const { return lanelet_map_; }

private:
  bool segment_in_range(const LaneSegment & segment, double center_x, double center_y) const;

  template <typename T>
  std::vector<float> create_line_tensor(
    const std::vector<T> & elements, const Eigen::Matrix4d & transform_matrix, double center_x,
    double center_y, int64_t num_elements, int64_t num_points, int64_t num_types) const;

  const LaneletMap lanelet_map_;
  const std::map<lanelet::Id, size_t> lanelet_id_to_index_;
};

}  // namespace autoware::oneplanner::vector_map

#endif  // AUTOWARE__ONEPLANNER__VECTOR_MAP__LANE_SEGMENTS_HPP_
