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

// Adapted from autoware_bevfusion's host postprocessing (autoware_universe): the BEV-IoU
// non-maximum suppression, the area-based class remapper and the box conversion.

#ifndef AUTOWARE__ONEPLANNER__POSTPROCESS__OBJECT_POSTPROCESS_HPP_
#define AUTOWARE__ONEPLANNER__POSTPROCESS__OBJECT_POSTPROCESS_HPP_

#include "autoware/oneplanner/bev_feature/detection_decode.hpp"

#include <Eigen/Core>

#include <autoware_perception_msgs/msg/detected_object.hpp>
#include <autoware_perception_msgs/msg/detected_objects.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace autoware::oneplanner
{

/**
 * @brief BEV-IoU non-maximum suppression over objects ordered by descending score.
 *
 * An object is dropped when its BEV IoU with a higher-ranked object within
 * `search_distance_2d` exceeds `iou_threshold`. Objects of different classes are compared,
 * except that a pedestrian never suppresses, and is never suppressed by, another class.
 */
class IouBevNms
{
public:
  IouBevNms(double search_distance_2d, double iou_threshold);

  std::vector<autoware_perception_msgs::msg::DetectedObject> apply(
    const std::vector<autoware_perception_msgs::msg::DetectedObject> & objects) const;

private:
  bool is_target_pair(
    const autoware_perception_msgs::msg::DetectedObject & a,
    const autoware_perception_msgs::msg::DetectedObject & b) const;

  double search_distance_2d_squared_;
  double iou_threshold_;
};

/**
 * @brief Reassigns an object's label by its BEV area.
 *
 * The three matrices are square over ObjectClassification labels, row-major (row = the
 * object's label): an object of label `from` becomes label `to` when
 * `allow_remapping_by_area[from][to]` and its area lies in `[min_area[from][to],
 * max_area[from][to]]`. A non-finite area bound reads as unbounded.
 */
class ClassRemapper
{
public:
  ClassRemapper(
    const std::vector<int64_t> & allow_remapping_by_area_matrix,
    const std::vector<double> & min_area_matrix, const std::vector<double> & max_area_matrix);

  void map_classes(autoware_perception_msgs::msg::DetectedObjects & objects) const;

private:
  Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> allow_remapping_by_area_;
  Eigen::MatrixXd min_area_;
  Eigen::MatrixXd max_area_;
  int num_labels_{0};
};

//! The ObjectClassification label of a class name; a name with no label is UNKNOWN.
uint8_t semantic_type(const std::string & class_name);

//! One detected object for a box, labelled through `class_names[box.label]`.
autoware_perception_msgs::msg::DetectedObject to_detected_object(
  const Box3D & box, const std::vector<std::string> & class_names);

}  // namespace autoware::oneplanner

#endif  // AUTOWARE__ONEPLANNER__POSTPROCESS__OBJECT_POSTPROCESS_HPP_
