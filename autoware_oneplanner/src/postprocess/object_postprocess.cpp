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

// Adapted from autoware_bevfusion's host postprocessing (autoware_universe).

#include "autoware/oneplanner/postprocess/object_postprocess.hpp"

#include <autoware/object_recognition_utils/geometry.hpp>
#include <autoware/object_recognition_utils/object_recognition_utils.hpp>
#include <autoware_utils_geometry/geometry.hpp>

#include <autoware_perception_msgs/msg/detected_object_kinematics.hpp>
#include <autoware_perception_msgs/msg/object_classification.hpp>
#include <autoware_perception_msgs/msg/shape.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace autoware::oneplanner
{

using autoware_perception_msgs::msg::DetectedObject;
using Label = autoware_perception_msgs::msg::ObjectClassification;

IouBevNms::IouBevNms(const double search_distance_2d, const double iou_threshold)
: search_distance_2d_squared_(search_distance_2d * search_distance_2d),
  iou_threshold_(iou_threshold)
{
  if (search_distance_2d < 0.0) {
    throw std::runtime_error("bev_feature.detection.iou_nms_search_distance_2d must be >= 0");
  }
  if (iou_threshold < 0.0 || iou_threshold > 1.0) {
    throw std::runtime_error("bev_feature.detection.iou_nms_threshold must be in [0, 1]");
  }
}

bool IouBevNms::is_target_pair(const DetectedObject & a, const DetectedObject & b) const
{
  namespace utils = autoware::object_recognition_utils;
  const auto label_a = utils::getHighestProbLabel(a.classification);
  const auto label_b = utils::getHighestProbLabel(b.classification);
  if (label_a != label_b && (label_a == Label::PEDESTRIAN || label_b == Label::PEDESTRIAN)) {
    return false;
  }
  return autoware_utils_geometry::calc_squared_distance2d(utils::getPose(a), utils::getPose(b)) <=
         search_distance_2d_squared_;
}

std::vector<DetectedObject> IouBevNms::apply(const std::vector<DetectedObject> & objects) const
{
  // Row: the object that may be suppressed; column: a higher-ranked one compared with it.
  Eigen::MatrixXd iou = Eigen::MatrixXd::Zero(
    static_cast<Eigen::Index>(objects.size()), static_cast<Eigen::Index>(objects.size()));
  for (size_t target = 0; target < objects.size(); ++target) {
    for (size_t source = 0; source < target; ++source) {
      if (!is_target_pair(objects[target], objects[source])) {
        continue;
      }
      const double value =
        autoware::object_recognition_utils::get2dIoU(objects[target], objects[source]);
      iou(static_cast<Eigen::Index>(target), static_cast<Eigen::Index>(source)) = value;
      // One overlap above the threshold suppresses the target whatever follows.
      if (value > iou_threshold_) {
        break;
      }
    }
  }

  std::vector<DetectedObject> kept;
  kept.reserve(objects.size());
  for (size_t i = 0; i < objects.size(); ++i) {
    if (iou.row(static_cast<Eigen::Index>(i)).maxCoeff() <= iou_threshold_) {
      kept.push_back(objects[i]);
    }
  }
  return kept;
}

ClassRemapper::ClassRemapper(
  const std::vector<int64_t> & allow_remapping_by_area_matrix,
  const std::vector<double> & min_area_matrix, const std::vector<double> & max_area_matrix)
{
  const size_t elements = min_area_matrix.size();
  const auto side = static_cast<int>(std::llround(std::sqrt(static_cast<double>(elements))));
  if (
    static_cast<size_t>(side) * static_cast<size_t>(side) != elements ||
    allow_remapping_by_area_matrix.size() != elements || max_area_matrix.size() != elements) {
    throw std::runtime_error(
      "the detection class remapper matrices must be square and of one size: they describe "
      "the same label-by-label table");
  }
  num_labels_ = side;

  // The parameters are row-major; Eigen maps column-major, hence the transposes.
  const Eigen::Map<const Eigen::Matrix<int64_t, Eigen::Dynamic, Eigen::Dynamic>> allow(
    allow_remapping_by_area_matrix.data(), side, side);
  allow_remapping_by_area_ = allow.transpose().cast<bool>();
  min_area_ = Eigen::Map<const Eigen::MatrixXd>(min_area_matrix.data(), side, side).transpose();
  max_area_ = Eigen::Map<const Eigen::MatrixXd>(max_area_matrix.data(), side, side).transpose();
  const auto unbounded = [](const double v) {
    return std::isfinite(v) ? v : std::numeric_limits<double>::max();
  };
  min_area_ = min_area_.unaryExpr(unbounded);
  max_area_ = max_area_.unaryExpr(unbounded);
}

void ClassRemapper::map_classes(autoware_perception_msgs::msg::DetectedObjects & objects) const
{
  for (auto & object : objects.objects) {
    const float bev_area = object.shape.dimensions.x * object.shape.dimensions.y;
    for (auto & classification : object.classification) {
      auto & label = classification.label;
      if (label >= num_labels_) {
        continue;  // a label the table does not cover
      }
      for (int to = 0; to < num_labels_; ++to) {
        if (
          allow_remapping_by_area_(label, to) && bev_area >= min_area_(label, to) &&
          bev_area <= max_area_(label, to)) {
          label = static_cast<uint8_t>(to);
          break;
        }
      }
    }
  }
}

uint8_t semantic_type(const std::string & class_name)
{
  if (class_name == "CAR") {
    return Label::CAR;
  }
  if (class_name == "TRUCK") {
    return Label::TRUCK;
  }
  if (class_name == "BUS") {
    return Label::BUS;
  }
  if (class_name == "TRAILER") {
    return Label::TRAILER;
  }
  if (class_name == "MOTORCYCLE") {
    return Label::MOTORCYCLE;
  }
  if (class_name == "BICYCLE") {
    return Label::BICYCLE;
  }
  if (class_name == "PEDESTRIAN") {
    return Label::PEDESTRIAN;
  }
  return Label::UNKNOWN;
}

DetectedObject to_detected_object(const Box3D & box, const std::vector<std::string> & class_names)
{
  DetectedObject object;
  object.existence_probability = box.score;

  Label classification;
  classification.probability = 1.0f;
  classification.label = box.label >= 0 && static_cast<size_t>(box.label) < class_names.size()
                           ? semantic_type(class_names[static_cast<size_t>(box.label)])
                           : Label::UNKNOWN;
  if (autoware::object_recognition_utils::isCarLikeVehicle(classification.label)) {
    object.kinematics.orientation_availability =
      autoware_perception_msgs::msg::DetectedObjectKinematics::SIGN_UNKNOWN;
  }
  object.classification.push_back(classification);

  object.kinematics.pose_with_covariance.pose.position =
    autoware_utils_geometry::create_point(box.x, box.y, box.z);
  object.kinematics.pose_with_covariance.pose.orientation =
    autoware_utils_geometry::create_quaternion_from_yaw(box.yaw);
  object.shape.type = autoware_perception_msgs::msg::Shape::BOUNDING_BOX;
  object.shape.dimensions =
    autoware_utils_geometry::create_translation(box.length, box.width, box.height);
  return object;
}

}  // namespace autoware::oneplanner
