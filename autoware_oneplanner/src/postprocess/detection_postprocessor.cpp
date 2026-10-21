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

#include "autoware/oneplanner/postprocess/detection_postprocessor.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace autoware::oneplanner
{

DetectionPostprocessor::DetectionPostprocessor(const Config & config)
: config_(config), iou_bev_nms_(config.iou_nms_search_distance_2d, config.iou_nms_threshold)
{
  if (config_.class_names.empty()) {
    throw std::runtime_error("bev_feature.detection.class_names is empty");
  }
  if (config_.circle_nms_dist_threshold < 0.0) {
    throw std::runtime_error("bev_feature.detection.circle_nms_dist_threshold must be >= 0");
  }
  // The matrices are indexed by ObjectClassification label, not by head class, so their side
  // is the label count and has nothing to do with class_names.size().
  if (
    !config_.allow_remapping_by_area_matrix.empty() || !config_.min_area_matrix.empty() ||
    !config_.max_area_matrix.empty()) {
    class_remapper_.emplace(
      config_.allow_remapping_by_area_matrix, config_.min_area_matrix, config_.max_area_matrix);
  }
}

std::vector<Box3D> circle_nms(const std::vector<Box3D> & boxes, const float distance_threshold)
{
  if (!(distance_threshold > 0.f)) {
    return boxes;
  }
  const float threshold_squared = distance_threshold * distance_threshold;
  std::vector<Box3D> kept;
  kept.reserve(boxes.size());
  std::vector<bool> suppressed(boxes.size(), false);
  for (size_t i = 0; i < boxes.size(); ++i) {
    if (suppressed[i]) {
      continue;
    }
    kept.push_back(boxes[i]);
    for (size_t j = i + 1; j < boxes.size(); ++j) {
      const float dx = boxes[i].x - boxes[j].x;
      const float dy = boxes[i].y - boxes[j].y;
      if (dx * dx + dy * dy < threshold_squared) {
        suppressed[j] = true;
      }
    }
  }
  return kept;
}

autoware_perception_msgs::msg::DetectedObjects DetectionPostprocessor::build(
  const std::vector<Box3D> & boxes, const std_msgs::msg::Header & header) const
{
  const auto kept = circle_nms(boxes, static_cast<float>(config_.circle_nms_dist_threshold));
  std::vector<autoware_perception_msgs::msg::DetectedObject> raw_objects;
  raw_objects.reserve(kept.size());
  for (const auto & box : kept) {
    raw_objects.push_back(to_detected_object(box, config_.class_names));
  }

  autoware_perception_msgs::msg::DetectedObjects message;
  message.header = header;
  message.objects = iou_bev_nms_.apply(raw_objects);
  if (class_remapper_) {
    class_remapper_->map_classes(message);
  }
  return message;
}

}  // namespace autoware::oneplanner
