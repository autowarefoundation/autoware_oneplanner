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

#include "autoware/oneplanner/bev_feature/history_selection.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace autoware::oneplanner
{

double substitute_reach_seconds(
  const double interval_seconds, const double tolerance_seconds,
  const double substitute_max_seconds)
{
  return std::min(substitute_max_seconds + tolerance_seconds, interval_seconds - tolerance_seconds);
}

std::vector<int64_t> select_history_slots(
  const std::vector<double> & stamps_newest_first, const int64_t frames,
  const double interval_seconds, const double tolerance_seconds,
  const double substitute_max_seconds)
{
  std::vector<int64_t> selection(static_cast<size_t>(frames), -1);
  if (stamps_newest_first.empty()) {
    return selection;
  }
  const double newest = stamps_newest_first.front();
  for (int64_t step = 0; step < frames; ++step) {
    const double target = newest - static_cast<double>(step) * interval_seconds;
    double best_error = std::numeric_limits<double>::infinity();
    for (size_t index = 0; index < stamps_newest_first.size(); ++index) {
      const double error = std::abs(stamps_newest_first[index] - target);
      if (error <= tolerance_seconds && error < best_error) {
        best_error = error;
        selection[static_cast<size_t>(step)] = static_cast<int64_t>(index);
      }
    }
    if (step == 0 || selection[static_cast<size_t>(step)] >= 0) {
      continue;
    }
    // The map for this step was dropped. Training's index stride would have handed it the
    // next older recorded scan, so the newest map older than the target stands in -- when it
    // is close enough to be a drop and not an outage, and never the next step's own map.
    const double max_age =
      substitute_reach_seconds(interval_seconds, tolerance_seconds, substitute_max_seconds);
    for (size_t index = 0; index < stamps_newest_first.size(); ++index) {
      const double age = target - stamps_newest_first[index];
      if (age > tolerance_seconds) {
        if (age <= max_age) {
          selection[static_cast<size_t>(step)] = static_cast<int64_t>(index);
        }
        break;
      }
    }
  }
  return selection;
}

}  // namespace autoware::oneplanner
