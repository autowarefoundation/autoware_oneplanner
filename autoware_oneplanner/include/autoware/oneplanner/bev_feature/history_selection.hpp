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

#ifndef AUTOWARE__ONEPLANNER__BEV_FEATURE__HISTORY_SELECTION_HPP_
#define AUTOWARE__ONEPLANNER__BEV_FEATURE__HISTORY_SELECTION_HPP_

#include <cstdint>
#include <vector>

namespace autoware::oneplanner
{

/**
 * @brief Which cached map fills each step of the BEV history.
 *
 * For each history step k, the index of the stamp closest to
 * `target = stamps[0] - k * interval_seconds` within tolerance; failing that (k >= 1), the
 * newest stamp older than the target by at most `substitute_max_seconds` (+ tolerance) and
 * short of the next step's target (- tolerance); -1 when neither exists.
 *
 * `stamps_newest_first` is ordered newest first; step 0 always selects index 0.
 */
std::vector<int64_t> select_history_slots(
  const std::vector<double> & stamps_newest_first, int64_t frames, double interval_seconds,
  double tolerance_seconds, double substitute_max_seconds);

//! How much older than its target a substitute may be: the bound, short of the next step.
double substitute_reach_seconds(
  double interval_seconds, double tolerance_seconds, double substitute_max_seconds);

}  // namespace autoware::oneplanner

#endif  // AUTOWARE__ONEPLANNER__BEV_FEATURE__HISTORY_SELECTION_HPP_
