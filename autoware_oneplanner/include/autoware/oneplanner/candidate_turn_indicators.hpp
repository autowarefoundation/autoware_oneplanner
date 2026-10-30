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

#ifndef AUTOWARE__ONEPLANNER__CANDIDATE_TURN_INDICATORS_HPP_
#define AUTOWARE__ONEPLANNER__CANDIDATE_TURN_INDICATORS_HPP_

#include <cstddef>
#include <type_traits>
#include <utility>
#include <vector>

namespace autoware::oneplanner
{

/// True when `Candidate` has a `turn_indicators_command` member. CandidateTrajectory has it from
/// autoware_internal_msgs 1.14.0 on; older versions, as in Autoware 1.9.0, lack it.
template <typename Candidate, typename = void>
struct HasTurnIndicatorsCommand : std::false_type
{
};

template <typename Candidate>
struct HasTurnIndicatorsCommand<
  Candidate, std::void_t<decltype(std::declval<Candidate &>().turn_indicators_command)>>
: std::true_type
{
};

/**
 * @brief Set the turn indicator of each candidate to the command of its batch.
 *
 * Candidate `k` belongs to batch `k % per_batch.size()`: the postprocessor appends one candidate
 * per batch for the prediction, then one per batch for each extra trajectory output. With one
 * command, every candidate gets it. The trajectory selector publishes the turn indicator of the
 * candidate that it picks.
 *
 * Does nothing when the message has no `turn_indicators_command` field, so the node still builds
 * against autoware_internal_msgs before 1.14.0. Header-only and templated, so both cases are
 * tested without ROS.
 */
template <typename Candidate, typename Command>
void set_candidate_turn_indicators(
  std::vector<Candidate> & candidates, const std::vector<Command> & per_batch)
{
  if constexpr (HasTurnIndicatorsCommand<Candidate>::value) {
    if (per_batch.empty()) {
      return;
    }
    for (size_t k = 0; k < candidates.size(); ++k) {
      candidates[k].turn_indicators_command = per_batch[k % per_batch.size()];
    }
  }
}

}  // namespace autoware::oneplanner

#endif  // AUTOWARE__ONEPLANNER__CANDIDATE_TURN_INDICATORS_HPP_
