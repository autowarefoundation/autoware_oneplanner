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

#ifndef AUTOWARE__ONEPLANNER__TIMESTAMPED_BUFFER_HPP_
#define AUTOWARE__ONEPLANNER__TIMESTAMPED_BUFFER_HPP_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <iterator>
#include <optional>
#include <utility>

namespace autoware::oneplanner
{

/**
 * @brief Samples kept by their stamp, oldest first, over a time window.
 *
 * Keyed by sensor time, not by callback count. One sample before the window is kept, so that
 * interpolation remains possible at the oldest requested stamp. A stamp older than the newest
 * one (a time jump, a replay restarted) clears the buffer.
 */
template <class T>
class TimestampedBuffer
{
public:
  struct Sample
  {
    int64_t stamp_ns;
    T value;
  };
  std::deque<Sample> samples;

  //! Add a sample and drop those older than `keep_ns` before it. A sample with the newest
  //! stamp replaces it. @return true when the stamp went backwards and the buffer was cleared.
  bool insert(int64_t stamp_ns, const T & value, int64_t keep_ns)
  {
    const bool rewound = !samples.empty() && stamp_ns < samples.back().stamp_ns;
    if (rewound) samples.clear();
    if (!samples.empty() && stamp_ns == samples.back().stamp_ns) {
      samples.back().value = value;
    } else {
      samples.push_back({stamp_ns, value});
    }
    while (samples.size() > 2 && samples[1].stamp_ns <= stamp_ns - keep_ns) {
      samples.pop_front();
    }
    return rewound;
  }

  //! Indices of the two samples bracketing `target_ns` (equal on an exact match), or
  //! std::nullopt outside the buffer.
  std::optional<std::pair<size_t, size_t>> bracket(int64_t target_ns) const
  {
    if (
      samples.empty() || target_ns < samples.front().stamp_ns ||
      target_ns > samples.back().stamp_ns)
      return std::nullopt;
    const auto hi = std::lower_bound(
      samples.begin(), samples.end(), target_ns,
      [](const Sample & sample, int64_t stamp) { return sample.stamp_ns < stamp; });
    const size_t index = static_cast<size_t>(hi - samples.begin());
    return std::make_pair(hi->stamp_ns == target_ns ? index : index - 1, index);
  }

  //! Whether every sample stamped at or before `target_ns` has arrived, for a stream delivered
  //! in stamp order: true once one stamped at or after it is here.
  bool covers(int64_t target_ns) const
  {
    return !samples.empty() && samples.back().stamp_ns >= target_ns;
  }

  //! The latest sample at or before `target_ns`, if it is at most `max_age_ns` older.
  const T * at_or_before(int64_t target_ns, int64_t max_age_ns) const
  {
    const auto after = std::upper_bound(
      samples.begin(), samples.end(), target_ns,
      [](int64_t stamp, const Sample & sample) { return stamp < sample.stamp_ns; });
    if (after == samples.begin()) return nullptr;
    const auto & sample = *std::prev(after);
    return target_ns - sample.stamp_ns <= max_age_ns ? &sample.value : nullptr;
  }
};

}  // namespace autoware::oneplanner

#endif  // AUTOWARE__ONEPLANNER__TIMESTAMPED_BUFFER_HPP_
