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

#ifndef AUTOWARE__ONEPLANNER__ROLLING_LATENCY_HPP_
#define AUTOWARE__ONEPLANNER__ROLLING_LATENCY_HPP_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <deque>
#include <vector>

namespace autoware::oneplanner
{

//! Nearest-rank percentiles of the last 512 timings, for the p95 / p99 debug topics.
class RollingLatency
{
public:
  //! Record one timing [ms]; a negative or non-finite value is ignored.
  void add(const double ms)
  {
    if (!std::isfinite(ms) || ms < 0) {
      return;
    }
    values_.push_back(ms);
    if (values_.size() > kWindow) {
      values_.pop_front();
    }
  }

  //! The `p` quantile (0 < p <= 1) of the window, or 0 when it is empty.
  double percentile(const double p) const
  {
    if (values_.empty()) {
      return 0.0;
    }
    std::vector<double> sorted(values_.begin(), values_.end());
    std::sort(sorted.begin(), sorted.end());
    return sorted[static_cast<size_t>(std::ceil(p * static_cast<double>(sorted.size()))) - 1];
  }

  void clear() { values_.clear(); }

private:
  static constexpr size_t kWindow = 512;
  std::deque<double> values_;
};

}  // namespace autoware::oneplanner

#endif  // AUTOWARE__ONEPLANNER__ROLLING_LATENCY_HPP_
