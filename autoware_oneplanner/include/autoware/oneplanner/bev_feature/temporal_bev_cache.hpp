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

#ifndef AUTOWARE__ONEPLANNER__BEV_FEATURE__TEMPORAL_BEV_CACHE_HPP_
#define AUTOWARE__ONEPLANNER__BEV_FEATURE__TEMPORAL_BEV_CACHE_HPP_

#include "autoware/oneplanner/bev_feature/history_selection.hpp"
#include "autoware/oneplanner/cuda_utils.hpp"
#include "autoware/oneplanner/pose_discontinuity.hpp"

#include <rclcpp/time.hpp>

#include <cuda_runtime_api.h>

#include <array>
#include <cstdint>
#include <deque>
#include <vector>

namespace autoware::oneplanner
{

/**
 * @class TemporalBevCache
 * @brief Current-to-past cache of BEV feature maps for temporal planners.
 *
 * The temporal-history semantics the models are trained with:
 *
 * - One feature map is accepted per sensor frame, at the sensor's own cadence, and every map
 *   inside the history window `(frames - 1) * interval_seconds + tolerance` is kept. The history
 *   interval is a property of the history, not of the sensor: a 0.2 s history on a 10 Hz LiDAR
 *   stores five maps and selects every second one.
 * - `build_history()` assembles `[frames, C, H, W]` current-to-past, selecting for each step k
 *   the cached map closest to `newest - k * interval_seconds` (within tolerance). Every slot is
 *   SE(2)-warped from its source ego frame into the frame the plan starts in: the newest map's
 *   own frame when planning at the cloud stamp (slot 0 is then copied, not warped), or the
 *   newer planning pose, which warps slot 0 as well.
 * - A dropped sensor frame is filled the way the training data indexes its recorded scans:
 *   by position, so a missing scan's slot takes the next OLDER map, never a newer one. Training
 *   windows never span a gap in the recording, so the model has not seen a substitute; it is
 *   the least stale real map there is. A history step whose map was dropped therefore takes
 *   the newest cached map older than its target by at most `substitute_max_seconds`; a step
 *   with no map that close (an outage, not a drop) leaves a hole, and `ready()` turns false
 *   until the window refills, instead of discarding the whole cache. Step 0 is never
 *   substituted. Only a non-monotonic timestamp (a time jump, a replay restart) or a pose
 *   discontinuity resets the cache, which `insert()` reports.
 * - Warmup either waits for a complete history or duplicates the newest map into the missing
 *   steps (`duplicate_current_on_warmup`).
 */
class TemporalBevCache
{
public:
  struct Config
  {
    int64_t frames{3};
    //! Gap between history frames; the model's ml_package file states its own.
    double interval_seconds{0.2};
    double interval_tolerance_seconds{0.02};
    //! How much older than its target a map may be to stand in for a dropped one: 0.2 s
    //! covers up to two consecutive dropped 10 Hz scans; a longer gap is an outage.
    double substitute_max_seconds{0.2};
    double bev_half_extent_m{122.4};
    bool duplicate_current_on_warmup{false};
    PoseContinuityLimits pose_limits;
  };

  enum class InsertResult { kConsecutive, kFirst, kGapReset, kPoseReset };

  /**
   * @throws std::runtime_error on invalid configuration or CUDA allocation failure.
   */
  TemporalBevCache(
    const Config & config, const int64_t channels, const int64_t height, const int64_t width);

  /**
   * @brief Insert the newest feature map (device `[C, H, W]`) with its source
   * ego pose.
   *
   * The pose is `[x, y, cos(yaw), sin(yaw)]` in the map frame, kept in double: map
   * coordinates reach 1e5 m, and the SE(2) warp composes pose differences, which float
   * storage would quantize to centimetres before the warp ever sees them. A stamp before
   * the newest cached stamp resets the cache first and reports `kGapReset`.
   * Maps that fall out of the history window are evicted (their device buffers
   * are recycled). The copy is queued on `stream` and not waited for: the
   * caller may reuse the source buffer through work queued on the same stream
   * afterwards, and nothing else.
   */
  InsertResult insert(
    const float * d_feature, const std::array<double, 4> & pose, const rclcpp::Time & stamp,
    cudaStream_t stream);

  bool ready() const;
  size_t device_bytes() const
  {
    return (slots_.size() + free_slots_.size() + config_.frames) * frame_elements_ * sizeof(float);
  }
  int64_t cached_frames() const { return static_cast<int64_t>(slots_.size()); }
  //! History steps the current selection fills with an older map for a dropped one.
  int64_t substituted_frames() const;
  int64_t frames() const { return config_.frames; }

  /**
   * @brief Assemble the `[frames, C, H, W]` current-to-past history on the GPU, in the ego
   * frame of `target_pose` (`[x, y, cos(yaw), sin(yaw)]`, map frame, like `insert()`'s).
   *
   * A slot whose pose equals `target_pose` is copied; every other one is warped. Requires
   * `ready()`. The returned device buffer is owned by the cache and valid until the next
   * `insert()`/`build_history()` call. Its contents are complete in stream order: a consumer
   * on `stream` reads the finished history, a host reader synchronizes first.
   */
  const float * build_history(cudaStream_t stream, const std::array<double, 4> & target_pose);

  void reset();

private:
  struct Slot
  {
    cuda::DeviceUniquePtr<float[]> feature;
    std::array<double, 4> pose{};
    rclcpp::Time stamp;
  };

  std::vector<int64_t> current_selection() const;

  bool warmup_complete_{false};
  Config config_;
  int64_t channels_;
  int64_t height_;
  int64_t width_;
  size_t frame_elements_;

  std::deque<Slot> slots_;        //!< Front = newest.
  std::vector<Slot> free_slots_;  //!< Evicted slots kept for device-buffer reuse.
  cuda::DeviceUniquePtr<float[]> history_;
};

}  // namespace autoware::oneplanner

#endif  // AUTOWARE__ONEPLANNER__BEV_FEATURE__TEMPORAL_BEV_CACHE_HPP_
