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

#ifndef AUTOWARE__ONEPLANNER__BEV_FEATURE__TRT_BEV_FEATURE_EXTRACTOR_HPP_
#define AUTOWARE__ONEPLANNER__BEV_FEATURE__TRT_BEV_FEATURE_EXTRACTOR_HPP_

#include "autoware/oneplanner/bev_feature/detection_decode.hpp"
#include "autoware/oneplanner/bev_feature/voxelizer.hpp"
#include "autoware/oneplanner/cuda_utils.hpp"
#include "autoware/oneplanner/trt/trt_engine.hpp"

#include <sensor_msgs/msg/point_cloud2.hpp>

#include <cuda_runtime_api.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace autoware::oneplanner
{

/**
 * @class TrtBevFeatureExtractor
 * @brief Runs the frozen BEVFusion-L feature extractor: point cloud -> BEV feature map.
 *
 * The deployment boundary of a OnePlanner model: voxelization is autoware_bevfusion's (see
 * Voxelizer), the sparse convolutions run inside the TensorRT engine (spconv through
 * `autoware_tensorrt_plugins`), and the engine's feature output (default `bev_feature`,
 * `[1, C, H, W]`) is exposed as a device buffer.
 *
 * The engine takes the same LiDAR-branch inputs as autoware_bevfusion (`voxels`,
 * `num_points_per_voxel`, `coors`, with a dynamic voxel count) and must expose the feature map
 * as an output with static dimensions.
 */
class TrtBevFeatureExtractor
{
public:
  struct Config
  {
    std::string onnx_path;
    std::string plugins_path;
    //! Engine cache path; empty puts `<onnx stem>.engine` beside the ONNX file.
    std::string engine_path;
    std::string precision{"fp16"};
    //! TensorRT builder workspace, shared with the planner engine (`trt_workspace_mib`).
    size_t max_workspace_size{4ULL << 30U};
    std::string feature_tensor{"bev_feature"};
    int64_t cloud_capacity{2000000};
    // Network description, read from the model's ml_package file with no fallback: a
    // default here would silently voxelize for a different network.
    int64_t max_points_per_voxel{0};
    std::vector<int64_t> voxels_num;       //!< [min, opt, max]
    std::vector<float> point_cloud_range;  //!< [x/y/z min, x/y/z max]
    std::vector<float> voxel_size;         //!< [x, y, z]
    bool use_intensity{false};

    /**
     * @brief The BEVFusion detection head the extractor graph may carry.
     *
     * The exported graph emits `bbox_pred [10, P]`, `score [P]` and `label_pred [P]` next to
     * the feature map: undecoded proposals, where a BEVFusion-L ONNX export for
     * autoware_bevfusion stops, because that node decodes and suppresses on the C++ side.
     * The decode here is its decode kernel (TransFusion coder, yaw-norm gate, per-(distance
     * band, class) score thresholds) with its arithmetic, fused with compaction; see
     * detection_decode.hpp.
     */
    struct Detection
    {
      //! Off by default: a graph without the head is still a valid extractor.
      bool enabled{false};
      std::string bbox_tensor{"bbox_pred"};
      std::string score_tensor{"score"};
      std::string label_tensor{"label_pred"};
      //! Proposal count the head emits; validated against the engine's own shapes.
      int64_t num_proposals{0};
      //! BEV stride the TransFusion coder decodes centres with.
      int64_t out_size_factor{8};
      //! The calibrated score thresholds: ascending band limits [m] and one threshold per
      //! class per band, distance-major (make_score_threshold_table).
      std::vector<double> distance_bin_upper_limits;
      std::vector<double> score_thresholds;
      //! One entry per class; a proposal whose (sin, cos) norm falls below its class's
      //! threshold is scored zero, as in autoware_bevfusion.
      std::vector<double> yaw_norm_thresholds;
    };
    Detection detection;
  };

  /**
   * @throws std::runtime_error on invalid configuration or engine setup failure.
   */
  TrtBevFeatureExtractor(const Config & config, cudaStream_t stream);

  int64_t channels() const { return channels_; }
  int64_t height() const { return height_; }
  int64_t width() const { return width_; }

  /**
   * @brief Extract the BEV feature map for one point cloud.
   *
   * The cloud is copied to the device once; it must use the Autoware `PointXYZIRC` layout
   * (the concatenated cloud's format) and be in `base_link`.
   * @return Device pointer to the `[C, H, W]` feature map (owned by the extractor, valid until
   *         the next call), or nullptr with `error` set. The map is complete in stream
   *         order: the work is queued on the constructor's stream and not waited for, so a
   *         consumer on that stream reads the finished map and a host reader synchronizes
   *         first. The one host wait inside is the voxel count, which sizes the engine's
   *         inputs.
   */
  const float * extract(const sensor_msgs::msg::PointCloud2 & cloud, std::string & error);

  /// True when the engine carries the detection head and it was configured on.
  bool detection_enabled() const { return detection_enabled_; }
  /**
   * @brief Decode the detection head's proposals of the last extract() into
   *        last_detections().
   *
   * Separate from extract() so that it can run after the trajectory is out: the planner
   * never reads these boxes, so nothing that consumes the trajectory should wait for them.
   * One kernel and one host wait: the survivors come back in a single copy and are put in
   * `autoware_bevfusion`'s order. A no-op when detection is disabled or the last extract()
   * has already been decoded.
   * @return false with `error` set when the decode kernels fail.
   */
  bool decode_detections(std::string & error);
  /**
   * @brief Proposals of the last decode_detections() that passed the score thresholds,
   *        score-descending, before any NMS.
   *
   * Empty when detection is disabled. Boxes are in the frame of the cloud that
   * produced them, in metres, still carrying every class the head knows.
   */
  const std::vector<Box3D> & last_detections() const { return last_detections_; }

  /// Voxel count of the last extract(), before clamping to the profile maximum.
  int64_t last_num_voxels() const { return last_num_voxels_; }
  /// False when the last cloud produced more voxels than the profile maximum and was clipped.
  bool last_voxels_within_range() const { return last_voxels_within_range_; }

private:
  void init_engine(const Config & config);
  void init_detection(const Config & config);
  /// Give every engine output an address, including ones this node never reads.
  void bind_unread_outputs();
  bool validate_cloud_layout(
    const sensor_msgs::msg::PointCloud2 & cloud, std::string & error) const;

  bev::VoxelizerConfig voxelizer_config_;
  std::int64_t min_num_voxels_{0};
  std::string feature_tensor_;
  cudaStream_t stream_;

  std::unique_ptr<bev::Voxelizer> voxelizer_;
  std::unique_ptr<trt::TrtEngine> engine_;

  // Device buffers (allocated once at maximum size)
  cuda::DeviceUniquePtr<uint8_t[]> cloud_d_;
  cuda::DeviceUniquePtr<float[]> points_d_;
  cuda::DeviceUniquePtr<float[]> voxel_features_d_;
  cuda::DeviceUniquePtr<int32_t[]> voxel_coords_d_;
  cuda::DeviceUniquePtr<int32_t[]> num_points_per_voxel_d_;
  cuda::DeviceUniquePtr<float[]> feature_d_;
  cuda::DeviceUniquePtr<float[]> bbox_pred_d_;
  cuda::DeviceUniquePtr<float[]> score_d_;
  cuda::DeviceUniquePtr<int64_t[]> label_pred_d_;
  // Detection decode: the thresholds, the compacted survivors and their count, and the
  // pinned host copy both come back to in one transfer.
  DetectionDecodeGeometry decode_geometry_{};
  cuda::DeviceUniquePtr<float[]> yaw_norm_thresholds_d_;
  cuda::DeviceUniquePtr<float[]> squared_upper_limits_d_;
  cuda::DeviceUniquePtr<float[]> score_thresholds_d_;
  cuda::DeviceUniquePtr<DecodedBox[]> decoded_d_;
  cuda::DeviceUniquePtr<int32_t[]> decoded_count_d_;
  cuda::HostUniquePtr<DecodedBox[]> decoded_h_;
  cuda::HostUniquePtr<int32_t[]> decoded_count_h_;
  std::vector<DecodedBox> decoded_;
  /// Outputs bound above, by engine tensor name; the rest get scratch buffers.
  std::vector<std::string> bound_outputs_;
  std::vector<cuda::DeviceUniquePtr<uint8_t[]>> unread_output_scratch_;

  int64_t channels_{0};
  int64_t height_{0};
  int64_t width_{0};
  int64_t last_num_voxels_{0};
  bool last_voxels_within_range_{true};
  bool detection_enabled_{false};
  bool detections_pending_{false};
  std::vector<Box3D> last_detections_;
};

}  // namespace autoware::oneplanner

#endif  // AUTOWARE__ONEPLANNER__BEV_FEATURE__TRT_BEV_FEATURE_EXTRACTOR_HPP_
