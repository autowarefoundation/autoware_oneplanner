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

#include "autoware/oneplanner/bev_feature/trt_bev_feature_extractor.hpp"

#include "autoware/oneplanner/bev_feature/detection_decode_kernel.hpp"
#include "autoware/oneplanner/types.hpp"

#include <sensor_msgs/msg/point_field.hpp>

#include <NvInfer.h>

#include <algorithm>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace autoware::oneplanner
{

namespace
{

//! The number of values per proposal in `bbox_pred`: x, y, z, log length, log width,
//! log height, sin(yaw), cos(yaw), vx, vy.
constexpr int64_t kNumBoxValues = 10;

bev::VoxelizerConfig make_voxelizer_config(const TrtBevFeatureExtractor::Config & config)
{
  if (config.voxels_num.size() != 3) {
    throw std::runtime_error("bev_feature.extractor.voxels_num must have 3 elements [min,opt,max]");
  }
  if (config.point_cloud_range.size() != 6) {
    throw std::runtime_error("bev_feature.extractor.point_cloud_range must have 6 elements");
  }
  if (config.voxel_size.size() != 3) {
    throw std::runtime_error("bev_feature.extractor.voxel_size must have 3 elements");
  }
  if (config.max_points_per_voxel <= 0 || config.cloud_capacity <= 0) {
    throw std::runtime_error(
      "bev_feature.extractor.max_points_per_voxel and cloud_capacity must be positive");
  }
  bev::VoxelizerConfig voxelizer;
  voxelizer.cloud_capacity = config.cloud_capacity;
  voxelizer.max_points_per_voxel = config.max_points_per_voxel;
  voxelizer.max_num_voxels = config.voxels_num[2];
  voxelizer.use_intensity = config.use_intensity;
  std::copy(
    config.point_cloud_range.begin(), config.point_cloud_range.end(),
    voxelizer.point_cloud_range.begin());
  std::copy(config.voxel_size.begin(), config.voxel_size.end(), voxelizer.voxel_size.begin());
  return voxelizer;
}

size_t element_size(const nvinfer1::DataType dtype)
{
  switch (dtype) {
    case nvinfer1::DataType::kFLOAT:
    case nvinfer1::DataType::kINT32:
      return 4;
    case nvinfer1::DataType::kHALF:
    case nvinfer1::DataType::kBF16:
      return 2;
    case nvinfer1::DataType::kINT64:
      return 8;
    case nvinfer1::DataType::kINT8:
    case nvinfer1::DataType::kUINT8:
    case nvinfer1::DataType::kBOOL:
    case nvinfer1::DataType::kFP8:
      return 1;
    default:
      return 0;
  }
}

std::vector<int64_t> to_shape(const nvinfer1::Dims & dims)
{
  return std::vector<int64_t>(dims.d, dims.d + dims.nbDims);
}

}  // namespace

TrtBevFeatureExtractor::TrtBevFeatureExtractor(const Config & config, cudaStream_t stream)
: voxelizer_config_(make_voxelizer_config(config)),
  min_num_voxels_(config.voxels_num[0]),
  feature_tensor_(config.feature_tensor),
  stream_(stream)
{
  voxelizer_ = std::make_unique<bev::Voxelizer>(voxelizer_config_, stream_);

  const auto capacity = static_cast<size_t>(voxelizer_config_.cloud_capacity);
  const auto features = static_cast<size_t>(voxelizer_config_.num_point_features());
  const auto max_voxels = static_cast<size_t>(voxelizer_config_.max_num_voxels);
  cloud_d_ = cuda::make_device_unique<uint8_t[]>(capacity * sizeof(bev::InputPoint));
  points_d_ = cuda::make_device_unique<float[]>(capacity * features);
  voxel_features_d_ = cuda::make_device_unique<float[]>(
    max_voxels * static_cast<size_t>(voxelizer_config_.max_points_per_voxel) * features);
  voxel_coords_d_ = cuda::make_device_unique<int32_t[]>(max_voxels * 3);
  num_points_per_voxel_d_ = cuda::make_device_unique<int32_t[]>(max_voxels);

  init_engine(config);
}

void TrtBevFeatureExtractor::init_engine(const Config & config)
{
  // The LiDAR-branch IO contract of autoware_bevfusion: a dynamic voxel count with a
  // min/opt/max optimization profile.
  const int64_t max_ppv = voxelizer_config_.max_points_per_voxel;
  const int64_t feature_size = voxelizer_config_.num_point_features();
  const auto voxel_dims = [max_ppv, feature_size](const int64_t count) {
    return nvinfer1::Dims{3, {count, max_ppv, feature_size}};
  };

  trt::EngineConfig engine_config;
  engine_config.onnx_path = config.onnx_path;
  engine_config.engine_path = config.engine_path;
  engine_config.precision = config.precision;
  engine_config.workspace_bytes = config.max_workspace_size;
  engine_config.plugin_paths = {config.plugins_path};
  engine_config.profile = {
    {"voxels", voxel_dims(config.voxels_num[0]), voxel_dims(config.voxels_num[1]),
     voxel_dims(config.voxels_num[2])},
    {"num_points_per_voxel", nvinfer1::Dims{1, {config.voxels_num[0]}},
     nvinfer1::Dims{1, {config.voxels_num[1]}}, nvinfer1::Dims{1, {config.voxels_num[2]}}},
    {"coors", nvinfer1::Dims{2, {config.voxels_num[0], 3}},
     nvinfer1::Dims{2, {config.voxels_num[1], 3}}, nvinfer1::Dims{2, {config.voxels_num[2], 3}}},
  };
  engine_ = std::make_unique<trt::TrtEngine>(engine_config);

  // The feature output must have static dimensions: the sparse voxel count only affects the
  // engine's internal tensors, and the scattered BEV grid is fixed by the model.
  if (!engine_->has_tensor(feature_tensor_)) {
    throw std::runtime_error(
      "The BEV feature extractor graph has no output named '" + feature_tensor_ + "'");
  }
  const auto dims = to_shape(engine_->tensor_shape(feature_tensor_.c_str()));
  const bool rank4 = dims.size() == 4 && dims[0] == 1;
  const bool rank3 = dims.size() == 3;
  if (!(rank3 || rank4) || std::any_of(dims.begin(), dims.end(), [](int64_t d) { return d < 1; })) {
    throw std::runtime_error(
      "BEV feature extractor output '" + feature_tensor_ +
      "' must have a static [1, C, H, W] or [C, H, W] shape; got " + shape_to_string(dims));
  }
  const size_t base = rank4 ? 1 : 0;
  channels_ = dims[base];
  height_ = dims[base + 1];
  width_ = dims[base + 2];

  feature_d_ = cuda::make_device_unique<float[]>(static_cast<size_t>(channels_ * height_ * width_));

  engine_->set_tensor_address("voxels", voxel_features_d_.get());
  engine_->set_tensor_address("num_points_per_voxel", num_points_per_voxel_d_.get());
  engine_->set_tensor_address("coors", voxel_coords_d_.get());
  engine_->set_tensor_address(feature_tensor_.c_str(), feature_d_.get());
  bound_outputs_.push_back(feature_tensor_);

  init_detection(config);
  bind_unread_outputs();
}

void TrtBevFeatureExtractor::bind_unread_outputs()
{
  // enqueueV3 requires an address for every output the engine declares, not just the ones
  // this node reads. The extractor graph carries a detection head whether or not the
  // deployment asked for boxes, so give the outputs nobody reads scratch buffers.
  const int32_t num_io = engine_->num_io_tensors();
  for (int32_t i = 0; i < num_io; ++i) {
    const char * name = engine_->io_tensor_name(i);
    if (engine_->is_input(name)) {
      continue;
    }
    if (std::find(bound_outputs_.begin(), bound_outputs_.end(), name) != bound_outputs_.end()) {
      continue;
    }

    const nvinfer1::Dims dims = engine_->tensor_shape(name);
    size_t count = 1;
    for (int32_t d = 0; d < dims.nbDims; ++d) {
      // A dynamic extent would make the size unknowable here.
      if (dims.d[d] < 0) {
        throw std::runtime_error(
          "The BEV feature extractor's output '" + std::string(name) +
          "' has a dynamic extent; its size cannot be bound ahead of inference");
      }
      count *= static_cast<size_t>(dims.d[d]);
    }
    const size_t bytes = count * element_size(engine_->tensor_dtype(name));
    if (bytes == 0) {
      throw std::runtime_error(
        "The BEV feature extractor's output '" + std::string(name) +
        "' has an unsupported data type; it cannot be given a buffer");
    }

    unread_output_scratch_.push_back(cuda::make_device_unique<uint8_t[]>(bytes));
    engine_->set_tensor_address(name, unread_output_scratch_.back().get());
    bound_outputs_.emplace_back(name);
  }
}

void TrtBevFeatureExtractor::init_detection(const Config & config)
{
  const auto & detection = config.detection;
  if (!detection.enabled) {
    return;
  }

  // A graph exported without the head is still a valid extractor: run without it rather than
  // failing a deployment that never asked for boxes. A graph that has the tensors but
  // disagrees on their shape is a real mismatch.
  const bool complete = engine_->has_tensor(detection.bbox_tensor) &&
                        engine_->has_tensor(detection.score_tensor) &&
                        engine_->has_tensor(detection.label_tensor);
  if (!complete) {
    return;
  }

  if (detection.num_proposals <= 0) {
    throw std::runtime_error(
      "bev_feature.detection is enabled but num_proposals is " +
      std::to_string(detection.num_proposals));
  }
  const auto class_count = static_cast<int64_t>(detection.yaw_norm_thresholds.size());
  if (class_count <= 0) {
    throw std::runtime_error(
      "bev_feature.detection.yaw_norm_thresholds is empty; the decode kernel indexes it by "
      "class and would read out of bounds");
  }

  // The decode kernel reads bbox_pred as [num_box_values, num_proposals] and the other two as
  // [num_proposals]; a head exported with another proposal count would be read at the wrong
  // stride.
  const auto expect = [this](const std::string & name, const std::vector<int64_t> & want) {
    const auto got = to_shape(engine_->tensor_shape(name.c_str()));
    if (got != want) {
      throw std::runtime_error(
        "The BEV feature extractor's detection output '" + name + "' has shape " +
        shape_to_string(got) + ", expected " + shape_to_string(want) +
        " (bev_feature.detection.num_proposals disagrees with the graph)");
    }
  };
  const int64_t proposals = detection.num_proposals;
  expect(detection.bbox_tensor, {kNumBoxValues, proposals});
  expect(detection.score_tensor, {proposals});
  expect(detection.label_tensor, {proposals});

  bbox_pred_d_ = cuda::make_device_unique<float[]>(static_cast<size_t>(kNumBoxValues * proposals));
  score_d_ = cuda::make_device_unique<float[]>(static_cast<size_t>(proposals));
  label_pred_d_ = cuda::make_device_unique<int64_t[]>(static_cast<size_t>(proposals));

  engine_->set_tensor_address(detection.bbox_tensor.c_str(), bbox_pred_d_.get());
  engine_->set_tensor_address(detection.score_tensor.c_str(), score_d_.get());
  engine_->set_tensor_address(detection.label_tensor.c_str(), label_pred_d_.get());
  bound_outputs_.push_back(detection.bbox_tensor);
  bound_outputs_.push_back(detection.score_tensor);
  bound_outputs_.push_back(detection.label_tensor);

  const auto table = make_score_threshold_table(
    detection.distance_bin_upper_limits, detection.score_thresholds,
    static_cast<size_t>(class_count));
  const std::vector<float> yaw_norm_thresholds(
    detection.yaw_norm_thresholds.begin(), detection.yaw_norm_thresholds.end());
  const auto upload = [this](const std::vector<float> & values) {
    auto device = cuda::make_device_unique<float[]>(values.size());
    ONEPLANNER_CHECK_CUDA(cudaMemcpyAsync(
      device.get(), values.data(), values.size() * sizeof(float), cudaMemcpyHostToDevice, stream_));
    return device;
  };
  yaw_norm_thresholds_d_ = upload(yaw_norm_thresholds);
  squared_upper_limits_d_ = upload(table.squared_upper_limits);
  score_thresholds_d_ = upload(table.thresholds);
  // The uploads read host vectors that end with this scope.
  ONEPLANNER_CHECK_CUDA(cudaStreamSynchronize(stream_));

  decode_geometry_.num_proposals = static_cast<int32_t>(proposals);
  decode_geometry_.num_classes = static_cast<int32_t>(class_count);
  decode_geometry_.num_bands = static_cast<int32_t>(table.squared_upper_limits.size());
  decode_geometry_.out_size_factor = static_cast<float>(detection.out_size_factor);
  decode_geometry_.voxel_size_x = config.voxel_size[0];
  decode_geometry_.voxel_size_y = config.voxel_size[1];
  decode_geometry_.min_x_range = config.point_cloud_range[0];
  decode_geometry_.min_y_range = config.point_cloud_range[1];

  const auto capacity = static_cast<size_t>(proposals);
  decoded_d_ = cuda::make_device_unique<DecodedBox[]>(capacity);
  decoded_count_d_ = cuda::make_device_unique<int32_t[]>(1);
  decoded_h_ = cuda::make_host_unique<DecodedBox[]>(capacity);
  decoded_count_h_ = cuda::make_host_unique<int32_t[]>(1);
  decoded_.reserve(capacity);
  last_detections_.reserve(capacity);
  detection_enabled_ = true;
}

bool TrtBevFeatureExtractor::validate_cloud_layout(
  const sensor_msgs::msg::PointCloud2 & cloud, std::string & error) const
{
  // The temporal warp places each map by the base_link odometry pose, so a cloud in any other
  // frame would be misplaced silently.
  if (cloud.header.frame_id != "base_link") {
    error = "Point cloud frame is '" + cloud.header.frame_id + "', expected 'base_link'";
    return false;
  }
  if (cloud.point_step != sizeof(bev::InputPoint)) {
    error = "Point cloud point_step is " + std::to_string(cloud.point_step) + ", expected " +
            std::to_string(sizeof(bev::InputPoint)) + " (the Autoware PointXYZIRC layout)";
    return false;
  }
  using sensor_msgs::msg::PointField;
  const auto has_field = [&cloud](
                           const std::string & name, const uint32_t offset, const uint8_t type) {
    return std::any_of(cloud.fields.begin(), cloud.fields.end(), [&](const PointField & field) {
      return field.name == name && field.offset == offset && field.datatype == type;
    });
  };
  if (
    !has_field("x", 0, PointField::FLOAT32) || !has_field("y", 4, PointField::FLOAT32) ||
    !has_field("z", 8, PointField::FLOAT32) || !has_field("intensity", 12, PointField::UINT8)) {
    error = "Point cloud fields do not match the Autoware PointXYZIRC layout";
    return false;
  }
  if (cloud.data.size() < static_cast<size_t>(cloud.width) * cloud.height * cloud.point_step) {
    error = "Point cloud data is shorter than width * height * point_step";
    return false;
  }
  return true;
}

const float * TrtBevFeatureExtractor::extract(
  const sensor_msgs::msg::PointCloud2 & cloud, std::string & error)
{
  if (!validate_cloud_layout(cloud, error)) {
    return nullptr;
  }

  size_t num_points = static_cast<size_t>(cloud.width) * cloud.height;
  if (num_points == 0) {
    error = "Point cloud is empty";
    return nullptr;
  }
  num_points = std::min(num_points, static_cast<size_t>(voxelizer_config_.cloud_capacity));

  // One upload of the cloud; the rest of the pipeline stays on the device.
  ONEPLANNER_CHECK_CUDA(cudaMemcpyAsync(
    cloud_d_.get(), cloud.data.data(), num_points * sizeof(bev::InputPoint), cudaMemcpyHostToDevice,
    stream_));
  ONEPLANNER_CHECK_CUDA(voxelizer_->generate_points(
    reinterpret_cast<const bev::InputPoint *>(cloud_d_.get()), num_points, points_d_.get()));

  // The voxel count comes back to the host, which is the one wait this function cannot
  // avoid: the engine's input shapes are set from it.
  const auto num_voxels = static_cast<int64_t>(voxelizer_->generate_voxels(
    points_d_.get(), static_cast<unsigned int>(num_points), voxel_features_d_.get(),
    voxel_coords_d_.get(), num_points_per_voxel_d_.get()));

  last_num_voxels_ = num_voxels;
  // As autoware_bevfusion: below the profile minimum the frame cannot run; above the maximum
  // it runs on a clipped voxel set and the diagnostics say so.
  last_voxels_within_range_ = num_voxels < voxelizer_config_.max_num_voxels;
  if (num_voxels < min_num_voxels_) {
    error = "Too few voxels (" + std::to_string(num_voxels) +
            ") for the optimization profile minimum (" + std::to_string(min_num_voxels_) + ")";
    return nullptr;
  }
  const auto clamped_voxels = std::min(num_voxels, voxelizer_config_.max_num_voxels);

  const int64_t max_ppv = voxelizer_config_.max_points_per_voxel;
  const int64_t feature_size = voxelizer_config_.num_point_features();
  engine_->set_input_shape("voxels", nvinfer1::Dims{3, {clamped_voxels, max_ppv, feature_size}});
  engine_->set_input_shape("num_points_per_voxel", nvinfer1::Dims{1, {clamped_voxels}});
  engine_->set_input_shape("coors", nvinfer1::Dims{2, {clamped_voxels, 3}});

  if (!engine_->enqueue(stream_)) {
    error = "Failed to enqueue the BEV feature extractor engine";
    return nullptr;
  }
  // Not waited for. The feature map is complete in stream order for whoever reads it on this
  // stream; the proposals stay on the device until decode_detections() asks for them.
  detections_pending_ = detection_enabled_;
  return feature_d_.get();
}

bool TrtBevFeatureExtractor::decode_detections(std::string & error)
{
  if (!detections_pending_) {
    return true;
  }
  detections_pending_ = false;
  // Ordered on the stream behind the extractor engine, so the proposals it reads are
  // finished. The survivor buffer is copied whole with its count: at most num_proposals
  // boxes, one transfer and one wait instead of a count round trip first.
  last_detections_.clear();
  const auto capacity = static_cast<size_t>(decode_geometry_.num_proposals);
  cudaError_t status = cudaMemsetAsync(decoded_count_d_.get(), 0, sizeof(int32_t), stream_);
  if (status == cudaSuccess) {
    status = launch_decode_detections(
      label_pred_d_.get(), bbox_pred_d_.get(), score_d_.get(), yaw_norm_thresholds_d_.get(),
      squared_upper_limits_d_.get(), score_thresholds_d_.get(), decode_geometry_, decoded_d_.get(),
      decoded_count_d_.get(), stream_);
  }
  if (status == cudaSuccess) {
    status = cudaMemcpyAsync(
      decoded_count_h_.get(), decoded_count_d_.get(), sizeof(int32_t), cudaMemcpyDeviceToHost,
      stream_);
  }
  if (status == cudaSuccess) {
    status = cudaMemcpyAsync(
      decoded_h_.get(), decoded_d_.get(), capacity * sizeof(DecodedBox), cudaMemcpyDeviceToHost,
      stream_);
  }
  if (status == cudaSuccess) {
    status = cudaStreamSynchronize(stream_);
  }
  if (status != cudaSuccess) {
    error = std::string("Decoding the detection head failed: ") + cudaGetErrorString(status);
    return false;
  }

  const auto count = std::min<size_t>(static_cast<size_t>(decoded_count_h_[0]), capacity);
  decoded_.assign(decoded_h_.get(), decoded_h_.get() + count);
  sort_like_bevfusion(decoded_);
  for (const auto & box : decoded_) {
    last_detections_.push_back(to_box3d(box));
  }
  return true;
}

}  // namespace autoware::oneplanner
