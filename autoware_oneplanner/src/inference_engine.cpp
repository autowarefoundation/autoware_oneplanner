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

#include "autoware/oneplanner/inference_engine.hpp"

#include <rclcpp/logging.hpp>

#include <NvInfer.h>

#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace autoware::oneplanner
{

namespace
{

//! What this node changes about a planner build, part of its engine identity: an engine built
//! under other settings is rebuilt once instead of reused with the arithmetic it was built with.
constexpr const char * kPlannerBuildSettings = "tf32-off+aux-streams-0+graph-precision";

size_t dtype_size(const TensorDataType dtype)
{
  switch (dtype) {
    case TensorDataType::kFLOAT32:
      return sizeof(float);
    case TensorDataType::kBOOL:
      return sizeof(uint8_t);
    case TensorDataType::kINT32:
      return sizeof(int32_t);
  }
  return sizeof(float);
}

TensorDataType to_tensor_dtype(const nvinfer1::DataType dtype, const std::string & name)
{
  switch (dtype) {
    case nvinfer1::DataType::kFLOAT:
      return TensorDataType::kFLOAT32;
    case nvinfer1::DataType::kBOOL:
      return TensorDataType::kBOOL;
    case nvinfer1::DataType::kINT32:
      return TensorDataType::kINT32;
    default:
      throw std::runtime_error(
        "Engine tensor '" + name +
        "' has an unsupported data type. Supported types: float32, bool, int32.");
  }
}

/**
 * @brief Let a graph that carries its own precision decide the builder's.
 *
 * An ONNX graph whose float tensors are partly float16 has had its precision chosen by the
 * exporter (for example a float16 transformer core with the metric coordinates at the graph's
 * two ends kept in float32). Under the plain FP16 flag the builder would be free to run those
 * float32 layers in half anyway, so this pins every layer to the dtype of its outputs and asks
 * the builder to obey. An all-float32 graph is left alone: the flag keeps meaning "any layer
 * may run in fp16".
 */
void obey_graph_precision(
  nvinfer1::INetworkDefinition & network, nvinfer1::IBuilderConfig & builder_config,
  const std::string & precision)
{
  const auto logger = rclcpp::get_logger("oneplanner");
  std::vector<std::pair<nvinfer1::ILayer *, nvinfer1::DataType>> typed_layers;
  int fp32_layers = 0;
  int fp16_layers = 0;
  for (int i = 0; i < network.getNbLayers(); ++i) {
    auto * layer = network.getLayer(i);
    // A plugin negotiates its own I/O formats. Pinning one under kOBEY_PRECISION_CONSTRAINTS
    // demands a combination it may not implement, and the builder then fails outright
    // ("No supported formats") rather than choosing a legal one. Leave plugins to negotiate;
    // the tensor types the exporter gave them still stand.
    if (
      layer->getType() == nvinfer1::LayerType::kSHAPE ||
      layer->getType() == nvinfer1::LayerType::kCONSTANT ||
      layer->getType() == nvinfer1::LayerType::kPLUGIN ||
      layer->getType() == nvinfer1::LayerType::kPLUGIN_V2 ||
      layer->getType() == nvinfer1::LayerType::kPLUGIN_V3 || layer->getNbOutputs() == 0) {
      continue;
    }
    const auto type = layer->getOutput(0)->getType();
    if (type != nvinfer1::DataType::kFLOAT && type != nvinfer1::DataType::kHALF) {
      continue;
    }
    bool uniform = true;
    for (int j = 1; j < layer->getNbOutputs(); ++j) {
      uniform = uniform && layer->getOutput(j)->getType() == type;
    }
    if (!uniform) {
      continue;
    }
    typed_layers.emplace_back(layer, type);
    (type == nvinfer1::DataType::kHALF ? fp16_layers : fp32_layers)++;
  }
  if (fp16_layers == 0) {
    return;  // an all-float32 graph: precision is the builder flag's business
  }
  if (precision != "fp16") {
    RCLCPP_WARN(
      logger,
      "The graph carries a float16 core (%d layers) but precision is \"%s\": the core is "
      "upcast and the engine runs slower than the exporter intended",
      fp16_layers, precision.c_str());
    return;
  }
  builder_config.setFlag(nvinfer1::BuilderFlag::kOBEY_PRECISION_CONSTRAINTS);
  for (auto & [layer, type] : typed_layers) {
    layer->setPrecision(type);
    for (int j = 0; j < layer->getNbOutputs(); ++j) {
      layer->setOutputType(j, type);
    }
  }
  RCLCPP_INFO(
    logger,
    "The graph carries its own precision: %d layers float32, %d layers float16; the builder "
    "obeys it",
    fp32_layers, fp16_layers);
}

}  // namespace

InferenceEngine::InferenceEngine(const Config & config)
{
  trt::EngineConfig engine_config;
  engine_config.onnx_path = config.model_path;
  engine_config.precision = config.precision;
  engine_config.workspace_bytes = config.max_workspace_size;
  engine_config.plugin_paths = {config.plugins_path};
  engine_config.build_settings = kPlannerBuildSettings;
  engine_config.configure = [precision = config.precision](
                              nvinfer1::INetworkDefinition & network,
                              nvinfer1::IBuilderConfig & builder_config) {
    // Single-stream execution: large transformer graphs otherwise allocate hundreds of MB of
    // auxiliary-stream scratch.
    builder_config.setMaxAuxStreams(0);
    // No TF32. TensorRT otherwise runs float32 convolutions and matrix products on TF32 tensor
    // cores (10-bit mantissa), and a float32 planner then disagrees with its own ONNX graph by
    // centimetres. The float16 layers of a float16-core graph are unaffected.
    builder_config.clearFlag(nvinfer1::BuilderFlag::kTF32);
    obey_graph_precision(network, builder_config, precision);
  };
  engine_ = std::make_unique<trt::TrtEngine>(engine_config);
  ONEPLANNER_CHECK_CUDA(cudaStreamCreate(&stream_));
  introspect_and_bind();
}

InferenceEngine::~InferenceEngine()
{
  if (stream_) {
    cudaStreamDestroy(stream_);
  }
}

namespace
{
//! cudaMalloc's alignment guarantee, and what a tensor address handed to TensorRT has to
//! satisfy; every slot in a block starts on it so any tensor can be read from its slot.
constexpr size_t kTensorAlignment = 256;

size_t align_up(const size_t bytes)
{
  return (bytes + kTensorAlignment - 1) / kTensorAlignment * kTensorAlignment;
}

bool is_aligned(const void * address)
{
  return reinterpret_cast<uintptr_t>(address) % kTensorAlignment == 0;
}
}  // namespace

void InferenceEngine::introspect_and_bind()
{
  const int32_t num_io = engine_->num_io_tensors();
  for (int32_t i = 0; i < num_io; ++i) {
    const char * name = engine_->io_tensor_name(i);
    const nvinfer1::Dims dims = engine_->tensor_shape(name);
    const auto nv_dtype = engine_->tensor_dtype(name);
    const bool is_input = engine_->is_input(name);

    TensorSpec spec;
    spec.name = name;
    spec.dtype = to_tensor_dtype(nv_dtype, spec.name);
    spec.shape.reserve(dims.nbDims);
    for (int32_t d = 0; d < dims.nbDims; ++d) {
      int64_t dim = dims.d[d];
      if (dim < 0) {
        if (d == 0) {
          dim = 1;  // Dynamic batch dimension: this node always runs with batch 1.
        } else {
          throw std::runtime_error(
            "Engine tensor '" + spec.name + "' has a dynamic non-batch dimension (dim " +
            std::to_string(d) + "). Export the model with static shapes.");
        }
      }
      spec.shape.push_back(dim);
    }
    if (is_input && dims.nbDims > 0 && dims.d[0] < 0) {
      nvinfer1::Dims resolved = dims;
      resolved.d[0] = 1;
      if (!engine_->set_input_shape(spec.name.c_str(), resolved)) {
        throw std::runtime_error(
          "Failed to set batch-1 shape for engine tensor '" + spec.name + "'");
      }
    }

    Binding binding;
    binding.spec = spec;
    binding.byte_size = static_cast<size_t>(spec.num_elements()) * dtype_size(spec.dtype);
    if (is_input) {
      binding.offset = input_block_bytes_;
      input_block_bytes_ += align_up(binding.byte_size);
      input_specs_.push_back(spec);
      input_bindings_.push_back(std::move(binding));
    } else {
      if (spec.dtype != TensorDataType::kFLOAT32) {
        throw std::runtime_error(
          "Engine output tensor '" + spec.name +
          "' is not float32; only float32 outputs are "
          "supported.");
      }
      binding.offset = output_block_bytes_;
      output_block_bytes_ += align_up(binding.byte_size);
      output_specs_.push_back(spec);
      output_bindings_.push_back(std::move(binding));
    }
  }
  if (output_specs_.empty()) {
    throw std::runtime_error("The engine has no output tensors.");
  }

  // One device block and one pinned block per direction. A tick's host tensors are staged
  // side by side in the pinned block and go up the bus in one copy per contiguous run,
  // instead of one pageable-memory copy each -- a dozen of those, each of which the
  // runtime has to stage and wait on, cost more than the bytes they move. The outputs come
  // back the same way, in one copy.
  device_inputs_ = cuda::make_device_unique<uint8_t[]>(input_block_bytes_);
  pinned_inputs_ = cuda::make_host_unique<uint8_t[]>(input_block_bytes_, cudaHostAllocDefault);
  device_outputs_ = cuda::make_device_unique<uint8_t[]>(output_block_bytes_);
  pinned_outputs_ = cuda::make_host_unique<uint8_t[]>(output_block_bytes_, cudaHostAllocDefault);
  for (auto & binding : input_bindings_) {
    binding.device = device_inputs_.get() + binding.offset;
    bind_address(binding, binding.device);
  }
  for (auto & binding : output_bindings_) {
    binding.device = device_outputs_.get() + binding.offset;
    binding.pinned = reinterpret_cast<const float *>(pinned_outputs_.get() + binding.offset);
    bind_address(binding, binding.device);
  }
}

void InferenceEngine::bind_address(Binding & binding, const void * address)
{
  if (binding.bound_address == address) {
    return;
  }
  if (!engine_->set_tensor_address(binding.spec.name.c_str(), const_cast<void *>(address))) {
    throw std::runtime_error(
      "Failed to bind device buffer for engine tensor '" + binding.spec.name + "'");
  }
  binding.bound_address = address;
}

std::string InferenceEngine::transfer_input(Binding & binding, const Tensor & tensor)
{
  const auto & spec = binding.spec;
  if (tensor.num_elements() != spec.num_elements()) {
    return "Input tensor '" + spec.name + "' has " + std::to_string(tensor.num_elements()) +
           " elements (shape " + shape_to_string(tensor.shape) + "), but the engine expects " +
           std::to_string(spec.num_elements()) + " (shape " + shape_to_string(spec.shape) + ")";
  }

  if (tensor.is_device()) {
    if (spec.dtype != TensorDataType::kFLOAT32) {
      return "Device-resident input '" + spec.name + "' requires a float32 engine tensor";
    }
    binding.host_fed = false;
    // The tensor was produced on this stream (InputProviderInterface::bind_stream), so the
    // network can read it where it is: no copy, and stream order guarantees it is complete.
    if (is_aligned(tensor.device_data)) {
      bind_address(binding, tensor.device_data);
      return "";
    }
    // A provider buffer that is not aligned for TensorRT goes through this tensor's slot.
    bind_address(binding, binding.device);
    ONEPLANNER_CHECK_CUDA(cudaMemcpyAsync(
      binding.device, tensor.device_data, binding.byte_size, cudaMemcpyDeviceToDevice, stream_));
    return "";
  }

  binding.host_fed = true;
  bind_address(binding, binding.device);
  uint8_t * staging = pinned_inputs_.get() + binding.offset;
  switch (spec.dtype) {
    case TensorDataType::kFLOAT32: {
      std::memcpy(staging, tensor.host_data.data(), binding.byte_size);
      break;
    }
    case TensorDataType::kBOOL: {
      for (size_t i = 0; i < tensor.host_data.size(); ++i) {
        staging[i] = tensor.host_data[i] > std::numeric_limits<float>::epsilon() ? 1 : 0;
      }
      break;
    }
    case TensorDataType::kINT32: {
      auto * staging_i32 = reinterpret_cast<int32_t *>(staging);
      for (size_t i = 0; i < tensor.host_data.size(); ++i) {
        staging_i32[i] = static_cast<int32_t>(tensor.host_data[i]);
      }
      break;
    }
  }
  return "";
}

InferenceEngine::Result InferenceEngine::infer(const TensorMap & inputs)
{
  Result result;
  for (auto & binding : input_bindings_) {
    const auto it = inputs.find(binding.spec.name);
    if (it == inputs.end()) {
      result.error_msg = "Missing input tensor '" + binding.spec.name + "'";
      return result;
    }
    const std::string error = transfer_input(binding, it->second);
    if (!error.empty()) {
      result.error_msg = error;
      return result;
    }
  }

  // Bindings are laid out in manifest order, so consecutive host-fed tensors are one
  // contiguous span of both blocks: copy each such run at once. Device-fed tensors sit
  // between runs and are skipped -- a BEV history is tens of megabytes that must never
  // ride along.
  const auto flush = [this](const size_t begin, const size_t end) {
    if (end > begin) {
      ONEPLANNER_CHECK_CUDA(cudaMemcpyAsync(
        device_inputs_.get() + begin, pinned_inputs_.get() + begin, end - begin,
        cudaMemcpyHostToDevice, stream_));
    }
  };
  size_t run_begin = 0;
  size_t run_end = 0;
  for (const auto & binding : input_bindings_) {
    if (binding.host_fed) {
      if (run_end == run_begin) {
        run_begin = binding.offset;
      }
      run_end = binding.offset + binding.byte_size;
    } else {
      flush(run_begin, run_end);
      run_begin = run_end = 0;
    }
  }
  flush(run_begin, run_end);

  if (!engine_->enqueue(stream_)) {
    result.error_msg = "Failed to enqueue inference";
    return result;
  }
  ONEPLANNER_CHECK_CUDA(cudaMemcpyAsync(
    pinned_outputs_.get(), device_outputs_.get(), output_block_bytes_, cudaMemcpyDeviceToHost,
    stream_));
  // The one host synchronization of the tick: everything a provider queued on this stream,
  // the network, and the output copy have completed when it returns.
  ONEPLANNER_CHECK_CUDA(cudaStreamSynchronize(stream_));

  TensorMap outputs;
  for (const auto & binding : output_bindings_) {
    const size_t count = static_cast<size_t>(binding.spec.num_elements());
    outputs.emplace(
      binding.spec.name,
      Tensor::from_host(
        binding.spec.shape, std::vector<float>(binding.pinned, binding.pinned + count)));
  }
  result.outputs = std::move(outputs);
  return result;
}

}  // namespace autoware::oneplanner
