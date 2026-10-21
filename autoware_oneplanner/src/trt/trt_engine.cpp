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

// The build configuration follows autoware_tensorrt_common's TrtCommon (autoware_universe),
// of which this is the subset the node needs.

#include "autoware/oneplanner/trt/trt_engine.hpp"

#include "autoware/oneplanner/cuda_utils.hpp"
#include "autoware/oneplanner/engine_identity.hpp"

#include <rclcpp/logging.hpp>

#include <NvInferPlugin.h>
#include <NvOnnxParser.h>
#include <dlfcn.h>

#include <cctype>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace autoware::oneplanner::trt
{

namespace
{

std::string dims_to_string(const nvinfer1::Dims & dims)
{
  std::ostringstream oss;
  for (int32_t i = 0; i < dims.nbDims; ++i) {
    oss << (i ? "x" : "") << dims.d[i];
  }
  return oss.str();
}

rclcpp::Logger node_logger()
{
  return rclcpp::get_logger("oneplanner");
}

}  // namespace

//! Forwards TensorRT's messages to the ROS log. Its INFO stream is verbose (a build prints
//! hundreds of lines), so it goes to DEBUG.
class TrtEngine::Logger : public nvinfer1::ILogger
{
public:
  void log(Severity severity, const char * message) noexcept override
  {
    const auto logger = rclcpp::get_logger("oneplanner.tensorrt");
    switch (severity) {
      case Severity::kINTERNAL_ERROR:
      case Severity::kERROR:
        RCLCPP_ERROR(logger, "%s", message);
        break;
      case Severity::kWARNING:
        RCLCPP_WARN(logger, "%s", message);
        break;
      default:
        RCLCPP_DEBUG(logger, "%s", message);
        break;
    }
  }
};

TrtEngine::TrtEngine(const EngineConfig & config)
: onnx_path_(config.onnx_path), logger_(std::make_unique<Logger>())
{
  if (config.precision != "fp32" && config.precision != "fp16") {
    throw std::runtime_error(
      "Unsupported precision '" + config.precision + "' (expected \"fp32\" or \"fp16\")");
  }
  if (!std::filesystem::is_regular_file(onnx_path_)) {
    throw std::runtime_error("No such ONNX file: " + onnx_path_);
  }
  engine_path_ = config.engine_path.empty()
                   ? std::filesystem::path(onnx_path_).replace_extension(".engine").string()
                   : config.engine_path;

  // Process-wide, as TrtCommon sets it: a host thread waiting for the device sleeps instead
  // of spinning a core.
  ONEPLANNER_CHECK_CUDA(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));
  load_plugins(config.plugin_paths);
  initLibNvInferPlugins(logger_.get(), "");
  runtime_.reset(nvinfer1::createInferRuntime(*logger_));
  if (!runtime_) {
    throw std::runtime_error("Failed to create the TensorRT runtime");
  }

  const std::string settings = build_identity(config);
  if (load_cached(settings)) {
    RCLCPP_INFO(node_logger(), "Loaded the cached engine %s", engine_path_.c_str());
  } else {
    build(config, settings);
  }
  context_.reset(engine_->createExecutionContext());
  if (!context_) {
    throw std::runtime_error("Failed to create an execution context for " + engine_path_);
  }
}

TrtEngine::~TrtEngine()
{
  // The context and the engine may call into plugin code, so they go before the libraries.
  context_.reset();
  engine_.reset();
  runtime_.reset();
  for (void * handle : plugin_handles_) {
    dlclose(handle);
  }
}

void TrtEngine::load_plugins(const std::vector<std::string> & plugin_paths)
{
  for (const auto & path : plugin_paths) {
    if (path.empty()) {
      continue;
    }
    void * handle = dlopen(path.c_str(), RTLD_LAZY);
    if (!handle) {
      // Failing here names the library; failing later names only the graph node whose
      // plugin is missing.
      throw std::runtime_error(
        "Could not load the TensorRT plugin library " + path + ": " + dlerror());
    }
    plugin_handles_.push_back(handle);
  }
}

std::string TrtEngine::build_identity(const EngineConfig & config) const
{
  int device = 0;
  ONEPLANNER_CHECK_CUDA(cudaGetDevice(&device));
  cudaDeviceProp properties{};
  ONEPLANNER_CHECK_CUDA(cudaGetDeviceProperties(&properties, device));

  std::ostringstream oss;
  oss << config.precision << "+trt" << NV_TENSORRT_MAJOR << "." << NV_TENSORRT_MINOR << "."
      << NV_TENSORRT_PATCH << "+sm" << properties.major << properties.minor;
  if (!config.build_settings.empty()) {
    oss << "+" << config.build_settings;
  }
  for (const auto & dims : config.profile) {
    oss << "+" << dims.tensor_name << ":" << dims_to_string(dims.min) << ":"
        << dims_to_string(dims.opt) << ":" << dims_to_string(dims.max);
  }
  std::string settings = oss.str();
  for (char & c : settings) {
    if (std::isspace(static_cast<unsigned char>(c))) {
      c = '_';
    }
  }
  return settings;
}

bool TrtEngine::load_cached(const std::string & settings)
{
  if (!std::filesystem::exists(engine_path_)) {
    return false;
  }
  if (!engine_identity_matches(onnx_path_, engine_path_, settings)) {
    RCLCPP_INFO(
      node_logger(), "%s does not match %s under the current settings; rebuilding",
      engine_path_.c_str(), onnx_path_.c_str());
    return false;
  }
  std::ifstream file(engine_path_, std::ios::binary);
  const std::vector<char> plan(
    (std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  engine_.reset(runtime_->deserializeCudaEngine(plan.data(), plan.size()));
  if (!engine_) {
    RCLCPP_WARN(node_logger(), "Could not deserialize %s; rebuilding", engine_path_.c_str());
    return false;
  }
  return true;
}

void TrtEngine::build(const EngineConfig & config, const std::string & settings)
{
  std::unique_ptr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(*logger_));
  if (!builder) {
    throw std::runtime_error("Failed to create the TensorRT builder");
  }
#if NV_TENSORRT_MAJOR < 10
  const auto flags =
    1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
#else
  const uint32_t flags = 0U;
#endif
  std::unique_ptr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(flags));
  std::unique_ptr<nvinfer1::IBuilderConfig> builder_config(builder->createBuilderConfig());
  if (!network || !builder_config) {
    throw std::runtime_error("Failed to create the TensorRT network or builder configuration");
  }
  if (config.precision == "fp16") {
    builder_config->setFlag(nvinfer1::BuilderFlag::kFP16);
  }
  builder_config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, config.workspace_bytes);

  std::unique_ptr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*network, *logger_));
  if (!parser->parseFromFile(
        onnx_path_.c_str(), static_cast<int32_t>(nvinfer1::ILogger::Severity::kERROR))) {
    std::string reason;
    for (int32_t i = 0; i < parser->getNbErrors(); ++i) {
      reason += std::string(i ? "; " : "") + parser->getError(i)->desc();
    }
    throw std::runtime_error("Failed to parse " + onnx_path_ + ": " + reason);
  }

  if (!config.profile.empty()) {
    auto * profile = builder->createOptimizationProfile();
    for (const auto & dims : config.profile) {
      profile->setDimensions(
        dims.tensor_name.c_str(), nvinfer1::OptProfileSelector::kMIN, dims.min);
      profile->setDimensions(
        dims.tensor_name.c_str(), nvinfer1::OptProfileSelector::kOPT, dims.opt);
      profile->setDimensions(
        dims.tensor_name.c_str(), nvinfer1::OptProfileSelector::kMAX, dims.max);
    }
    builder_config->addOptimizationProfile(profile);
  }
  if (config.configure) {
    config.configure(*network, *builder_config);
  }

  RCLCPP_INFO(
    node_logger(), "Building the TensorRT engine for %s (%s); this can take several minutes",
    onnx_path_.c_str(), config.precision.c_str());
  const auto started = std::chrono::steady_clock::now();
  std::unique_ptr<nvinfer1::IHostMemory> plan(
    builder->buildSerializedNetwork(*network, *builder_config));
  if (!plan) {
    throw std::runtime_error(
      "TensorRT could not build an engine from " + onnx_path_ +
      " (see the oneplanner.tensorrt log above; a larger trt_workspace_mib may help)");
  }
  engine_.reset(runtime_->deserializeCudaEngine(plan->data(), plan->size()));
  if (!engine_) {
    throw std::runtime_error("Failed to deserialize the engine just built from " + onnx_path_);
  }
  RCLCPP_INFO(
    node_logger(), "Built %s in %.0f s", engine_path_.c_str(),
    std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count());

  // Written through a rename, so an interrupted write never leaves a truncated engine that a
  // later start would try to load.
  const std::string temporary = engine_path_ + ".tmp";
  {
    std::ofstream file(temporary, std::ios::binary);
    file.write(static_cast<const char *>(plan->data()), static_cast<std::streamsize>(plan->size()));
    if (!file) {
      RCLCPP_WARN(
        node_logger(), "Could not write %s; the engine is used but not cached", temporary.c_str());
      return;
    }
  }
  std::filesystem::rename(temporary, engine_path_);
  record_engine_identity(onnx_path_, engine_path_, settings);
}

int32_t TrtEngine::num_io_tensors() const
{
  return engine_->getNbIOTensors();
}

const char * TrtEngine::io_tensor_name(const int32_t index) const
{
  return engine_->getIOTensorName(index);
}

bool TrtEngine::has_tensor(const std::string & name) const
{
  for (int32_t i = 0; i < engine_->getNbIOTensors(); ++i) {
    if (name == engine_->getIOTensorName(i)) {
      return true;
    }
  }
  return false;
}

bool TrtEngine::is_input(const char * name) const
{
  return engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT;
}

nvinfer1::Dims TrtEngine::tensor_shape(const char * name) const
{
  return engine_->getTensorShape(name);
}

nvinfer1::DataType TrtEngine::tensor_dtype(const char * name) const
{
  return engine_->getTensorDataType(name);
}

bool TrtEngine::set_input_shape(const char * name, const nvinfer1::Dims & dims)
{
  return context_->setInputShape(name, dims);
}

bool TrtEngine::set_tensor_address(const char * name, void * address)
{
  return context_->setTensorAddress(name, address);
}

bool TrtEngine::enqueue(cudaStream_t stream)
{
  return context_->enqueueV3(stream);
}

}  // namespace autoware::oneplanner::trt
