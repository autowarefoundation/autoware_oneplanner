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

#ifndef AUTOWARE__ONEPLANNER__TRT__TRT_ENGINE_HPP_
#define AUTOWARE__ONEPLANNER__TRT__TRT_ENGINE_HPP_

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace autoware::oneplanner::trt
{

//! Optimization profile of one dynamic input tensor.
struct ProfileDims
{
  std::string tensor_name;
  nvinfer1::Dims min;
  nvinfer1::Dims opt;
  nvinfer1::Dims max;
};

struct EngineConfig
{
  std::string onnx_path;
  //! Where the serialized engine is cached; empty puts `<onnx stem>.engine` beside the ONNX.
  std::string engine_path;
  //! "fp32" or "fp16": whether the builder may run layers in half precision.
  std::string precision{"fp32"};
  //! Upper bound on the builder's scratch memory, not an allocation.
  std::size_t workspace_bytes{4ULL << 30U};
  //! TensorRT plugin libraries to load before the ONNX file is parsed.
  std::vector<std::string> plugin_paths;
  //! One profile over the dynamic inputs; empty for a graph with static shapes.
  std::vector<ProfileDims> profile;
  //! Names whatever `configure` changes about the build, for the engine identity: an engine
  //! cached under other settings is rebuilt instead of reused. No whitespace.
  std::string build_settings;
  //! Adjusts the parsed network and the builder configuration before a build. Not called
  //! when a matching cached engine is loaded.
  std::function<void(nvinfer1::INetworkDefinition &, nvinfer1::IBuilderConfig &)> configure;
};

/**
 * @class TrtEngine
 * @brief One TensorRT engine built from an ONNX file and cached beside it.
 *
 * A cached engine is reused only when its identity matches: the SHA-256 of the ONNX file and
 * of the engine file, the precision, the build settings, the TensorRT version and the GPU.
 * Anything else -- a re-exported graph, another TensorRT, another GPU, an engine copied in
 * from elsewhere -- rebuilds it, so a stale engine never runs another graph's weights.
 */
class TrtEngine
{
public:
  /**
   * @brief Load the cached engine, or build and cache it, and create its execution context.
   * @throws std::runtime_error naming the file and the reason on any failure.
   */
  explicit TrtEngine(const EngineConfig & config);
  ~TrtEngine();

  TrtEngine(const TrtEngine &) = delete;
  TrtEngine & operator=(const TrtEngine &) = delete;

  int32_t num_io_tensors() const;
  const char * io_tensor_name(int32_t index) const;
  bool has_tensor(const std::string & name) const;
  bool is_input(const char * name) const;
  nvinfer1::Dims tensor_shape(const char * name) const;
  nvinfer1::DataType tensor_dtype(const char * name) const;

  bool set_input_shape(const char * name, const nvinfer1::Dims & dims);
  bool set_tensor_address(const char * name, void * address);
  bool enqueue(cudaStream_t stream);

  const std::string & engine_path() const { return engine_path_; }

private:
  class Logger;

  void load_plugins(const std::vector<std::string> & plugin_paths);
  //! The settings an engine is cached under (see engine_identity.hpp).
  std::string build_identity(const EngineConfig & config) const;
  bool load_cached(const std::string & settings);
  void build(const EngineConfig & config, const std::string & settings);

  std::string onnx_path_;
  std::string engine_path_;
  std::unique_ptr<Logger> logger_;
  std::vector<void *> plugin_handles_;
  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;
};

}  // namespace autoware::oneplanner::trt

#endif  // AUTOWARE__ONEPLANNER__TRT__TRT_ENGINE_HPP_
