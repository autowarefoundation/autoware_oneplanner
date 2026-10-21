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

#ifndef AUTOWARE__ONEPLANNER__INFERENCE_ENGINE_HPP_
#define AUTOWARE__ONEPLANNER__INFERENCE_ENGINE_HPP_

#include "autoware/oneplanner/cuda_utils.hpp"
#include "autoware/oneplanner/trt/trt_engine.hpp"
#include "autoware/oneplanner/types.hpp"

#include <cuda_runtime_api.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace autoware::oneplanner
{

/**
 * @class InferenceEngine
 * @brief Generic TensorRT execution over named tensors.
 *
 * Unlike model-specific wrappers, no tensor identity is known at compile time: the engine's IO
 * manifest is introspected at load time and exposed through `input_specs()` / `output_specs()`.
 * A dynamic batch dimension (-1 as dim 0) is resolved to 1; any other dynamic dimension is
 * rejected. Device buffers and pinned host output buffers are allocated and bound once.
 */
class InferenceEngine
{
public:
  struct Config
  {
    std::string model_path;    //!< ONNX model path; the built engine is cached alongside it.
    std::string plugins_path;  //!< Optional TensorRT plugin library path ("" to disable).
    //! "fp32" or "fp16": the builder flag. A graph that carries float16 tensors of its own
    //! decides layer by layer; see obey_graph_precision().
    std::string precision{"fp32"};
    //! TensorRT builder workspace: an upper bound on the builder's scratch, not an
    //! allocation. A property of the deployment host (`trt_workspace_mib`), not of the model.
    size_t max_workspace_size{4ULL << 30U};
  };

  struct Result
  {
    std::optional<TensorMap> outputs;
    std::string error_msg;
  };

  /**
   * @brief Build/load the engine and introspect its IO manifest.
   * @throws std::runtime_error on engine setup failure or unsupported IO tensors.
   */
  explicit InferenceEngine(const Config & config);
  ~InferenceEngine();

  InferenceEngine(const InferenceEngine &) = delete;
  InferenceEngine & operator=(const InferenceEngine &) = delete;

  const std::vector<TensorSpec> & input_specs() const { return input_specs_; }
  const std::vector<TensorSpec> & output_specs() const { return output_specs_; }
  //! The stream every tick runs on. Providers producing device-resident tensors submit
  //! their work here, so the network reads them in stream order without a host sync.
  cudaStream_t stream() const { return stream_; }

  /**
   * @brief Run inference. Every input spec must be present in `inputs` with a matching element
   * count; extra entries in `inputs` are ignored. Outputs are returned as host tensors.
   *
   * Host tensors are staged into one pinned block and cross the bus in as few copies as
   * their layout allows; a device-resident tensor is read by the network where it already
   * is, without a copy, when its address is suitably aligned. The only host synchronization
   * of the call is the one that waits for the outputs.
   */
  Result infer(const TensorMap & inputs);

private:
  struct Binding
  {
    TensorSpec spec;
    size_t byte_size{0};
    //! Byte offset of this tensor inside its direction's block (256-byte aligned).
    size_t offset{0};
    //! This tensor's slot in the device block: where a host tensor lands, and where a
    //! device tensor that cannot be read in place is copied to.
    uint8_t * device{nullptr};
    //! Outputs only: this tensor's slot in the pinned output block.
    const float * pinned{nullptr};
    //! The address the network currently reads this tensor from.
    const void * bound_address{nullptr};
    //! Inputs only: whether the last transfer filled the pinned staging slot (and so the
    //! slot must be part of the H2D copy) rather than pointing the network at a device tensor.
    bool host_fed{false};
  };
  void introspect_and_bind();
  /// Point the network at `address` for this binding, if it is not already there.
  void bind_address(Binding & binding, const void * address);
  /// Stage one input tensor: host data into the pinned block (with dtype conversion), or a
  /// device tensor bound in place / copied D2D. Returns an error message, "" on success.
  std::string transfer_input(Binding & binding, const Tensor & tensor);
  std::unique_ptr<trt::TrtEngine> engine_;
  std::vector<TensorSpec> input_specs_;
  std::vector<TensorSpec> output_specs_;
  std::vector<Binding> input_bindings_;
  std::vector<Binding> output_bindings_;
  // One block per direction, so a tick's host inputs go up in one copy per contiguous run
  // and its outputs come back in one.
  size_t input_block_bytes_{0};
  size_t output_block_bytes_{0};
  cuda::DeviceUniquePtr<uint8_t[]> device_inputs_;
  cuda::HostUniquePtr<uint8_t[]> pinned_inputs_;
  cuda::DeviceUniquePtr<uint8_t[]> device_outputs_;
  cuda::HostUniquePtr<uint8_t[]> pinned_outputs_;
  cudaStream_t stream_{nullptr};
};

}  // namespace autoware::oneplanner

#endif  // AUTOWARE__ONEPLANNER__INFERENCE_ENGINE_HPP_
