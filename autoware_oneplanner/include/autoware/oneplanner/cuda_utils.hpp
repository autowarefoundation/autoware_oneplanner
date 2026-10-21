// Copyright 2022-2026 TIER IV, Inc.
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

// Adapted from autoware_cuda_utils (autoware_universe). The smart pointers follow
// https://proc-cpuinfo.fixstars.com/2019/02/cuda_smart_pointer/ (CC0 1.0 Universal).

#ifndef AUTOWARE__ONEPLANNER__CUDA_UTILS_HPP_
#define AUTOWARE__ONEPLANNER__CUDA_UTILS_HPP_

#include <cuda_runtime_api.h>

#include <cstddef>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <type_traits>

namespace autoware::oneplanner::cuda
{

template <typename F, typename N>
void check_error(const ::cudaError_t e, F && file, N && line)
{
  if (e != ::cudaSuccess) {
    std::stringstream s;
    s << ::cudaGetErrorName(e) << " (" << e << ")@" << file << "#L" << line << ": "
      << ::cudaGetErrorString(e);
    throw std::runtime_error{s.str()};
  }
}

//! Frees device memory. Never throws: a deleter runs during unwinding, where an exception
//! would terminate the process, and a failed free at teardown has no one left to report to.
struct DeviceDeleter
{
  void operator()(void * p) const { static_cast<void>(::cudaFree(p)); }
};
template <typename T>
using DeviceUniquePtr = std::unique_ptr<T, DeviceDeleter>;

//! Frees pinned host memory; never throws, for the same reason as DeviceDeleter.
struct HostDeleter
{
  void operator()(void * p) const { static_cast<void>(::cudaFreeHost(p)); }
};
template <typename T>
using HostUniquePtr = std::unique_ptr<T, HostDeleter>;

}  // namespace autoware::oneplanner::cuda

#define ONEPLANNER_CHECK_CUDA(e) (::autoware::oneplanner::cuda::check_error(e, __FILE__, __LINE__))

namespace autoware::oneplanner::cuda
{

//! Device array of `n` elements.
template <typename T>
std::enable_if_t<std::is_array_v<T>, DeviceUniquePtr<T>> make_device_unique(const std::size_t n)
{
  using U = std::remove_extent_t<T>;
  U * p = nullptr;
  ONEPLANNER_CHECK_CUDA(::cudaMalloc(reinterpret_cast<void **>(&p), sizeof(U) * n));
  return DeviceUniquePtr<T>{p};
}

//! Pinned host array of `n` elements.
template <typename T>
std::enable_if_t<std::is_array_v<T>, HostUniquePtr<T>> make_host_unique(
  const std::size_t n, const unsigned int flags = cudaHostAllocDefault)
{
  using U = std::remove_extent_t<T>;
  U * p = nullptr;
  ONEPLANNER_CHECK_CUDA(::cudaHostAlloc(reinterpret_cast<void **>(&p), sizeof(U) * n, flags));
  return HostUniquePtr<T>{p};
}

}  // namespace autoware::oneplanner::cuda

#endif  // AUTOWARE__ONEPLANNER__CUDA_UTILS_HPP_
