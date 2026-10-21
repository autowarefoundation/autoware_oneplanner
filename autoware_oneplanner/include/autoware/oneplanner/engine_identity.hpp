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

#ifndef AUTOWARE__ONEPLANNER__ENGINE_IDENTITY_HPP_
#define AUTOWARE__ONEPLANNER__ENGINE_IDENTITY_HPP_

#include <string>

namespace autoware::oneplanner
{

/**
 * A cached TensorRT engine is tied to the bytes of the graph it was built from and to its
 * own bytes, not to file times: copying an old engine next to a re-exported graph must never
 * run the old weights. `<engine>.identity` records both digests and the build settings.
 */

//! Lower-case hex SHA-256 of a file. @throws std::runtime_error when it cannot be read.
std::string file_sha256(const std::string & path);

//! Whether `<engine>.identity` exists and names this graph, this engine and these settings.
//! `settings` must not contain whitespace.
bool engine_identity_matches(
  const std::string & onnx_path, const std::string & engine_path, const std::string & settings);

//! Write `<engine>.identity` for a freshly built engine (atomically, through a rename).
//! @throws std::runtime_error when it cannot be written.
void record_engine_identity(
  const std::string & onnx_path, const std::string & engine_path, const std::string & settings);

}  // namespace autoware::oneplanner

#endif  // AUTOWARE__ONEPLANNER__ENGINE_IDENTITY_HPP_
