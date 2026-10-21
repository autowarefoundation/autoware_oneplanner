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

#include "autoware/oneplanner/engine_identity.hpp"

#include <openssl/evp.h>

#include <array>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

namespace autoware::oneplanner
{

std::string file_sha256(const std::string & path)
{
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("Cannot read " + path);
  }
  const std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> context(
    EVP_MD_CTX_new(), EVP_MD_CTX_free);
  if (!context || EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr) != 1) {
    throw std::runtime_error("Cannot initialize SHA-256");
  }
  std::array<char, 1 << 16> buffer{};
  while (input.read(buffer.data(), buffer.size()) || input.gcount() > 0) {
    if (EVP_DigestUpdate(context.get(), buffer.data(), static_cast<size_t>(input.gcount())) != 1) {
      throw std::runtime_error("Cannot hash " + path);
    }
  }
  if (!input.eof()) {
    throw std::runtime_error("Reading " + path + " failed");
  }
  std::array<unsigned char, EVP_MAX_MD_SIZE> digest{};
  unsigned int length = 0;
  if (EVP_DigestFinal_ex(context.get(), digest.data(), &length) != 1) {
    throw std::runtime_error("Cannot finalize SHA-256");
  }
  std::ostringstream hex;
  for (unsigned int i = 0; i < length; ++i) {
    hex << std::hex << std::setfill('0') << std::setw(2) << static_cast<int>(digest[i]);
  }
  return hex.str();
}

bool engine_identity_matches(
  const std::string & onnx_path, const std::string & engine_path, const std::string & settings)
{
  std::ifstream input(engine_path + ".identity");
  std::string graph_digest;
  std::string engine_digest;
  std::string saved_settings;
  if (!(input >> graph_digest >> engine_digest >> saved_settings)) {
    return false;
  }
  return saved_settings == settings && graph_digest == file_sha256(onnx_path) &&
         engine_digest == file_sha256(engine_path);
}

void record_engine_identity(
  const std::string & onnx_path, const std::string & engine_path, const std::string & settings)
{
  const std::string temporary = engine_path + ".identity.tmp";
  {
    std::ofstream output(temporary);
    output << file_sha256(onnx_path) << "\n"
           << file_sha256(engine_path) << "\n"
           << settings << "\n";
    if (!output) {
      throw std::runtime_error("Cannot write " + temporary);
    }
  }
  std::filesystem::rename(temporary, engine_path + ".identity");
}

}  // namespace autoware::oneplanner
