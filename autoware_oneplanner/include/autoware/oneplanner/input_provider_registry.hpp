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

#ifndef AUTOWARE__ONEPLANNER__INPUT_PROVIDER_REGISTRY_HPP_
#define AUTOWARE__ONEPLANNER__INPUT_PROVIDER_REGISTRY_HPP_

#include "autoware/oneplanner/input_provider.hpp"

#include <rclcpp/rclcpp.hpp>

#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace autoware::oneplanner
{

/**
 * @brief Sensor input providers, looked up by the names given in `sensor_inputs`.
 *
 * A provider registers itself from its own translation unit, so adding one adds a source file
 * and never touches the node. The node only asks the registry for the names it was configured
 * with.
 */
using InputProviderFactory = std::function<std::unique_ptr<InputProviderInterface>(rclcpp::Node &)>;

void register_input_provider(const std::string & sensor, InputProviderFactory factory);

/// @throws std::runtime_error naming the registered providers when `sensor` is unknown.
std::unique_ptr<InputProviderInterface> make_input_provider(
  const std::string & sensor, rclcpp::Node & node);

std::vector<std::string> registered_input_providers();

/// Static registration helper; see ONEPLANNER_REGISTER_INPUT_PROVIDER.
struct InputProviderRegistrar
{
  InputProviderRegistrar(const std::string & sensor, InputProviderFactory factory)
  {
    register_input_provider(sensor, std::move(factory));
  }
};

}  // namespace autoware::oneplanner

#define ONEPLANNER_REGISTER_INPUT_PROVIDER(sensor, factory_expr)                               \
  static const ::autoware::oneplanner::InputProviderRegistrar oneplanner_registrar_##__LINE__( \
    (sensor), (factory_expr))

#endif  // AUTOWARE__ONEPLANNER__INPUT_PROVIDER_REGISTRY_HPP_
