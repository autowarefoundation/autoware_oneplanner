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

#include "autoware/oneplanner/candidate_turn_indicators.hpp"
#include "autoware/oneplanner/curvature_bias.hpp"
#include "autoware/oneplanner/engine_identity.hpp"
#include "autoware/oneplanner/planning_time.hpp"
#include "autoware/oneplanner/pose_discontinuity.hpp"
#include "autoware/oneplanner/postprocess/planning_factor.hpp"
#include "autoware/oneplanner/rolling_latency.hpp"
#include "autoware/oneplanner/signal_history.hpp"
#include "autoware/oneplanner/timestamped_buffer.hpp"
#include "autoware/oneplanner/training_ego_shape.hpp"
#include "autoware/oneplanner/turn_indicator_filter.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace autoware::oneplanner
{

TEST(TrainingEgoShape, PicksTheShapeWithTheClosestWheelBase)
{
  const auto shapes = parse_ego_shapes({3.0, 4.5, 1.8, 5.0, 8.0, 2.5});
  ASSERT_EQ(shapes.size(), 2u);
  EXPECT_DOUBLE_EQ(closest_ego_shape(shapes, 2.9).length, 4.5);
  EXPECT_DOUBLE_EQ(closest_ego_shape(shapes, 5.2).width, 2.5);
}

TEST(TrainingEgoShape, RefusesAVehicleTheModelNeverTrainedOn)
{
  const auto shapes = parse_ego_shapes({3.0, 4.5, 1.8, 5.0, 8.0, 2.5});
  for (const double unknown : {4.0, 1.0, std::numeric_limits<double>::quiet_NaN()}) {
    EXPECT_THROW(closest_ego_shape(shapes, unknown), std::runtime_error) << unknown;
  }
  EXPECT_THROW(closest_ego_shape({}, 3.0), std::runtime_error);
}

TEST(TrainingEgoShape, RefusesMalformedLists)
{
  EXPECT_THROW(parse_ego_shapes({3.0, 4.5}), std::runtime_error);
  EXPECT_THROW(parse_ego_shapes({3.0, -4.5, 1.8}), std::runtime_error);
}

TEST(EngineIdentity, MatchesOnlyTheGraphTheEngineAndTheSettingsItWasRecordedFor)
{
  const auto root =
    std::filesystem::temp_directory_path() /
    ("oneplanner_identity_" + std::to_string(::testing::UnitTest::GetInstance()->random_seed()));
  std::filesystem::create_directories(root);
  const auto graph = (root / "graph.onnx").string();
  const auto engine = (root / "graph.engine").string();
  std::ofstream(graph) << "original graph";
  std::ofstream(engine) << "engine one";

  EXPECT_FALSE(engine_identity_matches(graph, engine, "fp32"));
  record_engine_identity(graph, engine, "fp32");
  EXPECT_TRUE(engine_identity_matches(graph, engine, "fp32"));
  EXPECT_FALSE(engine_identity_matches(graph, engine, "fp16"));

  std::ofstream(engine) << "engine two";  // an engine copied in from elsewhere
  EXPECT_FALSE(engine_identity_matches(graph, engine, "fp32"));

  record_engine_identity(graph, engine, "fp32");
  const auto digest = file_sha256(graph);
  std::ofstream(graph) << "re-exported graph";
  EXPECT_NE(file_sha256(graph), digest);
  EXPECT_FALSE(engine_identity_matches(graph, engine, "fp32"));

  std::filesystem::remove_all(root);
}

TEST(PoseDiscontinuity, AllowsMotionAndFlagsJumps)
{
  const PoseContinuityLimits limits;
  EXPECT_FALSE(pose_discontinuous({100000, 0, 1, 0}, {100003, 0, 1, 0}, 0.1, limits));
  EXPECT_TRUE(pose_discontinuous({100000, 0, 1, 0}, {100020, 0, 1, 0}, 0.1, limits));
  EXPECT_TRUE(pose_discontinuous({0, 0, 1, 0}, {0, 0, 0, 1}, 0.1, limits));
  // Across the +-pi seam the yaw change is small.
  EXPECT_FALSE(pose_discontinuous({0, 0, -1, 0.001}, {0, 0, -1, -0.001}, 0.1, limits));
}

TEST(RollingLatency, ReportsPercentilesOverABoundedWindow)
{
  RollingLatency latency;
  for (int i = 1; i <= 100; ++i) {
    latency.add(i);
  }
  EXPECT_EQ(latency.percentile(0.95), 95);
  EXPECT_EQ(latency.percentile(0.99), 99);
  latency.add(-1);
  latency.add(std::numeric_limits<double>::infinity());
  EXPECT_EQ(latency.percentile(0.99), 99);
  for (int i = 0; i < 512; ++i) {
    latency.add(7);
  }
  EXPECT_EQ(latency.percentile(0.99), 7);
  latency.clear();
  EXPECT_EQ(latency.percentile(0.95), 0);
}

TEST(PlanningTime, ParsesAndChecksTheConvention)
{
  EXPECT_EQ(parse_planning_time("cloud_stamp"), PlanningTime::kCloudStamp);
  EXPECT_EQ(parse_planning_time("planning_time"), PlanningTime::kPlanningTime);
  EXPECT_THROW(parse_planning_time("now"), std::runtime_error);
  EXPECT_THROW(parse_planning_time(""), std::runtime_error);
  EXPECT_STREQ(planning_time_name(PlanningTime::kPlanningTime), "planning_time");

  EXPECT_NO_THROW(check_planning_time_inputs(PlanningTime::kCloudStamp, false));
  EXPECT_NO_THROW(check_planning_time_inputs(PlanningTime::kPlanningTime, true));
  EXPECT_THROW(check_planning_time_inputs(PlanningTime::kCloudStamp, true), std::runtime_error);
  EXPECT_THROW(check_planning_time_inputs(PlanningTime::kPlanningTime, false), std::runtime_error);
}

TEST(PlanningTime, PlansFromTheNewestOdometryNeverBeforeTheCloud)
{
  const int64_t cloud = 1'000'000'000;
  const int64_t odometry = cloud + 146'000'000;
  EXPECT_EQ(planning_stamp_ns(PlanningTime::kCloudStamp, cloud, odometry), cloud);
  EXPECT_EQ(planning_stamp_ns(PlanningTime::kPlanningTime, cloud, odometry), odometry);
  EXPECT_EQ(planning_stamp_ns(PlanningTime::kPlanningTime, cloud, cloud), cloud);
  EXPECT_FALSE(planning_stamp_ns(PlanningTime::kPlanningTime, cloud, cloud - 1));
}

// Tick for tick against the recursion documented in curvature_bias.hpp, evaluated in double
// precision (wheel base 3.0 m): held at rest and below 1 m/s, started on the first moving
// tick, signed through a reverse.
TEST(CurvatureBias, FollowsTheDocumentedRecursion)
{
  const double v[] = {0.0, 0.0, 0.5, 3.0, 6.0, 8.0, 8.0, 0.4, 0.0, 7.0, 9.0, -2.0, 10.0};
  const double r[] = {0.0, 0.01, 0.02, 0.05, 0.06, 0.08, -0.04, 0.0, 0.0, 0.07, 0.09, 0.01, 0.1};
  const double s[] = {0.01, 0.01, 0.02, 0.03, 0.02, 0.03, -0.01, 0.0, 0.0, 0.025, 0.03, 0.0, 0.028};
  const double b[] = {
    0.0,
    0.0,
    0.0,
    6.663665586273e-03,
    6.501200007133e-03,
    6.183986376932e-03,
    5.801110929120e-03,
    5.801110929120e-03,
    5.801110929120e-03,
    5.599387010759e-03,
    5.326155319383e-03,
    4.822542781762e-03,
    4.619739317756e-03};
  CurvatureBiasFilter filter(2.0, 1.0);
  for (int i = 0; i < 13; ++i) {
    filter.update(v[i], r[i], s[i], 3.0, 0.1);
    EXPECT_NEAR(filter.value(), b[i], 1e-7 * std::abs(b[i]) + 1e-12) << "tick " << i;
  }
}

TEST(CurvatureBias, ChecksThePackageAgainstTheGraph)
{
  EXPECT_THROW(CurvatureBiasFilter(0.0, 1.0), std::runtime_error);
  EXPECT_NO_THROW(check_curvature_bias_inputs(false, false));
  EXPECT_NO_THROW(check_curvature_bias_inputs(true, true));
  EXPECT_THROW(check_curvature_bias_inputs(false, true), std::runtime_error);
  EXPECT_THROW(check_curvature_bias_inputs(true, false), std::runtime_error);
}

namespace
{
constexpr int64_t kPoints = 2;
constexpr int64_t kChannels = 13;
constexpr int64_t kLightOffset = 8;
constexpr int64_t kRows = 3;

//! Lane rows whose light one-hot is `lights[row]`.
std::vector<float> rows_of(const std::vector<int> & lights)
{
  std::vector<float> data(kRows * kPoints * kChannels, 0.0f);
  for (size_t r = 0; r < lights.size(); ++r) {
    for (int64_t p = 0; p < kPoints; ++p) {
      data[(r * kPoints + p) * kChannels + kLightOffset + lights[r]] = 1.0f;
    }
  }
  return data;
}

//! The hot state of one row at one step of a 3-step timeline; -1 when all zeros.
int light(const std::vector<float> & timeline, const int64_t row, const int step)
{
  int hot = -1;
  for (int s = 0; s < SIGNAL_STATES; ++s) {
    if (timeline[(row * 3 + step) * SIGNAL_STATES + s] == 1.0f) {
      hot = s;
    }
  }
  return hot;
}
}  // namespace

TEST(SignalHistory, ARowIsFollowedByItsLaneletNotByItsSlot)
{
  const int green = 0;
  const int red = 2;
  const int none = 4;
  const int64_t a = 11;
  const int64_t b = 22;
  const int64_t e = 33;
  const int64_t tick = 500'000'000;
  SignalHistory history(3, 0.5);
  history.update(0 * tick, {a, b}, rows_of({green, none}), kRows, kPoints, kChannels, kLightOffset);
  history.update(1 * tick, {b, a}, rows_of({none, green}), kRows, kPoints, kChannels, kLightOffset);
  const auto now = history.update(
    2 * tick, {b, a, e}, rows_of({none, red, red}), kRows, kPoints, kChannels, kLightOffset);
  EXPECT_EQ(light(now, 1, 0), green);
  EXPECT_EQ(light(now, 1, 1), green);
  EXPECT_EQ(light(now, 1, 2), red);
  EXPECT_EQ(light(now, 0, 0), none);
  EXPECT_EQ(light(now, 0, 2), none);
  EXPECT_EQ(light(now, 2, 0), -1);
  EXPECT_EQ(light(now, 2, 1), -1);
  EXPECT_EQ(light(now, 2, 2), red);

  // Time going backwards (a replay restarted) forgets everything before.
  const auto reset =
    history.update(0, {a}, rows_of({red}), kRows, kPoints, kChannels, kLightOffset);
  EXPECT_EQ(light(reset, 0, 0), -1);
  EXPECT_EQ(light(reset, 0, 2), red);
}

TEST(SignalHistory, ARecordFurtherThanHalfAStepIsNotObserved)
{
  const int green = 0;
  const int red = 2;
  SignalHistory sparse(2, 0.5);
  sparse.update(0, {11}, rows_of({green}), kRows, kPoints, kChannels, kLightOffset);
  const auto late =
    sparse.update(800'000'000, {11}, rows_of({red}), kRows, kPoints, kChannels, kLightOffset);
  EXPECT_EQ(late[0], 0.0f);
  EXPECT_EQ(late[SIGNAL_STATES + red], 1.0f);
}

TEST(SignalHistory, ChecksThePackageAgainstTheGraph)
{
  EXPECT_THROW(SignalHistory(0, 0.5), std::runtime_error);
  EXPECT_NO_THROW(check_signal_history_inputs(false, false, false));
  EXPECT_NO_THROW(check_signal_history_inputs(true, true, true));
  EXPECT_THROW(check_signal_history_inputs(false, true, true), std::runtime_error);
  EXPECT_THROW(check_signal_history_inputs(true, false, false), std::runtime_error);
  EXPECT_THROW(check_signal_history_inputs(true, true, false), std::runtime_error);
}

TEST(TimestampedBuffer, InterpolatesRetainsAndResets)
{
  TimestampedBuffer<double> buffer;
  for (int i = 0; i <= 100; ++i) {
    buffer.insert(i * 10000000LL, i * 0.1, 300000000LL);
  }
  EXPECT_EQ(buffer.samples.front().stamp_ns, 700000000LL);
  const auto bracket = buffer.bracket(955000000LL);
  ASSERT_TRUE(bracket);
  const auto & lo = buffer.samples[bracket->first];
  const auto & hi = buffer.samples[bracket->second];
  const double alpha =
    static_cast<double>(955000000LL - lo.stamp_ns) / static_cast<double>(hi.stamp_ns - lo.stamp_ns);
  EXPECT_NEAR(lo.value + alpha * (hi.value - lo.value), 9.55, 1e-9);
  EXPECT_FALSE(buffer.bracket(1001000000LL));
  EXPECT_FALSE(buffer.bracket(699000000LL));

  const auto size = buffer.samples.size();
  buffer.insert(1000000000LL, 42, 300000000LL);  // a duplicate stamp replaces the sample
  EXPECT_EQ(buffer.samples.size(), size);
  EXPECT_EQ(buffer.samples.back().value, 42);
  EXPECT_EQ(*buffer.at_or_before(1000000001LL, 100), 42);
  EXPECT_FALSE(buffer.at_or_before(1000000200LL, 100));
  EXPECT_FALSE(buffer.at_or_before(0, 100));

  EXPECT_TRUE(buffer.insert(0, 0, 300000000LL));  // time going backwards resets
  EXPECT_EQ(buffer.samples.size(), 1u);
  EXPECT_EQ(buffer.bracket(0)->first, 0u);
}

// A stream delivered in stamp order has everything up to a target once a sample at or after it is
// in: what the node waits for before it reads the acceleration at the planning stamp.
TEST(TimestampedBuffer, CoversATargetOnceASampleAtOrAfterItArrived)
{
  TimestampedBuffer<double> buffer;
  EXPECT_FALSE(buffer.covers(0));
  buffer.insert(980000000LL, 1.0, 300000000LL);
  EXPECT_FALSE(buffer.covers(1000000000LL));  // the sample at the target is still in flight
  EXPECT_EQ(*buffer.at_or_before(1000000000LL, 200000000LL), 1.0);
  buffer.insert(1000000000LL, 2.0, 300000000LL);
  EXPECT_TRUE(buffer.covers(1000000000LL));
  EXPECT_EQ(*buffer.at_or_before(1000000000LL, 200000000LL), 2.0);
  EXPECT_TRUE(buffer.covers(990000000LL));
  buffer.insert(0, 0.0, 300000000LL);  // time going backwards resets
  EXPECT_FALSE(buffer.covers(1000000000LL));
}

// Irregular 50 Hz odometry with missing samples still yields a 31-pose 0.1 s grid anchored at
// a LiDAR stamp off the odometry grid.
TEST(TimestampedBuffer, SupportsATenHertzGridFromIrregularOdometry)
{
  TimestampedBuffer<double> history;
  for (int i = 0; i <= 200; ++i) {
    if (i == 53 || i == 109) {
      continue;
    }
    const int64_t stamp = i * 20000000LL + (i % 3) * 1000000LL;
    history.insert(stamp, static_cast<double>(stamp) * 1e-9 * 10.0, 4000000000LL);
  }
  const int64_t lidar_stamp = 3955000000LL;
  for (int step = 0; step < 31; ++step) {
    const int64_t target = lidar_stamp - (30 - step) * 100000000LL;
    const auto bracket = history.bracket(target);
    ASSERT_TRUE(bracket);
    const auto & a = history.samples[bracket->first];
    const auto & b = history.samples[bracket->second];
    const double ratio = a.stamp_ns == b.stamp_ns ? 0.0
                                                  : static_cast<double>(target - a.stamp_ns) /
                                                      static_cast<double>(b.stamp_ns - a.stamp_ns);
    EXPECT_NEAR(
      a.value + ratio * (b.value - a.value), static_cast<double>(target) * 1e-9 * 10.0, 1e-9);
  }
}

namespace
{
struct CandidateWithIndicator
{
  int turn_indicators_command{0};
};
struct CandidateWithoutIndicator
{
  int points{0};
};
}  // namespace

TEST(CandidateTurnIndicators, EachCandidateGetsTheCommandOfItsBatch)
{
  // Two batches: the prediction candidates of batch 0 and 1, then an extra output's.
  std::vector<CandidateWithIndicator> candidates(4);
  set_candidate_turn_indicators(candidates, std::vector<int>{2, 3});
  EXPECT_EQ(candidates[0].turn_indicators_command, 2);
  EXPECT_EQ(candidates[1].turn_indicators_command, 3);
  EXPECT_EQ(candidates[2].turn_indicators_command, 2);
  EXPECT_EQ(candidates[3].turn_indicators_command, 3);

  // No commands: the candidates keep their value.
  set_candidate_turn_indicators(candidates, std::vector<int>{});
  EXPECT_EQ(candidates[0].turn_indicators_command, 2);
}

TEST(CandidateTurnIndicators, MessageWithoutTheFieldIsLeftAlone)
{
  // CandidateTrajectory of autoware_internal_msgs before 1.14.0 has no turn_indicators_command.
  static_assert(HasTurnIndicatorsCommand<CandidateWithIndicator>::value);
  static_assert(!HasTurnIndicatorsCommand<CandidateWithoutIndicator>::value);
  std::vector<CandidateWithoutIndicator> candidates(2);
  set_candidate_turn_indicators(candidates, std::vector<int>{2});
  EXPECT_EQ(candidates[0].points, 0);
}

TEST(TurnIndicatorFilter, HoldsACommandSteady)
{
  const std::vector<float> off{4.f, 0.f, 0.f};
  const std::vector<float> left{0.f, 4.f, 0.f};
  const std::vector<float> right{0.f, 0.f, 4.f};
  const std::vector<float> unsure{0.f, 0.5f, 0.f};  // LEFT leads, but under 0.6

  // Starts DISABLE; a confident class must persist for confirm_cycles before it takes over.
  TurnIndicatorFilter filter({0.6, 3, 1.0});
  EXPECT_EQ(filter.command(), 1);
  EXPECT_EQ(filter.update(left, 10.0), 1);
  EXPECT_EQ(filter.update(left, 10.1), 1);
  EXPECT_EQ(filter.update(left, 10.2), 2);

  // One dissenting cycle resets the count, so a flicker never switches.
  EXPECT_EQ(filter.update(off, 10.3), 2);
  EXPECT_EQ(filter.update(left, 10.4), 2);
  EXPECT_EQ(filter.update(off, 10.5), 2);
  EXPECT_EQ(filter.update(off, 10.6), 2);

  // Three confirming cycles are not enough until hold_s has passed since the last change.
  EXPECT_EQ(filter.update(off, 10.7), 2);
  EXPECT_EQ(filter.update(off, 11.0), 2);
  EXPECT_EQ(filter.update(off, 11.3), 1);

  // A leading class below min_probability counts for nothing.
  for (int i = 0; i < 10; ++i) {
    EXPECT_EQ(filter.update(unsure, 20.0 + i), 1);
  }

  // LEFT to RIGHT directly, after the hold.
  for (int i = 0; i < 3; ++i) {
    filter.update(left, 30.0 + 0.1 * i);
  }
  EXPECT_EQ(filter.command(), 2);
  for (int i = 0; i < 3; ++i) {
    filter.update(right, 40.0 + 0.1 * i);
  }
  EXPECT_EQ(filter.command(), 3);

  // Malformed or non-finite output leaves the command alone.
  EXPECT_EQ(filter.update({0.f, 1.f}, 50.0), 3);
  EXPECT_EQ(filter.update({std::numeric_limits<float>::quiet_NaN(), 0.f, 0.f}, 50.1), 3);
}

namespace
{
autoware_planning_msgs::msg::TrajectoryPoint point(
  const double t, const float velocity, const float acceleration)
{
  autoware_planning_msgs::msg::TrajectoryPoint p;
  p.time_from_start.sec = static_cast<int32_t>(t);
  p.time_from_start.nanosec = static_cast<uint32_t>(std::round((t - std::floor(t)) * 1e9));
  p.pose.position.x = t;
  p.longitudinal_velocity_mps = velocity;
  p.acceleration_mps2 = acceleration;
  return p;
}
}  // namespace

TEST(PlanningFactor, DetectsAHeldStopAndTheSlowDownBeforeIt)
{
  std::vector<autoware_planning_msgs::msg::TrajectoryPoint> points;
  for (int i = 0; i < 30; ++i) {
    const double t = 0.1 * i;
    const float v = i < 10 ? 5.0f : (i < 15 ? 5.0f - static_cast<float>(i - 9) : 0.0f);
    const float a = i >= 9 && i < 15 ? -10.0f : 0.0f;
    points.push_back(point(t, v, a));
  }
  const auto result = detect_planning_factors(points, PlanningFactorDetectionConfig{});
  ASSERT_TRUE(result.stop);
  EXPECT_DOUBLE_EQ(result.stop->stop_pose.position.x, points[14].pose.position.x);
  ASSERT_TRUE(result.slowdown);
  EXPECT_DOUBLE_EQ(result.slowdown->start_pose.position.x, points[9].pose.position.x);
  EXPECT_DOUBLE_EQ(result.slowdown->end_pose.position.x, points[15].pose.position.x);
}

TEST(PlanningFactor, AStopThatDoesNotHoldIsNotReported)
{
  std::vector<autoware_planning_msgs::msg::TrajectoryPoint> points;
  for (int i = 0; i < 30; ++i) {
    points.push_back(point(0.1 * i, i == 5 ? 0.0f : 3.0f, 0.0f));
  }
  EXPECT_FALSE(detect_planning_factors(points, PlanningFactorDetectionConfig{}).stop);
  EXPECT_FALSE(detect_planning_factors({}, PlanningFactorDetectionConfig{}).stop);
}

}  // namespace autoware::oneplanner
