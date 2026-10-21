# autoware_oneplanner

## Purpose

`autoware_oneplanner` runs OnePlanner end-to-end planning models in Autoware. It turns the
concatenated LiDAR point cloud, the vector map, the route, traffic-light states and the ego
state into a trajectory, with TensorRT, at the LiDAR's rate.

The supported model architecture is [ResWorld](https://arxiv.org/abs/2602.10884): a planner
over a temporal history of LiDAR bird's-eye-view (BEV) features, here produced by a frozen
BEVFusion-L LiDAR backbone, plus vector-map and route context.

## Inner-workings / Algorithms

```text
LiDAR cloud ──► voxelization ──► BEV feature extractor (TensorRT) ──► temporal cache + SE(2) warp ─┐
                                    └─► detection head (optional) ──► ~/output/detected_objects    │
odometry, acceleration, steering ─► ego state, ego history                                         ├─► planner (TensorRT) ──► trajectory
map, route, traffic lights ───────► lanes, route lanes, polygons, line strings, goal               ─┘
```

- The node reads the planner engine's input manifest at start-up and asks its input providers
  to claim the tensors they can produce; an input nobody produces is a start-up error naming the
  tensor.
- The pass is driven by the point cloud callback, as autoware_bevfusion is. There is no timer.
- One BEV map is extracted per LiDAR frame. The planner reads the last K maps at the model's
  history interval, each warped into the frame the plan starts in.
- The trajectory is postprocessed with autoware_diffusion_planner's velocity profile and
  published as `Trajectory` and `CandidateTrajectories`.

See [docs/design.md](docs/design.md) for the design, and
[docs/model_contract.md](docs/model_contract.md) for what a model package must provide and
exactly what the node feeds it.

## Inputs / Outputs

### Inputs

| Name                      | Type                                                  | Description                                                                          |
| ------------------------- | ----------------------------------------------------- | ------------------------------------------------------------------------------------ |
| `~/input/pointcloud`      | `sensor_msgs/msg/PointCloud2`                         | Concatenated point cloud in `base_link`, Autoware `PointXYZIRC` layout               |
| `~/input/odometry`        | `nav_msgs/msg/Odometry`                               | Ego pose and twist                                                                   |
| `~/input/acceleration`    | `geometry_msgs/msg/AccelWithCovarianceStamped`        | Ego acceleration                                                                     |
| `~/input/steering`        | `autoware_vehicle_msgs/msg/SteeringReport`            | Steering tire angle                                                                  |
| `~/input/velocity`        | `autoware_vehicle_msgs/msg/VelocityReport`            | Vehicle speed, used where the odometry twist is exactly zero                         |
| `~/input/traffic_signals` | `autoware_perception_msgs/msg/TrafficLightGroupArray` | Traffic-light recognition                                                            |
| `~/input/route`           | `autoware_planning_msgs/msg/LaneletRoute`             | Mission route                                                                        |
| `~/input/vector_map`      | `autoware_map_msgs/msg/LaneletMapBin`                 | Lanelet2 map                                                                         |
| `~/input/turn_indicators` | `autoware_vehicle_msgs/msg/TurnIndicatorsReport`      | Turn-indicator state; not subscribed when `context.turn_indicators.enabled` is false |

A missing input skips that pass with a throttled warning and a WARN diagnostic naming it. At
start-up the node regularly runs before the publishers it reads; it waits for them.

### Outputs

| Name                                    | Type                                                        | Description                                                                                              |
| --------------------------------------- | ----------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| `~/output/trajectory`                   | `autoware_planning_msgs/msg/Trajectory`                     | Ego trajectory in `map`, a point at t = 0 then one per 0.1 s of the model's horizon                      |
| `~/output/trajectories`                 | `autoware_internal_planning_msgs/msg/CandidateTrajectories` | One candidate per batch and per extra trajectory output, with `GeneratorInfo`                            |
| `~/output/turn_indicators`              | `autoware_vehicle_msgs/msg/TurnIndicatorsCommand`           | When the model has a turn-indicator output                                                               |
| `~/output/detected_objects`             | `autoware_perception_msgs/msg/DetectedObjects`              | When the extractor carries a detection head and the model enables it; one per LiDAR frame                |
| `/planning/planning_factors/oneplanner` | `autoware_internal_planning_msgs/msg/PlanningFactorArray`   | Stop and slow-down read off the trajectory, when `planning_factor.enable_*` is set                       |
| `/diagnostics`                          | `diagnostic_msgs/msg/DiagnosticArray`                       | `inference_status`                                                                                       |
| `~/debug/...`                           | `autoware_internal_debug_msgs/msg/Float64Stamped`           | Processing time per stage and latency; see [docs/design.md](docs/design.md#diagnostics-and-debug-topics) |

## Parameters

Parameters are split the way autoware_bevfusion splits them, so that deploying a model never
edits this package:

| File                                 | Lives in                               | Holds                                                                                                      |
| ------------------------------------ | -------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| `config/oneplanner.param.yaml`       | this package                           | Deployment defaults, model-agnostic: artifact paths, TensorRT workspace, staleness bounds, postprocessing  |
| `ml_package_<model_name>.param.yaml` | the model directory, beside the graphs | The network: providers, tensor names, voxelization, temporal history, horizon, precision, input convention |

The launch file loads the package defaults first and the ml_package file second, so the model's
values win. Fields that describe the network are declared without defaults, as
autoware_bevfusion declares its ml_package fields: a model directory that lacks one fails at
start-up instead of running with another network's geometry.

The main deployment parameters (all of them, with defaults and ranges, are in
[schema/oneplanner.schema.json](schema/oneplanner.schema.json)):

| Name                                 | Default | Description                                                          |
| ------------------------------------ | ------- | -------------------------------------------------------------------- |
| `trt_workspace_mib`                  | 4096    | TensorRT builder workspace for both engines                          |
| `precision`                          | `fp32`  | Planner builder precision; the ml_package file states the model's    |
| `bev_feature.extractor.precision`    | `fp16`  | Extractor builder precision                                          |
| `bev_feature.max_delay_ms`           | 200     | Oldest cloud the node plans on, against the node clock               |
| `bev_feature.substitute_max_seconds` | 0.2     | How much older a map may stand in for a dropped LiDAR frame          |
| `bev_feature.warmup`                 | `wait`  | Until the history is complete: `wait`, or `duplicate_current`        |
| `input_timeout_seconds`              | 1.0     | Silence of the LiDAR after which `inference_status` reports the wait |
| `acceleration_max_wait_ms`           | 50      | Wait for the acceleration at the planning stamp; 0 never waits [ms]  |
| `postprocess.stopping_threshold`     | 0.3     | Speed below which a decelerating plan latches a stop [m/s]           |
| `planning_factor.enable_stop`        | false   | Publish the planned stop as a planning factor                        |
| `debug.dump_dir`                     | `""`    | Write the first passes' input and output tensors here (debug only)   |

The ml_package fields are described in
[schema/ml_package_oneplanner.schema.json](schema/ml_package_oneplanner.schema.json) and
[docs/model_contract.md](docs/model_contract.md); [config/ml_package_resworld.param.yaml](config/ml_package_resworld.param.yaml)
is an example.

## How to run

```bash
ros2 launch autoware_oneplanner oneplanner.launch.xml \
  model_path:=$HOME/autoware_data/oneplanner model_name:=resworld
```

The first start builds both TensorRT engines and caches them beside the ONNX files, which takes
several minutes; `build_only:=true` builds them and exits. To compose the node into an existing
container, pass `target_container:=<container>`; do not enable `use_intra_process_comms` for it,
because rclcpp refuses intra-process delivery for the latched route and map subscriptions.

On recorded data, put the node on the recording's clock (`use_sim_time`), as for any other
planning node: each input's staleness bound is measured against the node clock, and against
wall time every recorded frame is stale.

Autoware's RViz layout shows `/planning/trajectory`; remap `output_trajectory` there to see this
node's output.

## Assumptions / Known limits

- The input point cloud is in `base_link` with the `PointXYZIRC` layout (x, y, z float32,
  intensity uint8 at offset 12), the format of Autoware's concatenated cloud.
- The node runs the models it was written for: a ResWorld-architecture planner with a frozen
  BEVFusion-L feature extractor, trained with the `oneplanner_v1` input convention. A model
  trained on other inputs is refused rather than fed inputs it never saw.
- The trajectory step is 0.1 s; the horizon is the model's own.
- Nothing here makes a learned planner safe to drive with: see [DISCLAIMER.md](../DISCLAIMER.md).
  Run it behind Autoware's validation and control safeguards, on closed courses or with a safety
  driver.

## References

- J. Zhang, Z. Fu, Z. Xu, W. Dai, Q. Liu and Y. Wang, "ResWorld: Temporal Residual World Model
  for End-to-End Autonomous Driving", ICLR 2026. [arXiv:2602.10884](https://arxiv.org/abs/2602.10884)
- `autoware_bevfusion` and `autoware_diffusion_planner` in
  [autoware_universe](https://github.com/autowarefoundation/autoware_universe), from which the
  voxelization, detection postprocessing, map conversion and trajectory postprocessing are
  adapted.
