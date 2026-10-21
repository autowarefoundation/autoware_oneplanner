# Model contract

What a model package has to provide for `autoware_oneplanner` to run it, and exactly what the
node feeds it. Exporting a model, freezing its graph and generating its ml_package file happen in
the model's own training code, not in this package; this document is the interface between the
two.

## The model package

A model directory holds three files, named by the `model_name` launch argument:

| File                                 | Content                                                                                                                             |
| ------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------- |
| `<model_name>_planner.onnx`          | The planner graph, with static shapes (a dynamic batch dimension is run at 1).                                                      |
| `bevfusion_lidar_feature.onnx`       | The frozen BEVFusion-L LiDAR feature extractor, with the sparse-convolution operators of `autoware_tensorrt_plugins`.               |
| `ml_package_<model_name>.param.yaml` | The network description: providers, tensor names, voxelization geometry, temporal history, horizon, precision and input convention. |

The node builds a TensorRT engine for each graph on first start and caches it beside the ONNX
file as `<stem>.engine`, with a `<stem>.engine.identity` file recording the SHA-256 of the graph
and of the engine, the precision, the build settings, the TensorRT version and the GPU's compute
capability. An engine whose identity does not match is rebuilt, so the directory must be
writable. `config/ml_package_oneplanner.param.yaml` is an example of the ml_package file and
`schema/ml_package_oneplanner.schema.json` describes every field.

## Input convention `oneplanner_v1`

A model learns the statistics of the inputs it was trained on, so the node feeds one convention
and refuses a package that states another (`input_convention`). `oneplanner_v1` is:

- **Ego state** as recorded: the EKF velocity, acceleration and yaw rate unmodified at every
  speed, and the steering tire angle as the vehicle reports it. No low-speed zeroing, no clamp.
  The twist is the latest odometry sample at or before the planning time, not an interpolation;
  where that twist is exactly zero (an EKF stop filter), the latest velocity report at or before
  it stands in. The acceleration is the latest sample at or before the planning time too, and
  localization stamps it like the odometry but publishes it just after: a pass waits for the
  sample at its planning time for up to `acceleration_max_wait_ms` (then takes the latest
  earlier one), so what the model is fed does not depend on which message arrived first.
- **Map** converted as described in [Vector-map tensors](#vector-map-tensors): drivable lanes
  only, centerlines from the bounds, crosswalks as polygons, linearly resampled line strings,
  slot order by whole millimetres then map id.
- **Traffic lights** selected per light group at the LiDAR stamp: the elements of the latest
  message stamped at or before it, at most 500 ms old, in which the group has a recognised
  colour. A white light reads as unknown.
- **Ego shape** from the training data's per-vehicle table when the package states one
  (`context.training_ego_shapes`, the triple whose wheel base is closest to the vehicle's,
  within 0.5 m), otherwise from `vehicle_info`.

## When the plan starts

`planning_time` in the ml_package file states the instant the model was trained to plan from,
and the graph states it too, by whether it reads `sensor_latency`; the node refuses any
disagreement.

|                                          | `cloud_stamp`                                    | `planning_time`                                           |
| ---------------------------------------- | ------------------------------------------------ | --------------------------------------------------------- |
| ego state, ego history, map, route, goal | at the pacing cloud's stamp T                    | at the newest odometry sample, T + L                      |
| BEV maps (current and history)           | warped into the pose at T (the newest is copied) | warped into the pose at T + L, the newest included        |
| map crop, slot order, route start        | at T                                             | at T; only the points are re-expressed in the T + L frame |
| traffic lights                           | selected at T                                    | selected at T                                             |
| `sensor_latency`                         | absent                                           | L in seconds                                              |
| trajectory stamp                         | T                                                | T + L                                                     |

T + L is the newest odometry sample, never an extrapolation to the wall clock, so the ego state,
the trajectory stamp and `sensor_latency` describe one measured instant. An odometry that has not
reached T yet is waited for: training never saw a negative latency.

## The BEV feature extractor

| Tensor                 | Shape                  | Content                                                                         |
| ---------------------- | ---------------------- | ------------------------------------------------------------------------------- |
| `voxels` (in)          | `[M, P, F]`            | Points per voxel: x, y, z, intensity (when `use_intensity`), time lag (0)       |
| `num_points_per_voxel` | `[M]`                  | Valid points in each voxel                                                      |
| `coors` (in)           | `[M, 3]`               | Voxel coordinates z, y, x                                                       |
| `bev_feature` (out)    | `[1, C, H, W]`         | The BEV feature map; static shape. Name: `bev_feature.extractor.feature_tensor` |
| `bbox_pred` (out)      | `[10, N]`, optional    | Detection head: x, y, z, log l, log w, log h, sin, cos, vx, vy per proposal     |
| `score` (out)          | `[N]`, optional        | Proposal score                                                                  |
| `label_pred` (out)     | `[N]`, int64, optional | Proposal class                                                                  |

`M` is dynamic within `bev_feature.extractor.voxels_num` `[min, opt, max]`; a cloud with more
voxels runs on the first `max` and reports `is_num_voxels_within_range: false`. Voxelization is
spconv's hash voxelization over `point_cloud_range` with `voxel_size`, as in
autoware_bevfusion. The input cloud must be an Autoware `PointXYZIRC` cloud in `base_link`.

## The planner

The node reads the planner graph's input manifest at start-up and fills every input from one
of its providers; an input no provider produces is a start-up error naming the tensor. Inputs
may be float32, int32 (converted from the float value) or bool (true when the value is not
zero). All coordinates are in the ego frame of the planning pose (base_link, or the vehicle
centre with `shift_x`), x forward, y left, metres.

### Inputs

| Tensor                                                                  | Shape                         | Content                                                                                                                                         |
| ----------------------------------------------------------------------- | ----------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| `bev_feature_history` (`bev_feature.history_tensor`)                    | `[1, K, C, H, W]`             | The last K BEV maps, current first, each SE(2)-warped into the planning frame ([Temporal BEV history](#temporal-bev-history))                   |
| `ego_current_state`                                                     | `[1, 10]`                     | 0, 0, 1, 0 (the ego pose in its own frame), vx, vy [m/s], ax, ay [m/s²], steering tire angle [rad], yaw rate [rad/s]                            |
| `ego_agent_past`                                                        | `[1, T, 4]`                   | x, y, cos(yaw), sin(yaw) on a 0.1 s grid ending at the planning time, oldest first, interpolated between odometry samples                       |
| `lanes`                                                                 | `[1, S, 20, 33]`              | The S nearest lanes ([Vector-map tensors](#vector-map-tensors))                                                                                 |
| `lanes_speed_limit`, `lanes_has_speed_limit`                            | `[1, S, 1]`                   | Speed limit of each lane slot [m/s], 0 when it has none                                                                                         |
| `route_lanes`, `route_lanes_speed_limit`, `route_lanes_has_speed_limit` | `[1, R, 20, 33]`, `[1, R, 1]` | The route's lanes from the one closest to the ego onwards                                                                                       |
| `lanes_on_route`                                                        | `[1, S, 2]`                   | Per lane slot: 1 when the route runs through it, and its position along the route (0 at the route's first visible lane, 1 at its last)          |
| `polygons`                                                              | `[1, P, 40, 4]`               | x, y, one-hot type (intersection area, crosswalk) of the nearest polygons                                                                       |
| `line_strings`                                                          | `[1, L, 20, 4]`               | x, y, one-hot type (stop line, road border) of the nearest line-string pieces                                                                   |
| `goal_pose`                                                             | `[1, 4]`                      | Route goal: x, y, cos(yaw), sin(yaw)                                                                                                            |
| `ego_shape`                                                             | `[1, 3]`                      | Wheel base, length, width [m]                                                                                                                   |
| `turn_indicators`                                                       | `[1, T]`                      | `TurnIndicatorsReport` values over the last T reports, or DISABLE when `context.turn_indicators.enabled` is false                               |
| `static_objects`                                                        | any                           | Zeros                                                                                                                                           |
| `sensor_latency`                                                        | `[1, 1]`                      | L [s]; `planning_time: planning_time` only                                                                                                      |
| `ego_curvature_bias`                                                    | `[1, 1]`                      | Low-passed difference between the yaw-rate curvature and the steering curvature [1/m]; `context.curvature_bias.enabled` only                    |
| `lane_signals`, `route_signals`                                         | `[1, rows, steps, 5]`         | Each lane / route row's traffic-light one-hot over the last seconds, oldest first, following its lanelet; `context.signal_history.enabled` only |

The free dimensions (T, S, R, P, L, K and the BEV size) are the graph's own; the node sizes
everything from the manifest.

### Outputs

| Tensor                                         | Shape                         | Content                                                                                 |
| ---------------------------------------------- | ----------------------------- | --------------------------------------------------------------------------------------- |
| `postprocess.prediction_tensor`                | `[B, A, T, 4]` or `[B, T, 4]` | x, y, cos(yaw), sin(yaw) at 0.1 s, 0.2 s, ... in the planning frame; agent 0 is the ego |
| each of `postprocess.extra_trajectory_tensors` | `[B, T, 4]`                   | Further ego trajectories, published as extra candidates                                 |
| `postprocess.turn_indicator.tensor`, optional  | `[B, 3]`                      | Logits of DISABLE, ENABLE_LEFT, ENABLE_RIGHT                                            |

T must equal `postprocess.horizon_seconds / 0.1`. Outputs must be float32.

## Vector-map tensors

Map elements are selected in a square of half side 100 m around the ego (the cloud pose under
`planning_time`).

**Lanes** are the lanelets of subtype `road`, `highway`, `road_shoulder` and `bicycle_lane`. Both
bounds are resampled to 20 points evenly spaced in arc length, and the centerline is their
point-wise midpoint. A lane is in the crop when its mean, first or last centerline point lies
strictly inside the square. Lanes are ordered by the distance from the ego to the nearer of
their first and second-to-last centerline points, rounded to whole millimetres, then by lanelet
id. The 33 columns of a lane point are:

| Columns | Content                                                                                       |
| ------- | --------------------------------------------------------------------------------------------- |
| 0, 1    | Centerline x, y                                                                               |
| 2, 3    | Offset to the next centerline point (0 at the last point)                                     |
| 4, 5    | Left bound minus centerline                                                                   |
| 6, 7    | Right bound minus centerline                                                                  |
| 8 - 12  | Traffic light: green, amber, red, unknown (a referenced light with no or a white state), none |
| 13 - 22 | Left bound line type, one-hot                                                                 |
| 23 - 32 | Right bound line type, one-hot                                                                |

Line types, in order: crosswalk, curbstone, guard_rail, line_thick, line_thin,
pedestrian_marking, road_border, road_shoulder, virtual (also any other type), zebra_marking.
The speed limit is the lanelet's `speed_limit` attribute in km/h converted through mph
(× 0.621371 × 0.44704), as the training data stored it. A light state is the colour of the
group's element that governs the lane: the only recognised element, else the most confident
arrow in the lane's turn direction, else the most confident circle, else the most confident
element.

**The route** starts at the route lanelet whose centerline comes closest to the ego (in 2-D) and
continues along the route, skipping lanelets outside the crop.

**Polygons** are the `intersection_area` polygons followed by the crosswalk lanelets (outlined as
the left bound followed by the right bound reversed), resampled to 40 points. **Line strings**
are the `stop_line` and `road_border` line strings, cut into equal pieces of 20 points no more
than `context.line_string_max_step_m` apart, linear in arc length. Both are ordered by their
nearest point's distance in whole millimetres, then map id, then piece. A polygon or line
string is in the crop when any of its points lies strictly inside the square.

## Temporal BEV history

The node extracts one BEV map per LiDAR frame and keeps every map inside the history window
with the ego pose at its cloud stamp. History step k (0 = current) takes the map whose stamp is
closest to `newest - k * bev_feature.interval_seconds`, within
`bev_feature.interval_tolerance_seconds`. When that frame was dropped, the newest map older than
the target by at most `bev_feature.substitute_max_seconds` stands in (never the next step's own
map), the way the training data indexes its recorded scans; a longer gap is an outage and the
node waits until the history is complete again.

Every map is warped from its source ego frame into the planning frame: BEV rows are x forward,
columns y left, both spanning `[-bev_feature.bev_half_extent_m, +bev_feature.bev_half_extent_m]`;
sampling is bilinear with zero padding at pixel centres, matching PyTorch's
`grid_sample(mode="bilinear", padding_mode="zeros", align_corners=False)`. The relative transform
is composed in double precision on the host, since map coordinates reach 1e5 m. A localization
jump (see `localization_reset.*`) or a backward time step clears the cache.
