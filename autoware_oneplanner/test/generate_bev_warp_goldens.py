# Copyright 2026 TIER IV, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Generate the golden data of test/bev_warp_golden_test.cpp.

The node's CUDA kernel warps a BEV feature map from the ego frame it was recorded in into
another one, the way the models are trained to see it. This script is the reference: it
composes the relative SE(2) transform in float64, builds the sampling grid at pixel centres
in float32 and samples with PyTorch's ``grid_sample(mode="bilinear", padding_mode="zeros",
align_corners=False)``. BEV rows are x forward and columns y left, both spanning
``[-half_extent, +half_extent]``.

Usage (needs numpy and torch)::

    python3 generate_bev_warp_goldens.py --out data/bev_warp
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

CHANNELS, HEIGHT, WIDTH = 8, 32, 32
HALF_EXTENT_M = 122.4
SEED = 0


def pose(x, y, yaw_deg):
    """Return a pose as [x, y, cos(yaw), sin(yaw)]."""
    yaw = math.radians(yaw_deg)
    return [x, y, math.cos(yaw), math.sin(yaw)]


# (name, source pose, current pose), in a map frame.
CASES = [
    # Pure translation near the origin: one full cell forward (2 * 122.4 / 32 = 7.65 m).
    ("translation_origin", pose(0.0, 0.0, 0.0), pose(7.65, 0.0, 0.0)),
    # Rotation and translation far from the map origin, where float32 cancellation would lose
    # metre-scale information if the transform were not composed relative first.
    ("far_rotation_translation", pose(3700.0, 73670.0, 30.0), pose(3701.8, 73671.1, 35.0)),
    # Far from the origin, a sub-centimetre displacement (an ego standing still).
    ("far_tiny_shift", pose(3700.0, 73670.0, 30.0), pose(3700.002, 73670.0, 30.0)),
    # A quarter turn on the spot.
    ("quarter_turn", pose(10.0, -5.0, 0.0), pose(10.0, -5.0, 90.0)),
    # A motion that pushes part of the map out of range (zero padding).
    ("large_motion", pose(3700.0, 73670.0, 0.0), pose(3760.0, 73670.0, 10.0)),
]


def warp(feature, source_pose, current_pose, half_extent):
    """Warp a [1, C, H, W] map from the source pose's ego frame into the current one's."""
    source = torch.tensor(source_pose, dtype=torch.float64)
    current = torch.tensor(current_pose, dtype=torch.float64)
    source[2:] = source[2:] / torch.linalg.vector_norm(source[2:])
    current[2:] = current[2:] / torch.linalg.vector_norm(current[2:])

    # T_rel = T_source^-1 . T_current, composed once in float64.
    delta_x = current[0] - source[0]
    delta_y = current[1] - source[1]
    cos_rel = source[2] * current[2] + source[3] * current[3]
    sin_rel = source[2] * current[3] - source[3] * current[2]
    shift_x = source[2] * delta_x + source[3] * delta_y
    shift_y = -source[3] * delta_x + source[2] * delta_y
    cos_rel, sin_rel, shift_x, shift_y = (
        value.to(torch.float32) for value in (cos_rel, sin_rel, shift_x, shift_y)
    )

    _, _, height, width = feature.shape
    half = torch.tensor(half_extent, dtype=torch.float32)
    x = ((torch.arange(height, dtype=torch.float32) + 0.5) / height * 2.0 - 1.0) * half
    y = ((torch.arange(width, dtype=torch.float32) + 0.5) / width * 2.0 - 1.0) * half
    current_x, current_y = torch.meshgrid(x, y, indexing="ij")
    source_x = cos_rel * current_x - sin_rel * current_y + shift_x
    source_y = sin_rel * current_x + cos_rel * current_y + shift_y

    # grid_sample's last axis is (width, height), i.e. (y, x).
    grid = torch.stack((source_y / half, source_x / half), dim=-1)[None]
    return F.grid_sample(feature, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="output directory")
    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)
    feature = rng.standard_normal((1, CHANNELS, HEIGHT, WIDTH)).astype(np.float32)
    (out_dir / "feature.bin").write_bytes(feature.tobytes())

    manifest = {
        "channels": CHANNELS,
        "height": HEIGHT,
        "width": WIDTH,
        "half_extent_m": HALF_EXTENT_M,
        "feature": "feature.bin",
        "cases": [],
    }
    for name, source_pose, current_pose in CASES:
        golden = warp(torch.from_numpy(feature), source_pose, current_pose, HALF_EXTENT_M)
        golden_file = f"golden_{name}.bin"
        (out_dir / golden_file).write_bytes(golden.numpy().astype(np.float32).tobytes())
        manifest["cases"].append(
            {
                "name": name,
                "source_pose": source_pose,
                "current_pose": current_pose,
                "golden": golden_file,
            }
        )
    (out_dir / "cases.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {len(CASES)} cases to {out_dir}")


if __name__ == "__main__":
    main()
