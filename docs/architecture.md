# Architecture

This repository contains the real Franka teleoperation and D435 data-collection stack.

```text
robot/                    Franka HTTP client and environment adapter
camera/                   RealSense capture and point-cloud helpers
input/                    SpaceMouse input adapters
config/                   Typed configuration loader
configs/                  Editable YAML runtime configuration
scripts/control/          Manual teleoperation entry points
scripts/camera/           Camera inspection and recording tools
scripts/collection/       Demonstration and dataset collectors
docs/                     Hardware, data and design documentation
tests/                    Offline configuration and control tests
```

The normal control path is:

```text
SpaceMouse -> input/ -> scripts/control or scripts/collection
           -> robot/environment.py -> robot/client.py
           -> Franka HTTP service -> robot and gripper
```

The D435 path is separate from robot control. Camera scripts use `camera/` and
`pyrealsense2`; collection scripts combine camera frames, robot state and
SpaceMouse actions into NPZ or LeRobot datasets.

## Entry points

- `scripts/control/teleoperate_spacemouse.py`: robot and gripper control only.
- `scripts/camera/list_realsense_devices.py`: list connected D435 devices.
- `scripts/camera/view_two_realsense.py`: preview color/depth streams.
- `scripts/collection/collect_realsense_teleoperation.py`: compatibility entry;
  defaults to LeRobot and supports `--legacy-npz`.
- `scripts/collection/collect_lerobot_d435.py`: native LeRobot v3 collector.

All current teleoperation entry points read `configs/teleoperation.yaml`.
Command-line values override the YAML values where supported.
