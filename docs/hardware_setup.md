# Hardware Setup

## Required hardware

- Franka arm with the Franka HTTP service running on the robot-control host;
- Franka gripper connected to that service;
- one or two Intel RealSense D435 cameras for collection;
- 3Dconnexion SpaceMouse for manual teleoperation.

## Network

The client sends requests to `server_url` in `configs/teleoperation.yaml`.
Update that value to the host running the Franka service before starting control.
The client calls `/getstate` during startup, so the service must be reachable
before the SpaceMouse loop begins.

## Camera configuration

Use the device-list command before collection:

```bash
python -m franka_toolkit.scripts.camera.list_realsense_devices
```

Put the returned serial numbers in `top_serial`, `wrist_serial`, or
`legacy_camera_serial` as appropriate. The LeRobot collector currently expects
two `1280x720@30` RGB+depth streams. The legacy NPZ collector uses the separate
`legacy_camera_*` settings.

## Franka upper-computer service

The current upper computer at `192.168.1.11` exposes a Flask HTTP service on
port `5000`. The toolkit uses these `POST` endpoints:

| Endpoint | Payload | Purpose |
| --- | --- | --- |
| `/getstate` | none | Read pose, joints, velocities, gripper position, and telemetry |
| `/pose` | `{"arr": [x, y, z, qx, qy, qz, qw]}` | Send Cartesian pose |
| `/move_gripper` | `{"gripper_pos": number}` | Send an absolute gripper position |
| `/open_gripper` | none | Open the gripper |
| `/close_gripper` | none | Close the gripper |
| `/clearerr` | none | Clear the robot error state |
| `/set_joint_target` | `{"target": [7 joint values]}` | Send a joint target |
| `/jointreset` | none | Reset the robot joints |

The current `/getstate` response includes `pose`, `q`, `dq`, `gripper_pos`,
`force`, `jacobian`, `torque`, and `vel`. The client keeps the optional
telemetry fields when present, while remaining compatible with older responses.

## Safety checklist

1. Keep the emergency stop accessible.
2. Test with low `position_scale`, `rotation_scale` and `gripper_step`.
3. Confirm the SpaceMouse is neutral before enabling motion.
4. Keep people and obstacles outside the robot workspace.
5. Verify camera serial numbers and the output directory before recording.

No script in this repository starts the robot service. Start and validate that
service separately before running a teleoperation command.
