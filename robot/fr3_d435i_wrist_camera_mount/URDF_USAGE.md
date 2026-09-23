# FR3 + D435i assembled URDF

Use `urdf/fr3_d435i_official.urdf`. It contains the official FR3 kinematic chain, complete Franka Hand and finger links, the supplied CNC mount, and D435i color/depth/IMU frames. The package also contains the self-contained visual meshes referenced by the URDF.

For ROS 2:

```bash
colcon build --packages-select fr3_d435i_wrist_camera_mount
ros2 run robot_state_publisher robot_state_publisher \
  $(ros2 pkg prefix fr3_d435i_wrist_camera_mount)/share/fr3_d435i_wrist_camera_mount/urdf/fr3_d435i_official.urdf
```

The mount is attached to `fr3_hand`. The camera is placed against the lower 25-degree mount surface with 0.5 mm clearance; its width direction is parallel to the gripper side. For MuJoCo, run `python mujoco_fr3/view_fr3_d435i.py`.
