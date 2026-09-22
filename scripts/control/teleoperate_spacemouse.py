#!/usr/bin/env python3
"""Control a Franka with a SpaceMouse without cameras or data recording."""

import argparse
import select
import sys
import termios
import time
import tty

from franka_toolkit.config import DEFAULT_CONFIG_PATH, load_teleoperation_config

class TerminalKeyReader:
    """Read single keys without blocking the control loop."""

    def __init__(self):
        self.enabled = sys.stdin.isatty()
        self._settings = None

    def __enter__(self):
        if self.enabled:
            self._settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.enabled and self._settings is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._settings)

    def read(self):
        if not self.enabled:
            return None
        ready, _, _ = select.select([sys.stdin], [], [], 0)
        return sys.stdin.read(1) if ready else None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Use a SpaceMouse to teleoperate Franka without recording data."
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        type=str,
        help="teleoperation YAML config (default: %(default)s)",
    )
    parser.add_argument(
        "--server-url",
        default=None,
        help="override the server URL from the config",
    )
    parser.add_argument(
        "--position-scale",
        type=float,
        default=None,
        help="override translation scale from the config",
    )
    parser.add_argument(
        "--rotation-scale",
        type=float,
        default=None,
        help="override rotation scale from the config",
    )
    parser.add_argument(
        "--deadzone",
        type=float,
        default=None,
        help="override SpaceMouse deadzone from the config",
    )
    parser.add_argument(
        "--hz",
        type=float,
        default=None,
        help="override control-loop frequency from the config",
    )
    parser.add_argument(
        "--reset-on-start",
        action="store_true",
        default=None,
        help="move to the configured reset pose before enabling teleoperation",
    )
    args = parser.parse_args()
    config = load_teleoperation_config(args.config)
    args.server_url = args.server_url or config.server_url
    args.position_scale = config.position_scale if args.position_scale is None else args.position_scale
    args.rotation_scale = config.rotation_scale if args.rotation_scale is None else args.rotation_scale
    args.deadzone = config.deadzone if args.deadzone is None else args.deadzone
    args.hz = config.control_hz if args.hz is None else args.hz
    args.reset_on_start = config.reset_on_start if args.reset_on_start is None else args.reset_on_start
    args.gripper_step = config.gripper_step
    args.gripper_min = config.gripper_min
    args.gripper_max = config.gripper_max
    args.open_button = config.open_button
    args.close_button = config.close_button
    if not 0.0 <= args.deadzone < 1.0:
        parser.error("--deadzone must be in [0, 1)")
    if args.hz <= 0.0:
        parser.error("--hz must be greater than zero")
    if args.position_scale <= 0.0 or args.rotation_scale <= 0.0:
        parser.error("control scales must be greater than zero")
    return args


def print_pose(robot, rotation_type):
    pose = robot.currpos.copy()
    euler = rotation_type.from_quat(pose[3:]).as_euler("xyz")
    values = [*pose[:3], *euler, float(robot.curr_gripper_pos)]
    print("pose [x y z roll pitch yaw gripper]:", values)


def print_joints(robot, np):
    print("joint position (rad):", robot.q.copy())
    print("joint position (deg):", np.rad2deg(robot.q))
    print("joint velocity:", robot.dq.copy())


def main():
    args = parse_args()

    # Runtime imports keep --help usable even outside the robot conda environment.
    import numpy as np
    from scipy.spatial.transform import Rotation

    from franka_toolkit.input import SpaceMouseExpert
    from franka_toolkit.robot import Franka

    print("Franka SpaceMouse teleoperation (control only; no data recording)")
    print("Keys: q quit | r slow reset | p print pose | j print joints")
    print(
        f"SpaceMouse: button {args.close_button} close gripper | "
        f"button {args.open_button} open gripper (hold for continuous motion)"
    )
    print(f"Robot server: {args.server_url}")

    action_scales = [args.position_scale, args.rotation_scale, args.gripper_step]
    robot = Franka(
        action_scales=action_scales,
        server_url=args.server_url,
        gripper_min=args.gripper_min,
        gripper_max=args.gripper_max,
    )
    print_pose(robot, Rotation)

    wait_for_neutral = True
    if args.reset_on_start:
        print("Resetting robot before teleoperation...")
        robot.reset_slow()

    period = 1.0 / args.hz
    neutral_cycles = 0
    required_neutral_cycles = 5
    try:
        with TerminalKeyReader() as keyboard, SpaceMouseExpert() as spacemouse:
            print("Teleoperation active. Return the SpaceMouse to neutral to enable motion.")
            while True:
                loop_started = time.monotonic()
                key = keyboard.read()

                if key == "q":
                    break
                if key == "r":
                    print("Resetting robot slowly...")
                    robot.reset_slow()
                    wait_for_neutral = True
                    neutral_cycles = 0
                elif key == "p":
                    print_pose(robot, Rotation)
                elif key == "j":
                    print_joints(robot, np)

                action = np.asarray(
                    spacemouse.get_motion_state_transformed(), dtype=float
                ).copy()
                action[np.abs(action) < args.deadzone] = 0.0

                if wait_for_neutral:
                    if np.any(action):
                        neutral_cycles = 0
                        action.fill(0.0)
                    else:
                        neutral_cycles += 1
                        if neutral_cycles >= required_neutral_cycles:
                            wait_for_neutral = False
                            print("SpaceMouse is neutral; motion enabled.")

                buttons = spacemouse.get_action()[1]
                open_pressed = bool(buttons[args.open_button]) if args.open_button < len(buttons) else False
                close_pressed = bool(buttons[args.close_button]) if args.close_button < len(buttons) else False
                gripper_flag = 1 if open_pressed and not close_pressed else (
                    -1 if close_pressed and not open_pressed else 0
                )
                robot.move(action, gripper_flag=gripper_flag)

                elapsed = time.monotonic() - loop_started
                time.sleep(max(0.0, period - elapsed))
    except KeyboardInterrupt:
        print("\nInterrupted by Ctrl+C.")

    print("Teleoperation stopped. No data was saved.")


if __name__ == "__main__":
    main()
