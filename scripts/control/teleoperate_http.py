#!/usr/bin/env python3
"""SpaceMouse-only Franka teleoperation.

This is a control-only HTTP teleoperation entry point. It does not import or
initialize RealSense/OpenCV/Open3D data collection code. Robot communication is
done through the SERL/oneshot HTTP server, usually at http://192.168.1.11:5000/.
"""

import argparse
import os
import select
import site
import sys
import termios
import time
import tty

# Keep user-site packages from shadowing the conda env, especially numpy 2.x in
# ~/.local, which breaks scipy/open3d/sklearn binaries in oneshotRL.
os.environ.setdefault("PYTHONNOUSERSITE", "1")
try:
    user_site = site.getusersitepackages()
    sys.path = [p for p in sys.path if p != user_site]
except Exception:
    pass

import numpy as np
import requests
from scipy.spatial.transform import Rotation as R

from franka_toolkit.input.device import Spacemouse


RESET_POSE = np.array([0.59, 0.0777, 0.31378, 3.1099675, 0.0146619, -0.0078615])
RESET_JOINTS = np.array([0.0026492, 0.39198, 0.055768, -1.5627, -0.039242, 2.0566, -2.2338])
RANGE_LOW = np.array([-0.5, -0.4, -0.4, -3.14, -3.14, -3.14])
RANGE_HIGH = np.array([0.4, 0.4, 0.4, 3.14, 3.14, 3.14])

PRESET_POSES = {
    "1": np.array([0.38169, -0.21566, 0.14259, -2.9661, -0.051079, -3.0458, 204.01]),
    "2": np.array([0.37821, -0.21591, 0.10376, -2.9623, -0.072111, -3.041, 204.01]),
    "3": np.array([0.37815, -0.21616, 0.10713, -2.9411, -0.079579, -3.0374, 161.4]),
    "4": np.array([0.396, -0.2223, 0.20242, -2.9755, -0.019762, -3.0469, 160.59]),
    "5": np.array([0.35617, 0.16169, 0.18256, -3.0127, -0.052295, -2.9789, 159.74]),
    "6": np.array([0.36081, 0.16483, 0.11082, -3.0699, -0.076943, -2.9734, 159.73]),
    "7": np.array([0.36082, 0.16484, 0.11082, -3.0699, -0.076903, -2.9734, 201.77]),
    "8": np.array([0.37807, 0.16694, 0.25322, -3.0304, 0.012119, -2.9819, 199.0]),
    "9": np.array([0.49313, -0.10567, 0.10199, -3.0516, 0.031173, -3.0667, 0.24448]),
}


class TerminalKeyReader:
    def __init__(self):
        self.enabled = sys.stdin.isatty()
        self.old_settings = None

    def __enter__(self):
        if self.enabled:
            self.old_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.enabled and self.old_settings is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)

    def read_key(self):
        if not self.enabled:
            return None
        ready, _, _ = select.select([sys.stdin], [], [], 0)
        if not ready:
            return None
        return sys.stdin.read(1)


class FrankaHttpTeleop:
    def __init__(self, server_url, action_scales, timeout=2.0):
        self.url = server_url.rstrip("/") + "/"
        self.action_scales = np.array(action_scales, dtype=float)
        self.timeout = timeout
        self.currpos = np.zeros(7, dtype=float)
        self.q = np.zeros(7, dtype=float)
        self.dq = np.zeros(7, dtype=float)
        self.curr_gripper_pos = 0.0
        self.next_gripper_pos = 0.0
        self.xyz_low = (RESET_POSE + RANGE_LOW)[:3]
        self.xyz_high = (RESET_POSE + RANGE_HIGH)[:3]
        self.euler_low = (RESET_POSE + RANGE_LOW)[3:]
        self.euler_high = (RESET_POSE + RANGE_HIGH)[3:]
        self.update_state()

    def post(self, endpoint, json=None, timeout=None):
        response = requests.post(self.url + endpoint, json=json, timeout=timeout or self.timeout)
        response.raise_for_status()
        return response

    def update_state(self):
        state = self.post("getstate").json()
        self.currpos[:] = np.asarray(state["pose"], dtype=float)
        self.q[:] = np.asarray(state["q"], dtype=float)
        self.dq[:] = np.asarray(state["dq"], dtype=float)
        self.curr_gripper_pos = float(np.asarray(state["gripper_pos"]))
        return state

    def clear_errors(self):
        self.post("clearerr")

    def clip_safety_box(self, pose):
        pose = np.asarray(pose, dtype=float).copy()
        pose[:3] = np.clip(pose[:3], self.xyz_low, self.xyz_high)

        euler = R.from_quat(pose[3:]).as_euler("xyz")
        sign = np.sign(euler[0]) or 1.0
        euler[0] = sign * np.clip(abs(euler[0]), self.euler_low[0], self.euler_high[0])
        euler[1:] = np.clip(euler[1:], self.euler_low[1:], self.euler_high[1:])
        pose[3:] = R.from_euler("xyz", euler).as_quat()
        return pose

    def current_euler_state(self):
        euler = R.from_quat(self.currpos[3:]).as_euler("xyz")
        return np.concatenate([self.currpos[:3], euler, np.array([self.next_gripper_pos])])

    def send_pose(self, pose):
        self.clear_errors()
        self.post("pose", json={"arr": np.asarray(pose, dtype=np.float32).tolist()})

    def move_gripper_delta(self, delta):
        next_gripper_pos = 2550.0 * self.curr_gripper_pos + float(delta)
        self.next_gripper_pos = next_gripper_pos
        self.post("move_gripper", json={"gripper_pos": next_gripper_pos})

    def move(self, xyzrpy, buttons):
        xyzrpy = np.asarray(xyzrpy, dtype=float)
        nextpos = self.currpos.copy()
        nextpos[:3] += self.action_scales[0] * xyzrpy[:3]

        delta_rot = R.from_euler("xyz", xyzrpy[3:] * self.action_scales[1])
        nextpos[3:] = (delta_rot * R.from_quat(self.currpos[3:])).as_quat()

        gripper_delta = 0.0
        if buttons[0]:
            gripper_delta = self.action_scales[2]
        elif buttons[1]:
            gripper_delta = -self.action_scales[2]
        self.move_gripper_delta(gripper_delta)

        self.send_pose(self.clip_safety_box(nextpos))
        self.update_state()

    def move_absolute_euler(self, target):
        target = np.asarray(target, dtype=float)
        pose = self.currpos.copy()
        pose[:3] = target[:3]
        pose[3:] = R.from_euler("xyz", target[3:6]).as_quat()
        self.next_gripper_pos = float(target[6])
        self.post("move_gripper", json={"gripper_pos": self.next_gripper_pos})
        self.send_pose(self.clip_safety_box(pose))
        self.update_state()

    def reset(self):
        print("Reset requested: sending set_joint_target and jointreset.")
        try:
            response = self.post("set_joint_target", json={"target": RESET_JOINTS.tolist()}, timeout=5.0)
            print("set_joint_target:", response.text)
        except Exception as exc:
            print("set_joint_target failed; continuing to jointreset:", exc)
        response = self.post("jointreset", timeout=40.0)
        print("jointreset:", response.text)
        time.sleep(1.0)
        self.update_state()

    def print_state(self):
        state = self.current_euler_state()
        print("state xyz/rpy/gripper:", np.array2string(state, precision=5, suppress_small=True))

    def print_joints(self):
        print("joint rad:", np.array2string(self.q, precision=5, suppress_small=True))
        print("joint deg:", np.array2string(np.rad2deg(self.q), precision=3, suppress_small=True))
        print("joint vel:", np.array2string(self.dq, precision=5, suppress_small=True))


def print_help(args):
    print("=" * 72)
    print("Franka SpaceMouse control only - no camera, no data recording")
    print("Server:", args.server_url)
    print("Scales: pos=%.3f m, rot=%.3f rad, gripper=%.1f units" % tuple(args.scales))
    print("Keys: q quit | r reset | c clear errors | p print pose | j print joints")
    print("      0 print pose+joints for editing reset constants | 1-9 preset absolute poses")
    print("SpaceMouse: move knob for xyz/rpy; left/right buttons move gripper continuously")
    print("Startup reset is disabled by default. Use --reset-on-start if you need it.")
    print("=" * 72)


def parse_args():
    parser = argparse.ArgumentParser(description="SpaceMouse-only Franka teleoperation")
    parser.add_argument("--server-url", default="http://192.168.1.11:5000/", help="SERL/oneshot robot server URL")
    parser.add_argument("--scales", nargs=3, type=float, default=[0.10, 0.20, 40.0], metavar=("POS", "ROT", "GRIPPER"))
    parser.add_argument("--deadzone", type=float, default=0.3, help="SpaceMouse deadzone")
    parser.add_argument("--hz", type=float, default=20.0, help="Control loop frequency")
    parser.add_argument("--reset-on-start", action="store_true", help="Reset robot before teleoperation")
    return parser.parse_args()


def main():
    args = parse_args()
    print_help(args)

    robot = FrankaHttpTeleop(args.server_url, args.scales)
    robot.print_state()

    if args.reset_on_start:
        robot.reset()
        robot.print_state()

    dt = 1.0 / args.hz
    loop_count = 0
    with TerminalKeyReader() as keys, Spacemouse(deadzone=args.deadzone) as sm:
        print("Teleop active. Keep the terminal focused for keyboard shortcuts.")
        try:
            while True:
                start = time.time()
                key = keys.read_key()

                if key == "q":
                    print("Quit requested.")
                    break
                if key == "r":
                    robot.reset()
                elif key == "c":
                    robot.clear_errors()
                    print("clearerr sent")
                elif key == "p":
                    robot.print_state()
                elif key == "j":
                    robot.print_joints()
                elif key == "0":
                    robot.print_state()
                    robot.print_joints()
                elif key in PRESET_POSES:
                    print(f"Moving to preset {key}:", PRESET_POSES[key])
                    robot.move_absolute_euler(PRESET_POSES[key])

                mouse_action = sm.get_motion_state_transformed()
                buttons = [int(sm.is_button_pressed(0)), int(sm.is_button_pressed(1))]
                robot.move(mouse_action, buttons)

                loop_count += 1
                if loop_count % int(max(args.hz * 10, 1)) == 0:
                    robot.print_state()

                elapsed = time.time() - start
                time.sleep(max(0.0, dt - elapsed))
        except KeyboardInterrupt:
            print("Interrupted by Ctrl+C.")

    print("Teleop stopped.")


if __name__ == "__main__":
    main()
