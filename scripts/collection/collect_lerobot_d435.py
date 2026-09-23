"""Collect two-camera Franka demonstrations as a native LeRobot v3 dataset."""

import argparse
import select
import sys
import termios
import time
import tty
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from franka_toolkit.config import DEFAULT_CONFIG_PATH, load_teleoperation_config


class TerminalKeyReader:
    def __init__(self):
        self.enabled = sys.stdin.isatty()
        self.old_settings = None

    def __enter__(self):
        if self.enabled:
            self.old_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.enabled and self.old_settings is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)

    def read(self):
        if not self.enabled or not select.select([sys.stdin], [], [], 0)[0]:
            return None
        return sys.stdin.read(1).lower()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="teleoperation YAML config (default: %(default)s)",
    )
    parser.add_argument("--root", type=Path, default=Path("franka_toolkit/data/lerobot"))
    parser.add_argument("--repo-id")
    parser.add_argument("--task")
    parser.add_argument("--top-serial")
    parser.add_argument("--wrist-serial")
    parser.add_argument("--server-url")
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--fps", type=int)
    return parser.parse_args()


def start_camera(rs, serial: str, width: int, height: int, fps: int):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    pipeline.start(config)
    return pipeline, rs.align(rs.stream.color)


def read_camera(pipeline, align):
    frames = align.process(pipeline.wait_for_frames(timeout_ms=1000))
    color = frames.get_color_frame()
    depth = frames.get_depth_frame()
    if not color or not depth:
        return None
    # RealSense provides BGR; LeRobot's video contract is RGB.
    rgb = np.asarray(color.get_data())[:, :, ::-1].copy()
    depth_mm = np.asarray(depth.get_data(), dtype=np.uint16).copy()
    return rgb, depth_mm


def main():
    args = parse_args()
    config = load_teleoperation_config(args.config)
    repo_id = args.repo_id or config.repo_id
    task = args.task or config.task
    top_serial = args.top_serial or config.top_serial
    wrist_serial = args.wrist_serial or config.wrist_serial
    server_url = args.server_url or config.server_url
    width = args.width or config.camera_width
    height = args.height or config.camera_height
    fps = args.fps or config.camera_fps

    missing = [
        name
        for name, value in (
            ("repo_id", repo_id),
            ("task", task),
            ("top_serial", top_serial),
            ("wrist_serial", wrist_serial),
        )
        if not value
    ]
    if missing:
        raise ValueError(
            "Missing LeRobot settings: "
            + ", ".join(missing)
            + ". Set them in configs/teleoperation.yaml or pass the corresponding CLI option."
        )

    if width <= 0 or height <= 0 or fps <= 0:
        raise ValueError("camera width, height and fps must be > 0")
    if width != 1280 or height != 720 or fps != 30:
        raise ValueError("LeRobot D435 collector currently supports only 1280x720@30")
    print(f"Teleoperation config: {args.config}")
    print(
        f"Scales: position={config.position_scale}, rotation={config.rotation_scale}, "
        f"gripper_step={config.gripper_step}; buttons: open={config.open_button}, "
        f"close={config.close_button}"
    )
    try:
        import pyrealsense2 as rs
        from franka_toolkit.input import SpaceMouseExpert
        from franka_toolkit.lerobot_writer import LeRobotV3Writer
        from franka_toolkit.robot.client import Franka
    except ImportError as exc:
        raise RuntimeError(
            "Missing collection dependencies; create and activate environment_lerobot.yml"
        ) from exc

    if top_serial == wrist_serial:
        raise ValueError("top and wrist cameras must have different serial numbers")
    available = [
        device.get_info(rs.camera_info.serial_number) for device in rs.context().devices
    ]
    missing_serials = [serial for serial in (top_serial, wrist_serial) if serial not in available]
    if missing_serials:
        raise RuntimeError(f"RealSense devices not found: {missing_serials}; available={available}")

    pipelines = []
    writer = None
    try:
        top_pipeline, top_align = start_camera(
            rs, top_serial, width, height, fps
        )
        pipelines.append(top_pipeline)
        wrist_pipeline, wrist_align = start_camera(
            rs, wrist_serial, width, height, fps
        )
        pipelines.append(wrist_pipeline)
        for _ in range(10):
            read_camera(top_pipeline, top_align)
            read_camera(wrist_pipeline, wrist_align)

        robot = Franka(
            [config.position_scale, config.rotation_scale, config.gripper_step],
            server_url=server_url,
            gripper_min=config.gripper_min,
            gripper_max=config.gripper_max,
        )
        include_effort = robot.gripper_effort is not None
        writer = LeRobotV3Writer(
            args.root,
            repo_id,
            fps=fps,
            task=task,
            width=width,
            height=height,
            joint_count=7,
            include_gripper_effort=include_effort,
        )

        episode_frames = []
        collecting = False
        period = 1.0 / fps
        print("Keys: s start | t save episode | n discard | q save and quit")
        with TerminalKeyReader() as keys, SpaceMouseExpert() as mouse:
            while True:
                started = time.monotonic()
                key = keys.read()
                if key == "s" and not collecting:
                    episode_frames = []
                    collecting = True
                    print("Recording episode")
                elif key == "n" and collecting:
                    episode_frames = []
                    collecting = False
                    print("Episode discarded")
                elif key == "t" and collecting:
                    if episode_frames:
                        writer.write_episode(episode_frames)
                        print(f"Saved {len(episode_frames)} frames")
                    episode_frames = []
                    collecting = False
                elif key == "q":
                    if collecting and episode_frames:
                        writer.write_episode(episode_frames)
                        print(f"Saved {len(episode_frames)} frames")
                    break

                motion, buttons = mouse.get_action()
                motion = np.asarray(motion, dtype=float)
                motion[np.abs(motion) < config.deadzone] = 0.0
                open_pressed = bool(buttons[config.open_button]) if config.open_button < len(buttons) else False
                close_pressed = bool(buttons[config.close_button]) if config.close_button < len(buttons) else False
                gripper_flag = 1 if open_pressed and not close_pressed else (
                    -1 if close_pressed and not open_pressed else 0
                )
                robot.move(motion, gripper_flag)
                top = read_camera(top_pipeline, top_align)
                wrist = read_camera(wrist_pipeline, wrist_align)
                if top is None or wrist is None:
                    continue

                state = np.concatenate(
                    [np.asarray(robot.q, dtype=np.float32), [float(robot.curr_gripper_pos)]]
                ).astype(np.float32)
                ee_pose = np.concatenate(
                    [robot.currpos[:3], R.from_quat(robot.currpos[3:]).as_euler("xyz")]
                ).astype(np.float32)
                frame = {
                    "observation.image.top": top[0],
                    "observation.image.wrist": wrist[0],
                    "observation.depth.top": top[1],
                    "observation.depth.wrist": wrist[1],
                    "observation.state": state,
                    "observation.ee_pose": ee_pose,
                }
                if include_effort:
                    frame["observation.gripper_effort"] = np.asarray(
                        [robot.gripper_effort], dtype=np.float32
                    )
                if collecting:
                    episode_frames.append(frame)
                    if len(episode_frames) % fps == 0:
                        print(f"Recorded {len(episode_frames)} frames")
                time.sleep(max(0.0, period - (time.monotonic() - started)))
    finally:
        if writer is not None:
            writer.close()
        for pipeline in pipelines:
            pipeline.stop()


if __name__ == "__main__":
    main()
