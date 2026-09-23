"""LeRobot v3.0 episode writer used by the hardware collectors."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np


EE_NAMES = ["ee_x_m", "ee_y_m", "ee_z_m", "ee_rx_rad", "ee_ry_rad", "ee_rz_rad"]
VIDEO_KEYS = (
    "observation.image.top",
    "observation.image.wrist",
    "observation.depth.top",
    "observation.depth.wrist",
)


def dataset_features(
    width: int = 1280,
    height: int = 720,
    joint_count: int = 7,
    include_gripper_effort: bool = False,
) -> dict:
    """Return the exact user feature contract for this dataset."""
    rgb = {
        "dtype": "video",
        "shape": (3, height, width),
        "names": ["channels", "height", "width"],
        "info": {"is_depth_map": False},
    }
    depth = {
        "dtype": "video",
        "shape": (1, height, width),
        "names": ["channels", "height", "width"],
        "info": {"is_depth_map": True},
    }
    state_names = [f"joint{i}_rad" for i in range(1, joint_count + 1)]
    state_names.append("gripper_pos_raw")
    features = {
        "observation.image.top": deepcopy(rgb),
        "observation.image.wrist": deepcopy(rgb),
        "observation.depth.top": deepcopy(depth),
        "observation.depth.wrist": deepcopy(depth),
        "observation.state": {
            "dtype": "float32",
            "shape": (joint_count + 1,),
            "names": state_names,
        },
        "observation.ee_pose": {"dtype": "float32", "shape": (6,), "names": EE_NAMES},
        "action": {"dtype": "float32", "shape": (joint_count + 1,), "names": state_names},
        "next.done": {"dtype": "bool", "shape": (1,), "names": ["done"]},
    }
    if include_gripper_effort:
        features["observation.gripper_effort"] = {
            "dtype": "float32",
            "shape": (1,),
            "names": ["gripper_effort_raw"],
        }
    return features


def _load_lerobot():
    try:
        from lerobot.configs import DepthEncoderConfig, RGBEncoderConfig
        from lerobot.datasets import CODEBASE_VERSION, LeRobotDataset
    except ImportError as exc:
        raise RuntimeError(
            "LeRobot v3 writer requires Python 3.12 and lerobot[dataset]==0.6.1. "
            "Create the environment with environment_lerobot.yml."
        ) from exc
    if CODEBASE_VERSION != "v3.0":
        raise RuntimeError(f"LeRobot v3.0 is required, found {CODEBASE_VERSION}")
    return LeRobotDataset, RGBEncoderConfig, DepthEncoderConfig


class LeRobotV3Writer:
    """Create or resume a native LeRobot v3.0 dataset."""

    def __init__(
        self,
        root: str | Path,
        repo_id: str,
        fps: int = 30,
        task: str = "",
        width: int = 1280,
        height: int = 720,
        joint_count: int = 7,
        include_gripper_effort: bool = False,
    ):
        LeRobotDataset, RGBEncoderConfig, DepthEncoderConfig = _load_lerobot()
        self.root = Path(root).expanduser() / repo_id
        self.repo_id = repo_id
        self.fps = int(fps)
        self.task = task
        self.width = int(width)
        self.height = int(height)
        self.joint_count = int(joint_count)
        self.include_gripper_effort = bool(include_gripper_effort)
        self.features = dataset_features(
            self.width, self.height, self.joint_count, self.include_gripper_effort
        )
        rgb_encoder = RGBEncoderConfig(
            vcodec="h264", pix_fmt="yuv420p", g=2, crf=28, preset="faster"
        )
        depth_encoder = DepthEncoderConfig(
            vcodec="hevc",
            pix_fmt="gray12le",
            g=2,
            crf=30,
            extra_options={"x265-params": "lossless=1"},
            depth_min=0.01,
            depth_max=10.0,
            shift=3.5,
            use_log=True,
        )

        if (self.root / "meta" / "info.json").exists():
            self._validate_existing()
            self.dataset = LeRobotDataset.resume(
                repo_id=repo_id,
                root=self.root,
                rgb_encoder=rgb_encoder,
                depth_encoder=depth_encoder,
                image_writer_threads=4,
            )
        else:
            self.dataset = LeRobotDataset.create(
                repo_id=repo_id,
                root=self.root,
                fps=self.fps,
                robot_type="franka",
                features=self.features,
                use_videos=True,
                rgb_encoder=rgb_encoder,
                depth_encoder=depth_encoder,
                image_writer_threads=4,
                metadata_buffer_size=1,
                data_files_size_in_mb=100,
                video_files_size_in_mb=200,
            )
        self._closed = False

    def _validate_existing(self) -> None:
        info = json.loads((self.root / "meta" / "info.json").read_text())
        if info.get("codebase_version") != "v3.0":
            raise ValueError("existing dataset is not LeRobot v3.0")
        if int(info.get("fps", -1)) != self.fps:
            raise ValueError(f"existing dataset fps is {info.get('fps')}, requested {self.fps}")
        actual = info.get("features", {})
        for key, feature in self.features.items():
            if key not in actual:
                raise ValueError(f"existing dataset is missing feature {key}")
            if list(actual[key].get("shape", [])) != list(feature["shape"]):
                raise ValueError(f"feature shape mismatch for {key}")
            if actual[key].get("dtype") != feature["dtype"]:
                raise ValueError(f"feature dtype mismatch for {key}")
            if actual[key].get("names") != feature["names"]:
                raise ValueError(f"feature names mismatch for {key}")
            expected_depth = feature.get("info", {}).get("is_depth_map")
            actual_depth = actual[key].get("info", {}).get("is_depth_map")
            if expected_depth is not None and actual_depth != expected_depth:
                raise ValueError(f"depth marker mismatch for {key}")

        tasks_path = self.root / "meta" / "tasks.parquet"
        if tasks_path.exists():
            import pyarrow.parquet as pq

            tasks = pq.read_table(tasks_path, columns=["task"])["task"].to_pylist()
            if tasks and set(tasks) != {self.task}:
                raise ValueError(f"task mismatch: existing={tasks}, requested={self.task!r}")

    def _image(self, value, channels: int) -> np.ndarray:
        arr = np.asarray(value)
        if channels == 3 and arr.shape == (self.height, self.width, 3):
            return np.ascontiguousarray(arr, dtype=np.uint8)
        if channels == 1 and arr.shape == (self.height, self.width):
            arr = arr[..., None]
        if channels == 1 and arr.shape == (self.height, self.width, 1):
            return np.ascontiguousarray(arr, dtype=np.uint16)
        raise ValueError(
            f"expected {self.height}x{self.width}x{channels} frame, got {arr.shape}"
        )

    def write_episode(self, frames: Iterable[Dict], done: Optional[np.ndarray] = None) -> None:
        frames = list(frames)
        if not frames:
            raise ValueError("cannot write an empty episode")
        required = set(VIDEO_KEYS) | {"observation.state", "observation.ee_pose"}
        if self.include_gripper_effort:
            required.add("observation.gripper_effort")
        if any(not required.issubset(frame) for frame in frames):
            raise ValueError("each frame must contain two RGB streams, two depth streams, state and ee_pose")

        done_values = np.zeros(len(frames), dtype=bool)
        if done is not None:
            supplied = np.asarray(done, dtype=bool).reshape(-1)
            if len(supplied) != len(frames):
                raise ValueError("done must contain one value per frame")
            done_values[:] = supplied
        done_values[:-1] = False
        done_values[-1] = True

        for index, source in enumerate(frames):
            state = np.asarray(source["observation.state"], dtype=np.float32).reshape(
                self.joint_count + 1
            )
            ee_pose = np.asarray(source["observation.ee_pose"], dtype=np.float32).reshape(6)
            frame = {
                "observation.image.top": self._image(source["observation.image.top"], 3),
                "observation.image.wrist": self._image(source["observation.image.wrist"], 3),
                "observation.depth.top": self._image(source["observation.depth.top"], 1),
                "observation.depth.wrist": self._image(source["observation.depth.wrist"], 1),
                "observation.state": state,
                "observation.ee_pose": ee_pose,
                "action": state.copy(),
                "next.done": np.asarray([done_values[index]], dtype=bool),
                "task": self.task,
            }
            if self.include_gripper_effort:
                frame["observation.gripper_effort"] = np.asarray(
                    source["observation.gripper_effort"], dtype=np.float32
                ).reshape(1)
            self.dataset.add_frame(frame)
        self.dataset.save_episode()

    def close(self) -> None:
        if not self._closed:
            self.dataset.finalize()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
