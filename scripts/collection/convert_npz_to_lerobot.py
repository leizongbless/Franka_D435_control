"""Convert a collected NPZ episode to the LeRobot v3.0 layout.

The source NPZ must contain all four streams required by the Piper D435
contract: ``img_arrays_top``, ``img_arrays_wrist``, ``depth_arrays_top`` and
``depth_arrays_wrist``.  Existing legacy keys are not silently guessed.
"""

import argparse
from pathlib import Path

import numpy as np

from franka_toolkit.lerobot_writer import LeRobotV3Writer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("npz", type=Path)
    parser.add_argument("--root", type=Path, default=Path("data/lerobot"))
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--task", default="")
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    data = np.load(args.npz, allow_pickle=True)
    names = {
        "observation.image.top": "img_arrays_top",
        "observation.image.wrist": "img_arrays_wrist",
        "observation.depth.top": "depth_arrays_top",
        "observation.depth.wrist": "depth_arrays_wrist",
    }
    missing = [dst for dst, src in names.items() if src not in data]
    if missing:
        raise ValueError("NPZ lacks required Piper streams: " + ", ".join(missing))
    if "observation.state" not in data:
        raise ValueError(
            "NPZ must contain observation.state as joint angles plus gripper; "
            "legacy state_arrays contains ee pose and cannot be converted safely"
        )
    states = data["observation.state"]
    ee = data["observation.ee_pose"] if "observation.ee_pose" in data else data["state_ee"][:, :6]
    if states.ndim != 2 or states.shape[1] < 2:
        raise ValueError("observation.state must have shape (T, joint_count + 1)")
    if ee.shape[1] != 6:
        raise ValueError("state_ee/observation.ee_pose must have shape (T, 6)")
    lengths = [len(data[src]) for src in names.values()] + [len(states), len(ee)]
    if len(set(lengths)) != 1:
        raise ValueError(f"all streams must have equal length, got {lengths}")
    frames = []
    for i in range(lengths[0]):
        top = np.asarray(data["img_arrays_top"][i])
        wrist = np.asarray(data["img_arrays_wrist"][i])
        # Legacy OpenCV collectors store BGR; the LeRobot contract is RGB.
        top = top[..., ::-1] if top.ndim == 3 and top.shape[-1] == 3 else top
        wrist = wrist[..., ::-1] if wrist.ndim == 3 and wrist.shape[-1] == 3 else wrist
        frames.append({
            "observation.image.top": top,
            "observation.image.wrist": wrist,
            "observation.depth.top": data["depth_arrays_top"][i],
            "observation.depth.wrist": data["depth_arrays_wrist"][i],
            "observation.state": states[i],
            "observation.ee_pose": ee[i],
        })
    with LeRobotV3Writer(
        args.root,
        args.repo_id,
        fps=args.fps,
        task=args.task,
        width=frames[0]["observation.image.top"].shape[1],
        height=frames[0]["observation.image.top"].shape[0],
        joint_count=states.shape[1] - 1,
    ) as writer:
        writer.write_episode(frames)
    print(f"Wrote {len(frames)} frames to {args.root / args.repo_id}")


if __name__ == "__main__":
    main()
