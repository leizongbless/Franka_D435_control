# Franka D435 LeRobot v3.0 采集

采集入口 `collect_lerobot_d435.py` 使用 LeRobot 0.6.1 原生 API 写入数据，目录、
Parquet 分卷、视频、统计和续录索引均由 LeRobot 管理。

## 环境

LeRobot 0.6.1 要求 Python 3.12，与旧的 RL 环境依赖不同：

```bash
cd /path/to/franka_toolkit
conda env create -f environment_lerobot.yml
conda activate franka_lerobot
```

## 启动

从 `Reinforcement_Learning` 目录运行：

```bash
python -m franka_toolkit.scripts.collection.collect_lerobot_d435 \
  --root franka_toolkit/data/lerobot \
  --repo-id local/franka_task \
  --task "Put the object into the box" \
  --top-serial TOP_CAMERA_SERIAL \
  --wrist-serial WRIST_CAMERA_SERIAL \
  --server-url http://192.168.1.11:8000/
```

也可以把 `repo_id`、`task`、两个相机序列号、机器人地址和遥操作参数写入
`configs/teleoperation.yaml`，然后只传 `--config PATH`。命令行参数会覆盖 config 中的同名值。
SpaceMouse 两个侧键为持续相对控制：`close_button` 按住闭合、`open_button` 按住打开，
松开即停止发送夹爪命令。

快捷键：`s` 开始、`t` 保存当前 episode、`n` 丢弃、`q` 保存并退出。终端必须
保持焦点。采集要求两台相机同时提供 RGB 和深度；任一路缺失时整帧不写入。

## Franka 字段映射

目录和视频字段遵循 `piper_d435_dataset_format.md`，机器人字段按 Franka 硬件适配：

| 字段 | Franka 内容 |
|---|---|
| `observation.state` | 7 个关节角（rad）+ 夹爪服务原始值，共 8 维 |
| `action` | 同帧 `observation.state` 的拷贝 |
| `observation.ee_pose` | xyz（m）+ xyz 欧拉角（rad），6 维 |
| `observation.gripper_effort` | 仅当 `/getstate` 返回 `gripper_effort` 时注册和保存 |
| `next.done` | 仅 episode 末帧为 True |

不会为了匹配 Piper 维数而丢弃 Franka 第 7 关节，也不会为缺失的夹爪力矩填零。

RGB 输入由 RealSense BGR 转为 RGB；深度保持 D435 `z16` 的 uint16 毫米值，交给
LeRobot 的 `DepthEncoderConfig` 执行 12-bit log 量化和 HEVC Main12 无损编码。

## 加载

```python
from lerobot.datasets import LeRobotDataset

dataset = LeRobotDataset(
    "local/franka_task",
    root="franka_toolkit/data/lerobot/local/franka_task",
    depth_output_unit="m",
)
frame = dataset[0]
print(frame["observation.state"].shape)       # torch.Size([8])
print(frame["observation.image.top"].shape)  # torch.Size([3, 720, 1280])
print(frame["observation.depth.top"].shape)  # torch.Size([1, 720, 1280])
```

同一 `root/repo-id` 再次运行会自动续录。FPS、字段、分辨率或任务文本不匹配时会拒绝
写入，避免在同一数据集中混入不兼容 episode。
