# Piper D435 拖拽示教数据集格式说明

> 采集程序：`lerobot_collect.py` ｜ 数据集格式：**LeRobot v3.0** ｜ 帧率：30 Hz
> 可用原生 `LeRobotDataset` 直接加载，无需任何自定义解析代码。

---

## 1. 总览

| 项 | 值 |
|---|---|
| 数据集格式版本 | LeRobot `codebase_version: v3.0` |
| 帧率 fps | 30 |
| 图像分辨率 | 1280 × 720 |
| 图像路数 | 2 × RGB + 2 × 深度（俯视 top / 腕部 wrist） |
| 视频编码 | RGB: H.264（yuv420p, crf=28）｜深度: HEVC Main12 无损（gray12le） |
| 数值存储 | Apache Parquet（float32 / int64 / bool） |
| 每帧数据量 | 数值 ~120 B + 4 路视频帧 |

---

## 2. 目录结构

```
<dataset_root>/<repo_id>/                  # 例: lerobot_data/local/piper_task_brown_block_box/
├── data/                                  # 数值数据（parquet，按大小分卷，每卷 ≤100 MB）
│   └── chunk-000/
│       ├── file-000.parquet               # 每 episode 一行 → 一帧一行
│       └── file-001.parquet
├── videos/                                # 视频数据（每 episode 一个 mp4，按大小分卷，每卷 ≤200 MB）
│   ├── observation.image.top/
│   │   └── chunk-000/file-000.mp4
│   ├── observation.image.wrist/
│   │   └── chunk-000/file-000.mp4
│   ├── observation.depth.top/
│   │   └── chunk-000/file-000.mp4
│   └── observation.depth.wrist/
│   │   └── chunk-000/file-000.mp4
└── meta/
    ├── info.json                          # 数据集元信息（见 §6.1）
    ├── tasks.parquet                      # 任务文本表（见 §6.2）
    ├── episodes/
    │   └── chunk-000/file-000.parquet     # 每 episode 一行的元数据（见 §6.3）
    └── stats.json                         # 全数据集字段统计（min/max/mean/std/分位数）
```

要点：
- **视频一个 episode 一个 mp4 文件**，`file-XXX.mp4` 的 XXX 即 episode 编号（同一 chunk 内）。
- **数值 parquet 按累计大小分卷**（100 MB/卷），一卷可含多个 episode，卷内 `episode_index` 列区分归属。
- episode 编号全局唯一、从 0 递增；**续录时编号自动接续**（如已有 53 集，新录制从 53 开始）。

---

## 3. 单帧字段契约（核心）

每帧 = `data/*.parquet` 一行 + 4 个视频各一帧，配对依据是 `frame_index` / `episode_index` / `index`。

### 3.1 业务字段（由采集程序写入）

| 字段 | 类型 / 形状 | 内容与单位 | 落盘 |
|---|---|---|---|
| `observation.image.top` | video (3, 720, 1280) | 俯视相机 RGB，**RGB 通道序**，uint8 | H.264 mp4 |
| `observation.image.wrist` | video (3, 720, 1280) | 腕部相机 RGB，**RGB 通道序**，uint8 | H.264 mp4 |
| `observation.depth.top` | video (1, 720, 1280) | 俯视相机深度，原始 **uint16 毫米**（D435i z16） | 12-bit log 量化 HEVC mp4 |
| `observation.depth.wrist` | video (1, 720, 1280) | 腕部相机深度，原始 **uint16 毫米** | 同上 |
| `observation.state` | float32 (7,) | 6 关节角 **rad** + 夹爪位置 **raw（0.001 mm 单位的整数码值，未标定）** | parquet |
| `observation.gripper_effort` | float32 (1,) | 夹爪力矩 **raw（0.001 N·m 单位的整数码值，未标定）** | parquet |
| `observation.ee_pose` | float32 (6,) | 末端位姿：x/y/z **米** + rx/ry/rz **rad**（SDK 0.001 mm / 0.001° 换算为 SI） | parquet |
| `action` | float32 (7,) | **= 同帧 observation.state 的拷贝**（拖拽示教语义：人拖到哪，动作即哪） | parquet |
| `next.done` | bool (1,) | episode 结束标志：**每 episode 最后一帧 True，其余全部 False** | parquet |
| `task` | str | 任务文本（启动时 `--task` 参数指定，如 "Put the brown block into the storage box"），全数据集内同一任务为同一字符串 | tasks.parquet |

维度名（names）注册于 info.json：

```
observation.state / action:
  ["joint1_rad","joint2_rad","joint3_rad","joint4_rad","joint5_rad","joint6_rad","gripper_pos_raw_0p001mm"]
observation.gripper_effort: ["gripper_effort_raw_0p001Nm"]
observation.ee_pose: ["ee_x_m","ee_y_m","ee_z_m","ee_rx_rad","ee_ry_rad","ee_rz_rad"]
next.done: ["done"]
```

### 3.2 框架自动生成的索引字段（每帧一行内的元数据）

| 字段 | 类型 | 含义 |
|---|---|---|
| `frame_index` | int64 | 帧在 episode 内序号（0, 1, 2, …） |
| `timestamp` | float32 | `frame_index / 30`（秒，等间隔理想时间戳） |
| `episode_index` | int64 | 所属 episode 编号 |
| `index` | int64 | 全数据集帧序号（跨 episode 连续递增） |
| `task_index` | int64 | 指向 tasks.parquet 的行号 |

### 3.3 帧有效性规则（无占位值）

一帧**必须同时满足**以下条件才会写入，任一缺失则整帧丢弃（不存在补零/插值）：
- 6 关节反馈有效（0x2A1 状态帧已收到）
- 夹爪位置、夹爪力矩、末端位姿反馈有效
- 两路 RGB 图像有效
- **两路深度图有效**（任一路深度无效即丢整帧）

相机帧号未更新时会复用上一帧图像（日志中的"重复帧"计数），关节数值仍为当拍实时值——图像与状态为**弱同步**（latest() 取最近帧，无硬件时间戳对齐，单帧内相位差 ≤ ~33 ms）。

---

## 4. 视频编码与深度量化

### 4.1 RGB（H.264）

| 参数 | 值 |
|---|---|
| 编码器 | libx264（H.264 High Profile, 8-bit） |
| 像素格式 | yuv420p |
| CRF | 28（恒质量，实测码率 ~1–2 Mbps 静止场景） |
| GOP | g=2（每 2 帧一个 I 帧，随机帧访问解码延迟低，适配训练随机采样） |
| preset | faster |

### 4.2 深度（12-bit log 量化 + HEVC 无损）

深度不是直接压 uint8 视频，而是走 **lerobot 官方深度编码路径**（`is_depth_map: true`）：

**写入（量化）**：uint16 毫米深度 → log 域均匀量化到 **12-bit（4096 级）**

```
depth_min = 0.01 m, depth_max = 10.0 m, shift = 3.5, use_log = true
量化码值 q ∈ [0, 4095]，log 间距 → 近场（<0.5 m）步长亚毫米级，远场递增
```

**编码**：量化码值以 `gray12le`（12-bit 灰度）写入 HEVC Main12，`lossless=1`（x265 无损模式）——**编码环节零损失**，反量化误差只来自 12-bit 量化本身（实测 0.5–5 m 范围 max ~0.95 mm）。

**读取（反量化，自动完成）**：原生 `LeRobotDataset` 加载时按 info.json 中注册的量化参数自动还原为物理深度：

```python
from lerobot.datasets import LeRobotDataset

# 深度输出单位为米（默认输出 mm）
ds = LeRobotDataset("local/piper_task_brown_block_box",
                    root="lerobot_data/local",   # 含各数据集目录的父目录
                    depth_output_unit="m")
frame = ds[0]                                   # 或 ds.get_episode(0) 迭代
print(frame["observation.depth.top"].shape)      # torch [1, 720, 1280] float32，单位米
print(frame["observation.image.top"].shape)      # torch [3, 720, 1280] float32，0–1 归一化 RGB
```

### 4.3 深度无效像素约定

D435i 深度饱和值 65535 量化后反量化回 **10.0 m（= depth_max）**，与真实 10 m 深度不可区分。**下游请以 `depth >= 10.0` 判定无效像素并过滤**。

---

## 5. info.json 完整样例（实测）

```json
{
    "codebase_version": "v3.0",
    "fps": 30,
    "features": {
        "observation.image.top": {
            "dtype": "video", "shape": [3, 720, 1280],
            "names": ["channels", "height", "width"],
            "info": {
                "video.height": 720, "video.width": 1280,
                "video.codec": "h264", "video.pix_fmt": "yuv420p",
                "video.fps": 30, "video.channels": 3, "has_audio": false,
                "video.g": 2, "video.crf": 28, "video.preset": "faster",
                "video.fast_decode": 0, "video.video_backend": "pyav",
                "video.extra_options": {}, "is_depth_map": false
            }
        },
        "observation.image.wrist": { "...": "同 top" },
        "observation.depth.top": {
            "dtype": "video", "shape": [1, 720, 1280],
            "names": ["channels", "height", "width"],
            "info": {
                "is_depth_map": true, "depth_unit": "mm",
                "video.height": 720, "video.width": 1280,
                "video.codec": "hevc", "video.pix_fmt": "gray12le",
                "video.fps": 30, "video.channels": 1, "has_audio": false,
                "video.g": 2, "video.crf": 30, "video.preset": null,
                "video.fast_decode": 0, "video.video_backend": "pyav",
                "video.extra_options": { "x265-params": "lossless=1" },
                "video.depth_min": 0.01, "video.depth_max": 10.0,
                "video.shift": 3.5, "video.use_log": true
            }
        },
        "observation.depth.wrist": { "...": "同 top" },
        "observation.state": {
            "dtype": "float32", "shape": [7],
            "names": ["joint1_rad", "joint2_rad", "joint3_rad",
                       "joint4_rad", "joint5_rad", "joint6_rad",
                       "gripper_pos_raw_0p001mm"]
        },
        "observation.gripper_effort": {
            "dtype": "float32", "shape": [1], "names": ["gripper_effort_raw_0p001Nm"]
        },
        "observation.ee_pose": {
            "dtype": "float32", "shape": [6],
            "names": ["ee_x_m", "ee_y_m", "ee_z_m", "ee_rx_rad", "ee_ry_rad", "ee_rz_rad"]
        },
        "action": {
            "dtype": "float32", "shape": [7],
            "names": ["joint1_rad", "joint2_rad", "joint3_rad",
                       "joint4_rad", "joint5_rad", "joint6_rad",
                       "gripper_pos_raw_0p001mm"]
        },
        "next.done": { "dtype": "bool", "shape": [1], "names": ["done"] },
        "timestamp":    { "dtype": "float32", "shape": [1], "names": null },
        "frame_index":  { "dtype": "int64",   "shape": [1], "names": null },
        "episode_index":{ "dtype": "int64",   "shape": [1], "names": null },
        "index":        { "dtype": "int64",   "shape": [1], "names": null },
        "task_index":   { "dtype": "int64",   "shape": [1], "names": null }
    },
    "total_episodes": 53,
    "total_frames": 14079,
    "total_tasks": 1,
    "chunks_size": 1000,
    "data_files_size_in_mb": 100,
    "video_files_size_in_mb": 200,
    "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
    "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
    "robot_type": "piper",
    "splits": { "train": "0:53" }
}
```

---

## 6. meta 文件说明

### 6.1 `meta/info.json`
数据集总元信息：格式版本、fps、features 全量定义（含视频编码参数与深度量化参数）、总 episode/帧/task 数、分卷大小、路径模板。**续录兼容性校验依据**（fps 与 features 不一致会拒绝续录）。

### 6.2 `meta/tasks.parquet`

| 列 | 说明 |
|---|---|
| `task`（索引列） | 任务文本，一行一个 task |
| `task_index` | task 编号（帧数据通过 `task_index` 关联） |

单任务数据集仅一行，例：`{"task": "Put the brown block into the storage box", "task_index": 0}`。

### 6.3 `meta/episodes/chunk-XXX/file-XXX.parquet`（每 episode 一行）

关键列：

| 列 | 说明 |
|---|---|
| `episode_index` | episode 编号 |
| `tasks` | 该集任务文本 |
| `length` | 该集帧数 |
| `dataset_from_index` / `dataset_to_index` | 该集帧在全局 index 中的起止 |
| `data/chunk_index` / `data/file_index` | 该集数值 parquet 所在卷/文件 |
| `videos/<video_key>/chunk_index` / `file_index` | 该集每路视频所在卷/文件 |
| `videos/<video_key>/from_timestamp` / `to_timestamp` | 该集视频起止时间戳 |
| `stats/<feature>/min,max,mean,std,count,q01,q10,q50,q90,q99` | 该集各字段统计（含深度） |

### 6.4 `meta/stats.json`
全数据集级各字段统计（与 6.3 同构，聚合所有 episode），含全部 4 路视频字段与全部数值字段。

---

## 7. 加载与消费速查

```python
from lerobot.datasets import LeRobotDataset

ds = LeRobotDataset(
    "local/piper_task_brown_block_box",   # repo_id
    root="lerobot_data/local",            # 数据集目录的父目录
    depth_output_unit="m",                # 深度反量化为米（默认 mm）
)

print(len(ds))                            # 总帧数
ep = ds.get_episode(0)                    # 取整个 episode 0（按需迭代）

frame = ds[0]
# 数值字段: torch.Tensor
frame["observation.state"]                # float32 [7]
frame["observation.ee_pose"]              # float32 [6]
frame["next.done"]                        # bool [1]
# 图像字段: torch.Tensor, CHW, float32, 0–1 归一化
frame["observation.image.top"]            # [3, 720, 1280] RGB
frame["observation.depth.top"]            # [1, 720, 1280] 米
```

显示图像（注意 CHW → HWC 用 transpose，不要 reshape）：

```python
import torch
img = frame["observation.image.top"].permute(1, 2, 0).numpy()   # HWC, RGB
```

用 ffmpeg 单独查看某路视频文件时：RGB 为标准 H.264 可直接播放；**深度 mp4 是 HEVC Main12 gray12le 的量化码值视频，人眼直接看是灰度量级图，不是物理深度**，须走 LeRobotDataset 反量化。

---

## 8. 坑点备忘

| # | 坑点 | 说明 |
|---|---|---|
| 1 | **深度无效像素** | 反量化后 `>= 10.0 m` 即无效（65535 饱和），下游须过滤 |
| 2 | **图像通道序** | 视频内为 **RGB**（采集时已从相机 bgr8 翻转）；用 cv2 显示需再转 `cv2.cvtColor(img, cv2.COLOR_RGB2BGR)` |
| 3 | **张量布局** | LeRobotDataset 输出 **CHW** float32 0–1 归一化；转 HWC 用 `permute/transpose` |
| 4 | **夹爪是 raw 码值** | `gripper_pos_raw_0p001mm` / `gripper_effort_raw_0p001Nm` 未做标定，量纲换算仅按字面（0.001 单位）理解 |
| 5 | **timestamp 为理想值** | `frame_index / 30` 等间隔，非真实采集时刻；弱同步下图像/状态相位差 ≤ ~33 ms |
| 6 | **续录格式锁** | 同一数据集续录要求 fps + features（含 `is_depth_map`）一致，否则拒绝；格式升级须换新 `--repo-id` |
| 7 | **next.done 边界** | 每 episode 恰好一帧 True（末帧）；按 `next.done` 切分 episode 时无需额外处理 |
| 8 | **旧格式数据集不互通** | 早期 AV1、无深度、无 next.done 的数据集（如 `local/piper_d435_teach` 旧集）与新格式互不兼容续录 |
