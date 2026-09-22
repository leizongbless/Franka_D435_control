# Franka Toolkit

工具包按职责组织为：

- `robot/`：Franka 通信和机器人环境；
- `camera/`：RealSense SDK、点云采样和相机数据处理；
- `input/`：SpaceMouse 输入设备；
- `config/`、`configs/`：遥操作配置读取代码和可编辑 YAML；
- `scripts/collection/`：数据采集入口；
- `scripts/control/`：机械臂控制入口；
- `scripts/camera/`：相机检测、预览和录制入口；
- `docs/`：使用说明；
- `data/`：运行时生成的采集数据（不会随源码提交）。

从仓库父目录运行模块，例如：

```bash
python -m franka_toolkit.scripts.collection.collect_realsense_teleoperation
python -m franka_toolkit.scripts.control.teleoperate_spacemouse
python -m franka_toolkit.scripts.camera.list_realsense_devices
```

仓库建议克隆为 `franka_toolkit`，以保持 Python 模块路径不变：

```bash
git clone https://github.com/leizongbless/Franka_D435_control.git franka_toolkit
cd ..
```

## 只遥操作，不采集数据

以下入口只连接 Franka 和 SpaceMouse，不初始化相机、不缓存轨迹，也不会写入
NPZ、LeRobot 数据集或视频：

```bash
cd /path/to/parent/of/franka_toolkit
conda env create -f franka_toolkit/environment.yml
conda activate franka_toolkit
python -m franka_toolkit.scripts.control.teleoperate_spacemouse
```

遥操作参数统一在 `configs/teleoperation.yaml`，也可用 `--config PATH` 指定另一份配置。
夹爪侧键按住时按 `gripper_step` 连续相对移动，松开后保持当前位置。

环境只需创建一次；以后运行时从 `conda activate franka_toolkit` 开始即可。

## 查看两个 RealSense 摄像机

连接两个摄像机后，程序默认自动使用检测到的前两个设备，并在同一窗口中并排显示
`1280x720@30 FPS` 的彩色（BGR8）和深度（Z16）画面。启动时会写入并回读脚本中
定义的手动彩色曝光、白平衡及深度发射器参数；设备固件不支持的选项会打印为
`unsupported`：

```bash
cd /path/to/parent/of/franka_toolkit
conda activate franka_toolkit
python -m franka_toolkit.scripts.camera.view_two_realsense
```

按 `q` 或 `Esc` 退出。只查看彩色画面时添加 `--color-only`。也可以按序列号明确选择设备：

```bash
python -m franka_toolkit.scripts.camera.view_two_realsense \
  --serials CAMERA_1_SERIAL CAMERA_2_SERIAL
```

如果设备通过 USB 2 连接且没有暴露 RGB 流，脚本会自动显示该设备的红外画面；
需要彩色画面时，请将设备连接到 USB 3 端口并使用支持 USB 3 的数据线。

程序默认不会在启动时复位。需要先复位再操作时显式添加
`--reset-on-start`。机器人服务不是默认的 `192.168.1.11:5000` 时，可使用：

```bash
python -m franka_toolkit.scripts.control.teleoperate_spacemouse \
  --server-url http://ROBOT_IP:5000/
```

终端按键：`q` 退出、`r` 慢速复位、`p` 打印末端位姿、`j` 打印关节状态。
SpaceMouse 右键闭合夹爪，左键打开夹爪。启动或复位后，需要先让 SpaceMouse
回到中位，程序才会启用末端运动。

运行前确认机器人工作空间无人、急停可用，并先用小幅输入确认坐标方向。

新的双 D435 示教采集默认写入 LeRobot v3.0，而不是 NPZ：

```bash
conda env create -f franka_toolkit/environment_lerobot.yml
conda activate franka_lerobot
python -m franka_toolkit.scripts.collection.collect_lerobot_d435 \
  --repo-id local/franka_task \
  --task "Put the object into the box" \
  --top-serial TOP_CAMERA_SERIAL \
  --wrist-serial WRIST_CAMERA_SERIAL
```

目录结构见 [`docs/architecture.md`](docs/architecture.md)，硬件准备见
[`docs/hardware_setup.md`](docs/hardware_setup.md)，字段映射、按键和加载方法见
[`docs/lerobot_collection.md`](docs/lerobot_collection.md)。
`collect_realsense_teleoperation` 和 `collect_multicamera` 也默认转到该 LeRobot
采集器；只有显式传入 `--legacy-npz` 才运行原来的 NPZ 流程。其他专用采集脚本
仍保留其原有格式。

真实机械臂和相机操作前，请先确认网络、设备序列号和急停装置。
