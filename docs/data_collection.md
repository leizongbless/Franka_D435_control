# RealSense 遥操作数据采集说明

本文档说明 `collect_realsense_teleoperation.py` 的实际数据采集流程、控制方式、保存格式和使用注意事项。

> 这是一个直接控制实体 Franka 机械臂的脚本。运行前必须确认机器人工作空间、急停装置、网络连接和相机安装均处于安全状态。

## 1. 脚本用途

脚本通过 SpaceMouse 手动遥操作 Franka 机械臂，同时采集：

- 指定 RealSense 相机的彩色图像；
- 机械臂末端的笛卡尔位置和欧拉角姿态；
- 夹爪位置； - 根据图像帧生成一个对应的 MP4 视频。

数据先保存在 Python 内存列表中，按 `t` 或 `q` 后一次性写入 `.npz` 文件。脚本不会实时逐帧写盘，因此采集过程中异常退出可能导致尚未保存的数据丢失。

## 2. 启动命令

在项目根目录执行：

```bash
cd /path/to/parent/of/franka_toolkit
python -m franka_toolkit.scripts.collection.collect_realsense_teleoperation \
  --legacy-npz \
  --config franka_toolkit/configs/teleoperation.yaml \
  --save_path franka_toolkit/data/raw \
  --save_name data0.npz
```

遥操作参数统一放在 `configs/teleoperation.yaml`，也可以通过 `--config PATH` 指定另一份文件。
机器人地址、位置/旋转缩放、SpaceMouse 死区、控制频率、相机参数、重试次数和复位时长
都从 config 读取。夹爪按住 `close_button` 闭合、按住 `open_button` 打开，松开后停止发送
夹爪命令；两个按钮同时按下时保持当前位置。

参数说明：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--save_path` | `franka_toolkit/data/raw` | 输出目录；脚本会自动展开 `~` 并创建不存在的目录 |
| `--save_name` | `data0.npz` | 当前输出文件名 |

文件名需要符合 `data0.npz`、`data1.npz` 等形式。脚本会读取文件名中的编号，并在按 `t` 保存后自动将下一组命名为下一个编号。建议始终使用 `data数字.npz` 格式。

## 3. 初始化流程

脚本启动后按以下顺序工作：

1. 按 config 创建 `FrankaEnv([position_scale, rotation_scale, gripper_step])`。
2. 等待约 1 秒，让机器人连接初始化。
3. 枚举当前连接的 RealSense 设备并打印序列号。
4. 查找 config 中 `legacy_camera_serial` 指定的 RealSense。
5. 按 `legacy_camera_width`、`legacy_camera_height` 和 `legacy_camera_fps` 开启 BGR 彩色流。
6. 按 `camera_init_retries` 重试，每次重试前等待 `camera_retry_delay` 秒。
7. 按 `reset_duration` 和 `reset_steps` 慢速复位机器人。
8. 读取一帧初始图像，保持 config 中设置的 RealSense 原始分辨率。
9. 创建两个 OpenCV 窗口：
   - `RGB Image - Camera 2`：显示相机图像；
   - `Control Window`：显示当前控制提示。

如果目标相机不存在或初始化失败，脚本仍会继续运行。此时每次采样使用按 config
分辨率创建的全黑图像作为替代图像，因此必须检查终端输出中的相机状态。

## 4. 机器人控制方式

### 4.1 SpaceMouse 输入

主循环使用：

```python
mouse_action = sm.get_motion_state_transformed()
buttons = [sm.is_button_pressed(0), sm.is_button_pressed(1)]
```

其中 `mouse_action` 的前 6 个值用于末端增量控制，缩放和死区由 config 控制：

```text
[dx_input, dy_input, dz_input,
 droll_input, dpitch_input, dyaw_input]
```

脚本将它传给 `env.move_robot(mouse_action, buttons)`，底层 `Franka.move()` 使用以下缩放：

| 分量 | 缩放 | 含义 |
|---|---:|---|
| 平移前三维 | `position_scale` | 输入乘以该值后作为位置增量 |
| 旋转后三维 | `rotation_scale` | 输入乘以该值后作为 `xyz` 欧拉角增量，单位为弧度 |
| 夹爪 | `gripper_step` | 每个控制周期的相对步进，范围由 `gripper_min/max` 限制 |

机械臂每次收到动作后，会：

1. 基于当前末端位姿计算目标位姿；
2. 对目标位姿执行安全范围裁剪；
3. 通过 config 中的 `server_url` 发送 HTTP 请求；
4. 更新当前末端位姿、关节角、关节速度和夹爪状态。

### 4.2 SpaceMouse 按键

脚本按 config 中的 `open_button` 和 `close_button` 定义 SpaceMouse 按键：

- `close_button` 按住：每个控制周期将夹爪相对闭合 `gripper_step`；
- `open_button` 按住：每个控制周期将夹爪相对打开 `gripper_step`；
- 松开按钮：不发送夹爪命令，保持当前位置；两个按键同时按下：保持当前位置；
- 复位使用键盘 `r`。

### 4.3 安全边界

目标位姿会在 `robot/client.py` 中进行裁剪。脚本使用欧拉角状态读取接口，因此状态中的姿态不是四元数，而是 `xyz` 顺序的欧拉角。具体边界由 `RANGE_LOW`、`RANGE_HIGH` 和 `RESET_POSE` 决定；修改边界前应进行实体安全评估。

## 5. 采集时序

主循环目标间隔为 `1 / control_hz` 秒，并执行以下操作：

1. 读取 SpaceMouse 动作和按钮；
2. 读取 OpenCV 键盘事件；
3. 根据 SpaceMouse 控制机器人；
4. 从 RealSense 获取最新彩色帧；
5. 保持图像为 `640 x 480`，不做插值缩放；
6. 获取机器人当前状态；
7. 如果处于采集状态，并且距离上次保存至少 `1/15` 秒，则追加一个数据点。

脚本配置的数据保存频率目标是 `data_save_frequency`，相机帧率由 `legacy_camera_fps` 控制。图像和状态是在同一次 Python 循环中依次读取后组成样本，实际频率仍会受到机器人 HTTP 请求、相机读取和操作系统调度影响。

采集状态由 `collecting_data` 控制：

- 启动后默认为 `False`，只控制机器人和显示图像，不记录数据；
- 按 `s` 后变为 `True`，开始记录；
- 按 `t` 保存当前片段，然后清空内存并等待下一次 `s`；
- 按 `n` 直接丢弃当前片段；
- 按 `q` 退出主循环，并在当前正在采集时保存数据。

## 6. 键盘操作

| 按键 | 功能 |
|---|---|
| `s` | 开始采集当前片段 |
| `t` | 保存当前片段并继续运行；保存后需要再次按 `s` 才会记录下一段 |
| `n` | 丢弃当前片段，不保存 |
| `q` | 停止程序；如果当前正在采集，则保存当前片段 |
| `r` | 在未采集时慢速复位；采集过程中禁止复位 |
| `p` | 打印当前笛卡尔状态和即将记录的状态 |
| `g` | 打印夹爪状态和按钮状态 |
| `j` | 打印关节角度、角度制数值和关节速度 |
| `0` | 打印当前位置和关节角，提示如何设置新的复位位置 |
| `1` - `9` | 移动到脚本内置的 9 个绝对位姿 |

按 `1` 到 `9` 的预设位姿会直接调用绝对位姿控制。它们主要用于快速移动到常用位置，仍然会受到机器人安全边界限制。若在采集期间触发这些按键，当前循环仍可能把移动后的状态记录下来，因此正式采集时不建议使用。

## 7. `.npz` 文件格式

保存文件使用 `numpy.savez()`，包含三个键：

```python
data = np.load("data0.npz")
images = data["img_arrays"]
states = data["state_arrays"]
actions = data["action_arrays"]
```

### 7.1 `img_arrays`

```text
形状: (T, 480, 640, 3)
类型: uint8
通道: BGR
```

图像来自序列号 `213322070221` 的 RealSense 彩色流，保持 `640 x 480` 原始分辨率，不经过插值。图像不是 RGB 通道顺序；如果使用 PIL、Matplotlib 或某些深度学习输入管线，通常需要先执行 BGR 到 RGB 的转换。

### 7.2 `state_arrays`

```text
形状: (T, 7)
```

每一行的排列为：

```text
[x, y, z, roll, pitch, yaw, gripper]
```

- `x, y, z`：末端笛卡尔位置；
- `roll, pitch, yaw`：通过 `scipy.spatial.transform.Rotation` 从当前四元数转换得到的 `xyz` 欧拉角；
- `gripper`：夹爪位置值。

记录时会复制当前状态到 `state_for_recording`。夹爪动作已经在底层按相对步进累计，因此这里保存的是实际读取到的当前夹爪位置，不再额外人为加减固定值。

### 7.3 `action_arrays`

```text
形状: (T, 7)
```

需要特别注意：当前脚本中的 `action_arrays` 并没有保存 `mouse_action`，也没有保存原始的 SpaceMouse 6 维输入。实际代码保存的是后续采集时的 `state_for_recording`：

- 第一帧状态追加后，不追加动作；
- 从第二个采样点开始，将当前 `state_for_recording` 放入 `action_arrays`；
- 片段保存时再追加一个全零 7 维向量，使数组长度通常与状态数相同。

因此，当前文件中的 `action_arrays` 更接近“错位的状态序列”，不是严格意义上的动作标签。使用行为克隆或离线强化学习前，必须根据训练代码确认它的预期语义；如果需要真实动作标签，应修改采集逻辑，保存 `mouse_action` 及夹爪按钮对应的动作，而不是保存 `state_for_recording`。

当前脚本额外保存两个末端状态字段：

- `state_ee`：形状为 `(T, 7)`，排列为 `[x, y, z, roll, pitch, yaw, gripper]`，夹爪列使用实际状态，不额外减 `2.2`；
- `action_ee`：形状为 `(T, 7)`，按原有 `action_arrays` 的时间对齐方式保存对应末端状态，最后一行为全零填充。

需要注意：`state_arrays` 和 `action_arrays` 当前也不是关节数据，而是末端欧拉状态；新增字段用于明确标识末端状态语义。

## 8. 视频文件

保存 `data0.npz` 时，脚本还会生成：

```text
data0_cam.mp4
```

视频参数：

- 编码器：`mp4v`；
- 分辨率：`640 x 480`；
- 帧率：`15 FPS`；
- 帧内容：`img_arrays` 中的 BGR 图像。

视频只是根据已保存图像重新编码生成，并没有额外采集数据。若 `.npz` 保存失败，视频也不会生成。

## 9. 推荐采集流程

1. 确认机器人服务地址 `192.168.1.11:5000` 可访问。
2. 确认 SpaceMouse 能被系统识别。
3. 确认 config 中 `legacy_camera_serial` 与实际相机一致。
4. 启动脚本，观察相机初始化和机器人复位结果。
5. 不按 `s`，先用 SpaceMouse 小幅移动，确认方向和夹爪方向。
6. 使用 `r` 或左键连续双击慢速回到复位位置。
7. 按 `s` 开始一段示范。
8. 完成操作后按 `t` 保存该段，或按 `q` 保存并退出。
9. 查看终端打印的三个数组形状，并检查生成的 `.npz` 和 `_cam.mp4`。
10. 用 `n` 丢弃明显失败的片段，不要把失败数据混入训练集。

## 10. 常见问题

### 相机初始化失败

检查 USB 连接、RealSense 设备占用情况和序列号。脚本失败后仍会继续运行并记录黑图，所以不能只根据程序是否退出判断采集是否正常。

### `q` 没有生成文件

只有在 `collecting_data == True` 且至少记录过一帧时，退出时才会保存。启动后如果没有按 `s`，直接按 `q` 不会生成数据文件。

### 保存数组长度或动作含义异常

这是当前脚本的实现结果：动作数组首尾通过特殊逻辑补齐，且主体内容是状态而不是 SpaceMouse 原始动作。训练前应检查 `T`、状态与图像的时间对应关系，并按实际需求修正采集脚本或预处理脚本。

### 夹爪状态数值看起来异常

底层夹爪接口会把连续增量转换为发送给机器人服务的夹爪位置值。`gripper_step` 是控制步长，不保证等同于最终服务端返回值的物理单位；应以 `g` 键打印值和实际硬件表现共同判断。

## 11. 输出示例

正常保存时，终端会打印类似：

```text
img_arrays (Camera 2) shape: (T, 480, 640, 3)
state_arrays shape: (T, 7)
action_arrays shape: (T, 7)
摄像头视频已保存: .../data0_cam.mp4
```

其中 `T` 是当前片段按目标约 `15 Hz` 采集到的样本数，时长可粗略估算为 `T / 15` 秒。实际时间会受到机器人 HTTP 请求、相机读取和操作系统调度影响。



record:

pos01:
当前关节角度(弧度): [   0.001358     0.40461     0.05016     -1.5605   -0.039024      2.0176     0.89755]
当前关节角度(度): [    0.07781      23.182       2.874      -89.41     -2.2359       115.6      51.426]

pos02:
当前状态(笛卡尔): [     0.5724   -0.011479     0.24871     -3.0991    0.053779      3.1321     0.24113]
记录状态(包含夹爪): [     0.5724   -0.011479     0.24871     -3.0991    0.053779      3.1321     0.24113]

pos 03:
当前状态(笛卡尔): [     0.7417   0.0051162     0.17205     -3.1208     0.13166      3.1187     0.24113]
记录状态(包含夹爪): [     0.7417   0.0051162     0.17205     -3.1208     0.13166      3.1187     0.24113]
