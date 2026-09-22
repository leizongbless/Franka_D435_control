#!/usr/bin/env python3
"""
三相机 RGB 采集脚本
用法: python -m franka_toolkit.scripts.collection.collect_rgb_episodes --save_path data/raw --save_name episode_0.npz
"""

import os
import sys
import time
import argparse
import numpy as np
from datetime import datetime

from franka_toolkit.robot import Franka
from franka_toolkit.camera import RSCapture
from franka_toolkit.input import SpaceMouseExpert

# ============== 配置 ==============
CAMERA_SERIALS = ['135122078001', '213622073689', 'f1371022']
CAMERA_DIM = (640, 480)
CAMERA_FPS = 30
# ==================================


def parse_args():
    parser = argparse.ArgumentParser(description='三相机 RGB 数据采集')
    parser.add_argument('--save_path', type=str, required=True, help='保存目录')
    parser.add_argument('--save_name', type=str, default='data.npz', help='文件名 (默认: data.npz)')
    parser.add_argument('--no_robot', action='store_true', help='不连接机械臂（仅测试相机）')
    parser.add_argument('--no_spacemouse', action='store_true', help='不使用 SpaceMouse')
    parser.add_argument('--max_frames', type=int, default=0, help='最大帧数 (0=无限制)')
    parser.add_argument('--fps', type=float, default=10.0, help='目标采集帧率 (默认: 10)')
    return parser.parse_args()


def main():
    args = parse_args()

    # 创建保存目录
    os.makedirs(args.save_path, exist_ok=True)
    save_file = os.path.join(args.save_path, args.save_name)

    print("=" * 50)
    print("三相机 RGB 采集脚本")
    print("=" * 50)
    print(f"保存路径: {save_file}")
    print(f"目标帧率: {args.fps} fps")
    print()

    # ---------- 初始化相机 ----------
    print("正在初始化相机...")
    cameras = []
    for i, sn in enumerate(CAMERA_SERIALS):
        print(f"  [{i}] {sn} ...", end=" ", flush=True)
        cam = RSCapture(f'cam_{i}', serial_number=sn, dim=CAMERA_DIM, fps=CAMERA_FPS, depth=False)
        cameras.append(cam)
        print("OK")

    time.sleep(1.0)  # 等待流稳定
    print(f"相机初始化完成，共 {len(cameras)} 个\n")

    # ---------- 初始化机械臂 ----------
    robot = None
    if not args.no_robot:
        print("正在连接机械臂...")
        try:
            robot = Franka()
            print("机械臂连接成功\n")
        except Exception as e:
            print(f"机械臂连接失败: {e}")
            print("继续运行（无机械臂模式）\n")
            robot = None

    # ---------- 初始化 SpaceMouse ----------
    spacemouse = None
    if not args.no_spacemouse:
        print("正在初始化 SpaceMouse...")
        try:
            spacemouse = SpaceMouseExpert()
            print("SpaceMouse 初始化成功\n")
        except Exception as e:
            print(f"SpaceMouse 初始化失败: {e}")
            print("继续运行（无 SpaceMouse 模式）\n")
            spacemouse = None

    # ---------- 数据容器 ----------
    rgb_arrays_0 = []  # 相机 0
    rgb_arrays_1 = []  # 相机 1
    rgb_arrays_2 = []  # 相机 2
    eef_states = []    # 末端执行器状态
    timestamps = []    # 时间戳
    actions = []       # SpaceMouse 动作

    # ---------- 采集循环 ----------
    print("=" * 50)
    print("开始采集！")
    print("  - 按 Ctrl+C 停止并保存")
    if spacemouse:
        print("  - SpaceMouse 左键: 保存当前 episode")
        print("  - SpaceMouse 右键: 丢弃当前 episode")
    print("=" * 50)
    print()

    frame_interval = 1.0 / args.fps
    frame_count = 0
    start_time = time.time()

    try:
        while True:
            loop_start = time.time()

            # 获取 RGB 帧
            frames = [cam.get_latest_frame() for cam in cameras]

            # 检查帧是否有效
            if any(f is None for f in frames):
                print("警告: 某个相机帧为空，跳过")
                time.sleep(0.01)
                continue

            # 获取机械臂状态
            eef_state = None
            if robot:
                try:
                    eef_state = robot.get_ee_pose()  # [x, y, z, qx, qy, qz, qw] 或类似
                except:
                    eef_state = np.zeros(7)
            else:
                eef_state = np.zeros(7)

            # 获取 SpaceMouse 输入
            action = np.zeros(7)
            left_click = False
            right_click = False
            if spacemouse:
                try:
                    sm_state = spacemouse.get_action()
                    if isinstance(sm_state, dict):
                        action = sm_state.get('action', np.zeros(7))
                        left_click = sm_state.get('left_click', False)
                        right_click = sm_state.get('right_click', False)
                    else:
                        action = sm_state if sm_state is not None else np.zeros(7)
                except:
                    pass

            # SpaceMouse 按钮处理
            if right_click:
                print("\n[右键] 丢弃当前数据，重新开始")
                rgb_arrays_0.clear()
                rgb_arrays_1.clear()
                rgb_arrays_2.clear()
                eef_states.clear()
                timestamps.clear()
                actions.clear()
                frame_count = 0
                start_time = time.time()
                continue

            if left_click and frame_count > 0:
                print("\n[左键] 保存并退出")
                break

            # 存储数据
            rgb_arrays_0.append(frames[0].copy())
            rgb_arrays_1.append(frames[1].copy())
            rgb_arrays_2.append(frames[2].copy())
            eef_states.append(eef_state.copy() if hasattr(eef_state, 'copy') else np.array(eef_state))
            timestamps.append(time.time() - start_time)
            actions.append(action.copy() if hasattr(action, 'copy') else np.array(action))

            frame_count += 1

            # 显示进度
            if frame_count % 10 == 0:
                elapsed = time.time() - start_time
                actual_fps = frame_count / elapsed if elapsed > 0 else 0
                print(f"\r帧数: {frame_count:5d} | 时长: {elapsed:6.1f}s | 实际帧率: {actual_fps:5.1f} fps", end="", flush=True)

            # 检查最大帧数
            if args.max_frames > 0 and frame_count >= args.max_frames:
                print(f"\n达到最大帧数 {args.max_frames}，停止采集")
                break

            # 帧率控制
            elapsed_this_frame = time.time() - loop_start
            sleep_time = frame_interval - elapsed_this_frame
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n\n收到 Ctrl+C，停止采集...")

    # ---------- 关闭设备 ----------
    print("\n正在关闭设备...")
    for cam in cameras:
        cam.close()
    print("相机已关闭")

    # ---------- 保存数据 ----------
    if frame_count > 0:
        print(f"\n正在保存 {frame_count} 帧数据到 {save_file} ...")

        np.savez_compressed(
            save_file,
            # 三路 RGB
            rgb_0=np.stack(rgb_arrays_0),  # (N, 480, 640, 3)
            rgb_1=np.stack(rgb_arrays_1),  # (N, 480, 640, 3)
            rgb_2=np.stack(rgb_arrays_2),  # (N, 480, 640, 3)
            # 元数据
            eef_states=np.stack(eef_states),  # (N, 7)
            timestamps=np.array(timestamps),   # (N,)
            actions=np.stack(actions),         # (N, 7)
            # 配置信息
            camera_serials=np.array(CAMERA_SERIALS),
            camera_dim=np.array(CAMERA_DIM),
            save_time=np.array(datetime.now().isoformat()),
        )

        # 验证保存
        check = np.load(save_file)
        print("\n保存成功！数据结构:")
        print(f"  rgb_0:       {check['rgb_0'].shape} {check['rgb_0'].dtype}")
        print(f"  rgb_1:       {check['rgb_1'].shape} {check['rgb_1'].dtype}")
        print(f"  rgb_2:       {check['rgb_2'].shape} {check['rgb_2'].dtype}")
        print(f"  eef_states:  {check['eef_states'].shape}")
        print(f"  timestamps:  {check['timestamps'].shape}")
        print(f"  actions:     {check['actions'].shape}")
        print(f"  serials:     {check['camera_serials']}")
        check.close()
    else:
        print("\n没有采集到数据，不保存")

    print("\n完成！")


if __name__ == '__main__':
    main()
