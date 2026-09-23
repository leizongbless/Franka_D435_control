import sys

if __name__ == "__main__":
    if "--legacy-npz" not in sys.argv:
        from franka_toolkit.scripts.collection.collect_lerobot_d435 import main

        main()
        raise SystemExit
    sys.argv.remove("--legacy-npz")

# Legacy NPZ collector. Use --legacy-npz explicitly to run this path.
# state: [    0.38169    -0.21566     0.14259     -2.9661   -0.051079     -3.0458      204.01]
# state: [    0.37756    -0.21593     0.10673     -2.9417   -0.080626     -3.0375      204.01]
# state: [    0.37815    -0.21616     0.10713     -2.9411   -0.079579     -3.0374       161.4]
# state: [      0.396     -0.2223     0.20242     -2.9755   -0.019762     -3.0469      160.59]
# state: [    0.35617     0.16169     0.18256     -3.0127   -0.052295     -2.9789      159.74]
# state: [    0.36081     0.16483     0.11082     -3.0699   -0.076943     -2.9734      159.73]
# state: [    0.36082     0.16484     0.11082     -3.0699   -0.076903     -2.9734      201.77]
# state: [    0.37807     0.16694     0.25322     -3.0304    0.012119     -2.9819         199]
# state: [    0.49313    -0.10567     0.10199     -3.0516    0.031173     -3.0667     0.24448]push block
import os,sys,time,copy
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle as pkl
import datetime
import threading
from collections import deque
from typing import Tuple
import requests
import cprint
import torch 
import numpy as np
import cv2
import pyrealsense2 as rs

from scipy.spatial.transform import Rotation as R
from pyquaternion import Quaternion

from franka_toolkit.robot import Franka, FrankaEnv
from franka_toolkit.input import SpaceMouseExpert
from franka_toolkit.geometry import transform_camera_to_marker, interpolate_se3_euler
from franka_toolkit.camera import voxel_downsampling, furthest_point_sampling, PointNetEncoderXYZ, fast_furthest_point_sampling, RSCapture
from franka_toolkit.paths import RAW_DATA_ROOT
from franka_toolkit.config import DEFAULT_CONFIG_PATH, load_teleoperation_config

from ultralytics import YOLO

import argparse

def detect_realsense_cameras():
    """检测可用的RealSense相机"""
    ctx = rs.context()
    devices = ctx.query_devices()
    
    print(f"检测到 {len(devices)} 个RealSense设备:")
    available_serials = []
    for i, device in enumerate(devices):
        serial_number = device.get_info(rs.camera_info.serial_number)
        name = device.get_info(rs.camera_info.name)
        print(f"  设备 {i}: {name} (序列号: {serial_number})")
        available_serials.append(serial_number)
        
    return available_serials

def resize_image(image, target_width=256, target_height=256):
    """
    调整图像大小到指定尺寸
    """
    return cv2.resize(image, (target_width, target_height))


def configure_color_sensor(profile):
    """Enable color auto-exposure and report camera exposure controls."""
    sensor = profile.get_device().first_color_sensor()
    try:
        if sensor.supports(rs.option.enable_auto_exposure):
            sensor.set_option(rs.option.enable_auto_exposure, 1)
            print("已开启彩色相机自动曝光")
        if sensor.supports(rs.option.exposure):
            print(f"当前曝光值: {sensor.get_option(rs.option.exposure)}")
        if sensor.supports(rs.option.gain):
            print(f"当前增益值: {sensor.get_option(rs.option.gain)}")
    except Exception as e:
        print(f"彩色相机曝光配置失败，将使用相机默认设置: {e}")


class ColorFrameReader:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.latest_frame = None
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._read_loop, daemon=True)

    def start(self):
        self.thread.start()

    def _read_loop(self):
        while not self.stop_event.is_set():
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=1000)
                color_frame = frames.get_color_frame()
                if color_frame:
                    frame = np.asanyarray(color_frame.get_data()).copy()
                    with self.lock:
                        self.latest_frame = frame
            except Exception:
                continue

    def get_latest_frame(self):
        with self.lock:
            if self.latest_frame is None:
                return None
            return self.latest_frame.copy()

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=2.0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="save to npz")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--save_path", type=str, help="Output directory", default=str(RAW_DATA_ROOT))
    parser.add_argument("--save_name", type=str,  help="Path to the output datax.npz directory, x is belong to int such as 0,1,2...",default="data0.npz")
    
    args = parser.parse_args()
    config = load_teleoperation_config(args.config)
    print(f"Teleoperation config: {args.config}")
    save_path = args.save_path
    save_name = args.save_name
    try:
        if save_name[5] != '.':
            x = int(save_name[4:6])
        else:
            x = int(save_name[4])
        print("保存文件名:", save_name,end=" ")
        print("x=",x)
    except:
        print("请输入正确的文件名:Path to the output datax.npz directory, x is belong to int such as 0,1,2...")
        exit()
        
    accurate = [config.position_scale, config.rotation_scale, config.gripper_step]
    env = FrankaEnv(
        accurate,
        server_url=config.server_url,
        gripper_min=config.gripper_min,
        gripper_max=config.gripper_max,
    )
    time.sleep(1)

    img_arrays = []  # 摄像头2
    state_arrays = []
    action_arrays = [] # change into using delta action + gripper
    state_ee = []
    action_ee = []

    count = 0
    
    # 检测所有可用的RealSense摄像头
    print("=== 检测RealSense摄像头 ===")
    available_serials = detect_realsense_cameras()
    
    # 摄像头初始化重试机制
    max_retries = config.camera_init_retries
    retry_delay = config.camera_retry_delay
    
    pipeline_2 = None
    color_reader = None
    
    for attempt in range(max_retries):
        print(f"\n=== 摄像头初始化尝试 {attempt + 1}/{max_retries} ===")
        
        # 清理之前的连接
        for pipeline in [pipeline_2]:
            if pipeline is not None:
                try:
                    pipeline.stop()
                    time.sleep(0.3)
                except:
                    pass

        pipeline_2 = None
        
        # 等待设备释放
        if attempt > 0:
            print(f"等待 {retry_delay} 秒后重试...")
            time.sleep(retry_delay)
        
        success_count = 0
        
        # 检查目标摄像头是否可用
        current_serials = detect_realsense_cameras()
        
        # 初始化配置中的旧版采集相机。
        if config.legacy_camera_serial in current_serials:
            try:
                print(f"正在初始化摄像头 ({config.legacy_camera_serial})...")
                pipeline_2 = rs.pipeline()
                config_2 = rs.config()
                config_2.enable_device(config.legacy_camera_serial)
                config_2.enable_stream(
                    rs.stream.color,
                    config.legacy_camera_width,
                    config.legacy_camera_height,
                    rs.format.bgr8,
                    config.legacy_camera_fps,
                )
                
                profile_2 = pipeline_2.start(config_2)
                configure_color_sensor(profile_2)
                color_reader = ColorFrameReader(pipeline_2)
                color_reader.start()
                time.sleep(0.5)
                
                print(f"摄像头 ({config.legacy_camera_serial}) 初始化成功")
                success_count += 1
            except Exception as e:
                print(f"摄像头 ({config.legacy_camera_serial}) 初始化失败: {e}")
                pipeline_2 = None
        else:
            print(f"摄像头 ({config.legacy_camera_serial}) 不可用")

        # 检查是否摄像头初始化成功
        if success_count == 1:
            print("✅ 摄像头初始化成功！")
            break
        else:
            print("❌ 摄像头初始化失败")
            if attempt == max_retries - 1:
                print("⚠️  达到最大重试次数，将继续运行（失败的摄像头将使用默认图像）")

    print(f"\n摄像头最终状态: Camera2={'✅' if pipeline_2 else '❌'}")
    print("=== 摄像头初始化完成 ===\n")
    
    # 数据收集频率控制 (降低保存频率)
    data_save_frequency = config.data_save_frequency
    data_save_interval = 1.0 / data_save_frequency  # 约0.067秒间隔
    last_save_time = time.time()

    env.reset_slow(duration=config.reset_duration, steps=config.reset_steps)
    
    # 获取第二个摄像头的初始图像
    print("获取初始摄像头图像...")
    rgb_img = np.zeros(
        (config.legacy_camera_height, config.legacy_camera_width, 3), dtype=np.uint8
    )
    if color_reader is not None:
        initial_frame = color_reader.get_latest_frame()
        if initial_frame is not None:
            rgb_img = initial_frame
        else:
            print("无法获取第二个摄像头的初始图像")
    last_valid_rgb_img = rgb_img.copy()
    camera_frame_failures = 0
    
    init_state = env.get_euler_action()
    print(f"init_state = {init_state}")
    
    # 创建一个OpenCV窗口来接收键盘输入
    cv2.namedWindow('RGB Image - Camera 2', cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('Control Window', cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow('RGB Image - Camera 2', 0, 0)
    cv2.moveWindow('Control Window', 1050, 0)  # 将控制窗口移到最右边
    
    # 在窗口中显示提示信息
    control_image = np.zeros((300, 600, 3), dtype=np.uint8)
    cv2.putText(control_image, "Press 's' to START data collection", (20, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(control_image, "Press 'q' to STOP", (20, 150), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(control_image, "Press 'r' to RESET", (20, 200), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.imshow('RGB Image - Camera 2', rgb_img)
    
    cv2.imshow('RGB Image - Camera 2', rgb_img)
    cv2.imshow('Control Window', control_image)
    
    # 新增变量用于控制数据收集状态
    collecting_data = False
    last_buttons = [0, 0]
    wait_for_mouse_neutral = True
    last_neutral_debug_time = 0.0
    last_motion_debug_time = 0.0
    print("提示: 按下 's' 键开始收集数据，按下 'q' 键停止收集并保存数据")
    print("控制键说明:")
    print("  's' - 开始数据收集")
    print("  't' - 保存当前数据并继续")
    print("  'n' - 丢弃当前数据")
    print("  'r' - 重置机器人到初始位置")
    print("  'q' - 停止程序并保存")
    print("  'p' - 显示当前状态(笛卡尔坐标)")
    print("  'g' - 显示夹爪详细状态")
    print("  'j' - 显示当前关节角度")
    print("  '0' - 显示当前位置（用于设置新初始位置）")
    print("  '1'-'9' - 快速移动到预设位置")
    print("\n🤖 SpaceMouse控制:")
    print(f"  按钮 {config.close_button} 按住 - 夹爪闭合")
    print(f"  按钮 {config.open_button} 按住 - 夹爪打开")
    print(f"  夹爪控制步长: ±{config.gripper_step} / 控制周期")
    print(f"\n📊 数据收集频率: {data_save_frequency}Hz (每{data_save_interval:.2f}秒保存一次)")
    
    with SpaceMouseExpert() as sm:
        while True:
            mouse_action = sm.get_motion_state_transformed()
            buttons = [int(sm.is_button_pressed(0)),int(sm.is_button_pressed(1))]
            if time.time() - last_motion_debug_time >= 1.0:
                print(f"[SpaceMouse调试] action={np.asarray(mouse_action[:6])}, buttons={buttons}")
                last_motion_debug_time = time.time()
            if buttons != last_buttons:
                print(
                    f"[SpaceMouse调试] buttons={buttons} "
                    f"(open_button={config.open_button}, close_button={config.close_button})"
                )
                last_buttons = buttons.copy()
            gripper_buttons = buttons
            # print(f'mouse_action : {mouse_action}, buttons : {buttons}')
            
            # 使用OpenCV检查键盘输入
            key = cv2.waitKey(1) & 0xFF  # 增加等待时间以提高响应性
            
            # 检查是否按下 's' 键开始收集数据
            if key == ord('s') and not collecting_data:
                collecting_data = True
                last_save_time = time.time()  # 重置计时
                print(f"开始收集数据...（{data_save_frequency}Hz频率）")
                # 更新窗口提示信息
                control_image = np.zeros((300, 600, 3), dtype=np.uint8)
                cv2.putText(control_image, "Collecting data...", (20, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(control_image, "Press 't' to SAVE data and continue", (20, 150), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.putText(control_image, "Press 'q' to STOP and SAVE data", (20, 200), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                cv2.putText(control_image, "Press 'n' to DISCARD data", (20, 250), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                # cv2.imshow('RGB Image', rgb_img)
                # cv2.imshow('Control Window', control_image)
                
            # 检查是否按下 'q' 键停止收集数据
            if key == ord('q'):
                print("停止收集数据并保存...")
                break
            # 检查是否按下 't' 键停止收集数据
            if key == ord('t') and collecting_data:
                print("收集数据并保存...")
                # 添加最后一个动作
                if collecting_data and len(state_arrays) > 0:
                    action_arrays.append(np.zeros_like(state_arrays[-1]))
                    action_ee.append(np.zeros_like(state_ee[-1]))

                    
                    expanded_save_path = os.path.expanduser(save_path)
                    if not os.path.exists(expanded_save_path):
                        os.makedirs(expanded_save_path)
                    save_file = os.path.join(expanded_save_path, save_name)
                    
                    # 转换为numpy数组并打印形状
                    img_array = np.stack(img_arrays)
                    state_array = np.stack(state_arrays)
                    action_array = np.stack(action_arrays)
                    state_ee_array = np.stack(state_ee)
                    action_ee_array = np.stack(action_ee)
                    
                    print("=== 保存的数组形状信息 ===")
                    print(f"img_arrays (Camera 2) shape: {img_array.shape}")
                    print(f"state_arrays shape: {state_array.shape}")
                    print(f"action_arrays shape: {action_array.shape}")
                    print(f"state_ee shape: {state_ee_array.shape}")
                    print(f"action_ee shape: {action_ee_array.shape}")
                    print("========================")
                    
                    np.savez(save_file,
                        img_arrays=img_array,
                        state_arrays=state_array,
                        action_arrays=action_array,
                        state_ee=state_ee_array,
                        action_ee=action_ee_array,
                        )

                    print("Data saved successfully!")
                    
                    # 生成摄像头视频文件
                    video_name = save_name.replace('.npz', '_cam.mp4')
                    video_file = os.path.join(expanded_save_path, video_name)
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter(
                        video_file,
                        fourcc,
                        data_save_frequency,
                        (config.legacy_camera_width, config.legacy_camera_height),
                    )
                    
                    for frame in img_array:
                        video_writer.write(frame)
                    
                    video_writer.release()
                    print(f"摄像头视频已保存: {video_file}")
                    # 更新file_name
                    x+=1
                    save_name = f"data{x}.npz"
                collecting_data = False
                # 更新窗口提示信息
                control_image = np.zeros((300, 600, 3), dtype=np.uint8)
                cv2.putText(control_image, "Press 's' to START data collection", (20, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(control_image, "Press 'q' to STOP", (20, 150), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.putText(control_image, "Press 'r' to RESET", (20, 200), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                # cv2.imshow('RGB Image', rgb_img)
                # cv2.imshow('Control Window', control_image)

                # 清空数据列表
                img_arrays = []
                state_arrays = []
                action_arrays = []
                state_ee = []
                action_ee = []
                count = 0
                print(f"下一次数据将保存到: {expanded_save_path}/{save_name}")
            # 检查是否按下 'r' 键重置机器
            if key == ord('r'):
                if collecting_data:
                    print("无法重置：正在收集数据中，请先按't'或'n'停止收集")
                else:
                    # 显示重置前的位置信息
                    current_pos = env.get_euler_action(if_delta=False)
                    current_joints = env.robot.q
                    print(f"重置前位置 - 笛卡尔: {current_pos}")
                    print(f"重置前位置 - 关节角度(度): {np.rad2deg(current_joints)}")
                    
                    print("开始重置机器人到初始位置...")
                    try:
                        env.reset_slow(
                            duration=config.reset_duration,
                            steps=config.reset_steps,
                        )
                        print("机器人重置成功！")
                        wait_for_mouse_neutral = True
                        
                        # 验证重置后的位置
                        reset_pos = env.get_euler_action(if_delta=False)
                        reset_joints = env.robot.q
                        print(f"重置后位置 - 笛卡尔: {reset_pos}")
                        print(f"重置后位置 - 关节角度(度): {np.rad2deg(reset_joints)}")
                    except Exception as e:
                        print(f"重置失败: {e}")
                        print("可能的原因: 机器人位置超出安全范围或硬件连接问题")
                    
                    # 等待重置完成
                    time.sleep(0.5)

            # 按到n直接舍弃这组数据
            if key == ord('n') and collecting_data:
                print("停止收集数据...")
                # 清空数据列表
                img_arrays = []
                state_arrays = []
                action_arrays = []
                state_ee = []
                action_ee = []
                count = 0
                collecting_data = False
                # 更新窗口提示信息
                control_image = np.zeros((300, 600, 3), dtype=np.uint8)
                cv2.putText(control_image, "Press 's' to START data collection", (20, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(control_image, "Press 'q' to STOP", (20, 150), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.putText(control_image, "Press 'r' to RESET", (20, 200), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)


            # ---------use agent_action---------------
            # open_downbutton = buttons[0]
            # close_upbutton = buttons[-1]

            # move_gripper = 0
            # if open_downbutton != 0:
            #     move_gripper = 1

            # if close_upbutton != 0:
            #     move_gripper = -1
            # action = np.zeros((7))
            # action[:6] = mouse_action
            # action[6] = move_gripper

            # env.agent_action(action)

            # ---------use move_robot---------------
            if wait_for_mouse_neutral:
                mouse_motion_norm = np.linalg.norm(mouse_action[:6])
                if mouse_motion_norm <= 0.05:
                    wait_for_mouse_neutral = False
                    print("[SpaceMouse调试] 已回中，恢复末端运动控制")
                else:
                    mouse_action = np.zeros_like(mouse_action)
                    if time.time() - last_neutral_debug_time >= 1.0:
                        print("[SpaceMouse调试] 等待回中，暂不下发末端动作")
                        last_neutral_debug_time = time.time()
            mouse_action = np.asarray(mouse_action, dtype=float)
            mouse_action[np.abs(mouse_action) < config.deadzone] = 0.0
            env.move_robot(
                mouse_action,
                gripper_buttons,
                open_button=config.open_button,
                close_button=config.close_button,
            )

            # 获取第二个摄像头的数据
            if color_reader is not None:
                latest_frame = color_reader.get_latest_frame()
                if latest_frame is not None:
                    rgb_img = latest_frame
                    last_valid_rgb_img = rgb_img.copy()
                    camera_frame_failures = 0
                else:
                    rgb_img = last_valid_rgb_img
                    camera_frame_failures += 1
                    if camera_frame_failures % 30 == 1:
                        print(f"[相机调试] 后台尚未取得彩色帧，继续使用上一帧 (连续失败 {camera_frame_failures} 次)")
            else:
                rgb_img = last_valid_rgb_img
            
            # 获取当前机器人状态（不要修改原始状态）
            state = env.get_euler_action(if_delta=False)
            
            # 为数据记录创建状态副本，添加夹爪状态信息
            state_for_recording = state.copy()
            state_for_recording = state.copy()
            if key == ord('p'):
                print(f"当前状态(笛卡尔): {state}")
                print(f"记录状态(包含夹爪): {state_for_recording}")
                print(f"夹爪当前状态: {state[6]:.2f}mm")
            if key == ord('g'):
                # 显示夹爪详细状态
                gripper_pos = state[6]
                print(f"=== 夹爪状态详情 ===")
                print(f"当前夹爪位置: {gripper_pos:.2f}mm")
                print(
                    f"SpaceMouse按钮: open[{config.open_button}]={buttons[config.open_button]}, "
                    f"close[{config.close_button}]={buttons[config.close_button]}"
                )
                print(f"夹爪控制步长: {config.gripper_step}")
                if gripper_pos <= config.gripper_min + 1.0:
                    print("⚠️  夹爪可能已完全闭合")
                elif gripper_pos >= config.gripper_max - 1.0:
                    print("⚠️  夹爪可能已完全打开")
                print("====================")
            if key == ord('j'):
                # 获取当前关节角度
                joint_pos = env.robot.q  # 通过robot对象访问关节位置
                joint_vel = env.robot.dq  # 关节速度
                print(f"当前关节角度(弧度): {joint_pos}")
                print(f"当前关节角度(度): {np.rad2deg(joint_pos)}")
                print(f"关节速度: {joint_vel}")
            if key == ord('0'):
                # 保存当前位置为新的重置位置
                current_pos = env.get_euler_action(if_delta=False)
                current_joints = env.robot.q
                print(f"=== 当前位置信息 ===")
                print(f"笛卡尔坐标: {current_pos}")
                print(f"关节角度(弧度): {current_joints}")
                print(f"关节角度(度): {np.rad2deg(current_joints)}")
                print("要使用此位置作为初始位置，请修改franka.py中的RESET_JOINTS常量")
                print("=====================")
            if key == ord('1'):
                move = np.array([    0.38169 ,   -0.21566  ,   0.14259  ,   -2.9661 ,  -0.051079  ,   -3.0458   ,   204.01])
                env.agent_absolute_action(move)
            if key == ord('2'):
                move = np.array([    0.37821 ,   -0.21591 ,    0.10376 ,    -2.9623 ,  -0.072111  ,    -3.041   ,   204.01])
                env.agent_absolute_action(move)
            if key == ord('3'):
                move = np.array([    0.37815  ,  -0.21616  ,   0.10713   ,  -2.9411 ,  -0.079579  ,   -3.0374   ,    161.4])
                env.agent_absolute_action(move)
            if key == ord('4'):
                move = np.array([      0.396  ,   -0.2223   ,  0.20242  ,   -2.9755  , -0.019762  ,   -3.0469   ,   160.59])
                env.agent_absolute_action(move)
            if key == ord('5'):
                move = np.array([    0.35617  ,   0.16169   ,  0.18256  ,   -3.0127  , -0.052295  ,   -2.9789   ,   159.74])
                env.agent_absolute_action(move)
            if key == ord('6'):
                move = np.array([    0.36081  ,   0.16483   ,  0.11082  ,   -3.0699 ,  -0.076943  ,   -2.9734  ,    159.73])
                env.agent_absolute_action(move)
            if key == ord('7'):
                move = np.array([    0.36082   ,  0.16484  ,   0.11082  ,   -3.0699  , -0.076903   ,  -2.9734  ,    201.77])
                env.agent_absolute_action(move)
            if key == ord('8'):
                move = np.array([    0.37807  ,   0.16694  ,   0.25322  ,   -3.0304  ,  0.012119  ,   -2.9819   ,      199])
                env.agent_absolute_action(move)
            if key == ord('9'):
                move = np.array([    0.49313  ,  -0.10567  ,   0.10199   ,  -3.0516  ,  0.031173  ,   -3.0667   ,  0.24448])
                env.agent_absolute_action(move)
            
            # 只有在开始收集数据后才保存数据（带频率控制）
            if collecting_data:
                current_time = time.time()
                
                # 只有当距离上次保存超过设定间隔时才保存数据
                if current_time - last_save_time >= data_save_interval:
                    img_arrays.append(rgb_img)
                    state_arrays.append(state_for_recording)  # 使用包含夹爪状态的记录版本
                    state_ee.append(state.copy())  # 未做人为夹爪偏移的末端状态
                    if count != 0:
                        action_arrays.append(state_for_recording)  # 使用包含夹爪状态的记录版本
                        action_ee.append(state.copy())  # 与原有动作对齐的末端状态
                    count += 1
                    last_save_time = current_time
                    
                    # 每20个数据点显示一次进度（相当于4秒）
                    if count % 20 == 0:
                        print(f"📊 已收集 {count} 个数据点 ({count/data_save_frequency:.1f}秒)")

            # 持续更新摄像头和控制窗口的显示
            cv2.imshow('RGB Image - Camera 2', rgb_img)
            cv2.imshow('Control Window', control_image)
            
            # 减少延迟以提高夹爪响应速度
            time.sleep(1.0 / config.control_hz)
    
    # 关闭OpenCV窗口
    cv2.destroyAllWindows()
    
    # 添加最后一个动作
    if collecting_data and len(state_arrays) > 0:
        action_arrays.append(np.zeros_like(state_arrays[-1]))
        action_ee.append(np.zeros_like(state_ee[-1]))

        
        expanded_save_path = os.path.expanduser(save_path)
        if not os.path.exists(expanded_save_path):
            os.makedirs(expanded_save_path)
        save_file = os.path.join(expanded_save_path, save_name)
        
        # 转换为numpy数组并打印形状
        img_array = np.stack(img_arrays)
        state_array = np.stack(state_arrays)
        action_array = np.stack(action_arrays)
        state_ee_array = np.stack(state_ee)
        action_ee_array = np.stack(action_ee)
        
        print("=== 最终保存的数组形状信息 ===")
        print(f"img_arrays (Camera 2) shape: {img_array.shape}")
        print(f"state_arrays shape: {state_array.shape}")
        print(f"action_arrays shape: {action_array.shape}")
        print(f"state_ee shape: {state_ee_array.shape}")
        print(f"action_ee shape: {action_ee_array.shape}")
        print(f"数据收集时长: {img_array.shape[0]/data_save_frequency:.1f}秒")
        print(f"数据收集频率: {data_save_frequency}Hz")
        print("============================")
        
        np.savez(save_file,
            img_arrays=img_array,
            state_arrays=state_array,
            action_arrays=action_array,
            state_ee=state_ee_array,
            action_ee=action_ee_array,
            )

        print("Data saved successfully!")
        
        # 生成摄像头视频文件
        video_name = save_name.replace('.npz', '_cam.mp4')
        video_file = os.path.join(expanded_save_path, video_name)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            video_file,
            fourcc,
            data_save_frequency,
            (config.legacy_camera_width, config.legacy_camera_height),
        )
        
        for frame in img_array:
            video_writer.write(frame)
        
        video_writer.release()
        print(f"摄像头视频已保存: {video_file}")
    else:
        print("未收集到数据或取消保存")
    
    # 关闭所有摄像头
    if pipeline_2 is not None:
        if color_reader is not None:
            color_reader.stop()
        pipeline_2.stop()
        print("第二个摄像头已关闭")

    
    

        
