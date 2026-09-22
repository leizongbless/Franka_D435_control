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
import pyspacemouse
import pyrealsense2 as rs

from scipy.spatial.transform import Rotation as R
from pyquaternion import Quaternion

from franka_toolkit.robot import Franka, FrankaEnv
from franka_toolkit.input import SpaceMouseExpert
from franka_toolkit.input.device import Spacemouse
from franka_toolkit.geometry import transform_camera_to_marker, interpolate_se3_euler
from franka_toolkit.camera import voxel_downsampling, furthest_point_sampling, PointNetEncoderXYZ, fast_furthest_point_sampling, RSCapture
from franka_toolkit.paths import RAW_DATA_ROOT

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

def resize_image(image, target_width=320, target_height=192):
    """
    调整图像大小到指定尺寸
    """
    return cv2.resize(image, (target_width, target_height))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="save to npz")
    parser.add_argument("--save_path", type=str, help="Output directory", default=str(RAW_DATA_ROOT))
    parser.add_argument("--save_name", type=str,  help="Path to the output datax.npz directory, x is belong to int such as 0,1,2...",default="data0.npz")
    
    args = parser.parse_args()
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
        
    accurate = [0.10, 0.20, 40]
    env = FrankaEnv(accurate)  
    time.sleep(1)

    img_arrays = []  # 第一个摄像头
    img_arrays_2 = []  # 第二个摄像头
    img_arrays_3 = []  # 第三个摄像头
    state_arrays = []
    action_arrays = [] # change into using delta action + gripper

    count = 0
    
    # 检测所有可用的RealSense摄像头
    print("=== 检测RealSense摄像头 ===")
    available_serials = detect_realsense_cameras()
    
    # 目标摄像头序列号（现在管理所有三个）
    target_cameras = {
        '213322070221': 'camera_1',  # 主摄像头，现在也用RealSense SDK
        '213622073689': 'camera_2',
        'f1371022': 'camera_3'
    }
    
    # 摄像头初始化重试机制
    max_retries = 3
    retry_delay = 2
    
    pipeline_1 = None  # 现在也用RealSense SDK管理
    pipeline_2 = None
    pipeline_3 = None
    
    for attempt in range(max_retries):
        print(f"\n=== 摄像头初始化尝试 {attempt + 1}/{max_retries} ===")
        
        # 清理之前的连接
        for pipeline in [pipeline_1, pipeline_2, pipeline_3]:
            if pipeline is not None:
                try:
                    pipeline.stop()
                    time.sleep(0.3)
                except:
                    pass
        
        pipeline_1 = None
        pipeline_2 = None 
        pipeline_3 = None
        
        # 等待设备释放
        if attempt > 0:
            print(f"等待 {retry_delay} 秒后重试...")
            time.sleep(retry_delay)
        
        success_count = 0
        
        # 检查目标摄像头是否可用
        current_serials = detect_realsense_cameras()
        
        # 初始化第一个摄像头 (213322070221)
        if '213322070221' in current_serials:
            try:
                print("正在初始化第一个摄像头 (213322070221)...")
                pipeline_1 = rs.pipeline()
                config_1 = rs.config()
                config_1.enable_device('213322070221')
                config_1.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
                
                profile_1 = pipeline_1.start(config_1)
                time.sleep(0.5)
                
                print("第一个摄像头 (213322070221) 初始化成功")
                success_count += 1
            except Exception as e:
                print(f"第一个摄像头初始化失败: {e}")
                pipeline_1 = None
        else:
            print("第一个摄像头 (213322070221) 不可用")
        
        # 初始化第二个摄像头 (213622073689)
        if '213622073689' in current_serials:
            try:
                print("正在初始化第二个摄像头 (213622073689)...")
                pipeline_2 = rs.pipeline()
                config_2 = rs.config()
                config_2.enable_device('213622073689')
                config_2.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
                
                profile_2 = pipeline_2.start(config_2)
                time.sleep(0.5)
                
                print("第二个摄像头 (213622073689) 初始化成功")
                success_count += 1
            except Exception as e:
                print(f"第二个摄像头初始化失败: {e}")
                pipeline_2 = None
        else:
            print("第二个摄像头 (213622073689) 不可用")
        
        # 初始化第三个摄像头 (f1371022)
        if 'f1371022' in current_serials:
            try:
                print("正在初始化第三个摄像头 (f1371022)...")
                pipeline_3 = rs.pipeline()
                config_3 = rs.config()
                config_3.enable_device('f1371022')
                config_3.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
                
                profile_3 = pipeline_3.start(config_3)
                time.sleep(0.5)
                
                print("第三个摄像头 (f1371022) 初始化成功")
                success_count += 1
            except Exception as e:
                print(f"第三个摄像头初始化失败: {e}")
                pipeline_3 = None
        else:
            print("第三个摄像头 (f1371022) 不可用")
        
        # 检查是否所有摄像头都初始化成功
        if success_count == 3:
            print("✅ 所有摄像头初始化成功！")
            break
        else:
            print(f"❌ {3 - success_count} 个摄像头初始化失败")
            if attempt == max_retries - 1:
                print("⚠️  达到最大重试次数，将继续运行（失败的摄像头将使用默认图像）")
    
    print(f"\n摄像头最终状态: Camera1={'✅' if pipeline_1 else '❌'}, Camera2={'✅' if pipeline_2 else '❌'}, Camera3={'✅' if pipeline_3 else '❌'}")
    print("=== 摄像头初始化完成 ===\n")
    
    # 数据收集频率控制 (降低保存频率)
    data_save_frequency = 8  # 8Hz: 每0.125秒保存一次数据
    data_save_interval = 1.0 / data_save_frequency  # 0.125秒间隔
    last_save_time = time.time()

    env.reset()
    
    # 获取第一个摄像头的初始图像
    print("获取初始摄像头图像...")
    rgb_img = np.zeros((192, 320, 3), dtype=np.uint8)
    if pipeline_1 is not None:
        try:
            frames_1 = pipeline_1.wait_for_frames()
            color_frame_1 = frames_1.get_color_frame()
            if color_frame_1:
                rgb_img = np.asanyarray(color_frame_1.get_data())
                rgb_img = resize_image(rgb_img, 320, 192)
        except:
            print("无法获取第一个摄像头的初始图像")
    
    init_state = env.get_euler_action()
    print(f"init_state = {init_state}")
    
    # 创建一个OpenCV窗口来接收键盘输入
    cv2.namedWindow('RGB Image - Camera 1', cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('RGB Image - Camera 2', cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('RGB Image - Camera 3', cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('Control Window', cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow('RGB Image - Camera 1', 0, 0)
    cv2.moveWindow('RGB Image - Camera 2', 350, 0)
    cv2.moveWindow('RGB Image - Camera 3', 700, 0)
    cv2.moveWindow('Control Window', 1050, 0)  # 将控制窗口移到最右边
    
    # 在窗口中显示提示信息
    control_image = np.zeros((300, 600, 3), dtype=np.uint8)
    cv2.putText(control_image, "Press 's' to START data collection", (20, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(control_image, "Press 'q' to STOP", (20, 150), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(control_image, "Press 'r' to RESET", (20, 200), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.imshow('RGB Image - Camera 1', rgb_img)
    
    # 获取第二个摄像头的初始图像
    rgb_img_2 = np.zeros((192, 320, 3), dtype=np.uint8)  # 默认图像
    if pipeline_2 is not None:
        try:
            frames_2 = pipeline_2.wait_for_frames()
            color_frame_2 = frames_2.get_color_frame()
            if color_frame_2:
                rgb_img_2 = np.asanyarray(color_frame_2.get_data())
                rgb_img_2 = resize_image(rgb_img_2, 320, 192)
        except:
            print("无法获取第二个摄像头的初始图像")
    
    # 获取第三个摄像头的初始图像
    rgb_img_3 = np.zeros((192, 320, 3), dtype=np.uint8)  # 默认图像
    if pipeline_3 is not None:
        try:
            frames_3 = pipeline_3.wait_for_frames()
            color_frame_3 = frames_3.get_color_frame()
            if color_frame_3:
                rgb_img_3 = np.asanyarray(color_frame_3.get_data())
                rgb_img_3 = resize_image(rgb_img_3, 320, 192)
        except:
            print("无法获取第三个摄像头的初始图像")
    
    cv2.imshow('RGB Image - Camera 2', rgb_img_2)
    cv2.imshow('RGB Image - Camera 3', rgb_img_3)
    cv2.imshow('Control Window', control_image)
    
    # 新增变量用于控制数据收集状态
    collecting_data = False
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
    print("  左键 - 夹爪闭合")
    print("  右键 - 夹爪打开")
    print(f"  夹爪控制增量: ±{accurate[-1]}mm")
    print(f"\n📊 数据收集频率: {data_save_frequency}Hz (每{data_save_interval:.2f}秒保存一次)")
    
    with Spacemouse(deadzone=0.3) as sm:
        while True:
            mouse_action = sm.get_motion_state_transformed()
            buttons = [int(sm.is_button_pressed(0)),int(sm.is_button_pressed(1))]
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

                    
                    expanded_save_path = os.path.expanduser(save_path)
                    if not os.path.exists(expanded_save_path):
                        os.makedirs(expanded_save_path)
                    save_file = os.path.join(expanded_save_path, save_name)
                    
                    # 转换为numpy数组并打印形状
                    img_array = np.stack(img_arrays)  # 第一个摄像头
                    img_array_2 = np.stack(img_arrays_2)  # 第二个摄像头
                    img_array_3 = np.stack(img_arrays_3)  # 第三个摄像头
                    state_array = np.stack(state_arrays)
                    action_array = np.stack(action_arrays)
                    
                    print("=== 保存的数组形状信息 ===")
                    print(f"img_arrays (Camera 1) shape: {img_array.shape}")
                    print(f"img_arrays_2 (Camera 2) shape: {img_array_2.shape}")
                    print(f"img_arrays_3 (Camera 3) shape: {img_array_3.shape}")
                    print(f"state_arrays shape: {state_array.shape}")
                    print(f"action_arrays shape: {action_array.shape}")
                    print("========================")
                    
                    np.savez(save_file,
                        img_arrays=img_array,
                        img_arrays_2=img_array_2,  # 添加第二个摄像头数据
                        img_arrays_3=img_array_3,  # 添加第三个摄像头数据
                        state_arrays=state_array,
                        action_arrays=action_array,
                        )

                    print("Data saved successfully!")
                    
                    # 生成第一个摄像头的视频文件
                    video_name = save_name.replace('.npz', '_cam1.mp4')
                    video_file = os.path.join(expanded_save_path, video_name)
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter(video_file, fourcc, data_save_frequency, (320, 192))
                    
                    for frame in img_array:
                        video_writer.write(frame)
                    
                    video_writer.release()
                    print(f"摄像头1视频已保存: {video_file}")
                    
                    # 生成第二个摄像头的视频文件
                    video_name_2 = save_name.replace('.npz', '_cam2.mp4')
                    video_file_2 = os.path.join(expanded_save_path, video_name_2)
                    video_writer_2 = cv2.VideoWriter(video_file_2, fourcc, data_save_frequency, (320, 192))
                    
                    for frame in img_array_2:
                        video_writer_2.write(frame)
                    
                    video_writer_2.release()
                    print(f"摄像头2视频已保存: {video_file_2}")
                    
                    # 生成第三个摄像头的视频文件
                    video_name_3 = save_name.replace('.npz', '_cam3.mp4')
                    video_file_3 = os.path.join(expanded_save_path, video_name_3)
                    video_writer_3 = cv2.VideoWriter(video_file_3, fourcc, data_save_frequency, (320, 192))
                    
                    for frame in img_array_3:
                        video_writer_3.write(frame)
                    
                    video_writer_3.release()
                    print(f"摄像头3视频已保存: {video_file_3}")
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
                img_arrays_2 = []
                img_arrays_3 = []
                state_arrays = []
                action_arrays = []
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
                        env.reset()
                        print("机器人重置成功！")
                        
                        # 验证重置后的位置
                        reset_pos = env.get_euler_action(if_delta=False)
                        reset_joints = env.robot.q
                        print(f"重置后位置 - 笛卡尔: {reset_pos}")
                        print(f"重置后位置 - 关节角度(度): {np.rad2deg(reset_joints)}")
                    except Exception as e:
                        print(f"重置失败: {e}")
                        print("可能的原因: 机器人位置超出安全范围或硬件连接问题")
                    
                    # 等待重置完成
                    time.sleep(2)

            # 按到n直接舍弃这组数据
            if key == ord('n') and collecting_data:
                print("停止收集数据...")
                # 清空数据列表
                img_arrays = []
                img_arrays_2 = []
                img_arrays_3 = []
                state_arrays = []
                action_arrays = []
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
            env.move_robot(mouse_action, buttons)

            # 获取所有摄像头的图像数据
            # 获取第一个摄像头的数据
            rgb_img = np.zeros((192, 320, 3), dtype=np.uint8)  # 默认图像
            if pipeline_1 is not None:
                try:
                    frames_1 = pipeline_1.wait_for_frames(timeout_ms=50)
                    color_frame_1 = frames_1.get_color_frame()
                    if color_frame_1:
                        rgb_img = np.asanyarray(color_frame_1.get_data())
                        rgb_img = resize_image(rgb_img, 320, 192)
                except:
                    pass  # 使用默认图像
            
            # 获取第二个摄像头的数据
            rgb_img_2 = np.zeros((192, 320, 3), dtype=np.uint8)  # 默认图像
            if pipeline_2 is not None:
                try:
                    frames_2 = pipeline_2.wait_for_frames(timeout_ms=50)
                    color_frame_2 = frames_2.get_color_frame()
                    if color_frame_2:
                        rgb_img_2 = np.asanyarray(color_frame_2.get_data())
                        rgb_img_2 = resize_image(rgb_img_2, 320, 192)
                except:
                    pass  # 使用默认图像
            
            # 获取第三个摄像头的数据
            rgb_img_3 = np.zeros((192, 320, 3), dtype=np.uint8)  # 默认图像
            if pipeline_3 is not None:
                try:
                    frames_3 = pipeline_3.wait_for_frames(timeout_ms=50)
                    color_frame_3 = frames_3.get_color_frame()
                    if color_frame_3:
                        rgb_img_3 = np.asanyarray(color_frame_3.get_data())
                        rgb_img_3 = resize_image(rgb_img_3, 320, 192)
                except:
                    pass  # 使用默认图像
            
            # 获取当前机器人状态（不要修改原始状态）
            state = env.get_euler_action(if_delta=False)
            
            # 为数据记录创建状态副本，添加夹爪状态信息
            state_for_recording = state.copy()
            if buttons[0] == 1:
                state_for_recording[6] -= accurate[-1]
            if buttons[1] == 1:
                state_for_recording[6] += accurate[-1]
            if key == ord('p'):
                print(f"当前状态(笛卡尔): {state}")
                print(f"记录状态(包含夹爪): {state_for_recording}")
                print(f"夹爪当前状态: {state[6]:.2f}mm")
            if key == ord('g'):
                # 显示夹爪详细状态
                gripper_pos = state[6]
                print(f"=== 夹爪状态详情 ===")
                print(f"当前夹爪位置: {gripper_pos:.2f}mm")
                print(f"SpaceMouse按钮: 左={buttons[0]}, 右={buttons[1]}")
                print(f"夹爪控制增量: {accurate[-1]}")
                if gripper_pos <= 1.0:
                    print("⚠️  夹爪可能已完全闭合")
                elif gripper_pos >= 200.0:
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
                    img_arrays.append(rgb_img)  # 第一个摄像头
                    img_arrays_2.append(rgb_img_2)  # 第二个摄像头
                    img_arrays_3.append(rgb_img_3)  # 第三个摄像头
                    state_arrays.append(state_for_recording)  # 使用包含夹爪状态的记录版本
                    if count != 0:
                        action_arrays.append(state_for_recording)  # 使用包含夹爪状态的记录版本
                    count += 1
                    last_save_time = current_time
                    
                    # 每20个数据点显示一次进度（相当于4秒）
                    if count % 20 == 0:
                        print(f"📊 已收集 {count} 个数据点 ({count/data_save_frequency:.1f}秒)")

            # 持续更新三个摄像头和控制窗口的显示
            cv2.imshow('RGB Image - Camera 1', rgb_img)
            cv2.imshow('RGB Image - Camera 2', rgb_img_2)
            cv2.imshow('RGB Image - Camera 3', rgb_img_3)
            cv2.imshow('Control Window', control_image)
            
            # 减少延迟以提高夹爪响应速度
            time.sleep(0.05)  # 从0.1减少到0.05秒
    
    # 关闭OpenCV窗口
    cv2.destroyAllWindows()
    
    # 添加最后一个动作
    if collecting_data and len(state_arrays) > 0:
        action_arrays.append(np.zeros_like(state_arrays[-1]))

        
        expanded_save_path = os.path.expanduser(save_path)
        if not os.path.exists(expanded_save_path):
            os.makedirs(expanded_save_path)
        save_file = os.path.join(expanded_save_path, save_name)
        
        # 转换为numpy数组并打印形状
        img_array = np.stack(img_arrays)
        img_array_2 = np.stack(img_arrays_2)
        img_array_3 = np.stack(img_arrays_3)
        state_array = np.stack(state_arrays)
        action_array = np.stack(action_arrays)
        
        print("=== 最终保存的数组形状信息 ===")
        print(f"img_arrays (Camera 1) shape: {img_array.shape}")
        print(f"img_arrays_2 (Camera 2) shape: {img_array_2.shape}")
        print(f"img_arrays_3 (Camera 3) shape: {img_array_3.shape}")
        print(f"state_arrays shape: {state_array.shape}")
        print(f"action_arrays shape: {action_array.shape}")
        print(f"数据收集时长: {img_array.shape[0]/data_save_frequency:.1f}秒")
        print(f"数据收集频率: {data_save_frequency}Hz")
        print("============================")
        
        np.savez(save_file,
            img_arrays=img_array,
            img_arrays_2=img_array_2,
            img_arrays_3=img_array_3,
            state_arrays=state_array,
            action_arrays=action_array,
            )

        print("Data saved successfully!")
        
        # 生成第一个摄像头的视频文件
        video_name = save_name.replace('.npz', '_cam1.mp4')
        video_file = os.path.join(expanded_save_path, video_name)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_file, fourcc, data_save_frequency, (320, 192))
        
        for frame in img_array:
            video_writer.write(frame)
        
        video_writer.release()
        print(f"摄像头1视频已保存: {video_file}")
        
        # 生成第二个摄像头的视频文件
        video_name_2 = save_name.replace('.npz', '_cam2.mp4')
        video_file_2 = os.path.join(expanded_save_path, video_name_2)
        video_writer_2 = cv2.VideoWriter(video_file_2, fourcc, data_save_frequency, (320, 192))
        
        for frame in img_array_2:
            video_writer_2.write(frame)
        
        video_writer_2.release()
        print(f"摄像头2视频已保存: {video_file_2}")
        
        # 生成第三个摄像头的视频文件
        video_name_3 = save_name.replace('.npz', '_cam3.mp4')
        video_file_3 = os.path.join(expanded_save_path, video_name_3)
        video_writer_3 = cv2.VideoWriter(video_file_3, fourcc, data_save_frequency, (320, 192))
        
        for frame in img_array_3:
            video_writer_3.write(frame)
        
        video_writer_3.release()
        print(f"摄像头3视频已保存: {video_file_3}")
    else:
        print("未收集到数据或取消保存")
    
    # 关闭所有摄像头
    if pipeline_1 is not None:
        pipeline_1.stop()
        print("第一个摄像头已关闭")
    
    if pipeline_2 is not None:
        pipeline_2.stop()
        print("第二个摄像头已关闭")
    
    if pipeline_3 is not None:
        pipeline_3.stop()
        print("第三个摄像头已关闭")

    
    

        
