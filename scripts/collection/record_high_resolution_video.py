#!/usr/bin/env python3
"""
单个 RealSense 相机最高分辨率视频录制脚本
使用 1920x1080 分辨率，15 FPS 录制视频
按 's' 开始录制，'t' 停止并保存，'q' 退出
"""
import os
import sys
import time
import numpy as np
import cv2
import pyrealsense2 as rs
import argparse
from datetime import datetime
from franka_toolkit.paths import VIDEO_DATA_ROOT


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


def get_camera_max_resolution(serial_number):
    """获取相机支持的最高分辨率"""
    ctx = rs.context()
    devices = ctx.query_devices()
    
    for device in devices:
        if device.get_info(rs.camera_info.serial_number) == serial_number:
            sensors = device.query_sensors()
            for sensor in sensors:
                # 查找彩色传感器
                if sensor.is_sensor() and "RGB" in sensor.get_info(rs.camera_info.name):
                    profiles = sensor.get_stream_profiles()
                    max_width = 0
                    max_height = 0
                    for profile in profiles:
                        if profile.stream_type() == rs.stream.color:
                            video_profile = profile.as_video_stream_profile()
                            w = video_profile.width()
                            h = video_profile.height()
                            if w * h > max_width * max_height:
                                max_width = w
                                max_height = h
                    if max_width > 0:
                        print(f"相机 {serial_number} 支持的最高分辨率: {max_width}x{max_height}")
                        return max_width, max_height
    
    # 默认返回常见的高分辨率
    print(f"无法检测最高分辨率，使用默认 1920x1080")
    return 1920, 1080


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RealSense 高分辨率视频录制")
    parser.add_argument("--save_path", type=str, 
                        help="保存视频的文件夹路径", 
                        default=str(VIDEO_DATA_ROOT / "high_resolution"))
    parser.add_argument("--video_name", type=str, 
                        help="视频文件名（不含扩展名）", 
                        default="video")
    parser.add_argument("--fps", type=int, 
                        help="视频帧率", 
                        default=15)
    parser.add_argument("--serial", type=str, 
                        help="指定相机序列号（可选）", 
                        default=None)
    
    args = parser.parse_args()
    save_path = os.path.expanduser(args.save_path)
    video_base_name = args.video_name
    fps = args.fps
    
    # 创建保存目录
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print(f"创建保存目录: {save_path}")
    
    # 检测可用的相机
    print("=== 检测RealSense摄像头 ===")
    available_serials = detect_realsense_cameras()
    
    if len(available_serials) == 0:
        print("❌ 未检测到任何 RealSense 设备！")
        sys.exit(1)
    
    # 选择相机
    if args.serial and args.serial in available_serials:
        chosen_serial = args.serial
        print(f"使用指定的相机: {chosen_serial}")
    else:
        chosen_serial = available_serials[0]
        print(f"使用第一个检测到的相机: {chosen_serial}")
    
    # 获取最高分辨率
    max_width, max_height = get_camera_max_resolution(chosen_serial)
    
    # 初始化相机
    print(f"\n正在初始化相机 {chosen_serial}，分辨率: {max_width}x{max_height}, FPS: {fps}")
    pipeline = None
    
    try:
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(chosen_serial)
        config.enable_stream(rs.stream.color, max_width, max_height, rs.format.bgr8, fps)
        
        profile = pipeline.start(config)
        
        # 获取实际的流参数
        color_stream = profile.get_stream(rs.stream.color)
        intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
        actual_width = intrinsics.width
        actual_height = intrinsics.height
        
        print(f"✅ 相机初始化成功！")
        print(f"   实际分辨率: {actual_width}x{actual_height}")
        print(f"   目标帧率: {fps} FPS")
        
        time.sleep(0.5)
        
    except Exception as e:
        print(f"❌ 相机初始化失败: {e}")
        sys.exit(1)
    
    # 创建显示窗口
    window_name = f'RealSense Camera ({chosen_serial})'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 540)  # 显示时缩放以适应屏幕
    
    # 控制窗口
    cv2.namedWindow('Control', cv2.WINDOW_AUTOSIZE)
    control_img = np.zeros((200, 600, 3), dtype=np.uint8)
    cv2.putText(control_img, "Press 's' to START recording", (20, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(control_img, "Press 't' to STOP and SAVE", (20, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
    cv2.putText(control_img, "Press 'q' to QUIT", (20, 140), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow('Control', control_img)
    
    # 录制状态
    recording = False
    frames_buffer = []
    video_counter = 0
    frame_interval = 1.0 / fps  # 每帧之间的时间间隔
    last_save_time = time.time()
    
    print("\n=== 控制说明 ===")
    print("  's' - 开始录制")
    print("  't' - 停止录制并保存视频")
    print("  'q' - 退出程序")
    print(f"\n视频保存路径: {save_path}")
    print(f"视频格式: {video_base_name}_X.mp4 (X 为自动递增编号)")
    print(f"视频分辨率: {actual_width}x{actual_height} @ {fps} FPS")
    print("================\n")
    
    try:
        while True:
            # 获取帧
            try:
                frames = pipeline.wait_for_frames(timeout_ms=1000)
                color_frame = frames.get_color_frame()
                
                if not color_frame:
                    continue
                
                # 转换为 numpy 数组
                color_image = np.asanyarray(color_frame.get_data())
                
                # 如果正在录制，按照帧率保存
                if recording:
                    current_time = time.time()
                    if current_time - last_save_time >= frame_interval:
                        frames_buffer.append(color_image.copy())
                        last_save_time = current_time
                        
                        # 每30帧显示一次进度
                        if len(frames_buffer) % 30 == 0:
                            duration = len(frames_buffer) / fps
                            print(f"📹 录制中: {len(frames_buffer)} 帧 ({duration:.1f}秒)")
                    
                    # 在图像上添加录制标记
                    cv2.circle(color_image, (50, 50), 20, (0, 0, 255), -1)
                    cv2.putText(color_image, "REC", (80, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # 显示图像（缩放以适应屏幕）
                display_image = cv2.resize(color_image, (960, 540))
                cv2.imshow(window_name, display_image)
                
            except Exception as e:
                print(f"获取帧失败: {e}")
                continue
            
            # 检查键盘输入
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('s') and not recording:
                recording = True
                frames_buffer = []
                last_save_time = time.time()
                print(f"\n🎬 开始录制... ({fps} FPS)")
                
                # 更新控制窗口
                control_img = np.zeros((200, 600, 3), dtype=np.uint8)
                cv2.putText(control_img, "RECORDING...", (20, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                cv2.putText(control_img, "Press 't' to STOP and SAVE", (20, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                cv2.imshow('Control', control_img)
                
            elif key == ord('t') and recording:
                recording = False
                
                if len(frames_buffer) > 0:
                    # 生成文件名
                    video_filename = f"{video_base_name}_{video_counter}.mp4"
                    video_path = os.path.join(save_path, video_filename)
                    
                    print(f"\n💾 保存视频中...")
                    print(f"   文件名: {video_filename}")
                    print(f"   帧数: {len(frames_buffer)}")
                    print(f"   时长: {len(frames_buffer)/fps:.2f}秒")
                    
                    # 保存视频
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter(
                        video_path, fourcc, fps, (actual_width, actual_height)
                    )
                    
                    for frame in frames_buffer:
                        video_writer.write(frame)
                    
                    video_writer.release()
                    print(f"✅ 视频已保存: {video_path}\n")
                    
                    video_counter += 1
                else:
                    print("⚠️  缓冲区为空，未保存视频\n")
                
                frames_buffer = []
                
                # 恢复控制窗口
                control_img = np.zeros((200, 600, 3), dtype=np.uint8)
                cv2.putText(control_img, "Press 's' to START recording", (20, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(control_img, "Press 't' to STOP and SAVE", (20, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                cv2.putText(control_img, "Press 'q' to QUIT", (20, 140), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow('Control', control_img)
                
            elif key == ord('q'):
                print("\n退出程序...")
                break
        
    finally:
        # 清理资源
        if pipeline is not None:
            pipeline.stop()
            print("相机已关闭")
        
        cv2.destroyAllWindows()
        print(f"\n总共保存了 {video_counter} 个视频")
        print("程序结束")
