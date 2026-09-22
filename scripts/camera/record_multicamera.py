#!/usr/bin/env python3
"""
多摄像头RGB视频捕获工具 - 稳定版
修复了多线程显示问题
"""

import cv2
import numpy as np
import os
import time
import threading
from datetime import datetime
import argparse
from franka_toolkit.paths import VIDEO_DATA_ROOT

class MultiCameraCapture:
    def __init__(self, save_path, target_width=320, target_height=192):
        self.save_path = os.path.expanduser(save_path)
        self.target_width = target_width
        self.target_height = target_height
        self.cameras = []
        self.video_writers = []
        self.capture_threads = []
        self.recording = False
        self.frame_counts = []
        
        # 创建保存目录
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
            print(f"创建保存目录: {self.save_path}")
    
    def detect_cameras(self, max_cameras=20):
        """检测可用的摄像头设备"""
        print("=== 检测可用摄像头 ===")
        available_cameras = []
        
        for i in range(max_cameras):
            try:
                cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
                
                if cap.isOpened():
                    # 设置为RGB格式
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
                    cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)
                    
                    # 尝试读取一帧
                    ret, frame = cap.read()
                    if ret and frame is not None and len(frame.shape) == 3:
                        height, width, channels = frame.shape
                        # 检查是否是彩色图像
                        is_color = channels == 3 and not self._is_grayscale(frame)
                        color_type = "彩色" if is_color else "灰度"
                        print(f"摄像头{i}: ✅ {width}x{height}, {color_type}")
                        available_cameras.append({'index': i, 'is_color': is_color, 
                                                'resolution': (width, height), 'channels': channels})
                    cap.release()
            except:
                pass
        
        # 优先选择彩色摄像头
        color_cameras = [cam for cam in available_cameras if cam['is_color']]
        gray_cameras = [cam for cam in available_cameras if not cam['is_color']]
        
        print(f"\n📊 检测结果:")
        print(f"  彩色摄像头: {[cam['index'] for cam in color_cameras]}")
        print(f"  灰度摄像头: {[cam['index'] for cam in gray_cameras]}")
        
        return available_cameras
    
    def _is_grayscale(self, frame):
        """检查图像是否为灰度图像"""
        if len(frame.shape) != 3:
            return True
        # 检查R,G,B通道是否基本相同
        b, g, r = cv2.split(frame)
        return np.allclose(b, g, atol=10) and np.allclose(g, r, atol=10)
    
    def setup_cameras(self, camera_info_list):
        """设置摄像头"""
        print("\n=== 设置摄像头 ===")
        
        for i, cam_info in enumerate(camera_info_list):
            cam_idx = cam_info['index'] if isinstance(cam_info, dict) else cam_info
            
            try:
                cap = cv2.VideoCapture(cam_idx, cv2.CAP_V4L2)
                
                if cap.isOpened():
                    # 设置参数
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
                    cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    cap.set(cv2.CAP_PROP_FPS, 30)
                    
                    # 测试读取
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        is_color = "彩色" if not self._is_grayscale(frame) else "灰度"
                        print(f"摄像头{cam_idx}: ✅ 初始化成功 ({is_color})")
                        self.cameras.append(cap)
                        self.frame_counts.append(0)
                    else:
                        print(f"摄像头{cam_idx}: ❌ 读取失败")
                        cap.release()
                else:
                    print(f"摄像头{cam_idx}: ❌ 打开失败")
            except Exception as e:
                print(f"摄像头{cam_idx}: ❌ 错误 - {e}")
        
        print(f"成功初始化 {len(self.cameras)} 个摄像头")
        return len(self.cameras) > 0
    
    def setup_video_writers(self):
        """设置视频写入器"""
        print("\n=== 设置视频写入器 ===")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 10.0  # 修改为10Hz
        
        for i in range(len(self.cameras)):
            filename = f"camera_{i}_{timestamp}.mp4"
            filepath = os.path.join(self.save_path, filename)
            
            writer = cv2.VideoWriter(
                filepath, 
                fourcc, 
                fps, 
                (self.target_width, self.target_height)
            )
            
            if writer.isOpened():
                print(f"摄像头{i}: {filename} ✅")
                self.video_writers.append(writer)
            else:
                print(f"摄像头{i}: 创建写入器失败 ❌")
                return False
        
        return True
    
    def capture_camera_thread(self, camera_index):
        """单个摄像头的捕获线程 - 无GUI显示"""
        cap = self.cameras[camera_index]
        writer = self.video_writers[camera_index]
        
        print(f"摄像头{camera_index} 开始录制...")
        
        # 10Hz 帧率控制
        target_interval = 1.0 / 10.0  # 0.1秒间隔
        last_save_time = time.time()
        
        while self.recording:
            ret, frame = cap.read()
            if ret and frame is not None:
                current_time = time.time()
                
                # 只有当距离上次保存超过0.1秒时才保存帧
                if current_time - last_save_time >= target_interval:
                    # 确保是彩色图像（如果是灰度图，转换为3通道）
                    if len(frame.shape) == 2:
                        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    elif len(frame.shape) == 3 and frame.shape[2] == 1:
                        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    
                    # 调整尺寸
                    resized_frame = cv2.resize(frame, (self.target_width, self.target_height))
                    
                    # 写入视频
                    writer.write(resized_frame)
                    self.frame_counts[camera_index] += 1
                    last_save_time = current_time
                    
                    # 每100帧打印一次进度（相当于10秒）
                    if self.frame_counts[camera_index] % 100 == 0:
                        print(f"摄像头{camera_index}: {self.frame_counts[camera_index]} 帧 ({self.frame_counts[camera_index]/10}秒)")
                
                # 小延迟避免CPU过载
                time.sleep(0.001)
                
            else:
                print(f"摄像头{camera_index}: 读取失败")
                break
        
        print(f"摄像头{camera_index} 录制完成，共{self.frame_counts[camera_index]}帧 (时长: {self.frame_counts[camera_index]/10:.1f}秒)")
    
    def start_recording(self):
        """开始录制"""
        if len(self.cameras) == 0:
            print("❌ 没有可用的摄像头")
            return False
        
        if not self.setup_video_writers():
            print("❌ 视频写入器设置失败")
            return False
        
        print(f"\n🎬 开始录制 {len(self.cameras)} 个摄像头...")
        print(f"📁 保存路径: {self.save_path}")
        print(f"📐 视频尺寸: {self.target_width}x{self.target_height}")
        print(f"🎞️  帧率: 10 FPS")
        
        self.recording = True
        
        # 为每个摄像头创建录制线程
        for i in range(len(self.cameras)):
            thread = threading.Thread(target=self.capture_camera_thread, args=(i,))
            thread.daemon = True  # 设置为守护线程
            thread.start()
            self.capture_threads.append(thread)
        
        return True
    
    def stop_recording(self):
        """停止录制"""
        print("\n🛑 停止录制...")
        self.recording = False
        
        # 等待所有线程结束
        for thread in self.capture_threads:
            thread.join(timeout=5)  # 设置超时
        
        # 释放资源
        for cap in self.cameras:
            try:
                cap.release()
            except:
                pass
        
        for writer in self.video_writers:
            try:
                writer.release()
            except:
                pass
        
        # 打印统计信息
        print("\n📊 录制统计:")
        total_frames = 0
        for i, count in enumerate(self.frame_counts):
            duration = count / 10.0  # 10Hz帧率
            print(f"摄像头{i}: {count} 帧 (时长: {duration:.1f}秒)")
            total_frames += count
        total_duration = total_frames / len(self.frame_counts) / 10.0 if self.frame_counts else 0
        print(f"总计: {total_frames} 帧, 平均时长: {total_duration:.1f}秒")

def main():
    parser = argparse.ArgumentParser(description="多摄像头RGB视频捕获工具 - 稳定版")
    parser.add_argument("--save_path", type=str, 
                       default=str(VIDEO_DATA_ROOT / "multicamera"),
                       help="视频保存路径")
    parser.add_argument("--width", type=int, default=320,
                       help="输出视频宽度 (默认: 320)")
    parser.add_argument("--height", type=int, default=192,
                       help="输出视频高度 (默认: 192)")
    parser.add_argument("--cameras", type=str, default="auto",
                       help="指定摄像头索引，用逗号分隔 (例如: '6,12') 或 'auto' 自动检测")
    
    args = parser.parse_args()
    
    print("🎥 多摄像头RGB视频捕获工具 - 稳定版")
    print("=" * 50)
    
    # 创建捕获对象
    capture = MultiCameraCapture(args.save_path, args.width, args.height)
    
    if args.cameras == "auto":
        # 自动模式
        available = capture.detect_cameras()
        if available and capture.setup_cameras(available):
            try:
                duration = int(input("\n输入录制时长（秒，直接回车默认10秒）: ") or "10")
                print(f"准备录制 {duration} 秒...")
                
                if capture.start_recording():
                    print(f"开始录制 {duration} 秒...")
                    time.sleep(duration)
                    capture.stop_recording()
                    print("✅ 录制完成!")
                else:
                    print("❌ 录制启动失败")
            except (KeyboardInterrupt, ValueError):
                print("\n用户中断录制")
        else:
            print("❌ 摄像头设置失败")
    else:
        # 手动指定摄像头
        try:
            camera_indices = [int(x.strip()) for x in args.cameras.split(',')]
            print(f"使用指定摄像头: {camera_indices}")
            
            # 转换为camera_info格式
            camera_info = [{'index': idx, 'is_color': True} for idx in camera_indices]
            
            if capture.setup_cameras(camera_info):
                try:
                    duration = int(input("\n输入录制时长（秒，直接回车默认10秒）: ") or "5")
                    print(f"准备录制 {duration} 秒...")
                    
                    if capture.start_recording():
                        print(f"开始录制 {duration} 秒...")
                        time.sleep(duration)
                        capture.stop_recording()
                        print("✅ 录制完成!")
                    else:
                        print("❌ 录制启动失败")
                except (KeyboardInterrupt, ValueError):
                    print("\n用户中断录制")
            else:
                print("❌ 摄像头设置失败")
                
        except ValueError:
            print("❌ 摄像头索引格式错误")

if __name__ == "__main__":
    main()
