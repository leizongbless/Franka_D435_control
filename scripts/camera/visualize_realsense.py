#!/usr/bin/env python3
"""
RealSense 相机可视化工具
使用 RealSense SDK 实现多相机的实时可视化
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import time
import argparse

class RealSenseCameraVisualizer:
    def __init__(self):
        self.pipelines = []
        self.configs = []
        self.camera_info = []
        
    def detect_cameras(self):
        """检测可用的RealSense相机"""
        ctx = rs.context()
        devices = ctx.query_devices()
        
        print(f"检测到 {len(devices)} 个RealSense设备:")
        for i, device in enumerate(devices):
            serial_number = device.get_info(rs.camera_info.serial_number)
            name = device.get_info(rs.camera_info.name)
            print(f"  设备 {i}: {name} (序列号: {serial_number})")
            
        return [device.get_info(rs.camera_info.serial_number) for device in devices]
    
    def initialize_camera(self, serial_number=None, width=640, height=480, fps=30):
        """初始化指定序列号的相机"""
        pipeline = rs.pipeline()
        config = rs.config()
        
        if serial_number:
            config.enable_device(serial_number)
            
        # 启用彩色流
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        
        # 启用深度流（可选）
        try:
            config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
            has_depth = True
        except:
            has_depth = False
            print(f"相机 {serial_number} 不支持深度流")
        
        try:
            profile = pipeline.start(config)
            device = profile.get_device()
            name = device.get_info(rs.camera_info.name)
            actual_serial = device.get_info(rs.camera_info.serial_number)
            
            self.pipelines.append(pipeline)
            self.configs.append(config)
            self.camera_info.append({
                'serial': actual_serial,
                'name': name,
                'width': width,
                'height': height,
                'fps': fps,
                'has_depth': has_depth
            })
            
            print(f"相机初始化成功: {name} ({actual_serial})")
            return True
            
        except Exception as e:
            print(f"相机初始化失败 {serial_number}: {e}")
            return False
    
    def initialize_all_cameras(self, width=640, height=480, fps=30):
        """初始化所有检测到的相机"""
        serials = self.detect_cameras()
        
        for serial in serials:
            self.initialize_camera(serial, width, height, fps)
            
        print(f"成功初始化 {len(self.pipelines)} 个相机")
    
    def get_frames(self):
        """获取所有相机的帧"""
        frames_data = []
        
        for i, pipeline in enumerate(self.pipelines):
            try:
                frames = pipeline.wait_for_frames(timeout_ms=100)
                
                # 获取彩色帧
                color_frame = frames.get_color_frame()
                color_image = None
                if color_frame:
                    color_image = np.asanyarray(color_frame.get_data())
                
                # 获取深度帧（如果有）
                depth_frame = frames.get_depth_frame()
                depth_image = None
                if depth_frame and self.camera_info[i]['has_depth']:
                    depth_image = np.asanyarray(depth_frame.get_data())
                    # 转换深度图为可视化格式
                    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
                else:
                    depth_colormap = None
                
                frames_data.append({
                    'color': color_image,
                    'depth': depth_colormap,
                    'camera_info': self.camera_info[i]
                })
                
            except Exception as e:
                # 创建错误帧
                error_frame = np.zeros((self.camera_info[i]['height'], self.camera_info[i]['width'], 3), dtype=np.uint8)
                cv2.putText(error_frame, f"Camera {i} Error", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                frames_data.append({
                    'color': error_frame,
                    'depth': None,
                    'camera_info': self.camera_info[i]
                })
        
        return frames_data
    
    def visualize(self, show_depth=True, display_info=True):
        """开始可视化相机"""
        if not self.pipelines:
            print("没有可用的相机!")
            return
        
        print("开始相机可视化...")
        print("按键说明:")
        print("  'q' - 退出程序")
        print("  'c' - 切换彩色/深度显示")
        print("  's' - 保存当前帧")
        print("  'i' - 显示/隐藏相机信息")
        
        show_depth_mode = show_depth
        show_info = display_info
        frame_count = 0
        
        while True:
            frames_data = self.get_frames()
            
            # 处理键盘输入
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                show_depth_mode = not show_depth_mode
                print(f"深度显示: {'开启' if show_depth_mode else '关闭'}")
            elif key == ord('i'):
                show_info = not show_info
                print(f"信息显示: {'开启' if show_info else '关闭'}")
            elif key == ord('s'):
                self.save_frames(frames_data, frame_count)
            
            # 显示每个相机的画面
            for i, frame_data in enumerate(frames_data):
                camera_info = frame_data['camera_info']
                
                # 彩色图像
                if frame_data['color'] is not None:
                    color_img = frame_data['color'].copy()
                    
                    if show_info:
                        # 添加相机信息
                        info_text = [
                            f"Camera {i}: {camera_info['name'][:20]}",
                            f"Serial: {camera_info['serial']}",
                            f"Resolution: {camera_info['width']}x{camera_info['height']}",
                            f"FPS: {camera_info['fps']}"
                        ]
                        
                        for j, text in enumerate(info_text):
                            cv2.putText(color_img, text, (10, 25 + j*20), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    
                    window_name = f"Camera {i} - Color"
                    cv2.imshow(window_name, color_img)
                    
                    # 设置窗口位置
                    cv2.moveWindow(window_name, i * 320, 0)
                
                # 深度图像（如果有且启用）
                if show_depth_mode and frame_data['depth'] is not None:
                    depth_img = frame_data['depth'].copy()
                    
                    if show_info:
                        cv2.putText(depth_img, f"Camera {i} - Depth", (10, 25), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    window_name_depth = f"Camera {i} - Depth"
                    cv2.imshow(window_name_depth, depth_img)
                    
                    # 设置窗口位置
                    cv2.moveWindow(window_name_depth, i * 320, 500)
            
            frame_count += 1
        
        print("停止可视化...")
        self.cleanup()
    
    def save_frames(self, frames_data, frame_count):
        """保存当前帧"""
        timestamp = int(time.time())
        
        for i, frame_data in enumerate(frames_data):
            if frame_data['color'] is not None:
                color_filename = f"camera_{i}_color_{timestamp}_{frame_count}.jpg"
                cv2.imwrite(color_filename, frame_data['color'])
                print(f"保存彩色图像: {color_filename}")
            
            if frame_data['depth'] is not None:
                depth_filename = f"camera_{i}_depth_{timestamp}_{frame_count}.jpg"
                cv2.imwrite(depth_filename, frame_data['depth'])
                print(f"保存深度图像: {depth_filename}")
    
    def cleanup(self):
        """清理资源"""
        for pipeline in self.pipelines:
            pipeline.stop()
        
        cv2.destroyAllWindows()
        print("资源清理完成")

def main():
    parser = argparse.ArgumentParser(description='RealSense 相机可视化工具')
    parser.add_argument('--serial', type=str, help='指定相机序列号（可选）')
    parser.add_argument('--width', type=int, default=640, help='图像宽度（默认：640）')
    parser.add_argument('--height', type=int, default=480, help='图像高度（默认：480）')
    parser.add_argument('--fps', type=int, default=30, help='帧率（默认：30）')
    parser.add_argument('--no-depth', action='store_true', help='不显示深度图像')
    parser.add_argument('--no-info', action='store_true', help='不显示相机信息')
    
    args = parser.parse_args()
    
    visualizer = RealSenseCameraVisualizer()
    
    try:
        if args.serial:
            # 初始化指定序列号的相机
            success = visualizer.initialize_camera(args.serial, args.width, args.height, args.fps)
            if not success:
                print(f"无法初始化相机 {args.serial}")
                return
        else:
            # 初始化所有相机
            visualizer.initialize_all_cameras(args.width, args.height, args.fps)
        
        # 开始可视化
        visualizer.visualize(show_depth=not args.no_depth, display_info=not args.no_info)
        
    except KeyboardInterrupt:
        print("收到中断信号，正在退出...")
    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        visualizer.cleanup()

if __name__ == "__main__":
    main()