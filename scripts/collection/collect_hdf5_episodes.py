import os,sys,time,copy
import open3d as o3d
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
import h5py

from scipy.spatial.transform import Rotation as R
from pyquaternion import Quaternion

from franka_toolkit.robot import Franka, FrankaEnv
from franka_toolkit.input import SpaceMouseExpert
from franka_toolkit.geometry import transform_camera_to_marker, interpolate_se3_euler
from franka_toolkit.camera import voxel_downsampling, furthest_point_sampling, PointNetEncoderXYZ, fast_furthest_point_sampling, RSCapture
from franka_toolkit.paths import RAW_DATA_ROOT

from ultralytics import YOLO

class DataRecorder:
    def __init__(self, filename="teleoperation_data.h5"):
        self.filename = filename
        self.episode_count = 0
        self.current_episode_data = {
            'timestamps': [],
            'color_images': [],
            'depth_images': [],
            'eef_poses': [],
            'actions': [],
            'buttons': []
        }
        
    def start_new_episode(self):
        """开始新的episode录制"""
        self.current_episode_data = {
            'timestamps': [],
            'color_images': [],
            'depth_images': [],
            'eef_poses': [],
            'actions': [],
            'buttons': [],
            'camera2robot' : []
        }
        print(f"开始录制 episode {self.episode_count + 1}")
        
    def add_frame(self, frame, eef_action,camera_2_robotbase_vec7):
        """添加一帧数据"""
        timestamp = time.time()
        
        self.current_episode_data['timestamps'].append(timestamp)
        self.current_episode_data['color_images'].append(frame["color"].copy())
        self.current_episode_data['depth_images'].append(frame["depth"].copy())
        self.current_episode_data['eef_poses'].append(eef_action.copy())
        self.current_episode_data['camera2robot'].append(camera_2_robotbase_vec7.copy())
        
    def save_episode(self):
        """保存当前episode到HDF5文件"""
        if len(self.current_episode_data['timestamps']) == 0:
            print("没有数据可保存")
            return False
            
        try:
            with h5py.File(self.filename, 'a') as f:
                episode_name = f"episode_{self.episode_count:04d}"
                
                # 创建episode组
                episode_group = f.create_group(episode_name)
                
                # 保存元数据
                episode_group.attrs['recording_time'] = datetime.datetime.now().isoformat()
                episode_group.attrs['num_frames'] = len(self.current_episode_data['timestamps'])
                episode_group.attrs['duration'] = (self.current_episode_data['timestamps'][-1] - 
                                                 self.current_episode_data['timestamps'][0])
                
                # 保存数据
                episode_group.create_dataset('timestamps', 
                                           data=np.array(self.current_episode_data['timestamps']))
                episode_group.create_dataset('eef_poses', 
                                           data=np.array(self.current_episode_data['eef_poses']))
                
                # 保存图像数据（压缩存储）
                color_images = np.array(self.current_episode_data['color_images'])
                depth_images = np.array(self.current_episode_data['depth_images'])
                
                episode_group.create_dataset('color_images', 
                                           data=color_images,
                                           compression="gzip",
                                           compression_opts=9)
                episode_group.create_dataset('depth_images', 
                                           data=depth_images,
                                           compression="gzip", 
                                           compression_opts=9)
                
                # 保存数据形状信息
                episode_group.attrs['color_shape'] = color_images.shape
                episode_group.attrs['depth_shape'] = depth_images.shape
                episode_group.attrs['eef_pose_shape'] = np.array(self.current_episode_data['eef_poses']).shape
                episode_group.attrs['camera2robot'] = np.array(self.current_episode_data['camera2robot'][0])
            print(f"Episode {self.episode_count} 保存成功，共 {len(self.current_episode_data['timestamps'])} 帧")
            self.episode_count += 1
            return True
            
        except Exception as e:
            print(f"保存失败: {e}")
            return False
    
    def get_recording_status(self):
        """获取录制状态"""
        return {
            'current_frames': len(self.current_episode_data['timestamps']),
            'total_episodes': self.episode_count
        }

def user_confirmation(prompt):
    """获取用户确认"""
    while True:
        response = input(f"{prompt} (y/n): ").lower().strip()
        if response in ['y', 'yes']:
            return True
        elif response in ['n', 'no']:
            return False
        else:
            print("请输入 y 或 n")

def main():
    env = FrankaEnv()  
    time.sleep(1)
    
    # 初始化数据记录器
    recorder = DataRecorder(str(RAW_DATA_ROOT / "teleoperation.h5"))
    camera_2_robotbase_vec7 = np.array([0.69195025 ,  0.01557747 ,  0.58394862 , -0.23129771 , 0.66970116 , 0.65979304 ,-0.2503495])# qw qx qy qz
    try:
        while True:
            # 开始新的episode
            env.reset()
            time.sleep(5)
            
            print("开始录制...")
            print("按 'q' 结束当前episode录制")
            print("按 'ESC' 退出程序")
            recorder.start_new_episode()
            
            recording = True
            while recording:
                frame = env.get_obs()
                
                if frame is not None:
                    # 获取当前状态
                    eef_action = env.get_action(if_delta=False)
                    # mouse_action, buttons = env.mouse.get_action()
                    
                    # 记录数据
                    recorder.add_frame(frame, eef_action,camera_2_robotbase_vec7)
                    
                    # 显示RGB图像和状态信息
                    rgb_img = frame["color"].copy()
                    status = recorder.get_recording_status()
                    
                    # 在图像上显示录制状态
                    cv2.putText(rgb_img, f"Episode: {recorder.episode_count + 1}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(rgb_img, f"Frames: {status['current_frames']}", (10, 70), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(rgb_img, "Press 'q' to stop recording", (10, 110), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    cv2.imshow('Teleoperation Recording', rgb_img)
                    
                    # 控制机器人
                    # env.move_robot(mouse_action, buttons)
                    
                    # 检查按键
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):  # 结束当前episode
                        recording = False
                    elif key == 27:  # ESC键退出程序
                        print("用户中断程序")
                        return
                
                time.sleep(0.02)
            
            # 结束当前episode录制
            print(f"当前episode录制结束，共 {recorder.get_recording_status()['current_frames']} 帧")
            
            # 询问是否保存
            if user_confirmation("是否保存当前episode?"):
                if recorder.save_episode():
                    print("保存成功!")
                else:
                    print("保存失败!")
            
            # 询问是否继续录制下一个episode
            if not user_confirmation("是否开始录制下一个episode?"):
                print("录制结束")
                break
                
    except KeyboardInterrupt:
        print("程序被用户中断")
    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        cv2.destroyAllWindows()
        print(f"总共录制了 {recorder.episode_count} 个episode")
        print(f"数据保存在: {recorder.filename}")

if __name__ == "__main__":
    main()
