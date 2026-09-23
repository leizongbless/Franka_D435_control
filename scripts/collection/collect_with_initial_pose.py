# python -m franka_toolkit.scripts.collection.collect_with_initial_pose --save_path data/raw --save_name episode.npz
# state: [    0.38169    -0.21566     0.14259     -2.9661   -0.051079     -3.0458      204.01]
# state: [    0.37756    -0.21593     0.10673     -2.9417   -0.080626     -3.0375      204.01]
# state: [    0.37815    -0.21616     0.10713     -2.9411   -0.079579     -3.0374       161.4]
# state: [      0.396     -0.2223     0.20242     -2.9755   -0.019762     -3.0469      160.59]
# state: [    0.35617     0.16169     0.18256     -3.0127   -0.052295     -2.9789      159.74]
# state: [    0.36081     0.16483     0.11082     -3.0699   -0.076943     -2.9734      159.73]
# state: [    0.36082     0.16484     0.11082     -3.0699   -0.076903     -2.9734      201.77]
# state: [    0.37807     0.16694     0.25322     -3.0304    0.012119     -2.9819         199]
# state: [    0.49313    -0.10567     0.10199     -3.0516    0.031173     -3.0667     0.24448]push block
import os, sys, time, copy
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

from scipy.spatial.transform import Rotation as R
from pyquaternion import Quaternion

from franka_toolkit.robot import Franka, FrankaEnv
from franka_toolkit.input import SpaceMouseExpert
from franka_toolkit.input.device import Spacemouse
from franka_toolkit.geometry import transform_camera_to_marker, interpolate_se3_euler
from franka_toolkit.camera import voxel_downsampling, furthest_point_sampling, PointNetEncoderXYZ, \
    fast_furthest_point_sampling

from ultralytics import YOLO

import argparse


def setup_advanced_visualization():
    """设置更高级的3D可视化"""
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='3D Point Cloud', width=1280, height=720)

    # 设置渲染选项
    render_option = vis.get_render_option()
    render_option.point_size = 1.5
    render_option.background_color = np.array([0.1, 0.1, 0.1])  # 深色背景
    render_option.light_on = True

    # 添加坐标系
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    vis.add_geometry(coordinate_frame)

    return vis


def depth_to_pointcloud(depth_image, rgb_image=None, fx=615.0, fy=615.0, cx=320.0, cy=240.0):
    """
    将深度图像转换为点云
    fx, fy: 相机焦距
    cx, cy: 相机主点
    """
    if len(depth_image.shape) == 3:
        depth_image = depth_image.squeeze()
    height, width = depth_image.shape

    # 创建网格
    u = np.arange(width)
    v = np.arange(height)
    u, v = np.meshgrid(u, v)

    # 将像素坐标转换为3D坐标
    z = depth_image
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    # 创建点云
    points = np.stack([x, y, z], axis=-1).reshape(-1, 3)

    # 移除无效点（深度为0的点）
    valid_mask = z.reshape(-1) > 0
    points = points[valid_mask]

    # 创建Open3D点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # 如果有RGB图像，添加颜色
    if rgb_image is not None:
        colors = rgb_image.reshape(-1, 3)[valid_mask] / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="save to npz")
    parser.add_argument("--save_path", type=str, required=True, help="Path to the folder save files")
    parser.add_argument("--save_name", type=str, required=True,
                        help="Path to the output datax.npz directory, x is belong to int such as 0,1,2...")

    args = parser.parse_args()
    save_path = args.save_path
    save_name = args.save_name
    try:
        if save_name[5] != '.':
            x = int(save_name[4:6])
        else:
            x = int(save_name[4])
        print("保存文件名:", save_name, end=" ")
        print("x=", x)
    except:
        print("请输入正确的文件名:Path to the output datax.npz directory, x is belong to int such as 0,1,2...")
        exit()

    accurate = [0.04, 0.1, 20]
    env = FrankaEnv(accurate)
    time.sleep(1)

    img_arrays = []
    state_arrays = []
    point_cloud_arrays = []
    depth_arrays = []
    action_arrays = []  # change into using delta action + gripper

    count = 0

    env.reset()
    obs = np.zeros((2048, 3))
    frame = env.get_obs()
    rgb_img = frame["color"].copy()
    init_state = env.get_euler_action()
    print(f"init_state = {init_state}")

    # 点云可视化
    vis = setup_advanced_visualization()
    pointcloud_added = False
    pcd = o3d.geometry.PointCloud()

    # 创建一个OpenCV窗口来接收键盘输入
    cv2.namedWindow('RGB Image', cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('Control Window', cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow('RGB Image', 0, 0)
    cv2.moveWindow('Control Window', 650, 0)  # 将控制窗口移到RGB图像旁边

    # 在窗口中显示提示信息
    control_image = np.zeros((300, 600, 3), dtype=np.uint8)
    cv2.putText(control_image, "Press 's' to START data collection", (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(control_image, "Press 'q' to STOP", (20, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(control_image, "Press 'r' to RESET", (20, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.imshow('RGB Image', rgb_img)
    cv2.imshow('Control Window', control_image)

    # 新增变量用于控制数据收集状态
    collecting_data = False
    print("提示: 按下 's' 键开始收集数据，按下 'q' 键停止收集并保存数据")
    pose7 = [0.4789273668490163, 0.04029894306768502, 0.29108792750731516, -0.0714744170000644, 0.9970854964223268,
             -0.019703623557105518, 0.017991324505518727]
    resp = requests.post(f"http://192.168.17.123:5000/pose", json={"arr": pose7}, timeout=10)
    with Spacemouse(deadzone=0.3) as sm:
        while True:
            frame = env.get_obs()
            rgb_img = frame["color"].copy()
            mouse_action = sm.get_motion_state_transformed()
            buttons = [int(sm.is_button_pressed(0)), int(sm.is_button_pressed(1))]
            # print(f'mouse_action : {mouse_action}, buttons : {buttons}')

            # 使用OpenCV检查键盘输入
            key = cv2.waitKey(1) & 0xFF  # 增加等待时间以提高响应性

            # 检查是否按下 's' 键开始收集数据
            if key == ord('s') and not collecting_data:
                collecting_data = True
                print("开始收集数据...")
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
                    np.savez(save_file,
                             img_arrays=np.stack(img_arrays),
                             state_arrays=np.stack(state_arrays),
                             point_cloud_arrays=np.stack(point_cloud_arrays),
                             depth_arrays=np.stack(depth_arrays),
                             action_arrays=np.stack(action_arrays),
                             )

                    print("Data saved successfully!")

                    load_check = np.load(save_file, allow_pickle=True)
                    print(load_check.files)
                    print(f"img_arrays shape: {load_check['img_arrays'].shape}")
                    print(f"state_arrays shape: {load_check['state_arrays'].shape}")
                    print(f"point_cloud_arrays shape: {load_check['point_cloud_arrays'].shape}")
                    print(f"depth_arrays shape: {load_check['depth_arrays'].shape}")
                    print(f"action_arrays shape: {load_check['action_arrays'].shape}")
                    # 更新file_name
                    x += 1
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
                point_cloud_arrays = []  # change into using delta action + gripper
                depth_arrays = []
                action_arrays = []
                count = 0
                print(f"下一次数据将保存到: {expanded_save_path}/{save_name}")
            # 检查是否按下 'r' 键重置机器
            if key == ord('r') and not collecting_data:
                # env.reset()
                pose7=[0.4789273668490163,0.04029894306768502,0.29108792750731516,-0.0714744170000644,0.9970854964223268,-0.019703623557105518,0.017991324505518727]
                resp = requests.post(f"http://192.168.17.123:5000/pose", json={"arr": pose7}, timeout=10)
                resp.raise_for_status()
                print("机器已重置")

            # 按到n直接舍弃这组数据
            if key == ord('n') and collecting_data:
                print("停止收集数据...")
                # 清空数据列表
                img_arrays = []
                state_arrays = []
                point_cloud_arrays = []  # change into using delta action + gripper
                depth_arrays = []
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

            # get obs and action
            frame = env.get_obs()

            rgb_img = frame["color"].copy()
            pointcloud = frame["pointcloud"].copy()
            depth_img = frame["depth"].copy()
            state = env.get_euler_action(if_delta=True)
            if buttons[0] == 1:
                state[6] -= accurate[-1]
            if buttons[1] == 1:
                state[6] += accurate[-1]
            if key == ord('p'):
                print(f"state: {state}")
            if key == ord('1'):
                move = np.array([0.38169, -0.21566, 0.14259, -2.9661, -0.051079, -3.0458, 204.01])
                env.agent_absolute_action(move)
            if key == ord('2'):
                move = np.array([0.37821, -0.21591, 0.10376, -2.9623, -0.072111, -3.041, 204.01])
                env.agent_absolute_action(move)
            if key == ord('3'):
                move = np.array([0.37815, -0.21616, 0.10713, -2.9411, -0.079579, -3.0374, 161.4])
                env.agent_absolute_action(move)
            if key == ord('4'):
                move = np.array([0.396, -0.2223, 0.20242, -2.9755, -0.019762, -3.0469, 160.59])
                env.agent_absolute_action(move)
            if key == ord('5'):
                move = np.array([0.35617, 0.16169, 0.18256, -3.0127, -0.052295, -2.9789, 159.74])
                env.agent_absolute_action(move)
            if key == ord('6'):
                move = np.array([0.36081, 0.16483, 0.11082, -3.0699, -0.076943, -2.9734, 159.73])
                env.agent_absolute_action(move)
            if key == ord('7'):
                move = np.array([0.36082, 0.16484, 0.11082, -3.0699, -0.076903, -2.9734, 201.77])
                env.agent_absolute_action(move)
            if key == ord('8'):
                move = np.array([0.37807, 0.16694, 0.25322, -3.0304, 0.012119, -2.9819, 199])
                env.agent_absolute_action(move)
            if key == ord('9'):
                move = np.array([0.49313, -0.10567, 0.10199, -3.0516, 0.031173, -3.0667, 0.24448])
                env.agent_absolute_action(move)
            current_pcd = depth_to_pointcloud(depth_img, rgb_img,
                                              fx=461.262, fy=461.109,
                                              cx=296.000, cy=261.988)
            # 更新可视化
            pcd.points = current_pcd.points
            pcd.colors = current_pcd.colors
            if not pointcloud_added:
                vis.add_geometry(pcd)
                pointcloud_added = True
            else:
                vis.update_geometry(pcd)

            vis.poll_events()
            vis.update_renderer()

            # 只有在开始收集数据后才保存数据
            if collecting_data:
                img_arrays.append(rgb_img)
                state_arrays.append(state)
                point_cloud_arrays.append(pointcloud)
                depth_arrays.append(depth_img)
                if count != 0:
                    action_arrays.append(state)
                count += 1

            # 持续更新两个窗口的显示
            cv2.imshow('RGB Image', rgb_img)
            cv2.imshow('Control Window', control_image)

            time.sleep(0.1)

    # 关闭OpenCV窗口
    cv2.destroyAllWindows()
    vis.destroy_window()

    # 添加最后一个动作
    if collecting_data and len(state_arrays) > 0:
        action_arrays.append(np.zeros_like(state_arrays[-1]))

        expanded_save_path = os.path.expanduser(save_path)
        if not os.path.exists(expanded_save_path):
            os.makedirs(expanded_save_path)
        save_file = os.path.join(expanded_save_path, save_name)
        np.savez(save_file,
                 img_arrays=np.stack(img_arrays),
                 state_arrays=np.stack(state_arrays),
                 point_cloud_arrays=np.stack(point_cloud_arrays),
                 depth_arrays=np.stack(depth_arrays),
                 action_arrays=np.stack(action_arrays),
                 )

        print("Data saved successfully!")

        load_check = np.load(save_file, allow_pickle=True)
        print(load_check.files)
        print(f"img_arrays shape: {load_check['img_arrays'].shape}")
        print(f"state_arrays shape: {load_check['state_arrays'].shape}")
        print(f"point_cloud_arrays shape: {load_check['point_cloud_arrays'].shape}")
        print(f"depth_arrays shape: {load_check['depth_arrays'].shape}")
        print(f"action_arrays shape: {load_check['action_arrays'].shape}")
    else:
        print("未收集到数据或取消保存")




