import numpy as np

from scipy.spatial.transform import Rotation as R
from pyquaternion import Quaternion


def transform_camera_to_marker(vtx, R_cam2marker, t):
    # pc(N,3), rotation mat, translation
    R_marker2cam = R_cam2marker.T  
    vtx_marker = (vtx - t) @ R_marker2cam
    return vtx_marker

def interpolate_se3_euler(pose_start, pose_end, steps=10, euler_order='xyz'):
    """
    在欧拉角空间中插值两个 SE(3) 姿态（7元组：3平移 + 4四元数）
    
    参数:
        pose_start: [7] 起始姿态 (x, y, z, qx, qy, qz, qw)
        pose_end: [7] 结束姿态
        steps: 插值步数
        euler_order: 欧拉角顺序（默认为 'xyz'）
    
    返回:
        poses: List of [7] 插值后的 SE(3) 姿态
    """
    poses = []

    # 拆分平移和旋转
    t0 = pose_start[:3]
    t1 = pose_end[:3]
    r0 = R.from_quat(pose_start[3:])
    r1 = R.from_quat(pose_end[3:])

    # 转为欧拉角
    e0 = r0.as_euler(euler_order)
    e1 = r1.as_euler(euler_order)

    # 插值生成轨迹
    for i in range(steps):
        alpha = i / (steps - 1)
        t = (1 - alpha) * t0 + alpha * t1
        e = (1 - alpha) * e0 + alpha * e1
        quat = R.from_euler(euler_order, e).as_quat()
        poses.append(np.concatenate([t, quat]))

    return poses