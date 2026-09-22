import time

import requests
import torch
import numpy as np

from scipy.spatial.transform import Rotation as R, Slerp
from pyquaternion import Quaternion


RESET_POSE = [0.7417, 0.0051162, 0.17205, -3.1208, 0.13166, 3.1187]
CARTESIAN_RESET_POSE = np.array(
    [0.7417, 0.0051162, 0.17205, -3.1208, 0.13166, 3.1187, 0.24113],
    dtype=float,
)

# RESET_JOINTS = [    0.12081,    -0.84518,    0.043024,     -2.3642,     0.01638,      1.4503,     -2.3614]
RESET_JOINTS = [  0.0026492,     0.39198,    0.055768,     -1.5627,   -0.039242,      2.0566,     -2.2338]

# RANGE_LOW = [-0.1, -0.1, -0.3, -0.33, -0.33, -0.3333]
# RANGE_LOW = [-0.4, -0.3, -0.3, -3.14, -3.14, -3.14]
# RANGE_HIGH = [0.3, 0.3, 0.3, 3.14, 3.14, 3.14]

RANGE_LOW = [-0.5, -0.4, -0.4, -3.14, -3.14, -3.14]
RANGE_HIGH = [0.4, 0.4, 0.4, 3.14, 3.14, 3.14]
def quat_2_euler(quat):
    """calculates and returns: yaw, pitch, roll from given quaternion"""
    return R.from_quat(quat).as_euler("xyz")
def euler_2_quat(xyz):
    yaw, pitch, roll = xyz
    yaw = np.pi - yaw
    yaw_matrix = np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0, 0, 1.0],
        ]
    )
    pitch_matrix = np.array(
        [
            [np.cos(pitch), 0.0, np.sin(pitch)],
            [0.0, 1.0, 0.0],
            [-np.sin(pitch), 0, np.cos(pitch)],
        ]
    )
    roll_matrix = np.array(
        [
            [1.0, 0, 0], 
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)],
        ]
    )
    rot_mat = yaw_matrix.dot(pitch_matrix.dot(roll_matrix))
    return Quaternion(matrix=rot_mat).elements

class Franka:
    def __init__(
        self,
        action_scales=[0.04, 0.1, 20],
        server_url="http://192.168.1.11:8000/",
        gripper_min=0.0,
        gripper_max=255.0,
    ):


        # get state
        self.url = server_url.rstrip("/") + "/"
        self.api_v1 = self.url.rstrip("/").endswith(":8000") or "/api/v1" in self.url
        self.command_duration_s = 0.04

        self.resetpos = np.concatenate([RESET_POSE[:3], euler_2_quat(RESET_POSE[3:])])
        self.currpos = self.resetpos.copy() # [x,y,z,qx,qy,qz,qw]
        
        self.cur_joints = np.zeros((7),)

        self.q = np.zeros((7,))
        self.dq = np.zeros((7,))
        self.gripper_min = float(gripper_min)
        self.gripper_max = float(gripper_max)
        if self.gripper_min >= self.gripper_max:
            raise ValueError("gripper_min must be smaller than gripper_max")
        self.curr_gripper_pos = 0 # server-side raw position, normally 0-255
        self.next_gripper_pos = self.curr_gripper_pos
        self.gripper_binary_state = 0
        # Optional telemetry exposed by the current upper-computer service.
        self.force = np.zeros(3, dtype=float)
        self.jacobian = np.zeros((6, 7), dtype=float)
        self.torque = np.zeros(3, dtype=float)
        self.vel = np.zeros(6, dtype=float)
        self.gripper_effort = None

        self._update_currpos()

        self.xyz_low = (np.array(RESET_POSE) + np.array(RANGE_LOW))[:3]
        self.xyz_high = (np.array(RESET_POSE) + np.array(RANGE_HIGH))[:3]
        self.euler_low = (np.array(RESET_POSE) + np.array(RANGE_LOW))[3:]
        self.euler_high = (np.array(RESET_POSE) + np.array(RANGE_HIGH))[3:]

        self.action_scales = action_scales

        

    def clip_safety_box(self, pose):
        pose = np.asarray(pose, dtype=float).copy()
        pose[:3] = np.clip(pose[:3], self.xyz_low, self.xyz_high)
        euler = R.from_quat(pose[3:]).as_euler("xyz")
        euler = (euler + np.pi) % (2 * np.pi) - np.pi
        pose[3:] = R.from_euler("xyz", euler).as_quat()

        return pose

    def clip_safety_euler_box(self, pose):
    
        pose[:3] = np.clip(pose[:3], self.xyz_low, self.xyz_high)
        euler = R.from_quat(pose[3:]).as_euler("xyz")
        # Clip first euler angle separately due to discontinuity from pi to -pi
        sign = np.sign(euler[0])
        euler[0] = sign * (
                np.clip(
                    np.abs(euler[0]),
                    self.euler_low[0],
                    self.euler_high[0],
                )
        )
        euler[1:] = np.clip(
                euler[1:], self.euler_low[1:], self.euler_high[1:]
            )

        new_pos = np.zeros(6, dtype=np.float32)
        new_pos[:3] = pose[:3]
        new_pos[3:] = euler

        return new_pos

    def agent_move(self, xyz_delta, euler_delta, gripper):
        self.nextpos = self.currpos.copy()
        xyz_delta = xyz_delta if isinstance(xyz_delta, np.ndarray) else xyz_delta.numpy()
        self.nextpos[:3] = self.nextpos[:3] + xyz_delta
        euler_delta = euler_delta if isinstance(euler_delta, np.ndarray) else euler_delta.numpy()
        next_euler_pos = (
            R.from_euler('xyz', (euler_delta)) *
            R.from_quat(self.currpos[3:])
            ).as_quat()
        
        self.nextpos[3:] = next_euler_pos
        
        self.delta_xyz = np.array(xyz_delta)*self.action_scales[0]
        self.delta_euler = np.array(euler_delta)*self.action_scales[1]
        gripper = gripper.detach().cpu().numpy() if isinstance(gripper, torch.Tensor) else gripper
        arr = 1.*gripper
        self.next_gripper_pos = arr 

        data = {"gripper_pos": arr}
        headers = {"Content-Type":"application/json"}
        
        requests.post(self.url + "move_gripper", headers= headers, json=data)
        self._send_pos_command(self.clip_safety_box(self.nextpos))
    
        self._update_currpos()

    def agent_absolute_move(self, xyz_absolute, euler_absolute, gripper):
        self.nextpos = self.currpos.copy()
        
        # 处理位置输入（绝对坐标）
        xyz_absolute = xyz_absolute if isinstance(xyz_absolute, np.ndarray) else xyz_absolute.numpy()
        self.nextpos[:3] = xyz_absolute  # 直接使用绝对坐标
        
        # 处理姿态输入（绝对欧拉角）
        euler_absolute = euler_absolute if isinstance(euler_absolute, np.ndarray) else euler_absolute.numpy()
        
        # 将绝对欧拉角转换为四元数
        next_euler_pos = R.from_euler('xyz', euler_absolute).as_quat()
        self.nextpos[3:] = next_euler_pos
        
        # 计算实际执行的delta值（用于记录或监控）
        self.delta_xyz = (xyz_absolute - self.currpos[:3]) * self.action_scales[0]
        self.delta_euler = (euler_absolute - R.from_quat(self.currpos[3:]).as_euler('xyz')) * self.action_scales[1]
        
        # 处理夹爪控制（保持不变）
        gripper = gripper.detach().cpu().numpy() if isinstance(gripper, torch.Tensor) else gripper
        arr = 1. * gripper
        self.next_gripper_pos = arr 

        data = {"gripper_pos": arr}
        headers = {"Content-Type": "application/json"}
        
        requests.post(self.url + "move_gripper", headers=headers, json=data)
        self._send_pos_command(self.clip_safety_box(self.nextpos))
        # self._send_pos_command(self.nextpos)
        
        self._update_currpos()

    def move(self, xyzrpy, gripper_flag):
        xyz_delta = xyzrpy[:3]
        euler_delta = xyzrpy[3:]

        if np.any(np.abs(np.asarray(xyzrpy)) > 1e-6):
            print(f"[机器人调试] action={np.asarray(xyzrpy)} gripper={gripper_flag}")

        self.nextpos = self.currpos.copy()
        self.nextpos[:3] = self.nextpos[:3] + self.action_scales[0] * np.array(xyz_delta)
        
        self.delta_xyz = np.array(xyz_delta)*self.action_scales[0]
        self.delta_euler = np.array(euler_delta)*self.action_scales[1]
        
        
        next_euler_pos = (
            R.from_euler('xyz', (np.array(euler_delta)*self.action_scales[1])) *
            R.from_quat(self.currpos[3:])
            ).as_quat()
        self.nextpos[3:] = next_euler_pos

        
        gripper_commanded = gripper_flag in (1, -1)
        if gripper_commanded:
            # Button input is a direction. Accumulate it from the current
            # server position so holding a button produces continuous motion.
            self.move_gripper_delta(
                gripper_flag * self.action_scales[-1],
                update_state=False,
            )
        # The v1 command manager permits only one active command. A gripper
        # request and a Cartesian request cannot be accepted in the same
        # control tick, so give the held gripper button priority for that tick.
        response = None
        if not (self.api_v1 and gripper_commanded):
            response = self._send_pos_command(self.clip_safety_box(self.nextpos))
        if np.any(np.abs(np.asarray(xyzrpy)) > 1e-6):
            print(f"[机器人调试] pose status={response.status_code if response is not None else 'deferred'}")

        self._update_currpos()

    def reset(self):
        reset_gripper = 222
        for _ in range(4):
            self._send_gripper_command(reset_gripper,mode="continuous")

        reset_target = RESET_JOINTS
        data = {"target": reset_target}
        
        response = requests.post(
            self.url + "set_joint_target", 
            headers={"Content-Type": "application/json"},
            json=data
            )
        
        if response.status_code == 200:
            print("Robot reset to joint target successfully.")
            print(response.json())
        else:
            print(f"Failed to reset robot: {response.text}")
            # print(response.json())
        time.sleep(3) 

        # this code might take a few seconds
        resp = requests.post(self.url +"jointreset")
        print(resp.text)

        
        time.sleep(1)
        for _ in range(4):
            self._send_gripper_command(reset_gripper,mode="continuous")
        time.sleep(1)
        self._update_currpos()

    def reset_slow(self, duration=8.0, steps=40):
        self._update_currpos()
        start_position = self.currpos[:3].copy()
        start_rotation = R.from_quat(self.currpos[3:])
        target_position = CARTESIAN_RESET_POSE[:3]
        target_rotation = R.from_euler("xyz", CARTESIAN_RESET_POSE[3:6])
        rotations = Slerp(
            [0.0, 1.0],
            R.concatenate([start_rotation, target_rotation]),
        )
        interval = duration / steps

        for step in range(1, steps + 1):
            fraction = step / steps
            position = start_position + (target_position - start_position) * fraction
            orientation = rotations([fraction])[0].as_quat()
            pose = np.concatenate([position, orientation])
            self._send_pos_command(pose)
            time.sleep(interval)

        target_pose = np.concatenate([target_position, target_rotation.as_quat()])
        self._send_pos_command(target_pose)
        self._wait_for_pose(target_pose, timeout=5.0)

        response = requests.post(
            self.url + "move_gripper",
            headers={"Content-Type": "application/json"},
            json={"gripper_pos": float(CARTESIAN_RESET_POSE[6])},
        )
        response.raise_for_status()
        self.next_gripper_pos = float(CARTESIAN_RESET_POSE[6])
        self._update_currpos()


    def _send_gripper_command(self, pos: float, mode="binary"):
        if mode == "binary":
            if (
                pos <= 0
                and self.gripper_binary_state == 0
            ):  # close gripper
                requests.post(self.url + "close_gripper")
                time.sleep(0.6)
                self.gripper_binary_state = 1
                return True
            elif (
                pos >= 0
                and self.gripper_binary_state == 1
            ):  # open gripper
                requests.post(self.url + "open_gripper")
                time.sleep(0.6)
                self.gripper_binary_state = 0
                return True
            else:  # do nothing to the gripper
                return False
        elif mode == "continuous":
            # The service expects the commanded gripper value directly.
            arr = float(np.clip(pos, self.gripper_min, self.gripper_max))
            self.next_gripper_pos = arr
            data = {"gripper_pos": arr}
            headers = {"Content-Type":"application/json"}
            # print(data)
            requests.post(self.url + "move_gripper", headers= headers, json=data)

            return True 

    def move_gripper_delta(self, delta, update_state=True):
        """Move the gripper by a relative amount and clamp to its limits.

        A zero delta intentionally sends no request. This makes releasing a
        SpaceMouse button stop the gripper without changing its position.
        """
        delta = float(delta)
        if abs(delta) <= 1e-12:
            return False

        current = float(np.asarray(self.curr_gripper_pos).reshape(-1)[0])
        target = float(np.clip(current + delta, self.gripper_min, self.gripper_max))
        if abs(target - current) <= 1e-12:
            return False

        if self.api_v1:
            response = requests.post(
                self.url.rstrip("/") + "/api/v1/gripper/move",
                json={"width_m": target, "speed_m_s": 0.2}, timeout=5,
            )
        else:
            response = requests.post(
                self.url + "move_gripper",
                headers={"Content-Type": "application/json"},
                json={"gripper_pos": target},
            )
        response.raise_for_status()
        self.next_gripper_pos = target
        if update_state:
            self._update_currpos()
        return True

    def open_gripper(self):
        if self.api_v1:
            response = requests.post(
                self.url.rstrip("/") + "/api/v1/gripper/move",
                json={"width_m": self.gripper_max, "speed_m_s": 0.2}, timeout=5,
            )
            response.raise_for_status()
            self._update_currpos()
            return
        response = requests.post(self.url + "open_gripper")
        print(f"[夹爪调试] 调用 open_gripper: status={response.status_code}, response={response.text}")
        response.raise_for_status()
        self._update_currpos()

    def close_gripper(self):
        if self.api_v1:
            response = requests.post(
                self.url.rstrip("/") + "/api/v1/gripper/move",
                json={"width_m": self.gripper_min, "speed_m_s": 0.2}, timeout=5,
            )
            response.raise_for_status()
            self._update_currpos()
            return
        response = requests.post(self.url + "close_gripper")
        print(f"[夹爪调试] 调用 close_gripper: status={response.status_code}, response={response.text}")
        response.raise_for_status()
        self._update_currpos()

    def move_gripper_absolute(self, position):
        position = float(np.clip(position, self.gripper_min, self.gripper_max))
        if self.api_v1:
            response = requests.post(
                self.url.rstrip("/") + "/api/v1/gripper/move",
                json={"width_m": position, "speed_m_s": 0.2}, timeout=5,
            )
        else:
            response = requests.post(
                self.url + "move_gripper",
                headers={"Content-Type": "application/json"},
                json={"gripper_pos": position},
            )
        print(f"[夹爪调试] 下发绝对位置: gripper_pos={position}, status={response.status_code}")
        response.raise_for_status()
        self.next_gripper_pos = position
        self._update_currpos()
    
    def _send_pos_command(self, pos: np.ndarray):
        arr = np.array(pos).astype(np.float32)
        if self.api_v1:
            response = requests.post(
                self.url.rstrip("/") + "/api/v1/motions/cartesian-pose",
                json={
                    "position_m": arr[:3].tolist(),
                    "quaternion_xyzw": arr[3:].tolist(),
                    "duration_s": self.command_duration_s,
                }, timeout=5,
            )
            response.raise_for_status()
            return response

        self._recover()
        data = {"arr": arr.tolist()}

        
        

        response = requests.post(self.url + "pose", json=data)
        response.raise_for_status()
        return response

    def _wait_for_pose(self, target_pose, timeout=5.0, position_tolerance=0.01, angle_tolerance=0.08):
        target_pose = np.asarray(target_pose, dtype=float)
        deadline = time.time() + timeout
        while time.time() < deadline:
            self._update_currpos()
            position_error = np.linalg.norm(self.currpos[:3] - target_pose[:3])
            relative_rotation = R.from_quat(self.currpos[3:]).inv() * R.from_quat(target_pose[3:])
            angle_error = relative_rotation.magnitude()
            if position_error <= position_tolerance and angle_error <= angle_tolerance:
                return True
            time.sleep(0.1)

        print(
            "[复位调试] 目标未在超时时间内完全到达: "
            f"position_error={position_error:.4f}, angle_error={angle_error:.4f}"
        )
        return False

    def _recover(self):
        """Internal function to recover the robot from error state."""
        requests.post(self.url + "clearerr")      

    def _update_currpos(self):
        if self.api_v1:
            response = requests.get(
                self.url.rstrip("/") + "/api/v1/state", timeout=5
            )
            response.raise_for_status()
            state = response.json()
            robot = state["robot"]
            pose = robot.get("gripper_pose") or robot.get("flange_pose")
            if pose is None:
                raise RuntimeError("upper computer returned no Cartesian pose")
            self.currpos[:] = np.asarray(
                [*pose["position_m"], *pose["quaternion_xyzw"]], dtype=float
            )
            self.q[:] = np.asarray(robot.get("joint_position_rad", self.q), dtype=float)
            self.dq[:] = np.asarray(robot.get("joint_velocity_rad_s", self.dq), dtype=float)
            gripper = state.get("gripper") or {}
            if gripper.get("width_m") is not None:
                self.curr_gripper_pos = float(gripper["width_m"])
            self.gripper_effort = None
            self.force = np.zeros(3, dtype=float)
            self.jacobian = np.zeros((6, 7), dtype=float)
            self.torque = np.zeros(3, dtype=float)
            self.vel = np.asarray(self.dq[:6], dtype=float)
            return

        ps = requests.post(self.url + 'getstate').json()
        self.currpos[:] = np.array(ps["pose"])

        self.q[:] = np.array(ps["q"])
        self.dq[:] = np.array(ps["dq"])

        self.curr_gripper_pos = np.array(ps["gripper_pos"])
        self.gripper_effort = ps.get("gripper_effort")
        # Keep optional fields when available, while remaining compatible with
        # older upper-computer responses that omit them.
        if ps.get("force") is not None:
            self.force = np.asarray(ps["force"], dtype=float)
        if ps.get("jacobian") is not None:
            self.jacobian = np.asarray(ps["jacobian"], dtype=float)
        if ps.get("torque") is not None:
            self.torque = np.asarray(ps["torque"], dtype=float)
        if ps.get("vel") is not None:
            self.vel = np.asarray(ps["vel"], dtype=float)
