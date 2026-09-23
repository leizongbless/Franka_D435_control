
import numpy as np
from scipy.spatial.transform import Rotation as R
from ultralytics import YOLO

import cv2
from .client import Franka
from ..camera import RSCapture, voxel_downsampling, furthest_point_sampling, PointNetEncoderXYZ, fast_furthest_point_sampling
from ..geometry import transform_camera_to_marker, interpolate_se3_euler
from ..input import SpaceMouseExpert




class FrankaEnv:
    def __init__(
        self,
        action_scales=[0.04, 0.1, 20],
        server_url="http://192.168.1.11:8000/",
        gripper_min=0.0,
        gripper_max=255.0,
    ):

        self.robot = Franka(
            action_scales,
            server_url=server_url,
            gripper_min=gripper_min,
            gripper_max=gripper_max,
        )
        # '141722078044'
        # 'f1371022'
        # self.camera = RSCapture('viewer', serial_number = 'f1371022',depth=True)    # '141722078044'/'135122078001'f1371022 / 231622301322
        
        # #######################################################
        # # self.camera = RSCapture('viewer', serial_number = '213622073689',depth=False)    # 243722074077,213622073689
        
        # # Wait for camera to initialize
        # import time
        # print("Waiting for camera to initialize...")
        # time.sleep(3)
        
        # # Test camera connection
        # for i in range(5):
        #     test_frame = self.camera.get_latest_frame()
        #     if test_frame is not None:
        #         print("Camera initialized successfully")
        #         break
        #     time.sleep(1)
        # else:

        
        # self.yolo_model = YOLO('yolov8n.pt')

    
    

    def get_obs(self):
        # Try multiple times to get a valid frame
        for attempt in range(3):
            frames = self.camera.get_latest_frame()
            if frames is not None:
                break
            import time
            time.sleep(0.1)
        
        if frames is None:
            # Return a dummy observation with correct structure
            print("Warning: Could not get camera frame, returning dummy observation")
            dummy_color = np.zeros((480, 640, 3), dtype=np.uint8)
            return {"color": dummy_color}
        
        color_frame = frames["color_frame"]
        color = np.asarray(color_frame.get_data())
        output = {
            "color":color
        }        

        if self.camera.depth:
            depth_frame = frames['depth_frame']
            vtx_down = frames["vtx_down"].copy()
            
            depth = np.expand_dims(np.asarray(depth_frame.get_data()), axis=2)
            output['depth']  =depth
            output['pointcloud'] = vtx_down.copy()  # (2048,3)

        return output
    
    def move_robot(self, action, buttons, open_button=1, close_button=0):
        """Apply pose motion and one held-button gripper direction."""
        buttons = list(buttons)
        open_pressed = open_button < len(buttons) and buttons[open_button]
        close_pressed = close_button < len(buttons) and buttons[close_button]

        # Both buttons at once is intentionally neutral rather than unsafe.
        if open_pressed and not close_pressed:
            move_gripper = 1
        elif close_pressed and not open_pressed:
            move_gripper = -1
        else:
            move_gripper = 0

        self.robot.move(action, move_gripper)

    def agent_action(self, action):
        # delta xyz, euler, gripper
        xyz_delta = action[:3]
        euler_delta = action[3:6]
        gripper = action[-1]

        self.robot.agent_move(xyz_delta, euler_delta, gripper)

    def agent_absolute_action(self, action):
        # delta xyz, euler, gripper
        xyz_absolute = action[:3]
        euler_absolute = action[3:6]
        gripper = action[-1]

        self.robot.agent_absolute_move(xyz_absolute, euler_absolute, gripper)
        
        


    def get_action(self, if_delta=True):

        if if_delta:
            xyz_delta = self.robot.delta_xyz
            euler_delta = self.robot.delta_euler
            gripper_cmd = self.robot.next_gripper_pos
            return np.concatenate([xyz_delta, euler_delta, np.array([gripper_cmd])])
        else:
            # ee_cmd = self.robot.clip_safety_box(self.robot.nextpos)currpos
            ee_cmd = self.robot.clip_safety_box(self.robot.currpos)
            gripper_cmd = self.robot.next_gripper_pos
            # se3 array
            return np.concatenate([ee_cmd, np.array([gripper_cmd])])
        
    def get_euler_action(self, if_delta=False):

        if if_delta:
            xyz_delta = self.robot.delta_xyz
            euler_delta = self.robot.delta_euler
            gripper_cmd = self.robot.next_gripper_pos
            return np.concatenate([xyz_delta, euler_delta, np.array([gripper_cmd])])
        else:
            pose = self.robot.currpos.copy()
            ee_cmd = np.concatenate([
                pose[:3],
                R.from_quat(pose[3:]).as_euler("xyz"),
            ])
            gripper_cmd = self.robot.next_gripper_pos
            return np.concatenate([ee_cmd, np.array([gripper_cmd])])   

        
    def reset(self):
    
        self.robot.reset()
        
        return True

    def reset_slow(self, duration=8.0, steps=40):
        self.robot.reset_slow(duration=duration, steps=steps)
        return True

    def open_gripper(self):
        self.robot.open_gripper()

    def close_gripper(self):
        self.robot.close_gripper()

    def move_gripper_absolute(self, position):
        self.robot.move_gripper_absolute(position)

    def move_gripper_delta(self, delta):
        return self.robot.move_gripper_delta(delta)
    
    def visualize_yolo_results(self, color_img, results, window_name="YOLO Detection"):
    
        vis_img = color_img.copy()

        for result in results:
            boxes = result.boxes
            if len(boxes) > 0:
                for box in boxes:
                    x, y, w, h = box.xywh[0].cpu().numpy()
                    conf = box.conf[0].item()
                    cls_id = int(box.cls[0].item())

                    # 中心点 + 宽高 转 左上角 + 右下角
                    x1 = int(x - w / 2)
                    y1 = int(y - h / 2)
                    x2 = int(x + w / 2)
                    y2 = int(y + h / 2)

                    # 绘制矩形框
                    cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # 标签文字
                    label = f"ID:{cls_id} {conf:.2f}"
                    cv2.putText(vis_img, label, (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    print(f"Detected object at ({x}, {y}) with width {w} and height {h}")

        # 显示
        cv2.imshow(window_name, vis_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    def get_reward(self, agent_done):
        

        if agent_done == 0:
            print("mission failed.")
            return 0.0
        elif agent_done == 1:
            print("mission successed.")
            return 1.0


        # refer_xywh = [196,311,74,121]
            
        # if done == 1:   #########
        #     results = self.yolo_model.predict(
        #     source=color_img,
        #     classes=39,  # bottle id: 39 cup: 41
        #     conf=0.25,    # 置信度阈值
        #     imgsz=(640, 480),   
        #     verbose=True  # 不输出详细信息
        #     )

        #     # self.visualize_yolo_results(color_img, results, window_name="YOLO Detection")
        #     for result in results:
        #         boxes = result.boxes
        #         if len(boxes) > 0:
        #             for box in boxes:
        #                 x, y, w, h = box.xywh[0].cpu().numpy()
        #                 print(f"Detected object at ({x}, {y}) with width {w} and height {h}")
        #                 if abs(x - refer_xywh[0]) < 30 and abs(y - refer_xywh[1]) < 30 and abs(w - refer_xywh[2]) < 4 and abs(h - refer_xywh[3]) < 4:
        #                     print("success place!")
        #                     return 1.0
        # return 0.0
        
    def get_done(self,buttons):
        open_downbutton = buttons[0]
        close_upbutton = buttons[-1]

        if open_downbutton != 0 and close_upbutton != 0:
            return 1
        else:
            return 0
    
    def get_agent_done(self, buttons):
        fail_downbutton = buttons[0]
        success_upbutton = buttons[-1]

        if success_upbutton != 0:
            return 1
        elif fail_downbutton != 0:
            return 0
        elif success_upbutton == 0 and fail_downbutton == 0:
            return 2
