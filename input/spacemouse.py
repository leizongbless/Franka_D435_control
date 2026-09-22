import os,sys,time,copy

import datetime
import threading
from collections import deque
from typing import Tuple
import requests

import numpy as np

import pyspacemouse
import tkinter as tk
from tkinter import messagebox


class SpaceMouseExpert:

    def __init__(self):
        self.stop_event = threading.Event()
        self.device = pyspacemouse.open()
        if self.device is None:
            raise RuntimeError("未检测到可用的 SpaceMouse")

        self.state_lock = threading.Lock()
        self.latest_data = {"action": np.zeros(6), "buttons": [0, 0]}
        # Start a thread to continuously read the SpaceMouse state
        self.thread = threading.Thread(target=self._read_spacemouse)
        self.thread.daemon = True
        self.thread.start()

    def _read_spacemouse(self):
        while not self.stop_event.is_set():
            state = pyspacemouse.read()
            if state is None:
                continue
            with self.state_lock:
                self.latest_data["action"] = np.array(
                    [-state.y, state.x, state.z, -state.roll, -state.pitch, -state.yaw]
                )  # spacemouse axis matched with robot base frame
                self.latest_data["buttons"] = list(state.buttons)

    def get_action(self) -> Tuple[np.ndarray, list]:
        """Returns the latest action and button state of the SpaceMouse."""
        with self.state_lock:
            return self.latest_data["action"], self.latest_data["buttons"]

    def get_motion_state_transformed(self):
        return self.get_action()[0]

    def is_button_pressed(self, button_id):
        return bool(self.get_action()[1][button_id])

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_event.set()
        self.thread.join(timeout=1.0)
        pyspacemouse.close()
        
    def ask_success(self) -> bool:

        root = tk.Tk()
        root.withdraw()  # 隐藏主窗口
        result = messagebox.askyesno("Task Result", "任务是否成功？")
        root.destroy()
        return result