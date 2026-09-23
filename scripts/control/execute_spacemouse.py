#!/usr/bin/env python3
"""
机器人SpaceMouse控制程序
纯控制模式 - 不涉及任何摄像头和数据保存
"""

import os, sys, time
import numpy as np
import select

from franka_toolkit.robot import Franka, FrankaEnv
from franka_toolkit.input.device import Spacemouse


def check_keyboard_input():
    """非阻塞式检查键盘输入"""
    if select.select([sys.stdin], [], [], 0)[0]:
        return sys.stdin.read(1)
    return None


def main():
    print("=" * 50)
    print("机器人SpaceMouse控制程序 - 带键盘控制")
    print("=" * 50)
    print("控制说明:")
    print("  's' - 开始SpaceMouse操作模式")
    print("  't' - 停止操作，等待下一步指令")
    print("  'r' - 重置机器人到初始位置")
    print("  Ctrl+C - 退出程序")
    print("=" * 50)
    
    # 初始化机器人环境
    accurate = [0.04, 0.1, 20]
    env = FrankaEnv(accurate)  
    time.sleep(1)
    
    # 重置机器人到初始位置
    env.reset()
    init_state = env.get_euler_action()
    print(f"机器人初始状态: {init_state}")
    
    print("\n请按 's' 键开始操作...")
    
    # 控制状态
    operating = False
    
    with Spacemouse(deadzone=0.3) as sm:
        try:
            loop_count = 0
            while True:
                # 检查键盘输入
                key = check_keyboard_input()
                if key:
                    if key.lower() == 's':
                        operating = True
                        print("✅ 开始SpaceMouse操作模式")
                        print("SpaceMouse按钮: 控制夹爪开合")
                        
                    elif key.lower() == 't':
                        operating = False
                        print("⏸️ 停止操作模式，等待指令...")
                        print("按 's' 继续操作或 'r' 重置")
                        
                    elif key.lower() == 'r':
                        operating = False
                        env.reset()
                        init_state = env.get_euler_action()
                        print(f"🔄 机器人已重置到初始位置: {init_state}")
                        print("按 's' 开始操作")
                
                # 只有在操作模式下才响应SpaceMouse
                if operating:
                    # 获取SpaceMouse输入
                    mouse_action = sm.get_motion_state_transformed()
                    buttons = [int(sm.is_button_pressed(0)), int(sm.is_button_pressed(1))]
                    
                    # 控制机器人移动
                    env.move_robot(mouse_action, buttons)
                    
                    # 获取当前机器人状态
                    state = env.get_euler_action(if_delta=False) 
                    if buttons[0] == 1:
                        state[6] -= accurate[-1]
                    if buttons[1] == 1:
                        state[6] += accurate[-1]
                    
                    # 每200次循环打印一次状态（避免输出过多）
                    loop_count += 1
                    if loop_count % 200 == 0:
                        print(f"位置: [{state[0]:.3f}, {state[1]:.3f}, {state[2]:.3f}] | "
                              f"姿态: [{state[3]:.3f}, {state[4]:.3f}, {state[5]:.3f}] | "
                              f"夹爪: {state[6]:.1f}")
                else:
                    # 非操作模式下，重置loop_count
                    loop_count = 0
                
                time.sleep(0.01)  # 减少延迟，提高响应性
                
        except KeyboardInterrupt:
            print("\n收到Ctrl+C，程序退出")
            
    print("机器人控制程序已退出")


if __name__ == "__main__":
    main()
