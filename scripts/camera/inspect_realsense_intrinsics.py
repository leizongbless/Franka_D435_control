import pyrealsense2 as rs
import numpy as np
import cv2

def detect_realsense_intrinsics():
    """专门检测RealSense相机内参"""
    print("开始检测RealSense相机内参...")
    
    pipeline = rs.pipeline()
    config = rs.config()
    
    # 启用彩色和深度流
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    
    try:
        # 启动管道
        profile = pipeline.start(config)
        
        # 获取深度流的内参
        depth_profile = profile.get_stream(rs.stream.depth)
        depth_intrinsics = depth_profile.as_video_stream_profile().get_intrinsics()
        
        # 获取彩色流的内参  
        color_profile = profile.get_stream(rs.stream.color)
        color_intrinsics = color_profile.as_video_stream_profile().get_intrinsics()
        
        print("\n=== 深度相机内参 ===")
        print(f"焦距: fx={depth_intrinsics.fx:.3f}, fy={depth_intrinsics.fy:.3f}")
        print(f"主点: cx={depth_intrinsics.ppx:.3f}, cy={depth_intrinsics.ppy:.3f}")
        print(f"畸变模型: {depth_intrinsics.model.name}")
        print(f"畸变系数: {depth_intrinsics.coeffs}")
        
        print("\n=== 彩色相机内参 ===")
        print(f"焦距: fx={color_intrinsics.fx:.3f}, fy={color_intrinsics.fy:.3f}")
        print(f"主点: cx={color_intrinsics.ppx:.3f}, cy={color_intrinsics.ppy:.3f}")
        print(f"畸变模型: {color_intrinsics.model.name}")
        print(f"畸变系数: {color_intrinsics.coeffs}")
        
        # 保存内参供主程序使用
        intrinsics_dict = {
            'depth': {
                'fx': depth_intrinsics.fx,
                'fy': depth_intrinsics.fy,
                'cx': depth_intrinsics.ppx,
                'cy': depth_intrinsics.ppy,
                'coeffs': depth_intrinsics.coeffs
            },
            'color': {
                'fx': color_intrinsics.fx,
                'fy': color_intrinsics.fy,
                'cx': color_intrinsics.ppx,
                'cy': color_intrinsics.ppy,
                'coeffs': color_intrinsics.coeffs
            }
        }
        
        return intrinsics_dict
        
    except Exception as e:
        print(f"检测RealSense内参时出错: {e}")
        return None
    finally:
        pipeline.stop()

if __name__ == "__main__":
    intrinsics = detect_realsense_intrinsics()
    
    if intrinsics:
        print("\n=== 内参字典 (可在主程序中使用) ===")
        print(intrinsics)
        
        # 生成可以直接复制到主程序的代码
        print("\n=== 在主程序中使用的代码 ===")
        print(f"fx = {intrinsics['depth']['fx']:.3f}")
        print(f"fy = {intrinsics['depth']['fy']:.3f}")
        print(f"cx = {intrinsics['depth']['cx']:.3f}")
        print(f"cy = {intrinsics['depth']['cy']:.3f}")