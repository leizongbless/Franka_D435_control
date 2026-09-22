#!/usr/bin/env python3
"""
Simple RealSense live viewer.
Usage:
  python realsense_viewer.py            # use first connected camera
  python realsense_viewer.py --serial 135122078001
  python realsense_viewer.py --show_depth # also show depth colormap

Press 'q' to quit.
"""
import argparse
import time
import numpy as np
import cv2
import pyrealsense2 as rs


def list_serials():
    ctx = rs.context()
    devices = ctx.query_devices()
    serials = []
    for i, d in enumerate(devices):
        serial = d.get_info(rs.camera_info.serial_number)
        name = d.get_info(rs.camera_info.name)
        print(f"  [{i}] {name}  serial={serial}")
        serials.append(serial)
    return serials


def run_viewer(serial=None, show_depth=False, width=640, height=480, fps=30):
    serials = list_serials()
    if len(serials) == 0:
        print("No RealSense devices found")
        return

    if serial is None:
        serial = serials[0]
        print(f"No serial specified, using first device: {serial}")
    else:
        if serial not in serials:
            print(f"Requested serial {serial} not found. Available:")
            for s in serials:
                print(" ", s)
            return

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    if show_depth:
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

    try:
        profile = pipeline.start(config)
    except Exception as e:
        print("Failed to start pipeline:", e)
        return

    cv2.namedWindow('RealSense Color', cv2.WINDOW_AUTOSIZE)
    if show_depth:
        cv2.namedWindow('RealSense Depth', cv2.WINDOW_AUTOSIZE)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color = frames.get_color_frame()
            if not color:
                continue
            color_img = np.asanyarray(color.get_data())

            cv2.imshow('RealSense Color', color_img)

            if show_depth:
                depth = frames.get_depth_frame()
                if depth:
                    depth_img = np.asanyarray(depth.get_data())
                    # normalize and colormap for display
                    depth_clipped = np.clip(depth_img, 0, 10000)
                    depth_norm = (depth_clipped / np.max(depth_clipped + 1e-6) * 255).astype(np.uint8)
                    depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
                    cv2.imshow('RealSense Depth', depth_color)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RealSense live viewer')
    parser.add_argument('--serial', type=str, default=None, help='Camera serial number (optional)')
    parser.add_argument('--show_depth', action='store_true', help='Also show depth as colormap')
    args = parser.parse_args()

    run_viewer(serial=args.serial, show_depth=args.show_depth)
