#!/usr/bin/env python3
"""Preview two Intel RealSense cameras in one window."""

import argparse
import sys

import cv2
import numpy as np
import pyrealsense2 as rs


WINDOW_NAME = "Two RealSense Cameras"

DEPTH_OPTIONS = (
    (rs.option.visual_preset, 0.0),
    (rs.option.enable_auto_exposure, 1.0),
    (rs.option.emitter_enabled, 1.0),
    (rs.option.laser_power, 150.0),
    (rs.option.auto_exposure_limit_toggle, 0.0),
    (rs.option.auto_gain_limit_toggle, 0.0),
    (rs.option.emitter_on_off, 0.0),
    (rs.option.emitter_always_on, 0.0),
)

COLOR_OPTIONS = (
    (rs.option.enable_auto_exposure, 0.0),
    (rs.option.exposure, 500.0),
    (rs.option.gain, 0.0),
    (rs.option.enable_auto_white_balance, 0.0),
    (rs.option.white_balance, 4000.0),
    (rs.option.brightness, 0.0),
    (rs.option.contrast, 50.0),
    (rs.option.saturation, 64.0),
    (rs.option.sharpness, 50.0),
    (rs.option.gamma, 300.0),
    (rs.option.hue, 0.0),
    (rs.option.backlight_compensation, 0.0),
    (rs.option.power_line_frequency, 3.0),
    (rs.option.auto_exposure_priority, 0.0),
)


def connected_devices():
    """Return connected RealSense devices as (serial, name) pairs."""
    devices = []
    for device in rs.context().query_devices():
        serial = device.get_info(rs.camera_info.serial_number)
        name = device.get_info(rs.camera_info.name)
        devices.append((serial, name))
    return devices


def get_device(serial):
    for device in rs.context().query_devices():
        if device.get_info(rs.camera_info.serial_number) == serial:
            return device
    raise RuntimeError(f"Camera {serial} was disconnected")


def select_serials(requested_serials):
    devices = connected_devices()
    if not devices:
        raise RuntimeError("No RealSense cameras found")

    print("Connected RealSense cameras:")
    for index, (serial, name) in enumerate(devices):
        print(f"  [{index}] {name}, serial={serial}")

    available = {serial for serial, _ in devices}
    if requested_serials:
        missing = [serial for serial in requested_serials if serial not in available]
        if missing:
            raise RuntimeError(
                f"Camera serial(s) not found: {', '.join(missing)}"
            )
        if requested_serials[0] == requested_serials[1]:
            raise RuntimeError("Two different camera serial numbers are required")
        return requested_serials

    if len(devices) < 2:
        raise RuntimeError(f"Two cameras are required, but only {len(devices)} found")
    selected = [serial for serial, _ in devices[:2]]
    print(f"Using cameras: {selected[0]} and {selected[1]}")
    return selected


def device_has_color_stream(serial):
    return any(
        profile.stream_type() == rs.stream.color
        for sensor in get_device(serial).query_sensors()
        for profile in sensor.get_stream_profiles()
    )


def find_sensor(device, stream_type):
    for sensor in device.query_sensors():
        if any(
            profile.stream_type() == stream_type
            for profile in sensor.get_stream_profiles()
        ):
            return sensor
    return None


def apply_sensor_options(sensor, options, serial, sensor_name):
    if sensor is None:
        print(f"Warning: camera {serial} has no {sensor_name} sensor")
        return

    print(f"Camera {serial} {sensor_name} options:")
    for option, requested_value in options:
        option_name = str(option).split(".")[-1]
        if not sensor.supports(option):
            print(f"  {option_name}: unsupported")
            continue
        try:
            sensor.set_option(option, requested_value)
            actual_value = sensor.get_option(option)
        except RuntimeError as exc:
            raise RuntimeError(
                f"Camera {serial} failed to set {sensor_name} "
                f"{option_name}={requested_value}: {exc}"
            ) from exc
        print(f"  {option_name}: {actual_value}")


def start_camera(serial, width, height, fps, show_depth):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    device = get_device(serial)
    if device_has_color_stream(serial):
        stream_kind = "color"
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    else:
        stream_kind = "infrared"
        print(
            f"Warning: camera {serial} has no RGB stream; showing infrared instead. "
            "Connect it through USB 3 for RGB."
        )
        config.enable_stream(
            rs.stream.infrared, 1, width, height, rs.format.y8, fps
        )
    if show_depth:
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    if show_depth:
        apply_sensor_options(
            find_sensor(device, rs.stream.depth), DEPTH_OPTIONS, serial, "depth"
        )
    if stream_kind == "color":
        apply_sensor_options(
            find_sensor(device, rs.stream.color), COLOR_OPTIONS, serial, "color"
        )
    pipeline.start(config)
    align = rs.align(rs.stream.color) if show_depth and stream_kind == "color" else None
    return pipeline, align, stream_kind


def add_label(image, text):
    labeled = image.copy()
    cv2.rectangle(labeled, (0, 0), (labeled.shape[1], 38), (0, 0, 0), -1)
    cv2.putText(
        labeled,
        text,
        (10, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return labeled


def waiting_panel(width, height, serial, stream_kind, show_depth):
    panel_height = height * (2 if show_depth else 1)
    panel = np.zeros((panel_height, width, 3), dtype=np.uint8)
    cv2.putText(
        panel,
        f"Waiting for {stream_kind} frames",
        (20, max(50, panel_height // 2)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 165, 255),
        2,
        cv2.LINE_AA,
    )
    return add_label(panel, f"{stream_kind.title()} | {serial}")


def read_panel(pipeline, align, serial, stream_kind, show_depth):
    try:
        frames = pipeline.poll_for_frames()
    except RuntimeError as exc:
        raise RuntimeError(f"Camera {serial}: {exc}") from exc
    if not frames:
        return None
    if align is not None:
        frames = align.process(frames)

    if stream_kind == "color":
        image_frame = frames.get_color_frame()
    else:
        image_frame = frames.get_infrared_frame(1)
    if not image_frame:
        raise RuntimeError(f"No {stream_kind} frame received from camera {serial}")

    image = np.asanyarray(image_frame.get_data())
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    image = add_label(image, f"{stream_kind.title()} | {serial}")
    if not show_depth:
        return image

    depth_frame = frames.get_depth_frame()
    if not depth_frame:
        raise RuntimeError(f"No depth frame received from camera {serial}")
    depth_image = np.asanyarray(depth_frame.get_data())
    depth_colormap = cv2.applyColorMap(
        cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
    )
    depth_colormap = add_label(depth_colormap, f"Depth | {serial}")
    return cv2.vconcat([image, depth_colormap])


def run(serials, width, height, fps, show_depth):
    pipelines = []
    cameras = []
    try:
        for serial in serials:
            pipeline, align, stream_kind = start_camera(
                serial, width, height, fps, show_depth
            )
            pipelines.append(pipeline)
            cameras.append((pipeline, align, serial, stream_kind))

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        print("Press q or Esc in the preview window to quit")
        panels = [
            waiting_panel(width, height, serial, stream_kind, show_depth)
            for _, _, serial, stream_kind in cameras
        ]
        receiving = [False] * len(cameras)
        while True:
            for index, (pipeline, align, serial, stream_kind) in enumerate(cameras):
                panel = read_panel(
                    pipeline, align, serial, stream_kind, show_depth
                )
                if panel is not None:
                    panels[index] = panel
                    if not receiving[index]:
                        print(f"Receiving {stream_kind} frames from camera {serial}")
                        receiving[index] = True
            cv2.imshow(WINDOW_NAME, cv2.hconcat(panels))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        for pipeline in reversed(pipelines):
            try:
                pipeline.stop()
            except RuntimeError:
                pass
        cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--serials",
        nargs=2,
        metavar=("CAMERA_1", "CAMERA_2"),
        help="two camera serial numbers; default: first two connected cameras",
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    display_group = parser.add_mutually_exclusive_group()
    display_group.add_argument(
        "--show-depth",
        action="store_true",
        dest="show_depth",
        help="show aligned depth below each color image (default)",
    )
    display_group.add_argument(
        "--color-only",
        action="store_false",
        dest="show_depth",
        help="disable the depth stream and only show color",
    )
    parser.set_defaults(show_depth=True)
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        serials = select_serials(args.serials)
        run(serials, args.width, args.height, args.fps, args.show_depth)
    except (RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nStopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
