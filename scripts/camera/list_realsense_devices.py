import pyrealsense2 as rs
ctx = rs.context()
devices = ctx.devices
if len(devices) == 0:
    print("No RealSense devices found")
else:
    print("Found devices:")
    for d in devices:
        print(f"  Name: {d.get_info(rs.camera_info.name)}")
        print(f"  Serial: {d.get_info(rs.camera_info.serial_number)}")
