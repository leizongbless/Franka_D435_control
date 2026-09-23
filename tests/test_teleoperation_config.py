from franka_toolkit.config import load_teleoperation_config


def test_default_teleoperation_config_loads():
    config = load_teleoperation_config()

    assert config.position_scale > 0
    assert config.rotation_scale > 0
    assert config.gripper_step > 0
    assert {config.open_button, config.close_button} == {0, 1}
    assert config.legacy_camera_serial
