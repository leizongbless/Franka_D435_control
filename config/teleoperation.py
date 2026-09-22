"""Configuration shared by SpaceMouse teleoperation and data collection."""

from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Mapping


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "teleoperation.yaml"


@dataclass(frozen=True)
class TeleoperationConfig:
    # Robot and SpaceMouse control.
    server_url: str = "http://192.168.1.11:8000/"
    position_scale: float = 0.10
    rotation_scale: float = 0.20
    gripper_step: float = 0.005
    gripper_min: float = 0.0
    gripper_max: float = 0.08
    deadzone: float = 0.30
    control_hz: float = 20.0
    reset_on_start: bool = False
    open_button: int = 1
    close_button: int = 0

    # LeRobot collector.
    repo_id: str = ""
    task: str = ""
    top_serial: str = ""
    wrist_serial: str = ""
    camera_width: int = 1280
    camera_height: int = 720
    camera_fps: int = 30

    # Legacy NPZ collector.
    legacy_camera_serial: str = "261622077687"
    legacy_camera_width: int = 640
    legacy_camera_height: int = 480
    legacy_camera_fps: int = 30
    camera_init_retries: int = 3
    camera_retry_delay: float = 2.0
    data_save_frequency: float = 15.0
    reset_duration: float = 3.0
    reset_steps: int = 15

    def validate(self) -> "TeleoperationConfig":
        if self.position_scale <= 0 or self.rotation_scale <= 0 or self.gripper_step <= 0:
            raise ValueError("position_scale, rotation_scale and gripper_step must be > 0")
        if self.gripper_min >= self.gripper_max:
            raise ValueError("gripper_min must be smaller than gripper_max")
        if not 0 <= self.deadzone < 1:
            raise ValueError("deadzone must be in [0, 1)")
        if self.control_hz <= 0 or self.data_save_frequency <= 0:
            raise ValueError("control_hz and data_save_frequency must be > 0")
        if min(
            self.camera_width,
            self.camera_height,
            self.camera_fps,
            self.legacy_camera_width,
            self.legacy_camera_height,
            self.legacy_camera_fps,
        ) <= 0:
            raise ValueError("camera dimensions and frame rates must be > 0")
        if self.camera_init_retries < 1 or self.camera_retry_delay < 0:
            raise ValueError("camera retry settings are invalid")
        if self.reset_duration <= 0 or self.reset_steps < 1:
            raise ValueError("reset_duration and reset_steps are invalid")
        if (
            self.open_button == self.close_button
            or min(self.open_button, self.close_button) < 0
            or max(self.open_button, self.close_button) > 1
        ):
            raise ValueError(
                "open_button and close_button must be the two SpaceMouse button indices 0 and 1"
            )
        return self


def _coerce_values(data: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {field.name: field.type for field in fields(TeleoperationConfig)}
    unknown = sorted(set(data) - set(allowed))
    if unknown:
        raise ValueError(f"Unknown teleoperation config key(s): {', '.join(unknown)}")

    values: dict[str, Any] = {}
    for name, value in data.items():
        if value is None:
            continue
        expected = allowed[name]
        if expected is int:
            values[name] = int(value)
        elif expected is float:
            values[name] = float(value)
        elif expected is str:
            values[name] = str(value)
        elif expected is bool:
            if isinstance(value, str):
                normalized = value.strip().lower()
                if normalized not in {"true", "false", "1", "0", "yes", "no", "on", "off"}:
                    raise ValueError(f"Invalid boolean value for {name}: {value!r}")
                values[name] = normalized in {"true", "1", "yes", "on"}
            else:
                values[name] = bool(value)
        else:
            values[name] = value
    return values


def load_teleoperation_config(path: str | Path | None = None) -> TeleoperationConfig:
    """Load the editable YAML config, returning validated typed values."""

    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"Teleoperation config does not exist: {config_path}")
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required to read teleoperation.yaml; install pyyaml first"
        ) from exc

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, Mapping):
        raise ValueError(f"Teleoperation config must contain a YAML mapping: {config_path}")
    return replace(TeleoperationConfig(), **_coerce_values(raw)).validate()
