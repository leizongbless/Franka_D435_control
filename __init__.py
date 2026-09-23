"""Reusable Franka robot, camera, and teleoperation toolkit."""

__all__ = ["Franka", "FrankaEnv", "RSCapture", "SpaceMouseExpert"]


def __getattr__(name):
    if name in {"Franka", "FrankaEnv"}:
        from .robot import Franka, FrankaEnv

        return {"Franka": Franka, "FrankaEnv": FrankaEnv}[name]
    if name == "RSCapture":
        from .camera import RSCapture

        return RSCapture
    if name == "SpaceMouseExpert":
        from .input import SpaceMouseExpert

        return SpaceMouseExpert
    raise AttributeError(name)
