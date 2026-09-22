"""Robot communication and control primitives."""

from .client import Franka

__all__ = ["Franka", "FrankaEnv"]


def __getattr__(name):
    if name == "FrankaEnv":
        from .environment import FrankaEnv

        return FrankaEnv
    raise AttributeError(name)
