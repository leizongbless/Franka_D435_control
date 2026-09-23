"""Shared project paths for generated collection outputs."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data"
RAW_DATA_ROOT = DATA_ROOT / "raw"
VIDEO_DATA_ROOT = DATA_ROOT / "video"
