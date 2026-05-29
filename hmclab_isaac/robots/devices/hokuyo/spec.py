"""Hokuyo UST-10LX/20LX 2D LiDAR — single source of truth for sensor spec.

This module is the spec the rest of the codebase reads:
- USD authoring (see ``scripts/author_lidar_schemas.py``) bakes the
  RTX LiDAR ``cameraSensorType="lidar"`` schema into ``UST_10_20LX.usd``
  using ``RTX_CONFIG_NAME``.
- Isaac Lab's GPU RayCaster (parallel RL) ignores the USD schema and
  uses :class:`PatternBaseCfg` -based configs; build one from
  :data:`SPEC` if/when needed.

The matching JSON profile lives at ``profiles/Hokuyo_UST_20LX.json``;
that's what NVIDIA's RTX LiDAR plugin loads at runtime by name.
"""
from __future__ import annotations

import os

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

USD_PATH = os.path.join(_THIS_DIR, "UST_10_20LX.usd")
PROFILE_JSON = os.path.join(_THIS_DIR, "profiles", "Hokuyo_UST_20LX.json")
RTX_CONFIG_NAME = "Hokuyo_UST_20LX"

SPEC = {
    "make": "Hokuyo",
    "model": "UST-20LX",
    "scan_type": "rotary",
    "fov_h_deg": 270.0,
    "fov_v_deg": 0.0,
    "azimuth_resolution_deg": 0.25,
    "scan_rate_hz": 40.0,
    "near_range_m": 0.06,
    "far_range_m": 20.0,
    "channels": 1,
    "wavelength_nm": 905.0,
}

__all__ = ["USD_PATH", "PROFILE_JSON", "RTX_CONFIG_NAME", "SPEC"]
