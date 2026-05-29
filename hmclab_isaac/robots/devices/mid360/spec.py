"""Livox Mid-360 LiDAR — single source of truth for sensor spec.

Two simulation paths consume different parts of this:

- **Isaac Sim native RTX LiDAR** (USD-driven, GUI / ROS bridge / Replicator):
  reads the schema authored on ``assets/mid-360.usd`` by
  ``scripts/author_lidar_schemas.py`` and looks up the profile JSON
  ``profiles/Livox_Mid360.json`` by :data:`RTX_CONFIG_NAME`.

- **Isaac Lab GPU RayCaster** (parallel RL): ignores the USD schema
  and uses ``patterns/Mid360PatternCfg``, which loads the real Livox
  non-repetitive pattern from ``patterns/mid360-real-centr.csv``.

The RTX profile JSON is a coarse rotary approximation; the CSV-driven
Python pattern is the accurate one.
"""
from __future__ import annotations

import os

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

USD_PATH = os.path.join(_THIS_DIR, "Mid360.usd")
PROFILE_JSON = os.path.join(_THIS_DIR, "profiles", "Livox_Mid360.json")
PATTERN_CSV = os.path.join(_THIS_DIR, "patterns", "mid360-real-centr.csv")
RTX_CONFIG_NAME = "Livox_Mid360"

SPEC = {
    "make": "Livox",
    "model": "Mid-360",
    "scan_type": "non_repetitive_rotary",
    "fov_h_deg": 360.0,
    "fov_v_min_deg": -7.0,
    "fov_v_max_deg": 52.0,
    "near_range_m": 0.1,
    "far_range_m": 40.0,         # @ 80% reflectivity; up to 70m @ 20%
    "scan_rate_hz": 10.0,
    "points_per_second": 200_000,  # 800K / 4 s cycle
    "channels": 1,
    "wavelength_nm": 905.0,
}

__all__ = ["USD_PATH", "PROFILE_JSON", "PATTERN_CSV", "RTX_CONFIG_NAME", "SPEC"]
