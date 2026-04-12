"""F1Tenth 1/10-scale Ackermann vehicle with Livox Mid-360 LiDAR.

Codename: `f1tenth_mid360`.

Top-level exports:
  F1TENTH_MID360_CFG — ArticulationCfg template (replace `prim_path` per spawn)
  make_mid360         — Mid-360 LiDAR factory with this vehicle's mount offset
  make_front_camera   — Front camera factory with this vehicle's mount offset

Vehicle-specific submodules:
  patterns  — Livox Mid-360 non-repetitive scan pattern cfg
  sensors   — Pre-configured sensor factories
  ros2_graph — Standard ROS2 bridge graph (M7 stub)
"""

from .f1tenth_mid360_cfg import (
    F1TENTH_MID360_CFG,
    FRONT_CAMERA_MOUNT_OFFSET,
    LIDAR_MOUNT_OFFSET,
    MAX_STEER,
    TRACK_WIDTH,
    USD_PATH,
    WHEELBASE,
    WHEEL_RADIUS,
)
from .sensors import make_front_camera, make_mid360

__all__ = [
    "F1TENTH_MID360_CFG",
    "USD_PATH",
    "WHEELBASE",
    "TRACK_WIDTH",
    "WHEEL_RADIUS",
    "MAX_STEER",
    "LIDAR_MOUNT_OFFSET",
    "FRONT_CAMERA_MOUNT_OFFSET",
    "make_mid360",
    "make_front_camera",
]
