"""F1Tenth-Mid360 pre-configured sensor factories.

Thin wrappers around `hmclab_isaac.robots._sensors.*` that inject this
vehicle's specific mount offsets and the Livox Mid-360 pattern. Use these
from env cfgs so vehicle-specific details (mount height, pattern CSV) are
defined in exactly one place.
"""

from __future__ import annotations

from typing import Any, Sequence

from .f1tenth_mid360_cfg import (
    FRONT_CAMERA_MOUNT_OFFSET,
    LIDAR_MOUNT_OFFSET,
)


def make_mid360(
    robot_prim_path: str,
    mesh_targets: Sequence[Any],
    *,
    points_per_frame: int = 20000,
    downsample: int = 1,
    max_distance: float = 40.0,
    update_period: float = 0.1,
):
    """Standard Mid-360 LiDAR cfg for F1Tenth. Uses MultiMeshRayCaster so MARL
    envs with dynamic opponent targets work out of the box."""
    from hmclab_isaac.robots._sensors.lidar import make_multimesh_raycaster_cfg

    from .patterns import Mid360PatternCfg

    pattern_cfg = Mid360PatternCfg(
        points_per_frame=points_per_frame,
        frame_index=0,
        downsample=downsample,
    )
    return make_multimesh_raycaster_cfg(
        robot_prim_path=f"{robot_prim_path}/base_link",
        mesh_targets=mesh_targets,
        pattern_cfg=pattern_cfg,
        mount_offset=LIDAR_MOUNT_OFFSET,
        update_period=update_period,
        max_distance=max_distance,
    )


def make_front_camera(
    env_regex_ns: str,
    *,
    width: int = 640,
    height: int = 360,
    update_period: float = 0.033,
):
    """Standard front camera cfg for F1Tenth (RealSense D435-ish, 86° HFOV).

    Note: camera is spawned as a child of the env root (not the robot's
    `camera_link`). The mount offset is applied via the cfg so the camera
    visually appears at the right place in the scene.
    """
    from hmclab_isaac.robots._sensors.camera import make_tiled_camera_cfg

    return make_tiled_camera_cfg(
        prim_path=f"{env_regex_ns}/FrontCamera",
        width=width,
        height=height,
        mount_pos=FRONT_CAMERA_MOUNT_OFFSET,
        focal_length=24.0,
        horizontal_aperture=44.73,
        clipping_range=(0.01, 300.0),
        update_period=update_period,
        data_types=("rgb",),
    )
