"""LiDAR sensor factories (RayCaster / MultiMeshRayCaster based)."""

from __future__ import annotations

from typing import Any, Sequence


def make_raycaster_cfg(
    robot_prim_path: str,
    mesh_targets: Sequence[str],
    pattern_cfg: Any,
    mount_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    update_period: float = 0.1,
    max_distance: float = 40.0,
):
    """Static-scene LiDAR factory (only hits listed meshes, opponent not tracked).

    Args:
        robot_prim_path: Prim path of the robot body the sensor mounts on (e.g.
            "/World/envs/env_.*/Ego/base_link").
        mesh_targets: Static mesh prim paths the rays can hit.
        pattern_cfg: A ray pattern cfg (e.g. BpearlPatternCfg, Mid360PatternCfg).
        mount_offset: Sensor origin offset relative to the parent prim.
        update_period: Seconds between updates (0.1 = 10 Hz).
        max_distance: Max ray length in meters.
    """
    from isaaclab.sensors.ray_caster import RayCasterCfg

    return RayCasterCfg(
        prim_path=robot_prim_path,
        update_period=update_period,
        offset=RayCasterCfg.OffsetCfg(pos=mount_offset),
        pattern_cfg=pattern_cfg,
        max_distance=max_distance,
        mesh_prim_paths=list(mesh_targets),
    )


def make_multimesh_raycaster_cfg(
    robot_prim_path: str,
    mesh_targets: Sequence[Any],
    pattern_cfg: Any,
    mount_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    update_period: float = 0.1,
    max_distance: float = 40.0,
):
    """Dynamic-scene LiDAR factory (tracks moving prims; required for MARL).

    `mesh_targets` may contain strings (static mesh paths) and
    `MultiMeshRayCasterCfg.RaycastTargetCfg` entries for dynamic targets
    (e.g. opponent vehicles).
    """
    from isaaclab.sensors.ray_caster import MultiMeshRayCasterCfg

    return MultiMeshRayCasterCfg(
        prim_path=robot_prim_path,
        update_period=update_period,
        offset=MultiMeshRayCasterCfg.OffsetCfg(pos=mount_offset),
        pattern_cfg=pattern_cfg,
        max_distance=max_distance,
        mesh_prim_paths=list(mesh_targets),
    )
