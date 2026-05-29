"""HAMA_1 — DBXL_dw chassis + Livox Mid-360 LiDAR.

Single source of truth for **everything** about this vehicle variant:
geometry, solver, actuator (suspension / drive / steering) tuning,
and Mid-360 sensor parameters all live as module-level constants.
:func:`make_cfg` and :func:`make_mid360` read those constants as
defaults; pass keyword overrides to deviate per call site.

If you want to spawn an opponent without LiDAR, just don't call
:func:`make_mid360` for that instance — the Mid-360 visual is still
in the USD via reference, but no raycaster is created.

Variant detail: full double-wishbone with closed kinematic loops.
Articulation exposes 26 of 30 USD joints (the four
``*_upper_knuckle_joint`` ball joints are constraint-only). Shock
spring is modelled on ``*_sus1_joint`` (rocker) because the prismatic
``*_sus2_joint`` is rigidified by the closed loop.
"""
from __future__ import annotations

import os
from typing import Any, Sequence

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg


# ────────────────────────────────────────────────────────────────────
# Asset path
# ────────────────────────────────────────────────────────────────────
USD_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "HAMA_1.usd")


# ────────────────────────────────────────────────────────────────────
# Geometry
# ────────────────────────────────────────────────────────────────────
WHEELBASE = 0.571
TRACK_WIDTH = 0.520
WHEEL_RADIUS = 0.096
MAX_STEER = 0.45                # rad
INIT_HEIGHT = 0.15              # spawn z above ground


# ────────────────────────────────────────────────────────────────────
# Articulation solver / rigid-body limits
# ────────────────────────────────────────────────────────────────────
SOLVER_POSITION_ITERATIONS = 32
SOLVER_VELOCITY_ITERATIONS = 8
ENABLE_SELF_COLLISIONS = False
SLEEP_THRESHOLD = 0.005
STABILIZATION_THRESHOLD = 0.001
MAX_LINEAR_VELOCITY = 20.0
MAX_ANGULAR_VELOCITY = 2000.0
MAX_DEPENETRATION_VELOCITY = 10.0


# ────────────────────────────────────────────────────────────────────
# Actuator tuning
# ────────────────────────────────────────────────────────────────────
WHEEL_DRIVE_EFFORT_LIMIT = 8.0     # Nm per wheel
WHEEL_DRIVE_DAMPING = 12.0

STEERING_EFFORT_LIMIT = 200.0
STEERING_STIFFNESS = 2000.0
STEERING_DAMPING = 80.0

# Shock spring on the rocker joint (`*_sus1_joint`); prismatic
# `*_sus2_joint` is rigidified by the closed-loop linkage.
SHOCK_STIFFNESS = 400.0
SHOCK_DAMPING = 30.0

PASSIVE_PIVOT_DAMPING = 0.5


# ────────────────────────────────────────────────────────────────────
# Mid-360 LiDAR
# ────────────────────────────────────────────────────────────────────
# LIDAR_MOUNT_OFFSET is informational: it mirrors the USD-authored
# translate on `/HAMA/base_link/lidar`. The Python sensor factory
# tracks that Xform directly, so changing this constant alone does
# nothing — edit the USD's lidar Xform translate instead (or rebuild
# via scripts/build_vehicles.py).
LIDAR_MOUNT_OFFSET = (0.05, 0.0, 0.30)
LIDAR_RATE_HZ = 10.0
# Total ray throughput (rays/sec). 240K matches livox_laser_simulation
# Gazebo plugin (24K/frame @ 10 Hz). Livox datasheet is 200K.
LIDAR_THROUGHPUT_RAYS_PER_SEC = 240_000
LIDAR_MAX_DISTANCE = 40.0
LIDAR_MIN_RANGE = 0.1
LIDAR_DOWNSAMPLE = 1


# ────────────────────────────────────────────────────────────────────
# Actuator dict (built from the constants above)
# ────────────────────────────────────────────────────────────────────
ACTUATORS = {
    "wheel_drive": ImplicitActuatorCfg(
        joint_names_expr=[".*_wheel_joint"],
        effort_limit_sim=WHEEL_DRIVE_EFFORT_LIMIT,
        stiffness=0.0,
        damping=WHEEL_DRIVE_DAMPING,
    ),
    "steering": ImplicitActuatorCfg(
        joint_names_expr=["fl_steering_joint", "fr_steering_joint"],
        effort_limit_sim=STEERING_EFFORT_LIMIT,
        stiffness=STEERING_STIFFNESS,
        damping=STEERING_DAMPING,
    ),
    "shock_rocker": ImplicitActuatorCfg(
        joint_names_expr=[".*_sus1_joint"],
        effort_limit_sim=0.0,
        stiffness=SHOCK_STIFFNESS,
        damping=SHOCK_DAMPING,
    ),
    # `.*_knuckle_joint` covers both `_knuckle_joint` and any hidden
    # `_upper_knuckle_joint` if the closed-loop is removed later.
    "passive_pivots": ImplicitActuatorCfg(
        joint_names_expr=[
            ".*_under_arm_joint",
            ".*_upper_arm_joint",
            ".*_knuckle_joint",
            ".*_sus2_under_joint",
        ],
        effort_limit_sim=0.0,
        stiffness=0.0,
        damping=PASSIVE_PIVOT_DAMPING,
    ),
}


# ────────────────────────────────────────────────────────────────────
# Articulation cfg factory
# ────────────────────────────────────────────────────────────────────
def make_cfg() -> ArticulationCfg:
    return ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=USD_PATH,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                max_linear_velocity=MAX_LINEAR_VELOCITY,
                max_angular_velocity=MAX_ANGULAR_VELOCITY,
                max_depenetration_velocity=MAX_DEPENETRATION_VELOCITY,
                enable_gyroscopic_forces=True,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=ENABLE_SELF_COLLISIONS,
                solver_position_iteration_count=SOLVER_POSITION_ITERATIONS,
                solver_velocity_iteration_count=SOLVER_VELOCITY_ITERATIONS,
                sleep_threshold=SLEEP_THRESHOLD,
                stabilization_threshold=STABILIZATION_THRESHOLD,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, INIT_HEIGHT)),
        actuators=ACTUATORS,
    )


CFG = make_cfg()


# ────────────────────────────────────────────────────────────────────
# Mid-360 sensor factory
# ────────────────────────────────────────────────────────────────────
def make_mid360(
    robot_prim_path: str,
    mesh_targets: Sequence[Any],
    *,
    rate_hz: float | None = None,
    throughput_rays_per_sec: int | None = None,
    points_per_frame: int | None = None,
    max_distance: float | None = None,
    downsample: int | None = None,
    mount_offset: tuple[float, float, float] | None = None,
):
    """Mid-360 LiDAR cfg.

    Defaults read from this module's constants. Pass any keyword to
    override per call (e.g. ``rate_hz=100`` for a smoother viz, or
    ``points_per_frame=2400`` for a smaller per-frame batch).

    Args:
        robot_prim_path: Articulation prim path, e.g.
            ``/World/envs/env_.*/Ego``. The sensor mounts on
            ``<robot_prim_path>/base_link``.
        mesh_targets: Mix of static prim path strings and
            ``MultiMeshRayCasterCfg.RaycastTargetCfg`` for dynamic targets.
    """
    from hmclab_isaac.robots.devices.lidar import make_multimesh_raycaster_cfg
    from hmclab_isaac.robots.devices.mid360.patterns import Mid360PatternCfg

    rate = LIDAR_RATE_HZ if rate_hz is None else float(rate_hz)
    if points_per_frame is None:
        thru = (LIDAR_THROUGHPUT_RAYS_PER_SEC
                if throughput_rays_per_sec is None
                else int(throughput_rays_per_sec))
        points_per_frame = max(1, int(round(thru / rate)))

    pattern = Mid360PatternCfg(
        points_per_frame=points_per_frame,
        frame_index=0,
        downsample=LIDAR_DOWNSAMPLE if downsample is None else int(downsample),
    )
    # Track the lidar Xform directly: its USD-authored translate IS the
    # mount position. Moving the prim in GUI auto-propagates to sensor
    # pose. mount_offset is an *additional* override on top.
    return make_multimesh_raycaster_cfg(
        robot_prim_path=f"{robot_prim_path}/base_link/lidar",
        mesh_targets=mesh_targets,
        pattern_cfg=pattern,
        mount_offset=(tuple(mount_offset)
                      if mount_offset is not None else (0.0, 0.0, 0.0)),
        update_period=1.0 / rate,
        max_distance=(LIDAR_MAX_DISTANCE
                      if max_distance is None else float(max_distance)),
    )


__all__ = [
    "USD_PATH",
    # geometry
    "WHEELBASE", "TRACK_WIDTH", "WHEEL_RADIUS", "MAX_STEER", "INIT_HEIGHT",
    # solver
    "SOLVER_POSITION_ITERATIONS", "SOLVER_VELOCITY_ITERATIONS",
    "ENABLE_SELF_COLLISIONS", "SLEEP_THRESHOLD", "STABILIZATION_THRESHOLD",
    "MAX_LINEAR_VELOCITY", "MAX_ANGULAR_VELOCITY", "MAX_DEPENETRATION_VELOCITY",
    # actuators (constants + dict)
    "WHEEL_DRIVE_EFFORT_LIMIT", "WHEEL_DRIVE_DAMPING",
    "STEERING_EFFORT_LIMIT", "STEERING_STIFFNESS", "STEERING_DAMPING",
    "SHOCK_STIFFNESS", "SHOCK_DAMPING", "PASSIVE_PIVOT_DAMPING",
    "ACTUATORS",
    # LiDAR
    "LIDAR_MOUNT_OFFSET", "LIDAR_RATE_HZ", "LIDAR_THROUGHPUT_RAYS_PER_SEC",
    "LIDAR_MAX_DISTANCE", "LIDAR_MIN_RANGE", "LIDAR_DOWNSAMPLE",
    # factories
    "make_cfg", "make_mid360",
    "CFG",
]
