"""HAMA_2 — DBXL_simple chassis + Livox Mid-360 LiDAR.

Single source of truth for everything about this variant. See HAMA_1.py
for design rationale; only the suspension model differs (one prismatic
slider per corner, no wishbone, no closed loops). 10 articulation joints.
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
USD_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "HAMA_2.usd")


# ────────────────────────────────────────────────────────────────────
# Geometry
# ────────────────────────────────────────────────────────────────────
WHEELBASE = 0.571
TRACK_WIDTH = 0.520
WHEEL_RADIUS = 0.096
MAX_STEER = 0.45
INIT_HEIGHT = 0.15


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
WHEEL_DRIVE_EFFORT_LIMIT = 8.0
WHEEL_DRIVE_DAMPING = 12.0

STEERING_EFFORT_LIMIT = 200.0
STEERING_STIFFNESS = 2000.0
STEERING_DAMPING = 80.0

# Suspension on the prismatic slider (`*_sus2_joint`). No rocker on this
# variant; the slider's spring/damper carries the corner directly.
SUSPENSION_STIFFNESS = 2000.0
SUSPENSION_DAMPING = 100.0


# ────────────────────────────────────────────────────────────────────
# Mid-360 LiDAR
# ────────────────────────────────────────────────────────────────────
# Informational: mirrors the USD-authored translate on
# `/HAMA/base_link/lidar`. Python factory tracks that Xform directly,
# so changing this constant alone does nothing — edit the USD instead.
LIDAR_MOUNT_OFFSET = (0.05, 0.0, 0.30)
LIDAR_RATE_HZ = 10.0
LIDAR_THROUGHPUT_RAYS_PER_SEC = 240_000
LIDAR_MAX_DISTANCE = 40.0
LIDAR_MIN_RANGE = 0.1
LIDAR_DOWNSAMPLE = 1


# ────────────────────────────────────────────────────────────────────
# Actuator dict
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
    "suspension": ImplicitActuatorCfg(
        joint_names_expr=[".*_sus2_joint"],
        effort_limit_sim=0.0,
        stiffness=SUSPENSION_STIFFNESS,
        damping=SUSPENSION_DAMPING,
    ),
}


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
    """See ``HAMA_1.make_mid360`` — same semantics."""
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
    # Track the lidar Xform directly (USD-authored translate is the mount).
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
    "WHEELBASE", "TRACK_WIDTH", "WHEEL_RADIUS", "MAX_STEER", "INIT_HEIGHT",
    "SOLVER_POSITION_ITERATIONS", "SOLVER_VELOCITY_ITERATIONS",
    "ENABLE_SELF_COLLISIONS", "SLEEP_THRESHOLD", "STABILIZATION_THRESHOLD",
    "MAX_LINEAR_VELOCITY", "MAX_ANGULAR_VELOCITY", "MAX_DEPENETRATION_VELOCITY",
    "WHEEL_DRIVE_EFFORT_LIMIT", "WHEEL_DRIVE_DAMPING",
    "STEERING_EFFORT_LIMIT", "STEERING_STIFFNESS", "STEERING_DAMPING",
    "SUSPENSION_STIFFNESS", "SUSPENSION_DAMPING",
    "ACTUATORS",
    "LIDAR_MOUNT_OFFSET", "LIDAR_RATE_HZ", "LIDAR_THROUGHPUT_RAYS_PER_SEC",
    "LIDAR_MAX_DISTANCE", "LIDAR_MIN_RANGE", "LIDAR_DOWNSAMPLE",
    "make_cfg", "make_mid360",
    "CFG",
]
