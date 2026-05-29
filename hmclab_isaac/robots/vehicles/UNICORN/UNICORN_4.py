"""UNICORN_4 — modular SRC_dw chassis + Livox Mid-360 LiDAR + camera + NUC.

Composition file at ``UNICORN_4.usd`` uses external payloads for chassis
(``SRC_dw.usd``) and devices (``Mid360.usd``, ``SG2_AR0233C…usd``,
``NUC14PRO_slim.usd``), so editing any component USD propagates here
automatically.

Joint names match the current ``SRC_dw.usd`` authoring (no ``_joint``
suffix; ``fl_wheel`` not ``fl_wheel_joint``). The 4-bar suspension forms
a kinematic loop, so the prismatic ``*_shock`` joints are demoted by
PhysX to max-coord joints — their spring/damper is authored on the
joint via ``PhysicsDriveAPI:linear`` and this Python config does NOT
target them.
"""
from __future__ import annotations

import os
from typing import Any, Sequence

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg


USD_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "UNICORN_4.usd")


# ────────────────────────────────────────────────────────────────────
# Geometry (matches SRC_dw chassis as analyzed in COG inspection)
# ────────────────────────────────────────────────────────────────────
WHEELBASE = 0.344
TRACK_WIDTH = 0.259
WHEEL_RADIUS = 0.0525   # matches Cylinder collider radius in /refs/wheel
MAX_STEER = 0.45
INIT_HEIGHT = 0.15


# ────────────────────────────────────────────────────────────────────
# Solver / rigid-body limits
# ────────────────────────────────────────────────────────────────────
SOLVER_POSITION_ITERATIONS = 4
SOLVER_VELOCITY_ITERATIONS = 0
ENABLE_SELF_COLLISIONS = False
SLEEP_THRESHOLD = 0.005
STABILIZATION_THRESHOLD = 0.001
MAX_LINEAR_VELOCITY = 1000.0
MAX_ANGULAR_VELOCITY = 100000.0
MAX_DEPENETRATION_VELOCITY = 100.0


# ────────────────────────────────────────────────────────────────────
# Actuator tuning
# ────────────────────────────────────────────────────────────────────
WHEEL_DRIVE_EFFORT_LIMIT = 0.4
WHEEL_DRIVE_DAMPING = 1000.0

STEERING_EFFORT_LIMIT = 3.2
STEERING_STIFFNESS = 100.0
STEERING_DAMPING = 10.0

PASSIVE_PIVOT_DAMPING = 0.5


# ────────────────────────────────────────────────────────────────────
# Mid-360 LiDAR — mount Xform at <robot>/base_link/lidar (USD-authored)
# ────────────────────────────────────────────────────────────────────
LIDAR_MOUNT_PATH_TAIL = "/base_link/lidar"
LIDAR_RATE_HZ = 10.0
LIDAR_THROUGHPUT_RAYS_PER_SEC = 240_000
LIDAR_MAX_DISTANCE = 40.0
LIDAR_MIN_RANGE = 0.1
LIDAR_DOWNSAMPLE = 1


# ────────────────────────────────────────────────────────────────────
# Front camera — Isaac Lab creates a fresh UsdGeom.Camera at the
# `<robot>/base_link/cam` Xform mount (USD-authored translate/orient).
# The SG2 device USD ships a UsdGeom.Camera prim with detailed
# intrinsics, but Isaac Lab needs to verify the prim path *before*
# the payload is fully resolved during scene setup, so we let Isaac
# Lab spawn its own camera with the SG2 spec mirrored below.
# ────────────────────────────────────────────────────────────────────
# Camera lives as a direct child of the robot prim (/Robot/front_cam),
# positioned via offset matching the USD cam mount on base_link.
CAMERA_PRIM_NAME = "front_cam"
CAMERA_OFFSET_POS = (0.15, 0.0, 0.015)   # mirrors /SRC_dw/base_link/cam translate
CAMERA_OFFSET_ROT = (1.0, 0.0, 0.0, 0.0)  # identity (look forward)
CAMERA_RATE_HZ = 30.0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_DATA_TYPES = ["rgb", "distance_to_camera"]
# SG2 AR0233C optics
CAMERA_FOCAL_LENGTH = 5.0
CAMERA_HORIZONTAL_APERTURE = 20.955
CAMERA_CLIPPING_RANGE = (0.05, 100.0)


# ────────────────────────────────────────────────────────────────────
# Tire compliant-contact material
# ────────────────────────────────────────────────────────────────────
TIRE_STATIC_FRICTION = 1.2
TIRE_DYNAMIC_FRICTION = 1.0
TIRE_RESTITUTION = 0.0
TIRE_COMPLIANT_STIFFNESS = 1.0e5    # N/m
TIRE_COMPLIANT_DAMPING = 1.0e3      # N·s/m


ACTUATORS = {
    "wheel_drive": ImplicitActuatorCfg(
        joint_names_expr=[".*_wheel$"],
        effort_limit_sim=WHEEL_DRIVE_EFFORT_LIMIT,
        stiffness=0.0,
        damping=WHEEL_DRIVE_DAMPING,
    ),
    "steering": ImplicitActuatorCfg(
        joint_names_expr=["fl_steering", "fr_steering"],
        effort_limit_sim=STEERING_EFFORT_LIMIT,
        stiffness=STEERING_STIFFNESS,
        damping=STEERING_DAMPING,
    ),
    "passive_pivots": ImplicitActuatorCfg(
        joint_names_expr=[".*_shock_upper$", "RevoluteJoint.*"],
        effort_limit_sim=0.0,
        stiffness=0.0,
        damping=PASSIVE_PIVOT_DAMPING,
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
    """Mid-360 raycaster attached to ``<robot>/base_link/lidar`` Xform."""
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
    return make_multimesh_raycaster_cfg(
        robot_prim_path=f"{robot_prim_path}{LIDAR_MOUNT_PATH_TAIL}",
        mesh_targets=mesh_targets,
        pattern_cfg=pattern,
        mount_offset=(tuple(mount_offset)
                      if mount_offset is not None else (0.0, 0.0, 0.0)),
        update_period=1.0 / rate,
        max_distance=(LIDAR_MAX_DISTANCE
                      if max_distance is None else float(max_distance)),
    )


def make_camera(
    robot_prim_path: str,
    *,
    rate_hz: float | None = None,
    width: int | None = None,
    height: int | None = None,
    data_types: list[str] | None = None,
):
    """RGB(+depth) camera at ``<robot>/base_link/cam`` Xform mount.

    Isaac Lab creates a UsdGeom.Camera there using the SG2 AR0233C
    intrinsics specified by ``CAMERA_*`` constants. Mount pose comes
    from the USD-authored translate/orient on the ``cam`` Xform.

    Note: ``args.enable_cameras = True`` required at AppLauncher.
    """
    from isaaclab.sensors import CameraCfg

    return CameraCfg(
        prim_path=f"{robot_prim_path}/{CAMERA_PRIM_NAME}",
        update_period=1.0 / (CAMERA_RATE_HZ if rate_hz is None else float(rate_hz)),
        height=CAMERA_HEIGHT if height is None else int(height),
        width=CAMERA_WIDTH if width is None else int(width),
        data_types=list(CAMERA_DATA_TYPES if data_types is None else data_types),
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=CAMERA_FOCAL_LENGTH,
            horizontal_aperture=CAMERA_HORIZONTAL_APERTURE,
            clipping_range=CAMERA_CLIPPING_RANGE,
        ),
        offset=CameraCfg.OffsetCfg(
            pos=CAMERA_OFFSET_POS,
            rot=CAMERA_OFFSET_ROT,
            convention="ros",
        ),
    )


def apply_tire_material(
    robot_prim_path: str,
    *,
    static_friction: float | None = None,
    dynamic_friction: float | None = None,
    restitution: float | None = None,
    compliant_stiffness: float | None = None,
    compliant_damping: float | None = None,
) -> int:
    """Override tire material at runtime. See UNICORN_1.apply_tire_material —
    same semantics. Returns the number of material prims updated (0 if
    the chassis no longer ships a `tire_compliant` material binding)."""
    import re as _re

    import omni.usd
    from pxr import PhysxSchema, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    pat = _re.compile(
        robot_prim_path.replace(".*", "[^/]+")
        + r"/PhysicsMaterials/tire_compliant$"
    )

    sf = TIRE_STATIC_FRICTION if static_friction is None else float(static_friction)
    df = TIRE_DYNAMIC_FRICTION if dynamic_friction is None else float(dynamic_friction)
    rs = TIRE_RESTITUTION if restitution is None else float(restitution)
    s = TIRE_COMPLIANT_STIFFNESS if compliant_stiffness is None else float(compliant_stiffness)
    d = TIRE_COMPLIANT_DAMPING if compliant_damping is None else float(compliant_damping)

    n = 0
    for prim in stage.Traverse():
        if not pat.match(prim.GetPath().pathString):
            continue
        phys = UsdPhysics.MaterialAPI(prim)
        physx = PhysxSchema.PhysxMaterialAPI(prim)
        phys.CreateStaticFrictionAttr().Set(sf)
        phys.CreateDynamicFrictionAttr().Set(df)
        phys.CreateRestitutionAttr().Set(rs)
        physx.CreateCompliantContactStiffnessAttr().Set(s)
        physx.CreateCompliantContactDampingAttr().Set(d)
        n += 1
    return n


__all__ = [
    "USD_PATH",
    "WHEELBASE", "TRACK_WIDTH", "WHEEL_RADIUS", "MAX_STEER", "INIT_HEIGHT",
    "SOLVER_POSITION_ITERATIONS", "SOLVER_VELOCITY_ITERATIONS",
    "ENABLE_SELF_COLLISIONS", "SLEEP_THRESHOLD", "STABILIZATION_THRESHOLD",
    "MAX_LINEAR_VELOCITY", "MAX_ANGULAR_VELOCITY", "MAX_DEPENETRATION_VELOCITY",
    "WHEEL_DRIVE_EFFORT_LIMIT", "WHEEL_DRIVE_DAMPING",
    "STEERING_EFFORT_LIMIT", "STEERING_STIFFNESS", "STEERING_DAMPING",
    "PASSIVE_PIVOT_DAMPING",
    "ACTUATORS",
    "LIDAR_MOUNT_PATH_TAIL", "LIDAR_RATE_HZ", "LIDAR_THROUGHPUT_RAYS_PER_SEC",
    "LIDAR_MAX_DISTANCE", "LIDAR_MIN_RANGE", "LIDAR_DOWNSAMPLE",
    "CAMERA_PRIM_NAME", "CAMERA_OFFSET_POS", "CAMERA_OFFSET_ROT",
    "CAMERA_RATE_HZ", "CAMERA_WIDTH", "CAMERA_HEIGHT", "CAMERA_DATA_TYPES",
    "CAMERA_FOCAL_LENGTH", "CAMERA_HORIZONTAL_APERTURE", "CAMERA_CLIPPING_RANGE",
    "TIRE_STATIC_FRICTION", "TIRE_DYNAMIC_FRICTION", "TIRE_RESTITUTION",
    "TIRE_COMPLIANT_STIFFNESS", "TIRE_COMPLIANT_DAMPING",
    "make_cfg", "make_mid360", "make_camera", "apply_tire_material",
    "CFG",
]
