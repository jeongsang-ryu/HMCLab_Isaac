"""UNICORN_3 — modular SRC_dw chassis + Hokuyo UST-10/20LX 2D LiDAR + camera + NUC.

Composition file at ``UNICORN_3.usd`` uses external payloads for chassis
(``SRC_dw.usd``) and devices (``UST_10_20LX.usd``,
``SG2_AR0233C…usd``, ``NUC14PRO_slim.usd``), so editing any component
USD propagates here automatically.

Hokuyo model-level specs (FOV, channels, wavelength) live in
``devices/hokuyo/spec.SPEC``; only the *vehicle-side* knobs (mount
offset, scan rate, range overrides) are duplicated here so a single
file controls this variant fully.

Joint names match the current ``SRC_dw.usd`` authoring (no ``_joint``
suffix). The 4-bar suspension forms a kinematic loop; prismatic
``*_shock`` joints are demoted by PhysX to max-coord joints — their
spring/damper is authored on the joint via ``PhysicsDriveAPI:linear``
and this Python config does NOT target them.
"""
from __future__ import annotations

import os
from typing import Any, Sequence

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg


USD_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "UNICORN_3.usd")


# ────────────────────────────────────────────────────────────────────
# Geometry (matches SRC_dw chassis as analyzed in COG inspection)
# ────────────────────────────────────────────────────────────────────
WHEELBASE = 0.344
TRACK_WIDTH = 0.259
WHEEL_RADIUS = 0.0525
MAX_STEER = 0.785   # 45° each side — wider than MuSHR's ±28° for tighter cornering
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
# DCMotor — torque-velocity curve auto-caps torque as wheels accelerate,
# preventing the ImplicitActuator failure mode where wheels keep
# producing full torque even when slipping past grip limit.
WHEEL_DCMOTOR_SATURATION = 2.0
WHEEL_DCMOTOR_EFFORT_LIMIT = 0.4
WHEEL_DCMOTOR_VELOCITY_LIMIT = 400.0
WHEEL_DCMOTOR_DAMPING = 1000.0

# Pushed past realistic-servo (10Nm) to "industrial actuator" range.
# Stationary pre-steer is instantaneous, lock-to-lock < 10 ms.
STEERING_EFFORT_LIMIT = 25.0       # was 10
STEERING_STIFFNESS = 800.0         # was 300
STEERING_DAMPING = 60.0            # was 30
STEERING_VELOCITY_LIMIT = 60.0     # was 30

PASSIVE_PIVOT_DAMPING = 0.5


# ────────────────────────────────────────────────────────────────────
# Hokuyo UST 2D LiDAR — mount Xform at <robot>/base_link/lidar
# ────────────────────────────────────────────────────────────────────
LIDAR_MOUNT_PATH_TAIL = "/base_link/lidar"
LIDAR_RATE_HZ = 40.0                # real Hokuyo UST scan rate
LIDAR_HORIZONTAL_RES_DEG: float | None = None   # None → use spec.SPEC
LIDAR_MAX_DISTANCE: float | None = None         # None → use spec.SPEC


# ────────────────────────────────────────────────────────────────────
# Front camera — Isaac Lab creates a fresh UsdGeom.Camera at the
# `<robot>/base_link/cam` Xform mount (USD-authored translate/orient).
# ────────────────────────────────────────────────────────────────────
CAMERA_PRIM_NAME = "front_cam"
CAMERA_OFFSET_POS = (0.15, 0.0, 0.015)
CAMERA_OFFSET_ROT = (1.0, 0.0, 0.0, 0.0)
CAMERA_RATE_HZ = 30.0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_DATA_TYPES = ["rgb", "distance_to_camera"]
CAMERA_FOCAL_LENGTH = 5.0
CAMERA_HORIZONTAL_APERTURE = 20.955
CAMERA_CLIPPING_RANGE = (0.05, 100.0)


# ────────────────────────────────────────────────────────────────────
# Tire compliant-contact material
# ────────────────────────────────────────────────────────────────────
TIRE_STATIC_FRICTION = 1.0
TIRE_DYNAMIC_FRICTION = 1.0
TIRE_RESTITUTION = 0.0
TIRE_COMPLIANT_STIFFNESS = 5.0e5
TIRE_COMPLIANT_DAMPING = 5.0e3


ACTUATORS = {
    # 4WD with DCMotor model. Back-EMF curve caps torque automatically.
    "wheel_drive": DCMotorCfg(
        joint_names_expr=[".*_wheel$"],
        saturation_effort=WHEEL_DCMOTOR_SATURATION,
        effort_limit_sim=WHEEL_DCMOTOR_EFFORT_LIMIT,
        velocity_limit_sim=WHEEL_DCMOTOR_VELOCITY_LIMIT,
        stiffness=0.0,
        damping=WHEEL_DCMOTOR_DAMPING,
        friction=0.0,
    ),
    "steering": ImplicitActuatorCfg(
        joint_names_expr=["fl_steering", "fr_steering"],
        effort_limit_sim=STEERING_EFFORT_LIMIT,
        velocity_limit_sim=STEERING_VELOCITY_LIMIT,
        stiffness=STEERING_STIFFNESS,
        damping=STEERING_DAMPING,
        friction=0.0,
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


def make_hokuyo(
    robot_prim_path: str,
    mesh_targets: Sequence[Any],
    *,
    rate_hz: float | None = None,
    horizontal_res_deg: float | None = None,
    max_distance: float | None = None,
    mount_offset: tuple[float, float, float] | None = None,
):
    """Hokuyo UST 2D LiDAR cfg (270° single-channel scan).

    Defaults read from this module's constants and falls through to
    ``devices/hokuyo/spec.SPEC`` for any value left as ``None``.
    """
    from hmclab_isaac.robots.devices.lidar import make_multimesh_raycaster_cfg
    from hmclab_isaac.robots.devices.hokuyo.spec import SPEC
    from isaaclab.sensors.ray_caster.patterns import LidarPatternCfg

    rate = LIDAR_RATE_HZ if rate_hz is None else float(rate_hz)
    res = horizontal_res_deg if horizontal_res_deg is not None else (
        LIDAR_HORIZONTAL_RES_DEG
        if LIDAR_HORIZONTAL_RES_DEG is not None
        else float(SPEC["azimuth_resolution_deg"])
    )
    rng = max_distance if max_distance is not None else (
        LIDAR_MAX_DISTANCE
        if LIDAR_MAX_DISTANCE is not None
        else float(SPEC["far_range_m"])
    )

    fov = float(SPEC["fov_h_deg"])
    pattern = LidarPatternCfg(
        channels=int(SPEC["channels"]),
        vertical_fov_range=(0.0, 0.0),
        horizontal_fov_range=(-fov / 2.0, fov / 2.0),
        horizontal_res=float(res),
    )
    return make_multimesh_raycaster_cfg(
        robot_prim_path=f"{robot_prim_path}{LIDAR_MOUNT_PATH_TAIL}",
        mesh_targets=mesh_targets,
        pattern_cfg=pattern,
        mount_offset=(tuple(mount_offset)
                      if mount_offset is not None else (0.0, 0.0, 0.0)),
        update_period=1.0 / rate,
        max_distance=rng,
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
    Isaac Lab creates a UsdGeom.Camera with SG2 AR0233C intrinsics.
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


def make_tiled_camera(
    robot_prim_path: str,
    *,
    rate_hz: float = 50.0,
    width: int = 128,
    height: int = 128,
    data_types: list[str] | None = None,
):
    """TiledCamera variant of :func:`make_camera` for parallel RL training.

    TiledCameraCfg batches render products across all envs into one tile, which
    is what RL with N parallel envs needs (CameraCfg allocates one render
    product per env and runs out of VRAM fast).

    Defaults: 128×128 RGB @ 50 Hz. The camera is spawned **as a child of
    the USD-authored ``base_link/cam`` Xform** so it inherits that mount's
    pose (the SRC_dw chassis bakes the front-camera location into ``cam``).
    Offset is identity (no extra translation) — we trust the mount.
    """
    from isaaclab.sensors import TiledCameraCfg

    return TiledCameraCfg(
        prim_path=f"{robot_prim_path}/base_link/cam/{CAMERA_PRIM_NAME}",
        update_period=1.0 / float(rate_hz),
        height=int(height),
        width=int(width),
        data_types=list(["rgb"] if data_types is None else data_types),
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=CAMERA_FOCAL_LENGTH,
            horizontal_aperture=CAMERA_HORIZONTAL_APERTURE,
            clipping_range=CAMERA_CLIPPING_RANGE,
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            convention="world",
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
    """Override tire material at runtime. Returns count of prims updated."""
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
    "WHEEL_DCMOTOR_SATURATION", "WHEEL_DCMOTOR_EFFORT_LIMIT",
    "WHEEL_DCMOTOR_VELOCITY_LIMIT", "WHEEL_DCMOTOR_DAMPING",
    "STEERING_EFFORT_LIMIT", "STEERING_STIFFNESS", "STEERING_DAMPING",
    "STEERING_VELOCITY_LIMIT", "PASSIVE_PIVOT_DAMPING",
    "ACTUATORS",
    "LIDAR_MOUNT_PATH_TAIL", "LIDAR_RATE_HZ",
    "LIDAR_HORIZONTAL_RES_DEG", "LIDAR_MAX_DISTANCE",
    "CAMERA_PRIM_NAME", "CAMERA_OFFSET_POS", "CAMERA_OFFSET_ROT",
    "CAMERA_RATE_HZ", "CAMERA_WIDTH", "CAMERA_HEIGHT", "CAMERA_DATA_TYPES",
    "CAMERA_FOCAL_LENGTH", "CAMERA_HORIZONTAL_APERTURE", "CAMERA_CLIPPING_RANGE",
    "TIRE_STATIC_FRICTION", "TIRE_DYNAMIC_FRICTION", "TIRE_RESTITUTION",
    "TIRE_COMPLIANT_STIFFNESS", "TIRE_COMPLIANT_DAMPING",
    "make_cfg", "make_hokuyo", "make_camera", "make_tiled_camera",
    "apply_tire_material",
    "CFG",
]
