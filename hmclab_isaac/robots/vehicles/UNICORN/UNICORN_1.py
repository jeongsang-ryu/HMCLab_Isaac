"""UNICORN_1 — SRC_dw chassis + Livox Mid-360 LiDAR.

Single source of truth — see HAMA_1.py for the structure rationale.
SRC_dw differs from DBXL_dw only in chassis sub-USD topology (extra
``*_upper_knuckle_joint`` ball joints that are constraint-only); the
articulation surface and tuning are identical.
"""
from __future__ import annotations

import os
from typing import Any, Sequence

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg


USD_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "UNICORN_1.usd")


# ────────────────────────────────────────────────────────────────────
# Geometry
# ────────────────────────────────────────────────────────────────────
WHEELBASE = 0.331
TRACK_WIDTH = 0.270
WHEEL_RADIUS = 0.0525   # matches Cylinder collider radius in SRC_dw_bw.usd
MAX_STEER = 0.45
INIT_HEIGHT = 0.15


# ────────────────────────────────────────────────────────────────────
# Solver / rigid-body limits
# ────────────────────────────────────────────────────────────────────
SOLVER_POSITION_ITERATIONS = 4      # MuSHR exact (was 32 — caused vibration)
SOLVER_VELOCITY_ITERATIONS = 0
ENABLE_SELF_COLLISIONS = False
SLEEP_THRESHOLD = 0.005
STABILIZATION_THRESHOLD = 0.001
MAX_LINEAR_VELOCITY = 1000.0       # was 20 — too low cap clipped solver iter ω
MAX_ANGULAR_VELOCITY = 100000.0    # was 2000 — root cause of "drag wall"
MAX_DEPENETRATION_VELOCITY = 100.0


# ────────────────────────────────────────────────────────────────────
# Actuator tuning
# ────────────────────────────────────────────────────────────────────
WHEEL_DRIVE_EFFORT_LIMIT = 0.4      # 7 m/s² peak accel × 4 kg / 4 wheels × r
WHEEL_DRIVE_DAMPING = 1000.0        # MuSHR exact

STEERING_EFFORT_LIMIT = 3.2         # MuSHR exact
STEERING_STIFFNESS = 100.0
STEERING_DAMPING = 10.0

SHOCK_STIFFNESS = 400.0
SHOCK_DAMPING = 30.0

PASSIVE_PIVOT_DAMPING = 0.5


# ────────────────────────────────────────────────────────────────────
# Mid-360 LiDAR
# ────────────────────────────────────────────────────────────────────
# Informational: mirrors the USD-authored translate on
# `/UNICORN/base_link/lidar`. Python factory tracks that Xform directly,
# so changing this constant alone does nothing — edit the USD instead.
LIDAR_MOUNT_OFFSET = (0.05, 0.0, 0.30)
LIDAR_RATE_HZ = 10.0
LIDAR_THROUGHPUT_RAYS_PER_SEC = 240_000
LIDAR_MAX_DISTANCE = 40.0
LIDAR_MIN_RANGE = 0.1
LIDAR_DOWNSAMPLE = 1


# ────────────────────────────────────────────────────────────────────
# Front camera (UsdGeom.Camera authored in USD at /UNICORN/base_link/cam)
# ────────────────────────────────────────────────────────────────────
# Mount pose (translate/orient + intrinsics) lives in the USD's Camera
# prim. The Python factory attaches with ``spawn=None`` so it tracks
# whatever was authored. Edit the camera in GUI and re-save the USD.
CAMERA_RATE_HZ = 30.0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_DATA_TYPES = ["rgb", "distance_to_camera"]   # add "normals" / "semantic_segmentation" if needed


# ────────────────────────────────────────────────────────────────────
# Tire compliant-contact material (bound to /Flattened_Prototype_13/tire_0)
# ────────────────────────────────────────────────────────────────────
# Defaults authored in `hmclab_isaac/robots/materials/tire_compliant.usd`.
# To deviate per experiment without rebuilding the USD, edit the
# constants below and call ``apply_tire_material()`` after sim.reset()
# and before sim.step(). All four tires of every spawned env are
# updated.
TIRE_STATIC_FRICTION = 5.0
TIRE_DYNAMIC_FRICTION = 4.0
TIRE_RESTITUTION = 2.0
TIRE_COMPLIANT_STIFFNESS = 1.0e5    # N/m
TIRE_COMPLIANT_DAMPING = 1.0e3      # N·s/m


# Joint regex updated for the current chassis (SRC_dw.usd): no `_joint`
# suffix on names; the prismatic *_shock joints are NOT in the articulation
# (4-bar suspension loop demotes them to max-coord joints whose spring is
# authored directly on the joint via PhysicsDriveAPI:linear).
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
    """See HAMA_1.make_mid360 — same semantics."""
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


def make_camera(
    robot_prim_path: str,
    *,
    rate_hz: float | None = None,
    width: int | None = None,
    height: int | None = None,
    data_types: list[str] | None = None,
):
    """RGB(+depth) camera attached to the USD-authored Camera prim at
    ``<robot>/base_link/cam``.

    ``spawn=None`` means we don't create a new camera; we attach to
    whatever the user authored in USD. Move/rotate/edit intrinsics in
    GUI and re-save the USD — the Python factory tracks it.

    Note: the launcher must have ``args.enable_cameras = True`` for
    this sensor to actually render. data_types default to RGB +
    depth; add ``"normals"`` / ``"semantic_segmentation"`` etc. if
    needed.
    """
    from isaaclab.sensors import CameraCfg

    return CameraCfg(
        prim_path=f"{robot_prim_path}/base_link/cam",
        update_period=1.0 / (CAMERA_RATE_HZ if rate_hz is None else float(rate_hz)),
        height=CAMERA_HEIGHT if height is None else int(height),
        width=CAMERA_WIDTH if width is None else int(width),
        data_types=list(CAMERA_DATA_TYPES if data_types is None else data_types),
        spawn=None,
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
    """Override tire material properties at runtime.

    The chassis USD references a shared material at
    ``<robot>/PhysicsMaterials/tire_compliant`` and binds it to
    ``tire_0``. This function walks the composed stage, finds every
    spawned vehicle's copy of that material prim, and overwrites the
    physics + compliant-contact attributes.

    Defaults read from this module's ``TIRE_*`` constants; pass any
    keyword to deviate per call.

    Call after ``sim.reset()`` and before ``sim.step()``. Returns the
    number of material prims updated.
    """
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
    "SHOCK_STIFFNESS", "SHOCK_DAMPING", "PASSIVE_PIVOT_DAMPING",
    "ACTUATORS",
    "LIDAR_MOUNT_OFFSET", "LIDAR_RATE_HZ", "LIDAR_THROUGHPUT_RAYS_PER_SEC",
    "LIDAR_MAX_DISTANCE", "LIDAR_MIN_RANGE", "LIDAR_DOWNSAMPLE",
    "CAMERA_RATE_HZ", "CAMERA_WIDTH", "CAMERA_HEIGHT", "CAMERA_DATA_TYPES",
    "TIRE_STATIC_FRICTION", "TIRE_DYNAMIC_FRICTION", "TIRE_RESTITUTION",
    "TIRE_COMPLIANT_STIFFNESS", "TIRE_COMPLIANT_DAMPING",
    "make_cfg", "make_mid360", "make_camera", "apply_tire_material",
    "CFG",
]
