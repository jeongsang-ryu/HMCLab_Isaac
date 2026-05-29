"""UNICORN_2 — SRC_simple chassis + Hokuyo UST-20LX 2D LiDAR.

Single source of truth — all suspension / sensor params live here.
Hokuyo model-level specs (FOV, channels, wavelength) live in
``devices/hokuyo/spec.SPEC``; only the *vehicle-side* knobs (mount
offset, scan rate, range overrides) are duplicated here so a single
file controls this variant fully.
"""
from __future__ import annotations

import os
from typing import Any, Sequence

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg


USD_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "UNICORN_2.usd")


# ────────────────────────────────────────────────────────────────────
# Geometry
# ────────────────────────────────────────────────────────────────────
WHEELBASE = 0.331
TRACK_WIDTH = 0.250
WHEEL_RADIUS = 0.096
MAX_STEER = 0.488   # MuSHR drift env value (≈28°). Matches WheeledLab MushrDriftRLEnvCfg.actions.scale.
INIT_HEIGHT = 0.2  # was 0.5 — too high, suspension joint limits ±0.01 violently extended at spawn


# ────────────────────────────────────────────────────────────────────
# Solver / rigid-body limits
# ────────────────────────────────────────────────────────────────────
SOLVER_POSITION_ITERATIONS = 4      # match MuSHR — low iter gives numerical damping that absorbs vibration
SOLVER_VELOCITY_ITERATIONS = 0
ENABLE_SELF_COLLISIONS = False
SLEEP_THRESHOLD = 0.005
STABILIZATION_THRESHOLD = 0.001
MAX_LINEAR_VELOCITY = 1000.0       # was 20 — too low cap caused phantom drag
MAX_ANGULAR_VELOCITY = 100000.0    # was 2000 — same root-cause of "drag wall"
MAX_DEPENETRATION_VELOCITY = 100.0


# ────────────────────────────────────────────────────────────────────
# Actuator tuning
# ────────────────────────────────────────────────────────────────────
# DCMotor — push to wheelie limit. Wheel Iyy auto-like (0.0002) gives
# m_eff ≈ 4.29 kg; effort 0.95 Nm gives motor force 72.4 N at 97%
# of wheelie threshold 75 N. Edge of stable.
# Sim-to-real: matched to actual vehicle measurements.
# 7 m/s² peak accel × m_eff 4.25 kg × wheel_radius 0.0525 / 4 wheels = 0.39 Nm
WHEEL_DCMOTOR_SATURATION = 2.0       # back-EMF curve corner ~80% v_lim
WHEEL_DCMOTOR_EFFORT_LIMIT = 0.4     # 7 m/s² peak accel match
WHEEL_DCMOTOR_VELOCITY_LIMIT = 400.0   # 400 rad/s × 0.0525 m = 21 m/s no-load (real ~20 m/s)
WHEEL_DCMOTOR_DAMPING = 1000.0  # MuSHR exact

# Steering — exact match to WheeledLab.
STEERING_EFFORT_LIMIT = 3.2          # MuSHR exact
STEERING_STIFFNESS = 100.0           # MuSHR exact
STEERING_DAMPING = 10.0              # MuSHR exact
STEERING_VELOCITY_LIMIT = 10.0       # MuSHR exact

# Suspension — MuSHR exact (Coulomb only, NO viscous damping). Earlier
# we'd set damping=1000 thinking it would absorb bumpy terrain, but on
# flat ground it sucks 30+ N power from motor via tiny pitch oscillations.
SUSPENSION_STIFFNESS = 1.0e8
SUSPENSION_DAMPING = 0.0             # 0 not 1000 — eliminates "phantom drag"
SUSPENSION_FRICTION = 0.5


# ────────────────────────────────────────────────────────────────────
# Hokuyo UST-20LX 2D LiDAR
# (model-level spec is in devices/hokuyo/spec.SPEC; what's below is
# this *vehicle's* mounting and override knobs)
# ────────────────────────────────────────────────────────────────────
# Informational: mirrors the USD-authored translate on
# `/UNICORN/base_link/lidar`. Python factory tracks that Xform directly,
# so changing this constant alone does nothing — edit the USD instead.
LIDAR_MOUNT_OFFSET = (0.05, 0.0, 0.20)
LIDAR_RATE_HZ = 40.0                # real Hokuyo UST-20LX scan rate
LIDAR_HORIZONTAL_RES_DEG: float | None = None   # None → use spec.SPEC
LIDAR_MAX_DISTANCE: float | None = None         # None → use spec.SPEC


# ────────────────────────────────────────────────────────────────────
# Tire compliant-contact material (bound to /Flattened_Prototype_13/tire_0)
# ────────────────────────────────────────────────────────────────────
# Defaults authored in `hmclab_isaac/robots/materials/tire_compliant.usd`.
# Override per experiment by editing constants and calling
# ``apply_tire_material()`` after sim.reset() and before sim.step().
TIRE_STATIC_FRICTION = 1.0          # mid-range: enough grip without heavy drag from steering offset
TIRE_DYNAMIC_FRICTION = 1.0
TIRE_RESTITUTION = 0.0
# Compliant ON to absorb micro-bumps from track mesh facets.
TIRE_COMPLIANT_STIFFNESS = 5.0e5
TIRE_COMPLIANT_DAMPING = 5.0e3


# Joint regex updated for SRC_dw.usd chassis (no `_joint` suffix). The
# prismatic *_shock joints are NOT in the articulation (4-bar suspension
# loop demotes them); their spring/damper is authored directly on the
# joint via PhysicsDriveAPI:linear, so the suspension actuator is gone.
PASSIVE_PIVOT_DAMPING = 0.5

ACTUATORS = {
    # 4WD with DCMotor model. Torque-velocity curve (back-EMF) caps
    # torque automatically as wheels accelerate, preventing the
    # ImplicitActuator failure mode where wheels keep producing full
    # torque even when slipping past grip limit.
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
    """Hokuyo UST-20LX 2D LiDAR cfg.

    Defaults read from this module's constants (and falls through to
    ``devices/hokuyo/spec.SPEC`` for any value left as ``None``).
    Override per call to deviate.
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
    # Track the lidar Xform directly (USD-authored translate is the mount).
    return make_multimesh_raycaster_cfg(
        robot_prim_path=f"{robot_prim_path}/base_link/lidar",
        mesh_targets=mesh_targets,
        pattern_cfg=pattern,
        mount_offset=(tuple(mount_offset)
                      if mount_offset is not None else (0.0, 0.0, 0.0)),
        update_period=1.0 / rate,
        max_distance=rng,
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
    same semantics, applies to every spawned vehicle's composed copy of
    ``<robot>/PhysicsMaterials/tire_compliant``."""
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
    "SUSPENSION_STIFFNESS", "SUSPENSION_DAMPING",
    "ACTUATORS",
    "LIDAR_MOUNT_OFFSET", "LIDAR_RATE_HZ",
    "LIDAR_HORIZONTAL_RES_DEG", "LIDAR_MAX_DISTANCE",
    "TIRE_STATIC_FRICTION", "TIRE_DYNAMIC_FRICTION", "TIRE_RESTITUTION",
    "TIRE_COMPLIANT_STIFFNESS", "TIRE_COMPLIANT_DAMPING",
    "make_cfg", "make_hokuyo", "apply_tire_material",
    "CFG",
]
