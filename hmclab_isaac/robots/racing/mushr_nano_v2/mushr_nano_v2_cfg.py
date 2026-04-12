"""ArticulationCfg for the MuSHR.io Nano v2 racing rover.

Based on the joint structure documented in `README.md`:

    back/front left/right wheel_suspension  (passive springs)
    back/front left/right wheel_throttle    (drive — 4WD)
    front_left/right_wheel_steer            (steer)

Template — leave `prim_path` as default and `.replace(prim_path=...)` from
the caller. Wheelbase/wheel-radius constants are based on the real MuSHR.io
spec (1/10 scale RC car); measure from the USD if you need exact values.
"""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

_ASSET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
USD_PATH = os.path.join(_ASSET_DIR, "mushr_nano_v2.usd")

# MuSHR.io Nano spec (~1/10 RC car, similar class to F1Tenth)
WHEELBASE = 0.32
TRACK_WIDTH = 0.24
WHEEL_RADIUS = 0.05
MAX_STEER = 0.4  # rad (~23 deg)

MUSHR_NANO_V2_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=20.0,
            max_angular_velocity=2000.0,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=8,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.08),
        joint_pos={
            ".*_wheel_suspension": 0.0,
            ".*_wheel_throttle": 0.0,
            "front_.*_wheel_steer": 0.0,
        },
    ),
    # Realistic torque limits for a ~3 kg 1/10-scale RC vehicle. With the
    # original `effort_limit_sim=500` the 4WD throttle applied enough torque
    # to lift the nose and flip the car on hard acceleration. Lowering to
    # ~1 Nm per wheel gives peak force ~20 N/wheel, total ~80 N → ~27 m/s²
    # which settles instead of cartwheeling.
    actuators={
        # Rear wheels drive (RWD) — 4WD was causing wheel scrub + nose-lift.
        # Front throttle joints are left in a free-spin actuator so they
        # still behave as articulated joints but don't apply any torque.
        "throttle_rear": ImplicitActuatorCfg(
            joint_names_expr=["back_.*_wheel_throttle"],
            effort_limit_sim=12.0,
            stiffness=0.0,
            damping=15.0,
        ),
        "throttle_front_free": ImplicitActuatorCfg(
            joint_names_expr=["front_.*_wheel_throttle"],
            effort_limit_sim=0.001,
            stiffness=0.0,
            damping=0.0,
        ),
        "steering": ImplicitActuatorCfg(
            joint_names_expr=["front_.*_wheel_steer"],
            effort_limit_sim=20.0,
            stiffness=80.0,
            damping=10.0,
        ),
        "suspension": ImplicitActuatorCfg(
            joint_names_expr=[".*_wheel_suspension"],
            effort_limit_sim=0.0,
            stiffness=2000.0,  # passive spring approximation
            damping=50.0,
        ),
    },
)

def add_chassis_collider(prim_path: str) -> bool:
    """Add a box collider to the MuSHR chassis (`base_link`).

    The stock MuSHR Nano v2 USD only has colliders on the four wheel links
    and the laser/camera mounts. The chassis plate itself has no collider,
    so two vehicles can visually touch and then the top deck slides through
    the sibling's body. This helper attaches a `UsdPhysics.CollisionAPI`
    plus a dynamic-body-safe box approximation to each matching base_link.

    Args:
        prim_path: the top prim path used in the spawn (e.g. `/World/Ego`
            or `/World/envs/env_.*/Ego`). The helper traverses under this
            path and adds the collider to any `base_link` found.

    Returns True if at least one collider was added.
    """
    import omni.usd
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics, Vt

    stage = omni.usd.get_context().get_stage()
    added = False
    for prim in stage.Traverse():
        p = prim.GetPath().pathString
        if not p.startswith(prim_path.split("/env_")[0]):
            continue
        if not p.endswith("/base_link"):
            continue
        col_path = f"{p}/chassis_col"
        if stage.GetPrimAtPath(col_path).IsValid():
            continue
        # Unit cube (half-extents 0.5) — we scale-then-translate below so
        # the final AABB half-extents are (0.18, 0.10, 0.04) and the
        # bottom face sits at base_link z=+0.05 (above the wheel top).
        cube = UsdGeom.Cube.Define(stage, col_path)
        cube.CreateSizeAttr(1.0)
        xform = UsdGeom.Xformable(cube)
        xform.ClearXformOpOrder()
        # Scale first — transforms the unit cube to the body size
        xform.AddScaleOp().Set(Gf.Vec3f(0.36, 0.20, 0.08))
        # Then translate in the SCALED local space so the box sits above
        # the wheel line. With scale 0.08 in z, local z=1.0 is 0.08 m in
        # world — we want centre at 0.09 m, so translate z = 0.09 / 0.08 = 1.125.
        xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 1.125))
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        UsdGeom.Imageable(cube).MakeInvisible()
        added = True
    return added


__all__ = [
    "MUSHR_NANO_V2_CFG",
    "USD_PATH",
    "WHEELBASE",
    "TRACK_WIDTH",
    "WHEEL_RADIUS",
    "MAX_STEER",
    "add_chassis_collider",
]
