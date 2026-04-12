"""Build a 1/8-scale Serpent SRX8-inspired buggy USD from primitives.

This generates a MINIMAL articulation: chassis box + 4 cylinder wheels
connected by Ackermann-style joints (2 front steer + 4 throttle). No
detailed suspension — just rigid wheel mounts to keep the articulation
simple and fast to simulate. The chassis box carries a collision shape so
two SRX8 vehicles actually collide with each other.

Dimensions (approx 1/8 off-road buggy):
    chassis box: 0.50 × 0.30 × 0.12 m (L × W × H)
    wheelbase:   0.325 m
    track width: 0.28 m
    wheel radius: 0.055 m  (diameter 0.11)
    wheel width:  0.05 m

Output: hmclab_isaac/robots/racing/srx8/assets/srx8.usd

Joint topology (10-joint pattern — matches the MuSHR convention so our
existing _common.get_robot_spec drive/steer regexes work out of the box):
    base_link
      ├── back_left_wheel_suspension (prismatic, ±1 cm → stiff)
      │    └── back_left_wheel_suspension_link
      │         └── back_left_wheel_throttle (continuous revolute)
      │              └── back_left_wheel_link
      ├── back_right_wheel_suspension / throttle / link
      ├── front_left_wheel_suspension
      │    └── front_left_wheel_suspension_link
      │         └── front_left_wheel_steer (revolute ±30°)
      │              └── front_left_wheel_steer_link
      │                   └── front_left_wheel_throttle (continuous revolute)
      │                        └── front_left_wheel_link
      └── front_right_wheel_* (mirror)

This matches the mushr_nano_v2 10-joint layout, so all the regex-based
drive / steer lookups work.

Run:
    OMNI_KIT_ACCEPT_EULA=YES python scripts/build_srx8_usd.py
"""

from __future__ import annotations

import os

from isaaclab.app import AppLauncher

launcher = AppLauncher(headless=True)
simulation_app = launcher.app

import omni.usd  # noqa: E402
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics  # noqa: E402


# ----------------------------------------------------------------------
# Dimensions
# ----------------------------------------------------------------------
CHASSIS_L, CHASSIS_W, CHASSIS_H = 0.50, 0.30, 0.12
WHEELBASE = 0.325
TRACK_W = 0.28
WHEEL_R = 0.055
WHEEL_WIDTH = 0.05
SUSP_TRAVEL = 0.01          # ± 1 cm
MAX_STEER_RAD = 0.52        # ~30°

# All bodies are authored in a LOCAL frame anchored at `base_link`
# (which is always placed at the USD origin). This way
# `ArticulationCfg.init_state.pos` directly controls base_link world pose.
BASE_Z = 0.0                # base_link origin (wheel-hub height)
WHEEL_Z = 0.0               # wheel links sit at base_link height

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _xform(stage, path: str, translate=(0, 0, 0)) -> UsdGeom.Xform:
    x = UsdGeom.Xform.Define(stage, path)
    x.AddTranslateOp().Set(Gf.Vec3d(*translate))
    return x


def _rigid_body(stage, path: str, translate, mass: float) -> Usd.Prim:
    x = _xform(stage, path, translate)
    prim = x.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(prim)
    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.CreateMassAttr().Set(mass)
    return prim


def _box_visual_collision(stage, parent_path: str, half_extents, color=(0.3, 0.3, 0.35),
                          local_offset=(0.0, 0.0, 0.0)):
    cube = UsdGeom.Cube.Define(stage, f"{parent_path}/shape")
    cube.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(cube)
    xf.ClearXformOpOrder()
    # Scale first (shape the unit box), then translate — xform ops are
    # applied in list order so the translate is in the scaled local space.
    xf.AddScaleOp().Set(Gf.Vec3f(2 * half_extents[0], 2 * half_extents[1], 2 * half_extents[2]))
    if any(v != 0.0 for v in local_offset):
        tx = local_offset[0] / max(2 * half_extents[0], 1e-6)
        ty = local_offset[1] / max(2 * half_extents[1], 1e-6)
        tz = local_offset[2] / max(2 * half_extents[2], 1e-6)
        xf.AddTranslateOp().Set(Gf.Vec3d(tx, ty, tz))
    cube.CreateDisplayColorAttr().Set([Gf.Vec3f(*color)])
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())


def _cylinder_visual_collision(stage, parent_path: str, radius, height, color=(0.08, 0.08, 0.08)):
    cyl = UsdGeom.Cylinder.Define(stage, f"{parent_path}/shape")
    cyl.CreateAxisAttr("Y")   # cylinder axis along Y → wheel rolls around Y
    cyl.CreateRadiusAttr(radius)
    cyl.CreateHeightAttr(height)
    cyl.CreateDisplayColorAttr().Set([Gf.Vec3f(*color)])
    UsdPhysics.CollisionAPI.Apply(cyl.GetPrim())


def _prismatic_joint(stage, path, parent_path, child_path, axis="Z",
                     lower=-SUSP_TRAVEL, upper=SUSP_TRAVEL):
    j = UsdPhysics.PrismaticJoint.Define(stage, path)
    j.CreateBody0Rel().SetTargets([parent_path])
    j.CreateBody1Rel().SetTargets([child_path])
    j.CreateAxisAttr(axis)
    j.CreateLowerLimitAttr(float(lower))
    j.CreateUpperLimitAttr(float(upper))
    return j


def _revolute_joint(stage, path, parent_path, child_path, axis="Y",
                    lower_rad=None, upper_rad=None,
                    add_angular_drive: bool = False):
    """Create a USD revolute joint. **Limits are stored in DEGREES** per
    `UsdPhysicsRevoluteJoint` schema — so we convert from radians here.

    If `add_angular_drive=True`, attach an angular `DriveAPI` with zero
    target/stiffness/damping — Isaac Lab's ImplicitActuator will populate
    these at spawn time, but the API must exist on the joint first.
    """
    import math
    j = UsdPhysics.RevoluteJoint.Define(stage, path)
    j.CreateBody0Rel().SetTargets([parent_path])
    j.CreateBody1Rel().SetTargets([child_path])
    j.CreateAxisAttr(axis)
    if lower_rad is not None and upper_rad is not None:
        j.CreateLowerLimitAttr(float(math.degrees(lower_rad)))
        j.CreateUpperLimitAttr(float(math.degrees(upper_rad)))
    if add_angular_drive:
        drive = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "angular")
        drive.CreateTypeAttr("force")
        drive.CreateMaxForceAttr(1e7)
        drive.CreateTargetVelocityAttr(0.0)
        drive.CreateTargetPositionAttr(0.0)
        drive.CreateStiffnessAttr(0.0)
        drive.CreateDampingAttr(0.0)
    return j


# ----------------------------------------------------------------------
# Main builder
# ----------------------------------------------------------------------
def build():
    stage = Usd.Stage.CreateNew(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "hmclab_isaac/robots/racing/srx8/assets/srx8.usd",
        )
    )
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/srx8").GetPrim())

    root_path = "/srx8"
    UsdPhysics.ArticulationRootAPI.Apply(stage.GetPrimAtPath(root_path))

    # -------- base_link (chassis) --------
    # base_link origin is at wheel-hub height; chassis box collision/visual
    # is offset UPWARD so it sits above the wheels (bottom at z=+0.02,
    # top at z=+0.14 in local frame). Keeps the chassis from dragging on
    # the ground.
    base_link_path = f"{root_path}/base_link"
    _rigid_body(stage, base_link_path, (0, 0, BASE_Z), mass=2.8)
    _box_visual_collision(
        stage, base_link_path,
        half_extents=(CHASSIS_L / 2, CHASSIS_W / 2, CHASSIS_H / 2),
        color=(0.85, 0.85, 0.15),
        local_offset=(0.0, 0.0, 0.02 + CHASSIS_H / 2),  # +0.08 m
    )

    # Corner positions (in chassis frame, relative to base_link centre)
    half_wb = WHEELBASE / 2
    half_tw = TRACK_W / 2
    corners = {
        "back_left":   (-half_wb, +half_tw, 0.0),
        "back_right":  (-half_wb, -half_tw, 0.0),
        "front_left":  (+half_wb, +half_tw, 0.0),
        "front_right": (+half_wb, -half_tw, 0.0),
    }
    is_front = {"back_left": False, "back_right": False,
                "front_left": True, "front_right": True}

    # Build suspension + wheel chain per corner. Suspension link + wheel
    # link sit at the wheel-hub height (WHEEL_Z), while the chassis
    # (base_link) is above them at BASE_Z.
    for name, (cx, cy, _cz) in corners.items():
        susp_link_path = f"{root_path}/{name}_wheel_suspension_link"
        susp_world_pos = (cx, cy, WHEEL_Z)
        _rigid_body(stage, susp_link_path, susp_world_pos, mass=0.05)
        _box_visual_collision(
            stage, susp_link_path,
            half_extents=(0.015, 0.015, 0.015),  # tiny cube, mostly hidden
            color=(0.1, 0.1, 0.1),
        )

        # Suspension joint: prismatic on Z, ±1 cm (almost rigid)
        _prismatic_joint(
            stage,
            f"{root_path}/{name}_wheel_suspension",
            base_link_path,
            susp_link_path,
            axis="Z",
            lower=-SUSP_TRAVEL, upper=+SUSP_TRAVEL,
        )

        if is_front[name]:
            # front: susp_link → steer joint → steer_link → throttle → wheel
            steer_link_path = f"{root_path}/{name}_wheel_steer_link"
            _rigid_body(stage, steer_link_path, susp_world_pos, mass=0.05)
            _box_visual_collision(
                stage, steer_link_path,
                half_extents=(0.02, 0.02, 0.02),
                color=(0.1, 0.1, 0.1),
            )
            _revolute_joint(
                stage,
                f"{root_path}/{name}_wheel_steer",
                susp_link_path, steer_link_path,
                axis="Z",
                lower_rad=-MAX_STEER_RAD, upper_rad=+MAX_STEER_RAD,
                add_angular_drive=True,
            )
            wheel_parent = steer_link_path
        else:
            wheel_parent = susp_link_path

        # Wheel link
        wheel_link_path = f"{root_path}/{name}_wheel_link"
        _rigid_body(stage, wheel_link_path, susp_world_pos, mass=0.25)
        _cylinder_visual_collision(
            stage, wheel_link_path, radius=WHEEL_R, height=WHEEL_WIDTH,
        )
        _revolute_joint(
            stage,
            f"{root_path}/{name}_wheel_throttle",
            wheel_parent, wheel_link_path,
            axis="Y",
            add_angular_drive=True,
        )

    stage.Save()
    print(f"saved {stage.GetRootLayer().realPath}", flush=True)
    return stage.GetRootLayer().realPath


def main():
    path = build()
    # Verify the saved USD loads
    from pxr import Usd as _Usd
    s = _Usd.Stage.Open(path)
    n_prims = sum(1 for _ in s.Traverse())
    joints = [p for p in s.Traverse() if p.IsA(UsdPhysics.Joint)]
    print(f"verified: {n_prims} prims, {len(joints)} joints", flush=True)
    for j in joints:
        print(f"  {j.GetPath()} ({j.GetTypeName()})", flush=True)
    print("SRX8_BUILD_OK", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
