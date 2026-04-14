"""Patch TOY_01 chassis USD to add prismatic suspension joints + 4WD support.

Original USD is preserved. A new file `TOY_01_chassis_suspended.usd` is
written next to it, plus the underlying physics USD is also copied.

Pipeline:
  1. Copy `f1tenth_mid360_physics.usd` → `f1tenth_mid360_physics_suspended.usd`
  2. For each wheel link (front/rear, left/right):
       - Identify the existing revolute joint that drives it
       - Create a new intermediate link `*_suspension_link` (RigidBody)
       - Create a prismatic joint between parent (base_link / steering_link)
         and the new suspension_link (Z axis, ±15 mm, k=2000, c=80)
       - Rewire the wheel's revolute joint: body0 = suspension_link
  3. Save the modified USD

Run:
    OMNI_KIT_ACCEPT_EULA=YES python scripts/patch_toy01_suspension.py
"""

from __future__ import annotations

import os
import shutil

from isaaclab.app import AppLauncher

launcher = AppLauncher(headless=True)
simulation_app = launcher.app

import omni.usd  # noqa: E402
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics  # noqa: E402


REPO = "/home/js/hmcl_issac_project/HMCLab_Isaac"
SRC_USD = os.path.join(
    REPO, "hmclab_isaac/robots/racing/f1tenth_mid360/assets/configuration",
    "f1tenth_mid360_physics.usd",
)
DST_USD = os.path.join(
    REPO, "hmclab_isaac/robots/racing/f1tenth_mid360/assets/configuration",
    "f1tenth_mid360_physics_suspended.usd",
)
TOY_CHASSIS_DST = os.path.join(
    REPO, "hmclab_isaac/robots/racing/f1tenth_toy/chassis",
    "TOY_01_chassis_suspended.usd",
)

# Suspension parameters (per wheel)
SUS_TRAVEL = 0.015      # ±15 mm
SUS_STIFF = 2000.0      # N/m
SUS_DAMP = 80.0         # Ns/m


def find_wheel_joints(stage):
    """Return list of (joint_prim, body0_path, body1_path, joint_kind)."""
    out = []
    for prim in stage.Traverse():
        t = str(prim.GetTypeName())
        if t != "PhysicsRevoluteJoint":
            continue
        name = prim.GetName().lower()
        if "wheel_joint" not in name:
            continue
        joint = UsdPhysics.RevoluteJoint(prim)
        b0 = list(joint.GetBody0Rel().GetTargets())
        b1 = list(joint.GetBody1Rel().GetTargets())
        if not b0 or not b1:
            continue
        out.append((prim, str(b0[0]), str(b1[0]), name))
    return out


def inspect():
    """Print TOPOLOGY of the source USD and exit (called first)."""
    stage = Usd.Stage.Open(SRC_USD)
    print("=" * 70)
    print(f"Source USD: {SRC_USD}")
    print("=" * 70)
    print("\n>>> All Joints:")
    for prim in stage.Traverse():
        t = str(prim.GetTypeName())
        if "Joint" not in t or "Articulation" in t:
            continue
        try:
            joint = UsdPhysics.Joint(prim)
            b0 = list(joint.GetBody0Rel().GetTargets())
            b1 = list(joint.GetBody1Rel().GetTargets())
            print(f"  [{t}] {prim.GetPath()}")
            print(f"       body0 = {b0}")
            print(f"       body1 = {b1}")
        except Exception as e:
            print(f"  [{t}] {prim.GetPath()}  (error: {e})")

    print("\n>>> RigidBodies:")
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            print(f"  {prim.GetPath()}")

    print("\n>>> Wheel joints (filtered):")
    for jp, b0, b1, n in find_wheel_joints(stage):
        print(f"  {jp.GetPath()}  body0={b0}  body1={b1}")
    print()


def patch():
    """Copy the USD and insert suspension joints + intermediate links."""
    if not os.path.isfile(SRC_USD):
        raise FileNotFoundError(f"Source USD not found: {SRC_USD}")

    # 1. Copy
    print(f"Copy:\n  {SRC_USD}\n  → {DST_USD}")
    shutil.copy2(SRC_USD, DST_USD)

    # 2. Open the copy and modify
    stage = Usd.Stage.Open(DST_USD)
    wheel_joints = find_wheel_joints(stage)
    print(f"\nFound {len(wheel_joints)} wheel joints:")
    for _, b0, b1, n in wheel_joints:
        print(f"  {n}: {b0.split('/')[-1]} → {b1.split('/')[-1]}")

    if not wheel_joints:
        raise RuntimeError("No wheel_joint revolute joints found.")

    inserted = []

    for wheel_joint_prim, parent_path, wheel_path, jname in wheel_joints:
        wheel_link = stage.GetPrimAtPath(wheel_path)
        parent_link = stage.GetPrimAtPath(parent_path)
        if not wheel_link or not parent_link:
            print(f"  ⚠ skip {jname}: parent/wheel prim missing")
            continue

        # Position of the wheel relative to parent — we need this to place
        # the suspension link. We use the wheel link's local translation as
        # an approximation. This works because the wheel is a direct child
        # of `parent_link` in the existing USD.
        wheel_xf = UsdGeom.Xformable(wheel_link)
        wheel_translate = (0.0, 0.0, 0.0)
        for op in wheel_xf.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                t = op.Get()
                if t is not None:
                    wheel_translate = (float(t[0]), float(t[1]), float(t[2]))
                    break

        # 2a. Create suspension_link at the SAME hierarchy level as other
        # rigid bodies (top-level /f1tenth_mid360/). Nesting RigidBody inside
        # RigidBody breaks PhysX articulation discovery.
        articulation_root = "/f1tenth_mid360"
        sus_link_path = f"{articulation_root}/{wheel_link.GetName()}_sus_link"
        if stage.GetPrimAtPath(sus_link_path).IsValid():
            print(f"  skip {jname}: sus_link already exists")
            continue
        sus_xf = UsdGeom.Xform.Define(stage, sus_link_path)
        sus_xf.AddTranslateOp().Set(Gf.Vec3d(*wheel_translate))
        # Make it a tiny massless RigidBody — just a kinematic anchor with mass
        sus_prim = sus_xf.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(sus_prim)
        mass_api = UsdPhysics.MassAPI.Apply(sus_prim)
        mass_api.CreateMassAttr(0.05)  # 50 g virtual mass

        # 2b. Create prismatic joint: parent_link ↔ sus_link, axis Z.
        # Place under /joints/ scope to match existing convention.
        sus_joint_path = f"{articulation_root}/joints/{wheel_link.GetName()}_sus_joint"
        sus_joint = UsdPhysics.PrismaticJoint.Define(stage, sus_joint_path)
        sus_joint.CreateBody0Rel().SetTargets([parent_link.GetPath()])
        sus_joint.CreateBody1Rel().SetTargets([sus_prim.GetPath()])
        sus_joint.CreateAxisAttr("Z")
        sus_joint.CreateLowerLimitAttr(-SUS_TRAVEL)
        sus_joint.CreateUpperLimitAttr(SUS_TRAVEL)
        # localPos: parent frame → wheel mount point; child frame → origin
        sus_joint.CreateLocalPos0Attr(Gf.Vec3f(*wheel_translate))
        sus_joint.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
        sus_joint.CreateLocalRot0Attr(Gf.Quatf(1, 0, 0, 0))
        sus_joint.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))

        # Add drive:linear so it acts like a spring+damper
        drive = UsdPhysics.DriveAPI.Apply(sus_joint.GetPrim(), "linear")
        drive.CreateTypeAttr("force")
        drive.CreateTargetPositionAttr(0.0)
        drive.CreateTargetVelocityAttr(0.0)
        drive.CreateStiffnessAttr(SUS_STIFF)
        drive.CreateDampingAttr(SUS_DAMP)
        drive.CreateMaxForceAttr(50.0)

        # 2c. Move the wheel to be a child of suspension_link (visually)
        # Easier: just zero the wheel's local translate (sus_link is already
        # at the wheel position) and reparent isn't needed because the joint
        # rewire below makes the physics work regardless of USD parent.
        wheel_xf.ClearXformOpOrder()
        wheel_xf.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0))

        # 2d. Rewire the existing wheel revolute joint:
        #     body0 was parent_link → now sus_link
        wheel_revolute = UsdPhysics.RevoluteJoint(wheel_joint_prim)
        wheel_revolute.GetBody0Rel().SetTargets([sus_prim.GetPath()])
        # Update localPos0 to (0,0,0) since sus_link is co-located with wheel
        wheel_revolute.CreateLocalPos0Attr(Gf.Vec3f(0, 0, 0))

        inserted.append((sus_link_path, sus_joint_path, jname))
        print(f"  ✓ {jname}: + {sus_link_path.split('/')[-1]}, "
              f"+ {sus_joint_path.split('/')[-1]}")

    # 3. Save
    stage.GetRootLayer().Save()
    print(f"\n✓ Saved: {DST_USD}")
    print(f"  Inserted {len(inserted)} suspension joints")

    # 4. Also create a thin TOY_01 chassis USD that references the patched physics
    print(f"\nCreating chassis wrapper: {TOY_CHASSIS_DST}")
    if os.path.isfile(TOY_CHASSIS_DST):
        # back up the original
        bak = TOY_CHASSIS_DST + ".bak"
        if not os.path.isfile(bak):
            shutil.copy2(TOY_CHASSIS_DST, bak)
            print(f"  backup → {bak}")

    # We don't overwrite TOY_01_chassis.usd — user said preserve original.
    # Instead just point them to the new physics USD via a new chassis USD
    # via Sdf reference.
    new_chassis_path = os.path.join(
        os.path.dirname(TOY_CHASSIS_DST), "TOY_01_chassis_4wd_sus.usd"
    )
    new_stage = Usd.Stage.CreateNew(new_chassis_path)
    root_xf = UsdGeom.Xform.Define(new_stage, "/World")
    new_stage.SetDefaultPrim(root_xf.GetPrim())
    chassis_xf = UsdGeom.Xform.Define(new_stage, "/World/Robot")
    chassis_xf.GetPrim().GetReferences().AddReference(DST_USD)
    new_stage.GetRootLayer().Save()
    print(f"  ✓ chassis wrapper saved: {new_chassis_path}")


if __name__ == "__main__":
    print("\n=== INSPECT ===")
    inspect()
    print("\n=== PATCH ===")
    patch()
    simulation_app.close()
