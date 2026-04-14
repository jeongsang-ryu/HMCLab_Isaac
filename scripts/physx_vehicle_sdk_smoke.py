"""Smoke test: invoke the PhysX Vehicle SDK's own SharedComponentSetup
sample directly to validate the SDK works in our Isaac Sim install.

If this drives correctly with arrow keys disabled (we'll just push
accelerator from Python), we know the SDK is functional and our
custom Z-up authoring is the bug. If this also fails, the SDK isn't
usable in this Kit version.

Run:
    OMNI_KIT_ACCEPT_EULA=YES \
      /home/js/anaconda3/envs/hmclab_test/bin/python \
      scripts/physx_vehicle_sdk_smoke.py --steps 600
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=600)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = False
args.enable_cameras = False
launcher = AppLauncher(args)
sim_app = launcher.app

import isaaclab.sim as sim_utils  # noqa: E402


def main() -> int:
    # CPU pipeline (Vehicle SDK requirement)
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=1.0 / 60.0, device="cpu")
    )

    import omni.usd
    from pxr import UsdGeom, UsdPhysics, PhysxSchema, Gf, Sdf

    stage = omni.usd.get_context().get_stage()

    # The SDK sample expects a default prim. Create /World as default.
    if not stage.GetDefaultPrim() or not stage.GetDefaultPrim().IsValid():
        world = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(world.GetPrim())

    # Remove the auto-created /physicsScene because the SDK sample creates its
    # own scene under /World/PhysicsScene.
    auto_scene = stage.GetPrimAtPath("/physicsScene")
    if auto_scene and auto_scene.IsValid():
        stage.RemovePrim(auto_scene.GetPath())

    # Invoke the SDK's reference setup
    from omni.physxvehicle.scripts.samples import SharedComponentSetup
    SharedComponentSetup.create(stage)

    # Pick the first vehicle and apply throttle from Python
    veh_prim = stage.GetPrimAtPath("/World/Vehicle0")
    if not veh_prim or not veh_prim.IsValid():
        print("[smoke] vehicle prim missing", flush=True)
        return 1
    ctrl = PhysxSchema.PhysxVehicleControllerAPI(veh_prim)

    sim.reset()
    accel_attr = ctrl.GetAcceleratorAttr()
    steer_attr = ctrl.GetSteerAttr()

    dt = sim.get_physics_dt()
    for i in range(args.steps):
        accel_attr.Set(0.5)
        steer_attr.Set(0.0)
        sim.step()
        if i % 30 == 0:
            cache = UsdGeom.XformCache()
            mat = cache.GetLocalToWorldTransform(veh_prim)
            tr = mat.ExtractTranslation()
            print(
                f"[smoke] step={i} t={i*dt:.2f}s pos=({tr[0]:.2f},{tr[1]:.2f},{tr[2]:.2f})",
                flush=True,
            )
    print("SDK_SMOKE_DONE", flush=True)
    return 0


if __name__ == "__main__":
    import traceback
    rc = 0
    try:
        rc = main()
    except Exception:
        traceback.print_exc()
        rc = 1
    os._exit(rc)
