"""Car Tutorial 04 — View vehicle with Mid-360 + NUC14 mounted (GUI).

Run:
    python Itutorial/car/04_view_with_all_sensors.py --robot TOY_01
"""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--robot", type=str, default="TOY_01")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = False
launcher = AppLauncher(args)
simulation_app = launcher.app

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402
from pxr import Gf, UsdGeom  # noqa: E402
import omni.usd  # noqa: E402

from _common import get_robot_spec  # noqa: E402


def main():
    spec = get_robot_spec(args.robot)

    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=0.01, device="cuda:0")
    )
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=3000.0).func(
        "/World/Light", sim_utils.DomeLightCfg(intensity=3000.0)
    )

    cfg = spec.cfg.replace(prim_path="/World/Robot")
    cfg.spawn.func(cfg.prim_path, cfg.spawn, translation=(0, 0, spec.init_z))
    robot = Articulation(cfg)

    from hmclab_isaac.robots._sensors.mid360 import attach_mid360_visual
    attach_mid360_visual("/World/Robot", mount_link="base_link",
                         offset=(0.05, 0.0, 0.12))

    nuc_obj = os.path.abspath("hmclab_isaac/robots/_sensors/nuc/NUC14PRO_slim.obj")
    if os.path.exists(nuc_obj):
        stage = omni.usd.get_context().get_stage()
        base_link = stage.GetPrimAtPath("/World/Robot/base_link")
        if base_link.IsValid():
            nuc_path = "/World/Robot/base_link/nuc14_visual"
            nuc_prim = stage.DefinePrim(nuc_path)
            xf = UsdGeom.Xformable(nuc_prim)
            xf.ClearXformOpOrder()
            xf.AddTranslateOp().Set(Gf.Vec3d(-0.05, 0.0, 0.05))
            xf.AddScaleOp().Set(Gf.Vec3f(0.001, 0.001, 0.001))
            try:
                nuc_prim.GetReferences().AddReference(assetPath=nuc_obj)
                print(">>> NUC14 mounted")
            except Exception as e:
                print(f">>> NUC14 mount failed: {e}")

    print(f">>> {spec.name} + Mid-360 + NUC14 — close Kit window to exit")

    sim.reset()

    while simulation_app.is_running():
        try:
            sim.step()
            robot.update(sim.get_physics_dt())
        except Exception:
            break


if __name__ == "__main__":
    main()
