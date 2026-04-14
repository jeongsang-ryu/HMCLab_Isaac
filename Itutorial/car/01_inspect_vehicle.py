"""Car Tutorial 01 — Inspect a vehicle's joint structure (GUI).

Spawns a vehicle, prints joint/body info, then keeps the GUI open
so you can visually inspect the articulation in the Kit viewport.

Run:
    python Itutorial/car/01_inspect_vehicle.py --robot TOY_01
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
from _common import get_robot_spec  # noqa: E402


def main():
    spec = get_robot_spec(args.robot)

    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=0.01, device="cuda:0")
    )
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())

    cfg = spec.cfg.replace(prim_path="/World/Robot")
    cfg.spawn.func(cfg.prim_path, cfg.spawn, translation=cfg.init_state.pos)
    robot = Articulation(cfg)
    sim.reset()

    # ── Joint info ──
    print(f"\n{'='*60}")
    print(f"  Robot: {spec.name}")
    print(f"  Joints: {robot.num_joints}    Bodies: {robot.num_bodies}")
    print(f"{'='*60}")

    print(f"\n{'idx':>3}  {'joint name':<40} {'pos limits':>20} {'vel limit':>10}")
    print("-" * 78)
    for i, name in enumerate(robot.joint_names):
        lo = robot.data.joint_pos_limits[0, i, 0].item()
        hi = robot.data.joint_pos_limits[0, i, 1].item()
        vl = robot.data.joint_vel_limits[0, i].item()
        if abs(lo) > 1e10:
            lim_str = "unlimited"
        else:
            lim_str = f"[{lo:+.3f}, {hi:+.3f}]"
        vel_str = "inf" if vl > 1e10 else f"{vl:.1f}"
        print(f"  {i:>2}  {name:<40} {lim_str:>20} {vel_str:>10}")

    # ── Identify steer / drive ──
    steer_ids, drive_ids = spec.resolve_joints(robot.joint_names)
    print(f"\n  Steer joints (regex '{spec.steer_regex}'): {steer_ids}")
    print(f"  Drive joints (regex '{spec.drive_regex}'): {drive_ids}")

    # ── Body names ──
    print(f"\n  Bodies ({robot.num_bodies}):")
    for name in robot.body_names:
        print(f"    {name}")

    # ── Vehicle constants ──
    print(f"\n  Wheelbase:    {spec.wheelbase:.4f} m")
    print(f"  Wheel radius: {spec.wheel_radius:.4f} m")
    print(f"  Max steer:    {spec.max_steer:.3f} rad ({spec.max_steer * 57.296:.1f} deg)")
    print(f"{'='*60}")
    print("  GUI open — close Kit window to exit\n")

    while simulation_app.is_running():
        try:
            sim.step()
            robot.update(sim.get_physics_dt())
        except Exception:
            break


if __name__ == "__main__":
    main()
