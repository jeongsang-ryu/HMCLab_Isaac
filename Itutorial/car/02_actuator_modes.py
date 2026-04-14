"""Car Tutorial 02 — Compare actuator control modes (GUI).

Cycles through position / velocity / effort control on the vehicle
while the GUI stays open. Watch the wheels and steering respond
differently to each mode.

Run:
    python Itutorial/car/02_actuator_modes.py --robot TOY_01
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

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
    sim_utils.DomeLightCfg(intensity=2500.0).func(
        "/World/Light", sim_utils.DomeLightCfg(intensity=2500.0)
    )

    cfg = spec.cfg.replace(prim_path="/World/Robot")
    cfg.spawn.func(cfg.prim_path, cfg.spawn, translation=cfg.init_state.pos)
    robot = Articulation(cfg)
    sim.reset()

    steer_ids, drive_ids = spec.resolve_joints(robot.joint_names)
    n_j = robot.num_joints
    device = robot.device
    dt = sim.get_physics_dt()

    print(f"""
{'='*60}
  Actuator Modes Demo — {spec.name}
  Steer: {steer_ids}  Drive: {drive_ids}

  PhysX joint drive formula:
    torque = stiffness × (pos_target - pos)
           + damping   × (vel_target - vel)
           + effort_target
    actual = clamp(torque, -effort_limit, +effort_limit)

  Position mode: stiffness=high, damping=low
  Velocity mode: stiffness=0,    damping=high
  Effort mode:   stiffness=0,    damping=0
{'='*60}
  Running — close Kit window to exit
""")

    step = 0
    while simulation_app.is_running():
        try:
            cycle = (step // 200) % 3

            if cycle == 0:
                angle = 0.3 if (step // 100) % 2 == 0 else -0.3
                pos_t = torch.zeros(1, n_j, device=device)
                for idx in steer_ids:
                    pos_t[0, idx] = angle
                robot.set_joint_position_target(pos_t)
                if step % 200 == 0:
                    print(f"  [step {step}] Position control: steer={angle:+.1f} rad")

            elif cycle == 1:
                vel_t = torch.zeros(1, n_j, device=device)
                for idx in drive_ids:
                    vel_t[0, idx] = 30.0
                robot.set_joint_velocity_target(vel_t)
                if step % 200 == 0:
                    print(f"  [step {step}] Velocity control: wheel=30 rad/s")

            else:
                effort = torch.zeros(1, n_j, device=device)
                for idx in drive_ids:
                    effort[0, idx] = 2.0
                robot.set_joint_effort_target(effort)
                if step % 200 == 0:
                    print(f"  [step {step}] Effort control: torque=2.0 Nm")

            robot.write_data_to_sim()
            sim.step()
            robot.update(dt)
            step += 1
        except Exception:
            break


if __name__ == "__main__":
    main()
