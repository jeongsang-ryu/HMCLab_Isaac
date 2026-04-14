"""Car Tutorial 07 — Suspension stiffness/damping comparison (GUI).

3대의 차량을 다른 서스펜션 세팅으로 높이에서 떨어뜨려
바운스/안정화 거동을 비교합니다.

  차량 1: Soft   (stiffness=500,  damping=10)
  차량 2: Medium (stiffness=2000, damping=50)
  차량 3: Hard   (stiffness=8000, damping=200)

Controls:
  Enter: 낙하 시작
  R: 리셋

Run:
    python Itutorial/car/07_suspension_test.py --robot TOY_01
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--robot", type=str, default="TOY_01")
parser.add_argument("--drop_height", type=float, default=0.5)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = False
launcher = AppLauncher(args)
simulation_app = launcher.app

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import carb.input  # noqa: E402
import omni.appwindow  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402
from _common import get_robot_spec  # noqa: E402


class DropController:
    def __init__(self):
        self.drop = False
        self.reset_flag = False
        inp = carb.input.acquire_input_interface()
        kb = omni.appwindow.get_default_app_window().get_keyboard()
        inp.subscribe_to_keyboard_events(kb, self._on_key)

    def _on_key(self, event):
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input == carb.input.KeyboardInput.ENTER:
                self.drop = True
            elif event.input == carb.input.KeyboardInput.R:
                self.reset_flag = True
        return True


def spawn_with_suspension(spec, prim_path, pos_y, stiffness, damping, label):
    """Spawn vehicle with custom suspension parameters."""
    cfg = spec.cfg.replace(prim_path=prim_path)

    # Override suspension actuator if it exists
    new_actuators = dict(cfg.actuators)
    # Find suspension-like actuator and override
    for name, act in cfg.actuators.items():
        if "suspension" in name.lower():
            new_actuators[name] = ImplicitActuatorCfg(
                joint_names_expr=act.joint_names_expr,
                effort_limit_sim=0.0,
                stiffness=stiffness,
                damping=damping,
            )
    cfg = cfg.replace(actuators=new_actuators)

    init_z = spec.init_z + args.drop_height
    cfg = cfg.replace(init_state=cfg.init_state.replace(
        pos=(0.0, pos_y, init_z)
    ))
    cfg.spawn.func(cfg.prim_path, cfg.spawn,
                   translation=(0.0, pos_y, init_z))
    car = Articulation(cfg)
    print(f"  {label}: stiffness={stiffness}, damping={damping}")
    return car


def main():
    spec = get_robot_spec(args.robot)

    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=0.005, device="cuda:0")
    )
    sim_utils.GroundPlaneCfg(
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.5, dynamic_friction=1.3,
        ),
    ).func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=3000.0).func(
        "/World/Light", sim_utils.DomeLightCfg(intensity=3000.0)
    )

    print(f"\n  Spawning 3 cars at height {args.drop_height}m:")

    cars = [
        spawn_with_suspension(spec, "/World/Soft", -1.0,
                              stiffness=500, damping=10, label="SOFT  "),
        spawn_with_suspension(spec, "/World/Medium", 0.0,
                              stiffness=2000, damping=50, label="MEDIUM"),
        spawn_with_suspension(spec, "/World/Hard", 1.0,
                              stiffness=8000, damping=200, label="HARD  "),
    ]
    labels = ["SOFT  ", "MEDIUM", "HARD  "]

    sim.reset()
    ctrl = DropController()
    dt = sim.get_physics_dt()

    print(f"""
{'='*60}
  Suspension Drop Test — {spec.name}
  Drop height: {args.drop_height} m

  Left:   SOFT   (bouncy, slow to settle)
  Center: MEDIUM (balanced)
  Right:  HARD   (stiff, quick settle)

  Enter: drop    R: reset
{'='*60}
""")

    z_history = {l: [] for l in labels}
    step = 0

    while simulation_app.is_running():
        try:
            sim.step()
            for car in cars:
                car.update(dt)

            # Log Z positions
            if step % 10 == 0:
                status = []
                for label, car in zip(labels, cars):
                    z = car.data.root_pos_w[0, 2].item()
                    vel_z = car.data.root_lin_vel_w[0, 2].item()
                    z_history[label].append(z)
                    status.append(f"{label} z={z:.4f} vz={vel_z:+.3f}")
                print("  " + "  |  ".join(status), end="\r")

            step += 1
        except Exception:
            break


if __name__ == "__main__":
    main()
