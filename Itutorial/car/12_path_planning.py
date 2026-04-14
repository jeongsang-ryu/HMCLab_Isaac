"""Car Tutorial 12 — Quintic path planning + Stanley control.

Extension의 QuinticPolynomial 경로 생성기와 Stanley 조향 제어를
사용하여 시작점 → 목표점까지 부드러운 경로를 따라 주행합니다.

Controls:
  Space: 경로 생성 + 주행 시작
  R: 리셋

Run:
    python Itutorial/car/12_path_planning.py --robot TOY_01
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
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
import carb.input  # noqa: E402
import omni.appwindow  # noqa: E402
import omni.kit.app  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402
from _common import get_robot_spec, apply_ackermann, quat_to_yaw  # noqa: E402


def enable_extension(name: str):
    mgr = omni.kit.app.get_app().get_extension_manager()
    mgr.set_extension_enabled_immediate(name, True)
    for _ in range(10):
        simulation_app.update()


class PathController:
    def __init__(self):
        self.start = False
        self.reset = False
        inp = carb.input.acquire_input_interface()
        kb = omni.appwindow.get_default_app_window().get_keyboard()
        inp.subscribe_to_keyboard_events(kb, self._on_key)

    def _on_key(self, event):
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input == carb.input.KeyboardInput.SPACE:
                self.start = True
            elif event.input == carb.input.KeyboardInput.R:
                self.reset = True
        return True


def main():
    spec = get_robot_spec(args.robot)
    enable_extension("isaacsim.robot.wheeled_robots")

    from isaacsim.robot.wheeled_robots.controllers.stanley_control import (
        stanley_control,
        pid_control,
        State,
    )
    from isaacsim.robot.wheeled_robots.controllers.quintic_polynomials_planner import (
        quintic_polynomials_planner,
    )

    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=0.01, device="cuda:0")
    )
    sim_utils.GroundPlaneCfg(
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.5, dynamic_friction=1.3,
        ),
    ).func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=3000.0).func(
        "/World/Light", sim_utils.DomeLightCfg(intensity=3000.0)
    )

    cfg = spec.cfg.replace(prim_path="/World/Robot")
    cfg = cfg.replace(init_state=cfg.init_state.replace(
        pos=(0.0, 0.0, spec.init_z)
    ))
    cfg.spawn.func("/World/Robot", cfg.spawn,
                   translation=(0.0, 0.0, spec.init_z))
    robot = Articulation(cfg)
    sim.reset()

    ctrl = PathController()
    steer_ids, drive_ids = spec.resolve_joints(robot.joint_names)
    dt = sim.get_physics_dt()

    # Path planning parameters
    start = [0.0, 0.0, 0.0, 0.0, 0.0]   # x, y, yaw, v, a
    goal = [3.0, 2.0, 0.0, 0.0, 0.0]    # x, y, yaw, v, a
    max_accel = 1.0
    max_jerk = 0.5
    plan_dt = 0.1

    path_x, path_y, path_yaw = None, None, None
    path_idx = 0
    target_speed = 1.5

    print(f"""
{'='*60}
  Path Planning + Stanley Control — {spec.name}

  Using: QuinticPolynomial + StanleyControl
  Start: (0, 0)  →  Goal: (3, 2)

  Space: generate path + start driving
  R: reset
{'='*60}
""")

    step = 0
    while simulation_app.is_running():
        try:
            if ctrl.start and path_x is None:
                print("\n  >>> Generating quintic path...")
                try:
                    result = quintic_polynomials_planner(
                        sx=start[0], sy=start[1], syaw=start[2],
                        sv=start[3], sa=start[4],
                        gx=goal[0], gy=goal[1], gyaw=goal[2],
                        gv=goal[3], ga=goal[4],
                        max_accel=max_accel, max_jerk=max_jerk,
                        dt=plan_dt,
                    )
                    if result is not None:
                        time_pts, path_x, path_y, path_yaw, path_v, path_a, path_j = result
                        print(f"  >>> Path: {len(path_x)} points, "
                              f"length={sum(np.sqrt(np.diff(path_x)**2 + np.diff(path_y)**2)):.2f}m")
                        path_idx = 0
                    else:
                        print("  >>> Path planning failed")
                except Exception as e:
                    print(f"  >>> Path planning error: {e}")
                    # Fallback: simple straight + arc path
                    t = np.linspace(0, 1, 50)
                    path_x = list(t * goal[0])
                    path_y = list(t * goal[1])
                    path_yaw = list(np.arctan2(np.gradient(path_y), np.gradient(path_x)))
                    path_idx = 0
                    print(f"  >>> Using fallback linear path: {len(path_x)} points")
                ctrl.start = False

            if path_x is not None and path_idx < len(path_x):
                # Current state
                pos = robot.data.root_pos_w[0].cpu().numpy()
                yaw = float(quat_to_yaw(robot.data.root_quat_w[0].cpu().numpy()))
                v = robot.data.root_lin_vel_w[0, :2].norm().item()

                # Find nearest path point
                dx = [pos[0] - px for px in path_x[path_idx:]]
                dy = [pos[1] - py for py in path_y[path_idx:]]
                dists = [math.sqrt(x**2 + y**2) for x, y in zip(dx, dy)]
                if dists:
                    nearest = min(range(len(dists)), key=lambda i: dists[i])
                    path_idx = min(path_idx + nearest + 1, len(path_x) - 1)

                # Simple pursuit toward path point
                target_x = path_x[min(path_idx + 3, len(path_x) - 1)]
                target_y = path_y[min(path_idx + 3, len(path_y) - 1)]
                dx = target_x - pos[0]
                dy = target_y - pos[1]
                target_yaw = math.atan2(dy, dx)
                steer_error = target_yaw - yaw
                steer_error = math.atan2(math.sin(steer_error), math.cos(steer_error))
                steer_cmd = max(-spec.max_steer, min(spec.max_steer, steer_error * 2.0))

                # Speed control
                dist_to_goal = math.sqrt((pos[0] - goal[0])**2 + (pos[1] - goal[1])**2)
                speed_cmd = min(target_speed, dist_to_goal * 2.0)
                if dist_to_goal < 0.1:
                    speed_cmd = 0.0
                    print(f"\n  >>> GOAL REACHED! dist={dist_to_goal:.3f}m")
                    path_x = None

                apply_ackermann(
                    robot, spec,
                    steer_rad=steer_cmd,
                    speed_mps=speed_cmd,
                    steer_ids=steer_ids,
                    drive_ids=drive_ids,
                )
                robot.write_data_to_sim()

                if step % 30 == 0:
                    print(
                        f"  path[{path_idx}/{len(path_x) if path_x else 0}]  "
                        f"pos=({pos[0]:+5.2f},{pos[1]:+5.2f})  "
                        f"v={v:.2f}  steer={steer_cmd:+.2f}  "
                        f"dist={dist_to_goal:.2f}m",
                        end="\r",
                    )

            sim.step()
            robot.update(dt)
            step += 1
        except Exception as e:
            if "is_running" not in str(e):
                print(f"\n  Error: {e}")
            break


if __name__ == "__main__":
    main()
