"""Car Tutorial 11 — WheelBasePoseController (go-to-goal).

목표 좌표를 주면 자동으로 주행하는 closed-loop 컨트롤러.
클릭한 위치로 차량이 이동합니다.

Controls:
  1: 목표 (2, 0)
  2: 목표 (0, 2)
  3: 목표 (-2, 0)
  4: 목표 (0, -2)
  5: 목표 (0, 0) 원점

Run:
    python Itutorial/car/11_pose_controller.py --robot TOY_01
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
from _common import get_robot_spec, quat_to_yaw  # noqa: E402


def enable_extension(name: str):
    mgr = omni.kit.app.get_app().get_extension_manager()
    mgr.set_extension_enabled_immediate(name, True)
    for _ in range(10):
        simulation_app.update()


class GoalSelector:
    def __init__(self):
        self.goal = np.array([2.0, 0.0, 0.0])  # x, y, yaw
        inp = carb.input.acquire_input_interface()
        kb = omni.appwindow.get_default_app_window().get_keyboard()
        inp.subscribe_to_keyboard_events(kb, self._on_key)

    def _on_key(self, event):
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            goals = {
                carb.input.KeyboardInput.KEY_1: [2.0, 0.0, 0.0],
                carb.input.KeyboardInput.KEY_2: [0.0, 2.0, math.pi / 2],
                carb.input.KeyboardInput.KEY_3: [-2.0, 0.0, math.pi],
                carb.input.KeyboardInput.KEY_4: [0.0, -2.0, -math.pi / 2],
                carb.input.KeyboardInput.KEY_5: [0.0, 0.0, 0.0],
            }
            if event.input in goals:
                self.goal = np.array(goals[event.input])
                print(f"\n  >>> New goal: ({self.goal[0]:.1f}, {self.goal[1]:.1f}, "
                      f"yaw={math.degrees(self.goal[2]):.0f}°)")
        return True


def main():
    spec = get_robot_spec(args.robot)
    enable_extension("isaacsim.robot.wheeled_robots")

    from isaacsim.robot.wheeled_robots.controllers import (
        DifferentialController,
        WheelBasePoseController,
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

    track_width = 0.20
    diff_ctrl = DifferentialController(
        name="diff",
        wheel_radius=spec.wheel_radius,
        wheel_base=track_width,
    )
    pose_ctrl = WheelBasePoseController(
        name="pose",
        open_loop_wheel_controller=diff_ctrl,
        is_holonomic=False,
    )

    goal = GoalSelector()
    steer_ids, drive_ids = spec.resolve_joints(robot.joint_names)
    left_drive = [i for i in drive_ids if "left" in robot.joint_names[i]]
    right_drive = [i for i in drive_ids if "right" in robot.joint_names[i]]
    dt = sim.get_physics_dt()
    n_j = robot.num_joints
    device = robot.device

    print(f"""
{'='*60}
  Pose Controller (Go-to-Goal) — {spec.name}

  Using: WheelBasePoseController + DifferentialController
  Input:  target (x, y, yaw)
  Output: wheel velocities (automatic)

  1: goal (2, 0)     2: goal (0, 2)
  3: goal (-2, 0)    4: goal (0, -2)
  5: goal (0, 0)     origin
{'='*60}
""")

    step = 0
    while simulation_app.is_running():
        try:
            # Current pose
            pos = robot.data.root_pos_w[0].cpu().numpy()
            yaw = float(quat_to_yaw(robot.data.root_quat_w[0].cpu().numpy()))
            current_pos = np.array([pos[0], pos[1], yaw])

            # Pose controller → differential command
            actions = pose_ctrl.forward(
                start_position=current_pos[:2],
                start_orientation=current_pos[2],
                goal_position=goal.goal[:2],
                goal_orientation=goal.goal[2],
            )

            left_vel = float(actions.joint_velocities[0])
            right_vel = float(actions.joint_velocities[1])

            vel_t = torch.zeros(1, n_j, device=device)
            for i in left_drive:
                vel_t[0, i] = left_vel
            for i in right_drive:
                vel_t[0, i] = right_vel
            robot.set_joint_velocity_target(vel_t)
            robot.write_data_to_sim()

            sim.step()
            robot.update(dt)

            if step % 50 == 0:
                dist = np.linalg.norm(current_pos[:2] - goal.goal[:2])
                print(
                    f"  pos=({pos[0]:+5.2f},{pos[1]:+5.2f}) "
                    f"yaw={math.degrees(yaw):+6.1f}°  "
                    f"→ goal=({goal.goal[0]:+.1f},{goal.goal[1]:+.1f})  "
                    f"dist={dist:.3f}m",
                    end="\r",
                )

            step += 1
        except Exception as e:
            if "is_running" not in str(e):
                print(f"\n  Error: {e}")
            break


if __name__ == "__main__":
    main()
