"""Car Tutorial 10 — Isaac Sim DifferentialController.

Extension의 DifferentialController를 사용하여
linear velocity + angular velocity → 좌/우 바퀴 속도 변환을 테스트합니다.

TOY_01은 Ackermann 차량이지만, 뒷바퀴만 differential처럼
제어하는 것도 가능합니다 (스키드 스티어 근사).

Controls:
  W/S: 전진/후진 (linear velocity)
  A/D: 회전 (angular velocity)
  Space: 정지

Run:
    python Itutorial/car/10_differential_controller.py --robot TOY_01
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
import carb.input  # noqa: E402
import omni.appwindow  # noqa: E402
import omni.kit.app  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402
from _common import get_robot_spec  # noqa: E402


def enable_extension(name: str):
    mgr = omni.kit.app.get_app().get_extension_manager()
    mgr.set_extension_enabled_immediate(name, True)
    for _ in range(10):
        simulation_app.update()


class KeyboardInput:
    def __init__(self):
        self.linear = 0.0    # m/s
        self.angular = 0.0   # rad/s
        inp = carb.input.acquire_input_interface()
        kb = omni.appwindow.get_default_app_window().get_keyboard()
        inp.subscribe_to_keyboard_events(kb, self._on_key)

    def _on_key(self, event):
        active = event.type in (
            carb.input.KeyboardEventType.KEY_PRESS,
            carb.input.KeyboardEventType.KEY_REPEAT,
        )
        key = event.input
        if key in (carb.input.KeyboardInput.W, carb.input.KeyboardInput.UP):
            self.linear = 2.0 if active else 0.0
        elif key in (carb.input.KeyboardInput.S, carb.input.KeyboardInput.DOWN):
            self.linear = -1.0 if active else 0.0
        elif key in (carb.input.KeyboardInput.A, carb.input.KeyboardInput.LEFT):
            self.angular = 3.0 if active else 0.0
        elif key in (carb.input.KeyboardInput.D, carb.input.KeyboardInput.RIGHT):
            self.angular = -3.0 if active else 0.0
        elif key == carb.input.KeyboardInput.SPACE:
            self.linear = 0.0
            self.angular = 0.0
        return True


def main():
    spec = get_robot_spec(args.robot)
    enable_extension("isaacsim.robot.wheeled_robots")

    # Import after extension is enabled
    from isaacsim.robot.wheeled_robots.controllers import DifferentialController

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

    # DifferentialController setup
    track_width = 0.20  # distance between left/right wheels
    diff_ctrl = DifferentialController(
        name="diff_ctrl",
        wheel_radius=spec.wheel_radius,
        wheel_base=track_width,  # for diff drive, this is track width
    )

    kb = KeyboardInput()
    steer_ids, drive_ids = spec.resolve_joints(robot.joint_names)
    left_drive = [i for i in drive_ids if "left" in robot.joint_names[i]]
    right_drive = [i for i in drive_ids if "right" in robot.joint_names[i]]
    dt = sim.get_physics_dt()
    n_j = robot.num_joints
    device = robot.device

    print(f"""
{'='*60}
  Differential Controller Test — {spec.name}
  Wheel radius: {spec.wheel_radius}m  Track: {track_width}m

  Using: isaacsim.robot.wheeled_robots.DifferentialController
  Input:  (linear_vel, angular_vel)
  Output: (left_wheel_vel, right_wheel_vel)

  W/S: forward/back    A/D: rotate    Space: stop
{'='*60}
""")

    step = 0
    while simulation_app.is_running():
        try:
            # DifferentialController: (linear, angular) → (left, right) wheel vel
            command = [kb.linear, kb.angular]
            actions = diff_ctrl.forward(command)
            # actions = [left_vel, right_vel] in rad/s

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

            if step % 30 == 0:
                chassis_v = robot.data.root_lin_vel_w[0, :2].norm().item()
                actual_l = robot.data.joint_vel[0, left_drive].mean().item()
                actual_r = robot.data.joint_vel[0, right_drive].mean().item()
                print(
                    f"  cmd=({kb.linear:+.1f}, {kb.angular:+.1f})  "
                    f"→ L={left_vel:+6.1f} R={right_vel:+6.1f}  "
                    f"actual L={actual_l:+6.1f} R={actual_r:+6.1f}  "
                    f"v={chassis_v:.2f}",
                    end="\r",
                )

            step += 1
        except Exception as e:
            if "is_running" not in str(e):
                print(f"\n  Error: {e}")
            break


if __name__ == "__main__":
    main()
