"""Car Tutorial 08 — Motor torque-speed curve test (GUI).

차량을 고정하지 않고 자유 주행시키면서
실제 바퀴 토크와 속도를 기록하고 터미널에 실시간 출력합니다.

3가지 제어 모드를 순서대로 테스트:
  Phase 1: Velocity control (ImplicitActuator, damping=30)
  Phase 2: Effort control   (직접 토크 2 Nm)
  Phase 3: Speed ramp       (0 → max 속도 서서히 증가)

Controls:
  1, 2, 3: 모드 선택
  Enter: 시작/정지

Run:
    python Itutorial/car/08_motor_curves.py --robot TOY_01
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--robot", type=str, default="TOY_01")
parser.add_argument("--max_speed", type=float, default=5.0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = False
launcher = AppLauncher(args)
simulation_app = launcher.app

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import carb.input  # noqa: E402
import omni.appwindow  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402
from _common import get_robot_spec  # noqa: E402


class MotorTestController:
    def __init__(self):
        self.mode = 1         # 1=velocity, 2=effort, 3=ramp
        self.running = False
        inp = carb.input.acquire_input_interface()
        kb = omni.appwindow.get_default_app_window().get_keyboard()
        inp.subscribe_to_keyboard_events(kb, self._on_key)

    def _on_key(self, event):
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input == carb.input.KeyboardInput.ENTER:
                self.running = not self.running
            elif event.input == carb.input.KeyboardInput.KEY_1:
                self.mode = 1
                print("\n  >>> Mode 1: Velocity control")
            elif event.input == carb.input.KeyboardInput.KEY_2:
                self.mode = 2
                print("\n  >>> Mode 2: Effort control (constant torque)")
            elif event.input == carb.input.KeyboardInput.KEY_3:
                self.mode = 3
                print("\n  >>> Mode 3: Speed ramp (0 → max)")
        return True


def main():
    spec = get_robot_spec(args.robot)

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
    cfg.spawn.func(cfg.prim_path, cfg.spawn,
                   translation=(0.0, 0.0, spec.init_z))
    robot = Articulation(cfg)

    sim.reset()

    ctrl = MotorTestController()
    steer_ids, drive_ids = spec.resolve_joints(robot.joint_names)
    n_j = robot.num_joints
    device = robot.device
    dt = sim.get_physics_dt()

    max_wheel_vel = args.max_speed / spec.wheel_radius

    print(f"""
{'='*60}
  Motor Curve Test — {spec.name}
  Max speed: {args.max_speed} m/s
  Max wheel vel: {max_wheel_vel:.1f} rad/s
  Wheel radius: {spec.wheel_radius} m

  1: Velocity control (set_joint_velocity_target)
  2: Effort control   (set_joint_effort_target, 2 Nm)
  3: Speed ramp       (velocity 0 → max over 5 sec)
  Enter: start/stop

  Output columns:
    wheel_vel = actual wheel angular velocity (rad/s)
    chassis_v = forward speed (m/s)
    slip      = (wheel_v × r - chassis_v) / max(wheel_v × r, 0.01)
{'='*60}
""")

    step = 0
    ramp_progress = 0.0

    while simulation_app.is_running():
        try:
            if ctrl.running:
                vel_t = torch.zeros(1, n_j, device=device)
                eff_t = torch.zeros(1, n_j, device=device)

                if ctrl.mode == 1:
                    # Velocity control — full speed target
                    for di in drive_ids:
                        vel_t[0, di] = max_wheel_vel
                    robot.set_joint_velocity_target(vel_t)

                elif ctrl.mode == 2:
                    # Effort control — constant 2 Nm per wheel
                    for di in drive_ids:
                        eff_t[0, di] = 2.0
                    robot.set_joint_effort_target(eff_t)

                elif ctrl.mode == 3:
                    # Speed ramp — gradually increase
                    ramp_progress = min(ramp_progress + dt / 5.0, 1.0)
                    target = max_wheel_vel * ramp_progress
                    for di in drive_ids:
                        vel_t[0, di] = target
                    robot.set_joint_velocity_target(vel_t)

                robot.write_data_to_sim()

            sim.step()
            robot.update(dt)

            # Print telemetry
            if step % 20 == 0 and ctrl.running:
                wheel_w = robot.data.joint_vel[0, drive_ids].mean().item()
                wheel_v = wheel_w * spec.wheel_radius
                chassis_v = robot.data.root_lin_vel_w[0, :2].norm().item()
                slip = (wheel_v - chassis_v) / max(abs(wheel_v), 0.01)

                mode_str = ["", "VEL_CTRL", "EFF_CTRL", "RAMP    "][ctrl.mode]
                bar_len = int(min(abs(wheel_w) / max_wheel_vel, 1.0) * 20)
                bar = "█" * bar_len + "░" * (20 - bar_len)

                print(
                    f"  [{mode_str}] "
                    f"wheel={wheel_w:6.1f} rad/s  "
                    f"chassis={chassis_v:5.2f} m/s  "
                    f"slip={slip:+.2f}  "
                    f"|{bar}|",
                    end="\r",
                )

            step += 1
        except Exception:
            break


if __name__ == "__main__":
    main()
