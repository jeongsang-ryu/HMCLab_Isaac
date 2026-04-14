"""Car Tutorial 09 — Isaac Sim built-in Ackermann Controller.

Extension `isaacsim.robot.wheeled_robots`의 AckermannController를
사용하여 내/외륜 차이가 반영된 정확한 Ackermann 조향을 테스트합니다.

비교:
  좌측: 우리의 apply_ackermann (좌우 같은 각도)
  우측: Extension AckermannController (좌우 다른 각도)

Controls:
  W/S: 속도 증감
  A/D: 조향
  Space: 정지

Run:
    python Itutorial/car/09_ackermann_controller.py --robot TOY_01
"""

from __future__ import annotations

import argparse
import math
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
        self.speed = 0.0
        self.steer = 0.0
        inp = carb.input.acquire_input_interface()
        kb = omni.appwindow.get_default_app_window().get_keyboard()
        inp.subscribe_to_keyboard_events(kb, self._on_key)

    def _on_key(self, event):
        active = event.type in (
            carb.input.KeyboardEventType.KEY_PRESS,
            carb.input.KeyboardEventType.KEY_REPEAT,
        )
        released = event.type == carb.input.KeyboardEventType.KEY_RELEASE
        key = event.input

        if key in (carb.input.KeyboardInput.W, carb.input.KeyboardInput.UP):
            self.speed = 2.0 if active else 0.0
        elif key in (carb.input.KeyboardInput.S, carb.input.KeyboardInput.DOWN):
            self.speed = -1.0 if active else 0.0
        elif key in (carb.input.KeyboardInput.A, carb.input.KeyboardInput.LEFT):
            self.steer = 0.4 if active else 0.0
        elif key in (carb.input.KeyboardInput.D, carb.input.KeyboardInput.RIGHT):
            self.steer = -0.4 if active else 0.0
        elif key == carb.input.KeyboardInput.SPACE:
            self.speed = 0.0
            self.steer = 0.0
        return True


def compute_ackermann_angles(steer_angle, wheelbase, track_width):
    """Pure Ackermann geometry — inner wheel turns more than outer.

    Returns (left_angle, right_angle).
    """
    if abs(steer_angle) < 1e-6:
        return 0.0, 0.0
    R = wheelbase / math.tan(abs(steer_angle))
    left = math.atan(wheelbase / (R - track_width / 2))
    right = math.atan(wheelbase / (R + track_width / 2))
    if steer_angle > 0:
        return left, right     # left turn: left wheel turns more
    else:
        return -right, -left   # right turn: right wheel turns more


def main():
    spec = get_robot_spec(args.robot)

    # Enable wheeled robots extension
    enable_extension("isaacsim.robot.wheeled_robots")

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

    # ── 2대 스폰: 비교용 ──
    def spawn(prim_path, y_offset):
        cfg = spec.cfg.replace(prim_path=prim_path)
        cfg = cfg.replace(init_state=cfg.init_state.replace(
            pos=(0.0, y_offset, spec.init_z)
        ))
        cfg.spawn.func(prim_path, cfg.spawn,
                       translation=(0.0, y_offset, spec.init_z))
        return Articulation(cfg)

    car_simple = spawn("/World/Simple", -0.5)    # 좌: 단순 동일 각도
    car_acker = spawn("/World/Ackermann", 0.5)   # 우: 정확한 Ackermann

    sim.reset()

    kb = KeyboardInput()
    dt = sim.get_physics_dt()

    steer_ids_s, drive_ids_s = spec.resolve_joints(car_simple.joint_names)
    steer_ids_a, drive_ids_a = spec.resolve_joints(car_acker.joint_names)

    # Identify left/right steer joints
    left_steer_a = [i for i in steer_ids_a if "left" in car_acker.joint_names[i]]
    right_steer_a = [i for i in steer_ids_a if "right" in car_acker.joint_names[i]]

    # Track width (approximate from spec or joint positions)
    track_width = 0.20  # f1tenth default

    print(f"""
{'='*60}
  Ackermann Steering Comparison — {spec.name}
  Wheelbase: {spec.wheelbase}m  Track: {track_width}m

  Left car:  Simple (same angle both wheels)
  Right car: Ackermann (inner turns more)

  WASD to drive, Space to stop
{'='*60}
""")

    step = 0
    while simulation_app.is_running():
        try:
            speed_cmd = kb.speed
            steer_cmd = kb.steer
            wheel_vel = speed_cmd / spec.wheel_radius

            n_j = car_simple.num_joints
            device = car_simple.device

            # ── Car 1: Simple (same angle) ──
            pos_s = torch.zeros(1, n_j, device=device)
            vel_s = torch.zeros(1, n_j, device=device)
            for i in steer_ids_s:
                pos_s[0, i] = steer_cmd
            for i in drive_ids_s:
                vel_s[0, i] = wheel_vel
            car_simple.write_joint_position_to_sim(
                pos_s[:, steer_ids_s], joint_ids=list(steer_ids_s)
            )
            car_simple.set_joint_velocity_target(vel_s)

            # ── Car 2: Proper Ackermann ──
            left_angle, right_angle = compute_ackermann_angles(
                steer_cmd, spec.wheelbase, track_width
            )
            pos_a = torch.zeros(1, n_j, device=device)
            vel_a = torch.zeros(1, n_j, device=device)
            for i in left_steer_a:
                pos_a[0, i] = left_angle
            for i in right_steer_a:
                pos_a[0, i] = right_angle
            for i in drive_ids_a:
                vel_a[0, i] = wheel_vel
            car_acker.write_joint_position_to_sim(
                pos_a[:, steer_ids_a], joint_ids=list(steer_ids_a)
            )
            car_acker.set_joint_velocity_target(vel_a)

            car_simple.write_data_to_sim()
            car_acker.write_data_to_sim()
            sim.step()
            car_simple.update(dt)
            car_acker.update(dt)

            if step % 30 == 0 and abs(steer_cmd) > 0.01:
                print(
                    f"  steer={steer_cmd:+.2f}  "
                    f"Simple: L={steer_cmd:+.3f} R={steer_cmd:+.3f}  |  "
                    f"Ackermann: L={left_angle:+.3f} R={right_angle:+.3f}  "
                    f"(diff={abs(left_angle)-abs(right_angle):.3f})",
                    end="\r",
                )

            step += 1
        except Exception:
            break


if __name__ == "__main__":
    main()
