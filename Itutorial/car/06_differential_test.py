"""Car Tutorial 06 — Differential drive modes comparison (GUI).

3대의 차량을 나란히 스폰하고, 각각 다른 differential 모드로
같은 원호를 주행시켜 거동 차이를 비교합니다.

  차량 1: Locked diff — 좌우 동일 속도 (코너에서 안쪽 바퀴 미끄러짐)
  차량 2: Open diff   — Ackermann 기하학 적용 (내/외륜 속도 차이)
  차량 3: LSD         — 제한된 속도 차이 (중간)

Controls:
  Space: 주행 시작/정지
  W/S: 속도 증감
  A/D: 조향각 변경

Run:
    python Itutorial/car/06_differential_test.py --robot TOY_01
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

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402
from _common import get_robot_spec  # noqa: E402


class Controller:
    def __init__(self):
        self.running = False
        self.speed = 2.0
        self.steer = 0.3
        inp = carb.input.acquire_input_interface()
        kb = omni.appwindow.get_default_app_window().get_keyboard()
        inp.subscribe_to_keyboard_events(kb, self._on_key)

    def _on_key(self, event):
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input == carb.input.KeyboardInput.ENTER:
                self.running = not self.running
                print(f"\n  >>> {'RUNNING' if self.running else 'STOPPED'}")
            elif event.input in (carb.input.KeyboardInput.W, carb.input.KeyboardInput.UP):
                self.speed = min(self.speed + 0.5, 5.0)
                print(f"\n  >>> speed={self.speed:.1f}")
            elif event.input in (carb.input.KeyboardInput.S, carb.input.KeyboardInput.DOWN):
                self.speed = max(self.speed - 0.5, 0.5)
                print(f"\n  >>> speed={self.speed:.1f}")
            elif event.input in (carb.input.KeyboardInput.A, carb.input.KeyboardInput.LEFT):
                self.steer = min(self.steer + 0.05, 0.5)
                print(f"\n  >>> steer={self.steer:.2f}")
            elif event.input in (carb.input.KeyboardInput.D, carb.input.KeyboardInput.RIGHT):
                self.steer = max(self.steer - 0.05, 0.0)
                print(f"\n  >>> steer={self.steer:.2f}")
        return True


def spawn_vehicle(spec, prim_path, pos_y):
    cfg = spec.cfg.replace(prim_path=prim_path)
    cfg = cfg.replace(init_state=cfg.init_state.replace(
        pos=(0.0, pos_y, spec.init_z)
    ))
    cfg.spawn.func(prim_path, cfg.spawn,
                   translation=(0.0, pos_y, spec.init_z))
    return Articulation(cfg)


def compute_diff_speeds(speed, steer_angle, wheelbase, track_width, mode):
    """Compute left/right wheel speeds based on differential mode.

    Returns (left_speed_mps, right_speed_mps).
    """
    if abs(steer_angle) < 0.001:
        return speed, speed

    # Turn radius from bicycle model
    R = wheelbase / math.tan(abs(steer_angle))

    if mode == "locked":
        # Both wheels same speed — inner wheel will slip
        return speed, speed

    elif mode == "open":
        # Ackermann: inner slower, outer faster
        # v_left / v_right = (R - d/2) / (R + d/2) for left turn
        if steer_angle > 0:  # left turn
            left = speed * (R - track_width / 2) / R
            right = speed * (R + track_width / 2) / R
        else:  # right turn
            left = speed * (R + track_width / 2) / R
            right = speed * (R - track_width / 2) / R
        return left, right

    elif mode == "lsd":
        # Limited slip: allow max 30% speed difference
        if steer_angle > 0:
            left = speed * (R - track_width / 2) / R
            right = speed * (R + track_width / 2) / R
        else:
            left = speed * (R + track_width / 2) / R
            right = speed * (R - track_width / 2) / R
        # Limit difference to 30%
        avg = (left + right) / 2
        max_diff = avg * 0.3
        if abs(left - right) > max_diff * 2:
            if left > right:
                left = avg + max_diff
                right = avg - max_diff
            else:
                left = avg - max_diff
                right = avg + max_diff
        return left, right

    return speed, speed


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

    cars = [
        spawn_vehicle(spec, "/World/Locked", -1.0),
        spawn_vehicle(spec, "/World/Open", 0.0),
        spawn_vehicle(spec, "/World/LSD", 1.0),
    ]
    modes = ["locked", "open", "lsd"]
    labels = ["LOCKED", "OPEN  ", "LSD   "]

    sim.reset()

    ctrl = Controller()
    dt = sim.get_physics_dt()
    track_width = 0.20

    joint_info = []
    for car in cars:
        steer_ids, drive_ids = spec.resolve_joints(car.joint_names)
        left_drive = [i for i in drive_ids if "left" in car.joint_names[i]]
        right_drive = [i for i in drive_ids if "right" in car.joint_names[i]]
        joint_info.append((steer_ids, left_drive, right_drive))

    print(f"""
{'='*60}
  Differential Test — {spec.name}
  Wheelbase: {spec.wheelbase}m  Track: {track_width}m

  Left:   LOCKED (same speed both wheels → inner slip)
  Center: OPEN   (Ackermann speed ratio → smooth turn)
  Right:  LSD    (limited 30% speed diff → compromise)

  Enter: start/stop
  W/S: speed ±0.5    A/D: steer ±0.05
{'='*60}
""")

    step = 0
    while simulation_app.is_running():
        try:
            for idx, (car, (steer_ids, left_d, right_d)) in enumerate(
                zip(cars, joint_info)
            ):
                n_j = car.num_joints
                device = car.device

                # Steering (force write)
                steer_val = ctrl.steer if ctrl.running else 0.0
                steer_t = torch.full(
                    (1, len(steer_ids)), steer_val, device=device
                )
                car.write_joint_position_to_sim(
                    steer_t, joint_ids=list(steer_ids)
                )

                # Drive
                if ctrl.running:
                    left_spd, right_spd = compute_diff_speeds(
                        ctrl.speed, ctrl.steer, spec.wheelbase,
                        track_width, modes[idx]
                    )
                    left_wvel = left_spd / spec.wheel_radius
                    right_wvel = right_spd / spec.wheel_radius
                else:
                    left_wvel = 0.0
                    right_wvel = 0.0

                # Force write wheel velocities directly
                if left_d:
                    left_t = torch.full(
                        (1, len(left_d)), left_wvel, device=device
                    )
                    car.write_joint_velocity_to_sim(
                        left_t, joint_ids=list(left_d)
                    )
                if right_d:
                    right_t = torch.full(
                        (1, len(right_d)), right_wvel, device=device
                    )
                    car.write_joint_velocity_to_sim(
                        right_t, joint_ids=list(right_d)
                    )

            sim.step()
            for car in cars:
                car.update(dt)

            if step % 50 == 0 and ctrl.running:
                parts = []
                for label, car, mode, (_, left_d, right_d) in zip(
                    labels, cars, modes, joint_info
                ):
                    lv = car.data.joint_vel[0, left_d].mean().item() \
                        if left_d else 0.0
                    rv = car.data.joint_vel[0, right_d].mean().item() \
                        if right_d else 0.0
                    cv = car.data.root_lin_vel_w[0, :2].norm().item()

                    left_cmd, right_cmd = compute_diff_speeds(
                        ctrl.speed, ctrl.steer, spec.wheelbase,
                        track_width, mode
                    )
                    parts.append(
                        f"{label} cmd=({left_cmd:.1f}/{right_cmd:.1f}) "
                        f"L={lv*spec.wheel_radius:.2f} R={rv*spec.wheel_radius:.2f} "
                        f"v={cv:.2f}"
                    )
                print("  " + "  |  ".join(parts), end="\r")

            step += 1
        except Exception:
            break


if __name__ == "__main__":
    main()
