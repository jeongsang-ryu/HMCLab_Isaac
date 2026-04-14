"""Car Tutorial 06 — 구동 방식 비교 테스트 (GUI).

같은 차량에서 숫자 키로 구동 모드를 전환하며 차이를 체감합니다.

  1: Velocity Control  — set_joint_velocity_target (목표 속도 지정)
  2: Effort Control    — set_joint_effort_target (직접 토크 지정)
  3: Force Write       — write_joint_velocity_to_sim (속도 강제 설정)
  4: Position Control  — write_joint_position_to_sim (각도 강제 설정)
  5: Ramp Velocity     — 0에서 max까지 서서히 증가
  6: Effort + Steering — 토크로 구동 + 조향 (가장 현실적)

WASD: 전진/후진/좌/우
Enter: 시작/정지

Run:
    python Itutorial/car/06_drive_modes.py --robot TOY_01
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
parser.add_argument("--max_speed", type=float, default=3.0)
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


MODE_NAMES = {
    1: "VELOCITY TARGET  — set_joint_velocity_target()",
    2: "EFFORT (TORQUE)  — set_joint_effort_target()",
    3: "FORCE WRITE VEL  — write_joint_velocity_to_sim()",
    4: "FORCE WRITE POS  — write_joint_position_to_sim()",
    5: "RAMP VELOCITY    — 0 → max over 3 sec",
    6: "EFFORT + STEER   — torque drive + steering",
}


class Controller:
    def __init__(self):
        self.mode = 1
        self.running = False
        self.throttle = 0.0   # -1 ~ +1
        self.steering = 0.0   # -1 ~ +1
        inp = carb.input.acquire_input_interface()
        kb = omni.appwindow.get_default_app_window().get_keyboard()
        inp.subscribe_to_keyboard_events(kb, self._on_key)

    def _on_key(self, event):
        pressed = event.type == carb.input.KeyboardEventType.KEY_PRESS
        active = event.type in (
            carb.input.KeyboardEventType.KEY_PRESS,
            carb.input.KeyboardEventType.KEY_REPEAT,
        )
        key = event.input

        # Mode selection
        if pressed:
            mode_keys = {
                carb.input.KeyboardInput.KEY_1: 1,
                carb.input.KeyboardInput.KEY_2: 2,
                carb.input.KeyboardInput.KEY_3: 3,
                carb.input.KeyboardInput.KEY_4: 4,
                carb.input.KeyboardInput.KEY_5: 5,
                carb.input.KeyboardInput.KEY_6: 6,
            }
            if key in mode_keys:
                self.mode = mode_keys[key]
                print(f"\n  >>> Mode {self.mode}: {MODE_NAMES[self.mode]}")
            elif key == carb.input.KeyboardInput.ENTER:
                self.running = not self.running
                print(f"\n  >>> {'RUNNING' if self.running else 'STOPPED'}")

        # WASD
        if key in (carb.input.KeyboardInput.W, carb.input.KeyboardInput.UP):
            self.throttle = 1.0 if active else 0.0
        elif key in (carb.input.KeyboardInput.S, carb.input.KeyboardInput.DOWN):
            self.throttle = -0.5 if active else 0.0
        elif key in (carb.input.KeyboardInput.A, carb.input.KeyboardInput.LEFT):
            self.steering = 1.0 if active else 0.0
        elif key in (carb.input.KeyboardInput.D, carb.input.KeyboardInput.RIGHT):
            self.steering = -1.0 if active else 0.0

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
    cfg.spawn.func("/World/Robot", cfg.spawn,
                   translation=(0.0, 0.0, spec.init_z))
    robot = Articulation(cfg)
    sim.reset()

    ctrl = Controller()
    steer_ids, drive_ids = spec.resolve_joints(robot.joint_names)
    left_drive = [i for i in drive_ids if "left" in robot.joint_names[i]]
    right_drive = [i for i in drive_ids if "right" in robot.joint_names[i]]
    n_j = robot.num_joints
    device = robot.device
    dt = sim.get_physics_dt()

    max_wheel_vel = args.max_speed / spec.wheel_radius
    ramp_progress = 0.0
    wheel_pos_accum = 0.0  # for mode 4

    print(f"""
{'='*60}
  Drive Modes Test — {spec.name}
  Max speed: {args.max_speed} m/s ({max_wheel_vel:.1f} rad/s)
  Wheel radius: {spec.wheel_radius} m

  1: Velocity target   (PD가 속도 유지 시도)
  2: Effort/torque     (직접 Nm 지정, 가속됨)
  3: Force write vel   (PD 무시, 속도 강제)
  4: Force write pos   (바퀴 각도 직접 제어)
  5: Ramp velocity     (서서히 가속)
  6: Effort + steering (현실적 구동)

  WASD: 조작    Enter: 시작/정지    1~6: 모드 변경
{'='*60}
""")

    step = 0
    while simulation_app.is_running():
        try:
            # ── Steering (모든 모드 공통) ──
            steer_val = ctrl.steering * spec.max_steer if ctrl.running else 0.0
            if steer_ids:
                steer_t = torch.full(
                    (1, len(steer_ids)), steer_val, device=device
                )
                robot.write_joint_position_to_sim(
                    steer_t, joint_ids=list(steer_ids)
                )

            # ── Drive (모드별 다름) ──
            if ctrl.running and abs(ctrl.throttle) > 0.01:
                target_vel = ctrl.throttle * max_wheel_vel
                target_effort = ctrl.throttle * 1.0  # Nm

                if ctrl.mode == 1:
                    # ── Velocity Target: PD 컨트롤러가 속도 유지 ──
                    vel_t = torch.zeros(1, n_j, device=device)
                    for di in drive_ids:
                        vel_t[0, di] = target_vel
                    robot.set_joint_velocity_target(vel_t)

                elif ctrl.mode == 2:
                    # ── Effort: 직접 토크, 가속이 계속됨 ──
                    eff_t = torch.zeros(1, n_j, device=device)
                    for di in drive_ids:
                        eff_t[0, di] = target_effort
                    robot.set_joint_effort_target(eff_t)

                elif ctrl.mode == 3:
                    # ── Force Write Vel: PD 무시, 바퀴 속도 강제 ──
                    for d_list in (left_drive, right_drive):
                        if d_list:
                            v_t = torch.full(
                                (1, len(d_list)), target_vel, device=device
                            )
                            robot.write_joint_velocity_to_sim(
                                v_t, joint_ids=list(d_list)
                            )

                elif ctrl.mode == 4:
                    # ── Force Write Pos: 바퀴 각도 직접 제어 ──
                    # 매 step마다 각도를 누적시켜 회전
                    wheel_pos_accum += target_vel * dt
                    for d_list in (left_drive, right_drive):
                        if d_list:
                            p_t = torch.full(
                                (1, len(d_list)), wheel_pos_accum, device=device
                            )
                            robot.write_joint_position_to_sim(
                                p_t, joint_ids=list(d_list)
                            )

                elif ctrl.mode == 5:
                    # ── Ramp: 0 → max 서서히 증가 ──
                    ramp_progress = min(ramp_progress + dt / 3.0, 1.0)
                    ramp_vel = target_vel * ramp_progress
                    for d_list in (left_drive, right_drive):
                        if d_list:
                            v_t = torch.full(
                                (1, len(d_list)), ramp_vel, device=device
                            )
                            robot.write_joint_velocity_to_sim(
                                v_t, joint_ids=list(d_list)
                            )

                elif ctrl.mode == 6:
                    # ── Effort + Steering: 토크로 구동 ──
                    # velocity target=0 으로 해서 damping 안 싸우게
                    vel_t = torch.zeros(1, n_j, device=device)
                    robot.set_joint_velocity_target(vel_t)
                    eff_t = torch.zeros(1, n_j, device=device)
                    for di in drive_ids:
                        eff_t[0, di] = target_effort * 3.0
                    robot.set_joint_effort_target(eff_t)

            else:
                # 정지
                if ctrl.mode in (3, 4, 5):
                    for d_list in (left_drive, right_drive):
                        if d_list:
                            v_t = torch.zeros(1, len(d_list), device=device)
                            robot.write_joint_velocity_to_sim(
                                v_t, joint_ids=list(d_list)
                            )
                else:
                    vel_t = torch.zeros(1, n_j, device=device)
                    robot.set_joint_velocity_target(vel_t)
                ramp_progress = 0.0

            robot.write_data_to_sim()
            sim.step()
            robot.update(dt)

            # ── Telemetry ──
            if step % 30 == 0:
                wheel_w = robot.data.joint_vel[0, drive_ids].mean().item()
                wheel_v = wheel_w * spec.wheel_radius
                chassis_v = robot.data.root_lin_vel_w[0, :2].norm().item()
                slip = (wheel_v - chassis_v) / max(abs(wheel_v), 0.01) \
                    if abs(wheel_v) > 0.01 else 0.0

                mode_short = ["", "VEL_TGT", "EFFORT ", "FRC_VEL",
                              "FRC_POS", "RAMP   ", "EFF+STR"][ctrl.mode]
                bar_len = int(min(abs(chassis_v) / args.max_speed, 1.0) * 20)
                bar = "█" * bar_len + "░" * (20 - bar_len)

                print(
                    f"  [{mode_short}] "
                    f"wheel={wheel_w:+6.1f}rad/s "
                    f"chassis={chassis_v:5.2f}m/s "
                    f"slip={slip:+.2f} "
                    f"|{bar}|",
                    end="\r",
                )

            step += 1
        except Exception:
            break


if __name__ == "__main__":
    main()
