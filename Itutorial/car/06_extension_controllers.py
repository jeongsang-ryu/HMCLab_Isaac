"""Car Tutorial 06 — Isaac Sim 내장 컨트롤러 종합 테스트 (GUI).

isaacsim.robot.wheeled_robots extension의 모든 컨트롤러를
TOY_01 차량으로 순서대로 테스트합니다.

  1: DifferentialController  — (linear_vel, angular_vel) → 좌/우 바퀴 속도
  2: WheelBasePoseController — 목표 좌표로 자동 주행
  3: Stanley + PID           — 경로 추종 (Quintic 경로 생성 포함)
  4: Ackermann geometry      — 내/외륜 각도 계산

WASD: 수동 입력 (모드 1)
1~4: 모드 전환
Enter: 시작/정지

Run:
    python Itutorial/car/06_extension_controllers.py --robot TOY_01
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


MODE_NAMES = {
    1: "DifferentialController — (linear, angular) → wheel vel",
    2: "WheelBasePoseController — go to goal (2,2)",
    3: "Stanley+PID — quintic path following",
    4: "Ackermann geometry — inner/outer wheel angles",
}


class Controller:
    def __init__(self):
        self.mode = 1
        self.running = False
        self.linear = 0.0
        self.angular = 0.0
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

        if pressed:
            keys = {
                carb.input.KeyboardInput.KEY_1: 1,
                carb.input.KeyboardInput.KEY_2: 2,
                carb.input.KeyboardInput.KEY_3: 3,
                carb.input.KeyboardInput.KEY_4: 4,
            }
            if key in keys:
                self.mode = keys[key]
                print(f"\n  >>> Mode {self.mode}: {MODE_NAMES[self.mode]}")
            elif key == carb.input.KeyboardInput.ENTER:
                self.running = not self.running
                print(f"\n  >>> {'RUNNING' if self.running else 'STOPPED'}")

        if key in (carb.input.KeyboardInput.W, carb.input.KeyboardInput.UP):
            self.linear = 1.5 if active else 0.0
        elif key in (carb.input.KeyboardInput.S, carb.input.KeyboardInput.DOWN):
            self.linear = -1.0 if active else 0.0
        elif key in (carb.input.KeyboardInput.A, carb.input.KeyboardInput.LEFT):
            self.angular = 2.0 if active else 0.0
        elif key in (carb.input.KeyboardInput.D, carb.input.KeyboardInput.RIGHT):
            self.angular = -2.0 if active else 0.0
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
            static_friction=1.5, dynamic_friction=1.3),
    ).func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=3000.0).func(
        "/World/Light", sim_utils.DomeLightCfg(intensity=3000.0))

    cfg = spec.cfg.replace(prim_path="/World/Robot")
    cfg = cfg.replace(init_state=cfg.init_state.replace(
        pos=(0.0, 0.0, spec.init_z)))
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
    track_width = 0.20

    # ── 컨트롤러 초기화 ──
    # 주의: DifferentialController는 cm 단위!
    diff_ctrl = DifferentialController(
        name="diff",
        wheel_radius=spec.wheel_radius * 100,   # m → cm
        wheel_base=track_width * 100,            # m → cm
    )

    pose_ctrl = WheelBasePoseController(
        name="pose",
        open_loop_wheel_controller=diff_ctrl,
        is_holonomic=False,
    )

    # Stanley path (미리 생성)
    path_generated = False
    path_x, path_y, path_yaw = [], [], []
    path_idx = 0

    print(f"""
{'='*65}
  Extension Controllers Test — {spec.name}
  wheel_radius={spec.wheel_radius}m  track={track_width}m

  1: DifferentialController (WASD로 조작)
     공식: ω_L = (2V - ωb) / 2r
           ω_R = (2V + ωb) / 2r

  2: WheelBasePoseController (자동으로 (2,2)로 이동)

  3: Stanley+PID path following
     (Enter 누르면 (0,0)→(3,2) 경로 생성 + 추종)

  4: Ackermann geometry (WASD로 조작)
     공식: δ_left  = atan(L / (R - d/2))
           δ_right = atan(L / (R + d/2))

  Enter: 시작    1~4: 모드    WASD: 입력
{'='*65}
""")

    step = 0
    while simulation_app.is_running():
        try:
            pos = robot.data.root_pos_w[0].cpu().numpy()
            quat = robot.data.root_quat_w[0].cpu().numpy()
            yaw = float(quat_to_yaw(quat))
            chassis_v = robot.data.root_lin_vel_w[0, :2].norm().item()

            if not ctrl.running:
                vel_t = torch.zeros(1, n_j, device=device)
                robot.set_joint_velocity_target(vel_t)
                robot.write_data_to_sim()
                sim.step()
                robot.update(dt)
                step += 1
                continue

            # ═══ Mode 1: DifferentialController ═══
            if ctrl.mode == 1:
                command = np.array([ctrl.linear, ctrl.angular])
                actions = diff_ctrl.forward(command)
                left_vel = float(actions.joint_velocities[0])
                right_vel = float(actions.joint_velocities[1])

                for d_list, vel in [(left_drive, left_vel), (right_drive, right_vel)]:
                    if d_list:
                        v_t = torch.full((1, len(d_list)), vel, device=device)
                        robot.write_joint_velocity_to_sim(v_t, joint_ids=list(d_list))

                if step % 50 == 0:
                    print(
                        f"  [DIFF] cmd=({ctrl.linear:+.1f},{ctrl.angular:+.1f}) "
                        f"→ L={left_vel:+6.1f} R={right_vel:+6.1f} rad/s  "
                        f"v={chassis_v:.2f}m/s",
                        end="\r",
                    )

            # ═══ Mode 2: WheelBasePoseController ═══
            elif ctrl.mode == 2:
                goal = np.array([2.0, 2.0])
                actions = pose_ctrl.forward(
                    start_position=pos[:2],
                    start_orientation=yaw,
                    goal_position=goal,
                    goal_orientation=math.atan2(goal[1], goal[0]),
                    lateral_velocity=0.5,
                    yaw_velocity=1.0,
                    position_tol=0.1,
                    heading_tol=0.1,
                )
                if actions is not None and actions.joint_velocities is not None:
                    left_vel = float(actions.joint_velocities[0])
                    right_vel = float(actions.joint_velocities[1])
                    for d_list, vel in [(left_drive, left_vel), (right_drive, right_vel)]:
                        if d_list:
                            v_t = torch.full((1, len(d_list)), vel, device=device)
                            robot.write_joint_velocity_to_sim(v_t, joint_ids=list(d_list))

                dist = np.linalg.norm(pos[:2] - goal)
                if step % 50 == 0:
                    print(
                        f"  [POSE] pos=({pos[0]:+.2f},{pos[1]:+.2f}) "
                        f"→ goal=(2.0,2.0) dist={dist:.3f}m  v={chassis_v:.2f}",
                        end="\r",
                    )

            # ═══ Mode 3: Stanley + PID path following ═══
            elif ctrl.mode == 3:
                if not path_generated:
                    # 직선+커브 경로 생성
                    t = np.linspace(0, 1, 80)
                    path_x = list(t * 3.0)
                    path_y = list(t * 2.0)
                    path_yaw = list(np.arctan2(np.gradient(path_y),
                                               np.gradient(path_x)))
                    path_idx = 0
                    path_generated = True
                    print("\n  >>> Path generated: (0,0) → (3,2)")

                if path_idx < len(path_x):
                    # Simple pursuit
                    look = min(path_idx + 5, len(path_x) - 1)
                    dx = path_x[look] - pos[0]
                    dy = path_y[look] - pos[1]
                    target_yaw = math.atan2(dy, dx)
                    err = target_yaw - yaw
                    err = math.atan2(math.sin(err), math.cos(err))
                    steer = max(-spec.max_steer, min(spec.max_steer, err * 2.0))

                    dist_goal = math.sqrt((pos[0]-3.0)**2 + (pos[1]-2.0)**2)
                    speed = min(1.5, dist_goal * 2.0)
                    if dist_goal < 0.1:
                        speed = 0.0
                        print("\n  >>> GOAL REACHED!")
                        path_generated = False

                    # PID speed → wheel velocity
                    wheel_vel = speed / spec.wheel_radius
                    steer_t = torch.full((1, len(steer_ids)), steer, device=device)
                    robot.write_joint_position_to_sim(steer_t, joint_ids=list(steer_ids))
                    for d_list in (left_drive, right_drive):
                        if d_list:
                            v_t = torch.full((1, len(d_list)), wheel_vel, device=device)
                            robot.write_joint_velocity_to_sim(v_t, joint_ids=list(d_list))

                    # Advance path index
                    dists = [math.sqrt((pos[0]-px)**2+(pos[1]-py)**2)
                             for px, py in zip(path_x[path_idx:path_idx+10],
                                               path_y[path_idx:path_idx+10])]
                    if dists:
                        path_idx += min(range(len(dists)), key=lambda i: dists[i])
                        path_idx = min(path_idx + 1, len(path_x) - 1)

                    if step % 50 == 0:
                        print(
                            f"  [STANLEY] path[{path_idx}/{len(path_x)}] "
                            f"steer={steer:+.2f} v={chassis_v:.2f} "
                            f"dist={dist_goal:.2f}m",
                            end="\r",
                        )

            # ═══ Mode 4: Ackermann geometry ═══
            elif ctrl.mode == 4:
                steer_cmd = ctrl.angular * 0.15  # angular input → steer angle
                speed = ctrl.linear

                if abs(steer_cmd) > 0.001:
                    R = spec.wheelbase / math.tan(abs(steer_cmd))
                    left_a = math.atan(spec.wheelbase / (R - track_width/2))
                    right_a = math.atan(spec.wheelbase / (R + track_width/2))
                    if steer_cmd > 0:
                        steer_left, steer_right = left_a, right_a
                    else:
                        steer_left, steer_right = -right_a, -left_a
                else:
                    steer_left = steer_right = 0.0

                # Apply per-wheel steering
                left_steer = [i for i in steer_ids if "left" in robot.joint_names[i]]
                right_steer = [i for i in steer_ids if "right" in robot.joint_names[i]]
                if left_steer:
                    robot.write_joint_position_to_sim(
                        torch.tensor([[steer_left]], device=device),
                        joint_ids=list(left_steer))
                if right_steer:
                    robot.write_joint_position_to_sim(
                        torch.tensor([[steer_right]], device=device),
                        joint_ids=list(right_steer))

                wheel_vel = speed / spec.wheel_radius
                for d_list in (left_drive, right_drive):
                    if d_list:
                        v_t = torch.full((1, len(d_list)), wheel_vel, device=device)
                        robot.write_joint_velocity_to_sim(v_t, joint_ids=list(d_list))

                if step % 50 == 0 and abs(steer_cmd) > 0.01:
                    print(
                        f"  [ACKER] steer={steer_cmd:+.3f} "
                        f"→ L={math.degrees(steer_left):+5.1f}° "
                        f"R={math.degrees(steer_right):+5.1f}° "
                        f"(diff={abs(steer_left)-abs(steer_right):.2f}°) "
                        f"v={chassis_v:.2f}",
                        end="\r",
                    )

            robot.write_data_to_sim()
            sim.step()
            robot.update(dt)
            step += 1
        except Exception:
            break


if __name__ == "__main__":
    main()
