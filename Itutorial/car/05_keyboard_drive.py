"""Car Tutorial 05 — Keyboard driving (GUI).

WASD로 차량을 직접 조작합니다.

Controls:
  W / ↑  : 전진
  S / ↓  : 후진
  A / ←  : 좌회전
  D / →  : 우회전
  Enter  : 브레이크 (속도 0)
  ESC    : 종료

Run:
    python Itutorial/car/05_keyboard_drive.py --robot TOY_01
    python Itutorial/car/05_keyboard_drive.py --robot TOY_01 --track mini_oval_flat
"""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--robot", type=str, default="TOY_01")
parser.add_argument("--track", type=str, default=None)
parser.add_argument("--max_speed", type=float, default=3.0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = False
launcher = AppLauncher(args)
simulation_app = launcher.app

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import math  # noqa: E402

import carb.input  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402

from _common import apply_ackermann, get_robot_spec  # noqa: E402


class KeyboardController:
    """WASD + Arrow key listener."""

    def __init__(self):
        self.throttle = 0.0   # -1 (reverse) ~ +1 (forward)
        self.steering = 0.0   # -1 (right) ~ +1 (left)
        self.brake = False

        self._input = carb.input.acquire_input_interface()
        import omni.appwindow
        self._keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        self._sub = self._input.subscribe_to_keyboard_events(
            self._keyboard, self._on_key
        )

    def _on_key(self, event):
        pressed = event.type == carb.input.KeyboardEventType.KEY_PRESS
        repeat = event.type == carb.input.KeyboardEventType.KEY_REPEAT
        released = event.type == carb.input.KeyboardEventType.KEY_RELEASE
        active = pressed or repeat

        key = event.input

        if key in (carb.input.KeyboardInput.W, carb.input.KeyboardInput.UP):
            self.throttle = 1.0 if active else 0.0
        elif key in (carb.input.KeyboardInput.S, carb.input.KeyboardInput.DOWN):
            self.throttle = -1.0 if active else 0.0
        elif key in (carb.input.KeyboardInput.A, carb.input.KeyboardInput.LEFT):
            self.steering = 1.0 if active else 0.0
        elif key in (carb.input.KeyboardInput.D, carb.input.KeyboardInput.RIGHT):
            self.steering = -1.0 if active else 0.0
        elif key == carb.input.KeyboardInput.ENTER:
            self.brake = active

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

    # Track (optional)
    if args.track:
        from hmclab_isaac.envs.racing._base import default_track_path
        from hmclab_isaac.worlds.racing import RacingTrack, spawn_circuit
        from hmclab_isaac.worlds.racing.duct_track import spawn_duct_track

        track = RacingTrack.load(default_track_path(args.track))
        spawn_circuit(track, prim_path="/World/Road", surface_only=True,
                      color=(0.35, 0.35, 0.35))
        spawn_duct_track(track, prim_path="/World/Duct",
                         pipe_radius=0.15, pipe_offset=1.0, rib_spacing=0.8)

        start_pos, start_rpy = track.spawn_pose(progress=0.0)
        yaw = float(start_rpy[2])
        init_pos = (float(start_pos[0]), float(start_pos[1]),
                    float(start_pos[2]) + spec.init_z)
        init_rot = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
    else:
        init_pos = (0.0, 0.0, spec.init_z)
        init_rot = None

    cfg = spec.cfg.replace(prim_path="/World/Robot")
    if init_rot:
        cfg = cfg.replace(
            init_state=cfg.init_state.replace(pos=init_pos, rot=init_rot)
        )
    else:
        cfg = cfg.replace(init_state=cfg.init_state.replace(pos=init_pos))
    cfg.spawn.func(cfg.prim_path, cfg.spawn,
                   translation=init_pos,
                   orientation=init_rot if init_rot else (1, 0, 0, 0))
    robot = Articulation(cfg)

    sim.reset()

    kb = KeyboardController()
    steer_ids, drive_ids = spec.resolve_joints(robot.joint_names)
    dt = sim.get_physics_dt()

    print(f"""
{'='*50}
  Keyboard Drive — {spec.name}
  Max speed: {args.max_speed} m/s

  W/↑ : forward    S/↓ : reverse
  A/← : left       D/→ : right
  Enter : brake
  Close window to exit
{'='*50}
""")

    step = 0
    while simulation_app.is_running():
        try:
            if kb.brake:
                speed = 0.0
                steer = 0.0
            else:
                speed = kb.throttle * args.max_speed
                steer = kb.steering * spec.max_steer

            apply_ackermann(
                robot, spec,
                steer_rad=steer,
                speed_mps=speed,
                steer_ids=steer_ids,
                drive_ids=drive_ids,
            )
            robot.write_data_to_sim()
            sim.step()
            robot.update(dt)

            if step % 50 == 0:
                vel = robot.data.root_lin_vel_w[0, :2].norm().item()
                pos = robot.data.root_pos_w[0]
                print(
                    f"  speed={vel:.2f} m/s  "
                    f"pos=({pos[0].item():.2f}, {pos[1].item():.2f})  "
                    f"throttle={kb.throttle:+.0f}  steer={kb.steering:+.0f}",
                    end="\r",
                )
            step += 1
        except Exception:
            break


if __name__ == "__main__":
    main()
