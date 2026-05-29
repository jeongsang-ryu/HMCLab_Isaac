"""Keyboard + joystick teleop for MuSHR (WheeledLab cfg) — UNICORN_2 reference.

Uses the MuSHR USD + actuator cfg from the bundled WheeledLab repo:
  reference_repos/WheeledLab/source/wheeledlab_assets/data/Robots/UWRLL/mushr_nano_v2.usd

The actuator setup follows WheeledLab's MUSHR_SUS_CFG (4WD + suspension):
  - DCMotor throttle (saturation 1.05 Nm, effort 0.25 Nm)
  - ImplicitActuator steering (k=100, c=10, effort 3.2 Nm)
  - Passive suspension (k=1e8, c=0, friction 0.5)

Keyboard:
    ↑ / ↓   throttle, ← / →   steer, Space brake, L reset

Joystick (default Xbox-style):
    Left stick Y   → throttle (up = forward)
    Right stick X  → steer
    Button 0 (A)   → brake

Run:
    OMNI_KIT_ACCEPT_EULA=YES python scripts/keyboard_teleop_mushr.py
    OMNI_KIT_ACCEPT_EULA=YES python scripts/keyboard_teleop_mushr.py \\
      --track mini_oval_banked
"""
from __future__ import annotations

import argparse
import os
import re

from isaaclab.app import AppLauncher

from hmclab_isaac.utils.teleop_input import add_input_args, make_controller

TRACKS_DIR = (
    "/home/js/hmcl_issac_project/HMCLab_Isaac/"
    "hmclab_isaac/worlds/racing/_tracks_data"
)
TRACK_CHOICES = sorted(
    f.removesuffix(".txt") for f in os.listdir(TRACKS_DIR) if f.endswith(".txt")
)

parser = argparse.ArgumentParser()
parser.add_argument("--max-wheel-rps", type=float, default=152.0,
                    help="commanded max wheel rotational velocity (rad/s). "
                         "152 × 0.0525 m wheel ≈ 8 m/s top speed. "
                         "MuSHR's drag-limited natural top is also ~8 m/s "
                         "with effort_limit_sim=0.25 Nm.")
parser.add_argument("--dt", type=float, default=0.005)
parser.add_argument("--track", choices=["", *TRACK_CHOICES], default="",
                    help="If set, spawn a 3D racing track instead of flat ground.")
parser.add_argument("--track-wall-height", type=float, default=0.5)
parser.add_argument("--track-surface-only", action="store_true")
parser.add_argument("--track-z-offset", type=float, default=0.20,
                    help="Lift vehicle this much above the track surface (m).")
parser.add_argument("--track-subdivide", type=int, default=10,
                    help="Subdivide each centerline segment into N pieces "
                         "for a smoother (less faceted) surface.")
add_input_args(parser)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = False
args.enable_cameras = False
launcher = AppLauncher(args)
sim_app = launcher.app

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import ArticulationCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402

USD_PATH = (
    "/home/js/hmcl_issac_project/reference_repos/WheeledLab/"
    "source/wheeledlab_assets/data/Robots/UWRLL/mushr_nano_v2.usd"
)

MUSHR_MAX_STEER = 0.4   # ~23 deg


# ─────────────── Actuator cfg (WheeledLab MUSHR_SUS_CFG) ───────────────

ACTUATORS = {
    "steering_joints": ImplicitActuatorCfg(
        joint_names_expr=["front_left_wheel_steer", "front_right_wheel_steer"],
        velocity_limit_sim=10.0,
        effort_limit_sim=3.2,
        stiffness=100.0,
        damping=10.0,
        friction=0.0,
    ),
    "throttle_joints": DCMotorCfg(
        joint_names_expr=[".*throttle"],
        saturation_effort=1.05,
        effort_limit_sim=0.25,
        velocity_limit_sim=450.0,
        stiffness=0.0,
        damping=1000.0,
        friction=0.0,
    ),
    "suspension": ImplicitActuatorCfg(
        joint_names_expr=[".*_suspension"],
        effort_limit_sim=None,
        velocity_limit_sim=None,
        stiffness=1.0e8,
        damping=0.0,
        friction=0.5,
    ),
}


MUSHR_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=100000.0,
            max_depenetration_velocity=100.0,
            max_contact_impulse=0.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.05),
        joint_pos={
            "back_left_wheel_throttle":   0.0,
            "back_right_wheel_throttle":  0.0,
            "front_left_wheel_steer":     0.0,
            "front_right_wheel_steer":    0.0,
            "front_left_wheel_throttle":  0.0,
            "front_right_wheel_throttle": 0.0,
            "front_left_wheel_suspension":  0.0,
            "front_right_wheel_suspension": 0.0,
            "back_left_wheel_suspension":   0.0,
            "back_right_wheel_suspension":  0.0,
        },
    ),
    actuators=ACTUATORS,
)


# ─────────────── main ───────────────

def main():
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=args.dt, device="cuda:0")
    )
    sim_utils.DomeLightCfg(intensity=2000.0, color=(0.95, 0.95, 0.95)).func(
        "/World/Light",
        sim_utils.DomeLightCfg(intensity=2000.0, color=(0.95, 0.95, 0.95)),
    )

    spawn_pos = (0.0, 0.0, 0.0)
    spawn_yaw = 0.0

    if args.track:
        import numpy as np
        from hmclab_isaac.worlds.racing import RacingTrack, spawn_circuit
        track = RacingTrack.load(os.path.join(TRACKS_DIR, f"{args.track}.txt"))

        # Densify centerline for smoother surface
        sub = max(1, int(args.track_subdivide))
        if sub > 1:
            n_old = len(track.positions)
            new_pos, new_rpy, new_w = [], [], []
            last = n_old if track.closed else n_old - 1
            for i in range(last):
                j = (i + 1) % n_old
                for k in range(sub):
                    t = k / sub
                    new_pos.append(track.positions[i] * (1 - t) + track.positions[j] * t)
                    new_rpy.append(track.rpy[i] * (1 - t) + track.rpy[j] * t)
                    new_w.append(track.widths[i] * (1 - t) + track.widths[j] * t)
            if not track.closed:
                new_pos.append(track.positions[-1])
                new_rpy.append(track.rpy[-1])
                new_w.append(track.widths[-1])
            track = type(track)(
                name=track.name,
                positions=np.array(new_pos),
                rpy=np.array(new_rpy),
                widths=np.array(new_w),
                closed=track.closed,
            )
            print(f"[teleop] track {args.track!r} densified {n_old} → "
                  f"{len(track.positions)} pts (subdivide={sub})", flush=True)

        spawn_circuit(
            track, "/World/track",
            wall_height=args.track_wall_height,
            surface_only=args.track_surface_only,
            color=(0.4, 0.4, 0.4),
        )
        spawn_pos = (
            float(track.positions[0, 0]),
            float(track.positions[0, 1]),
            float(track.positions[0, 2]) + args.track_z_offset,
        )
        spawn_yaw = float(track.rpy[0, 2])
        sim.set_camera_view(
            eye=(spawn_pos[0] + 2.0, spawn_pos[1] + 2.0, spawn_pos[2] + 1.5),
            target=spawn_pos,
        )
    else:
        sim.set_camera_view(eye=(1.5, 1.5, 1.0), target=(0.0, 0.0, 0.1))
        gp = sim_utils.GroundPlaneCfg(
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.5, dynamic_friction=1.3, restitution=0.0,
            ),
        )
        gp.func("/World/ground", gp)
        for i, (x, y) in enumerate([(3.0, 0.0), (0.0, 3.0), (-3.0, -1.5)]):
            cube_cfg = sim_utils.MeshCuboidCfg(
                size=(0.6, 0.6, 1.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.7, 0.2, 0.2)),
            )
            cube_cfg.func(f"/World/Obstacle_{i}", cube_cfg, translation=(x, y, 0.5))

    # Override init_state with track start (or default)
    import math
    init_state = MUSHR_CFG.init_state.replace(pos=spawn_pos)
    if args.track:
        cy, sy = math.cos(spawn_yaw / 2.0), math.sin(spawn_yaw / 2.0)
        init_state = init_state.replace(rot=(cy, 0.0, 0.0, sy))
    robot_cfg = MUSHR_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        init_state=init_state,
    )

    class _SceneCfg(InteractiveSceneCfg):
        robot = robot_cfg

    scene = InteractiveScene(_SceneCfg(num_envs=1, env_spacing=2.5,
                                       replicate_physics=True))
    sim.reset()

    art = scene["robot"]
    joint_names = art.joint_names
    steer_re = re.compile(r"_wheel_steer$")
    throttle_re = re.compile(r"_wheel_throttle$")
    steer_ids = [i for i, n in enumerate(joint_names) if steer_re.search(n)]
    drive_ids = [i for i, n in enumerate(joint_names) if throttle_re.search(n)]

    print(f"\n[teleop] MuSHR (WheeledLab MUSHR_SUS)", flush=True)
    print(f"[teleop] joints  = {len(joint_names)}", flush=True)
    print(f"[teleop] steer   = {[joint_names[i] for i in steer_ids]}", flush=True)
    print(f"[teleop] drive   = {[joint_names[i] for i in drive_ids]}  (4WD)", flush=True)
    print(f"[teleop] DCMotor effort_limit_sim=0.25 Nm/wheel, sat=1.05 Nm",
          flush=True)
    print("\n[teleop] Click on the GUI window to capture keyboard.", flush=True)
    print("[teleop] Keys: ↑/↓ throttle, ←/→ steer, Space brake, L reset.",
          flush=True)
    if args.joystick:
        print(f"[teleop] Joystick: stick X = steer (axis {args.js_axis_steer}), "
              f"stick Y = throttle (axis {args.js_axis_throttle})  "
              f"button {args.js_button_brake} = brake.\n",
              flush=True)
    else:
        print("[teleop] Joystick disabled.\n", flush=True)

    ctrl = make_controller(args)

    dev = sim.device
    steer_target = torch.zeros((1, len(steer_ids)), device=dev)
    drive_target = torch.zeros((1, len(drive_ids)), device=dev)

    step_count = 0
    while sim_app.is_running():
        thr, steer_in = ctrl.read()
        wheel_speed = thr * args.max_wheel_rps
        steer_angle = steer_in * MUSHR_MAX_STEER

        drive_target.fill_(wheel_speed)
        steer_target.fill_(steer_angle)
        art.set_joint_velocity_target(drive_target, joint_ids=drive_ids)
        art.set_joint_position_target(steer_target, joint_ids=steer_ids)
        art.write_data_to_sim()

        if step_count % int(0.5 / args.dt) == 0:
            steer_actual = art.data.joint_pos[0, steer_ids].tolist()
            wheel_w = art.data.joint_vel[0, drive_ids].tolist()       # rad/s
            try:
                wheel_tau = art.data.applied_torque[0, drive_ids].tolist()
            except (AttributeError, RuntimeError):
                wheel_tau = [float("nan")] * len(drive_ids)
            v = art.data.root_lin_vel_w[0]
            speed = float((v[0] ** 2 + v[1] ** 2) ** 0.5)
            wheel_names = [joint_names[i].replace("_wheel_throttle", "")
                           for i in drive_ids]
            steer_names = [joint_names[i].replace("_wheel_steer", "")
                           for i in steer_ids]
            print(f"\n[diag] src={ctrl.last_source}  thr={thr:+.2f}  "
                  f"steer_in={steer_in:+.2f}  v_chassis={speed:5.2f} m/s",
                  flush=True)
            print(f"  wheel cmd_vel = {wheel_speed:+7.1f} rad/s  (sent to all 4 wheels)",
                  flush=True)
            print(f"  wheel actual ω (rad/s):  "
                  + "  ".join(f"{n}={w:+7.2f}"
                              for n, w in zip(wheel_names, wheel_w)),
                  flush=True)
            print(f"  wheel applied τ (Nm):    "
                  + "  ".join(f"{n}={t:+6.3f}"
                              for n, t in zip(wheel_names, wheel_tau)),
                  flush=True)
            print(f"  steer cmd = {steer_angle:+.4f} rad   actual:  "
                  + "  ".join(f"{n}={p:+.4f}"
                              for n, p in zip(steer_names, steer_actual)),
                  flush=True)

        sim.step()
        scene.update(args.dt)
        step_count += 1

    ctrl.close()
    return 0


if __name__ == "__main__":
    import traceback
    rc = 0
    try:
        rc = main()
    except Exception:
        traceback.print_exc()
        rc = 1
    os._exit(rc)
