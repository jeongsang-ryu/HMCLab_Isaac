"""Keyboard + joystick teleop driving the bare SRC_simple chassis (no LiDAR).

Loads ``hmclab_isaac/robots/chassis/SRX/SRC_simple.usd`` as a standalone
articulation and reuses ``UNICORN_2.py``'s actuator tuning so the driving
feel matches the assembled vehicle, minus sensor overhead.

Keyboard:
    ↑ / ↓   throttle forward / reverse
    ← / →   steer left / right
    Space   brake
    L       reset

Joystick (default Xbox-style):
    Left stick Y    → throttle (up = forward)
    Right stick X   → steering
    Button 0 (A)    → brake
    Use ``scripts/joystick_probe.py`` to find your pad's axis/button indices,
    then override with ``--js-axis-steer N`` etc.

Run:
    OMNI_KIT_ACCEPT_EULA=YES python scripts/keyboard_teleop_chassis.py
    OMNI_KIT_ACCEPT_EULA=YES python scripts/keyboard_teleop_chassis.py \\
      --track mini_oval_banked
"""
from __future__ import annotations

import argparse
import math
import os

from isaaclab.app import AppLauncher

from hmclab_isaac.utils.teleop_input import add_input_args, make_controller

# CHASSIS_USD = (
#     "/home/js/hmcl_issac_project/HMCLab_Isaac/"
#     "hmclab_isaac/robots/chassis/SRX/SRC_dw_bw.usd"
# )
CHASSIS_USD = (
    "/home/js/hmcl_issac_project/HMCLab_Isaac/"
    "hmclab_isaac/robots/vehicles/UNICORN/UNICORN_3.usd"
)
TRACKS_DIR = (
    "/home/js/hmcl_issac_project/HMCLab_Isaac/"
    "hmclab_isaac/worlds/racing/_tracks_data"
)
TRACK_CHOICES = sorted(
    f.removesuffix(".txt") for f in os.listdir(TRACKS_DIR) if f.endswith(".txt")
)

parser = argparse.ArgumentParser()
parser.add_argument("--max-wheel-rps", type=float, default=600.0,
                    help="commanded max wheel rotational velocity (rad/s). "
                         "Default 200 × 0.0525 m wheel = 10.5 m/s cap. "
                         "Drag-limited natural top ≈ 10 m/s.")
parser.add_argument("--dt", type=float, default=0.005)
parser.add_argument("--track", choices=["", *TRACK_CHOICES], default="",
                    help="If set, spawn a 3D racing track instead of flat ground.")
parser.add_argument("--track-wall-height", type=float, default=0.5)
parser.add_argument("--track-surface-only", action="store_true")
parser.add_argument("--track-z-offset", type=float, default=0.20)
parser.add_argument("--track-subdivide", type=int, default=10)
parser.add_argument("--ground-static-friction", type=float, default=1.5,
                    help="Static μ for the ground material. Step 5: 1.5 "
                         "gives margin 86%% with motor force 53 N vs grip "
                         "62 N. Realistic value, no sim-only crazy μ.")
parser.add_argument("--ground-dynamic-friction", type=float, default=1.3,
                    help="Dynamic μ for the ground material. Step 5.")
parser.add_argument("--friction-combine-mode", default="multiply",
                    choices=["average", "min", "max", "multiply"],
                    help="MuSHR drift env uses 'multiply' (effective μ = "
                         "tire μ × ground μ).")
add_input_args(parser)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = False
args.enable_cameras = False
launcher = AppLauncher(args)
sim_app = launcher.app

import re  # noqa: E402

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import ArticulationCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402

# Reuse UNICORN_1 tuning constants but rebuild ACTUATORS locally — the
# current SRC_dw_bw.usd dropped the `_joint` suffix on joint names and
# renamed the suspension joints, so UNICORN_1's regexes no longer match.
# Mapping: `*_wheel_joint`→`*_wheel`, `fl_steering_joint`→`fl_steering`,
# `*_sus1_joint`→`*_shock` (prismatic shock travel).
from hmclab_isaac.robots.vehicles.UNICORN.UNICORN_1 import (  # noqa: E402
    ENABLE_SELF_COLLISIONS,
    INIT_HEIGHT,
    MAX_ANGULAR_VELOCITY,
    MAX_DEPENETRATION_VELOCITY,
    MAX_LINEAR_VELOCITY,
    MAX_STEER,
    PASSIVE_PIVOT_DAMPING,
    SLEEP_THRESHOLD,
    SOLVER_POSITION_ITERATIONS,
    SOLVER_VELOCITY_ITERATIONS,
    STABILIZATION_THRESHOLD,
    STEERING_DAMPING,
    STEERING_EFFORT_LIMIT,
    STEERING_STIFFNESS,
    WHEEL_DRIVE_DAMPING,
    WHEEL_DRIVE_EFFORT_LIMIT,
)


# Articulation joint names exposed by SRC_dw_bw.usd:
#   wheels (4):           fl_wheel, fr_wheel, rl_wheel, rr_wheel
#   steering (2, front):  fl_steering, fr_steering
#   rocker pivots (4):    *_shock_upper                 (free, revolute Z)
#   passive pivots (16):  RevoluteJoint, RevoluteJoint_0..14
#
# Note: the prismatic *_shock joints are NOT in the articulation — the
# 4-bar suspension closes a loop and PhysX demotes them to max-coord
# joints. Their spring/damper is authored directly in the USD via
# PhysicsDriveAPI:linear, so we don't (and can't) target them here.
ACTUATORS = {
    "wheel_drive": ImplicitActuatorCfg(
        joint_names_expr=[".*_wheel$"],
        effort_limit_sim=WHEEL_DRIVE_EFFORT_LIMIT,
        stiffness=0.0,
        damping=WHEEL_DRIVE_DAMPING,
    ),
    "steering": ImplicitActuatorCfg(
        joint_names_expr=["fl_steering", "fr_steering"],
        effort_limit_sim=STEERING_EFFORT_LIMIT,
        stiffness=STEERING_STIFFNESS,
        damping=STEERING_DAMPING,
    ),
    "passive_pivots": ImplicitActuatorCfg(
        joint_names_expr=[".*_shock_upper$", "RevoluteJoint.*"],
        effort_limit_sim=0.0,
        stiffness=0.0,
        damping=PASSIVE_PIVOT_DAMPING,
    ),
}


def _make_chassis_cfg(spawn_pos, spawn_yaw):
    cy, sy = math.cos(spawn_yaw / 2.0), math.sin(spawn_yaw / 2.0)
    return ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=CHASSIS_USD,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                max_linear_velocity=MAX_LINEAR_VELOCITY,
                max_angular_velocity=MAX_ANGULAR_VELOCITY,
                max_depenetration_velocity=MAX_DEPENETRATION_VELOCITY,
                enable_gyroscopic_forces=True,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=ENABLE_SELF_COLLISIONS,
                solver_position_iteration_count=SOLVER_POSITION_ITERATIONS,
                solver_velocity_iteration_count=SOLVER_VELOCITY_ITERATIONS,
                sleep_threshold=SLEEP_THRESHOLD,
                stabilization_threshold=STABILIZATION_THRESHOLD,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=spawn_pos,
            rot=(cy, 0.0, 0.0, sy),
        ),
        actuators=ACTUATORS,
    )


def main():
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=args.dt, device="cuda:0")
    )
    sim_utils.DomeLightCfg(intensity=2000.0, color=(0.95, 0.95, 0.95)).func(
        "/World/Light",
        sim_utils.DomeLightCfg(intensity=2000.0, color=(0.95, 0.95, 0.95)),
    )

    spawn_pos = (0.0, 0.0, INIT_HEIGHT)
    spawn_yaw = 0.0

    if args.track:
        import numpy as np
        from hmclab_isaac.worlds.racing import RacingTrack, spawn_circuit
        track = RacingTrack.load(os.path.join(TRACKS_DIR, f"{args.track}.txt"))

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
        print(f"[teleop] spawned track {args.track!r}: "
              f"start=({spawn_pos[0]:.2f}, {spawn_pos[1]:.2f}, {spawn_pos[2]:.2f}), "
              f"yaw={spawn_yaw:.3f} rad", flush=True)
    else:
        sim.set_camera_view(eye=(2.5, 2.5, 1.5), target=(0.0, 0.0, 0.2))
        # Ground material — exact match to WheeledLab's DriftTerrainImporterCfg
        # (mushr_drift_env_cfg.py). With tire μ=1.0 and combine='multiply':
        # effective μ_s = 1.0 × 1.1 = 1.1, effective μ_d = 1.0 × 1.0 = 1.0.
        gp_mat = sim_utils.RigidBodyMaterialCfg(
            static_friction=args.ground_static_friction,
            dynamic_friction=args.ground_dynamic_friction,
            restitution=0.0,
            friction_combine_mode=args.friction_combine_mode,
            restitution_combine_mode=args.friction_combine_mode,
        )
        gp = sim_utils.GroundPlaneCfg(physics_material=gp_mat)
        gp.func("/World/ground", gp)
        for i, (x, y) in enumerate([(3.0, 0.0), (0.0, 3.0), (-3.0, -1.5)]):
            cube_cfg = sim_utils.MeshCuboidCfg(
                size=(0.6, 0.6, 1.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.7, 0.2, 0.2)),
            )
            cube_cfg.func(f"/World/Obstacle_{i}", cube_cfg, translation=(x, y, 0.5))
        print(f"[teleop] ground material: μ_s={args.ground_static_friction:.2f}, "
              f"μ_d={args.ground_dynamic_friction:.2f}, "
              f"combine={args.friction_combine_mode!r} "
              f"(MuSHR-drift-env exact)", flush=True)

    chassis_cfg = _make_chassis_cfg(spawn_pos, spawn_yaw).replace(
        prim_path="/World/envs/env_.*/Robot",
    )

    class _SceneCfg(InteractiveSceneCfg):
        robot = chassis_cfg

    scene = InteractiveScene(_SceneCfg(num_envs=1, env_spacing=2.5,
                                       replicate_physics=True))
    sim.reset()

    art = scene["robot"]
    joint_names = art.joint_names
    # SRC_dw_bw.usd uses bare names like `fl_wheel` / `fl_steering` (no
    # `_joint` suffix). Anchor the regex on `^` for steering so we don't
    # also pick up `_steering_*` siblings like a future link, and on `$`
    # for wheel so `_shock_upper`-style names can't sneak in.
    steer_re = re.compile(r"^(fl|fr|rl|rr)_steering$")
    drive_re = re.compile(r"^(fl|fr|rl|rr)_wheel$")
    steer_ids = [i for i, n in enumerate(joint_names) if steer_re.search(n)]
    drive_ids = [i for i, n in enumerate(joint_names) if drive_re.search(n)]

    print(f"\n[teleop] chassis = SRC_dw_bw.usd", flush=True)
    print(f"[teleop] joints  = {len(joint_names)}", flush=True)
    print(f"[teleop] steer   = {[joint_names[i] for i in steer_ids]}",
          flush=True)
    print(f"[teleop] drive   = {[joint_names[i] for i in drive_ids]}  (4WD)",
          flush=True)
    print("\n[teleop] Click on the GUI window first to capture keyboard.",
          flush=True)
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
        steer_angle = steer_in * MAX_STEER
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
            wheel_names = [joint_names[i].replace("_wheel", "") for i in drive_ids]
            steer_names = [joint_names[i].replace("_steering", "") for i in steer_ids]
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
