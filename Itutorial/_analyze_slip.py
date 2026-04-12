"""Diagnostic: slip vs torque on the 3D banked oval.

Runs a single vehicle with pure-pursuit around the 3D track and logs:
  - chassis forward speed  (m/s)
  - mean rear-wheel angular speed × wheel_radius (m/s equivalent)
  - slip ratio (wheel_v − chassis_v) / max(wheel_v, 0.1)
  - z-height (to detect uphill vs downhill)

If slip ratio stays ≳ 0.3 on uphills, the wheels are spinning (friction-
limited). If slip ratio ≈ 0 but chassis stalls, torque is insufficient.

Usage:
    OMNI_KIT_ACCEPT_EULA=YES python Itutorial/_analyze_slip.py --robot mushr
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--robot", type=str, default="mushr")
parser.add_argument("--track", type=str, default="mini_oval_3d")
parser.add_argument("--steps", type=int, default=500)
parser.add_argument("--speed", type=float, default=3.0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = False
launcher = AppLauncher(args)
simulation_app = launcher.app

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402

from _common import (  # noqa: E402
    apply_ackermann,
    apply_high_friction_to_mesh,
    get_robot_spec,
    post_spawn_fix,
    pure_pursuit_steer,
    quat_to_yaw,
    spawn_high_friction_ground,
)
from hmclab_isaac.envs.racing._base import default_track_path  # noqa: E402
from hmclab_isaac.worlds.racing import RacingTrack, spawn_circuit  # noqa: E402
from hmclab_isaac.worlds.racing.duct_track import spawn_duct_track  # noqa: E402


def _yaw_to_quat(yaw: float):
    return (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))


def main() -> int:
    spec = get_robot_spec(args.robot)
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=0.01, device="cuda:0")
    )
    spawn_high_friction_ground("/World/ground")
    sim_utils.DomeLightCfg(intensity=2500.0).func(
        "/World/Light", sim_utils.DomeLightCfg(intensity=2500.0)
    )

    track = RacingTrack.load(default_track_path(args.track))
    spawn_circuit(track, prim_path="/World/Road", surface_only=True,
                  color=(0.35, 0.35, 0.35))
    apply_high_friction_to_mesh("/World/Road/mesh")
    spawn_duct_track(
        track, prim_path="/World/Duct",
        pipe_radius=0.15, pipe_offset=1.1, rib_spacing=0.8,
    )

    pos0, rpy0 = track.spawn_pose(progress=0.0)
    yaw0 = float(rpy0[2])
    q = _yaw_to_quat(yaw0)
    init = (float(pos0[0]), float(pos0[1]), float(pos0[2]) + spec.init_z)
    cfg = spec.cfg.replace(prim_path="/World/Ego")
    cfg = cfg.replace(init_state=cfg.init_state.replace(pos=init, rot=q))
    cfg.spawn.func(cfg.prim_path, cfg.spawn, translation=init, orientation=q)
    post_spawn_fix(spec, "/World/Ego")
    ego = Articulation(cfg)
    sim.reset()

    steer_ids, drive_ids = spec.resolve_joints(ego.joint_names)
    centerline = torch.as_tensor(
        track.positions[:, :2], dtype=torch.float32, device=ego.device
    )

    # Gravity / mass constants for analytical check
    mass = 3.0 if spec.name == "mushr" else 2.5
    g = 9.81
    Fg = mass * g                                      # N
    req_torque_per_w_flat = 0.0                        # rolling resistance ignored
    # At pitch θ (positive = climbing), along-slope gravity force = mg sin θ
    # Required drive torque at wheels = (mg sin θ) * wheel_radius
    print(
        f"[ANALYSIS] robot={spec.name}  mass={mass} kg  wheel_r={spec.wheel_radius} m",
        flush=True,
    )

    rows = []
    for i in range(args.steps):
        pos_xy = ego.data.root_pos_w[:, :2]
        yaw_t = quat_to_yaw(ego.data.root_quat_w)
        spd = ego.data.root_lin_vel_w[:, :2].norm(dim=-1)
        steer = pure_pursuit_steer(
            pos_xy, yaw_t, centerline, 0.7, spec.wheelbase, speed=spd
        )
        apply_ackermann(
            ego, spec, steer_rad=steer, speed_mps=args.speed,
            steer_ids=steer_ids, drive_ids=drive_ids,
        )
        ego.write_data_to_sim()
        sim.step()
        ego.update(sim.get_physics_dt())

        if i % 5 == 0:
            chassis_v = float(ego.data.root_lin_vel_w[0, :2].norm().item())
            wheel_w = float(ego.data.joint_vel[0, drive_ids].mean().item())
            wheel_v = wheel_w * spec.wheel_radius
            slip = (wheel_v - chassis_v) / max(abs(wheel_v), 0.1)
            z = float(ego.data.root_pos_w[0, 2].item())
            # Pitch estimated from quaternion
            q = ego.data.root_quat_w[0]
            pitch = math.asin(max(-1.0, min(1.0, 2.0 * (float(q[0]) * float(q[2]) - float(q[3]) * float(q[1])))))
            req_t = Fg * math.sin(pitch) * spec.wheel_radius  # total (both rear wheels)
            rows.append(
                (i, z, math.degrees(pitch), chassis_v, wheel_v, slip, req_t)
            )

    print(
        f"{'step':>5} {'z':>6} {'pitch°':>8} {'chassis':>9} {'wheel':>8} "
        f"{'slip':>7} {'req_τ':>7}",
        flush=True,
    )
    for row in rows:
        i, z, pitch_d, cv, wv, slip, req_t = row
        print(
            f"{i:>5} {z:>6.2f} {pitch_d:>8.2f} {cv:>9.2f} {wv:>8.2f} "
            f"{slip:>7.2f} {req_t:>7.3f}",
            flush=True,
        )

    # Summary
    slips = [r[5] for r in rows]
    req_torques = [r[6] for r in rows]
    max_slip = max(slips) if slips else 0.0
    max_req = max(req_torques) if req_torques else 0.0
    # Available torque at wheels (from cfg effort_limit_sim × num rear wheels)
    if spec.name == "mushr":
        avail_per_w = 0.8
    else:
        avail_per_w = 3.0
    avail_total = avail_per_w * len(drive_ids)

    print("----", flush=True)
    print(f"max_slip_ratio={max_slip:.2f}  max_req_torque={max_req:.3f} Nm  "
          f"available_torque={avail_total:.2f} Nm", flush=True)
    if max_slip > 0.3:
        verdict = "FRICTION-LIMITED (wheels spin faster than chassis)"
    elif max_req > avail_total * 0.8:
        verdict = "TORQUE-LIMITED (wheels barely spin, not enough force)"
    else:
        verdict = "NEITHER — check other causes (e.g. terrain, actuator lag)"
    print(f"VERDICT: {verdict}", flush=True)
    os._exit(0)


if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception:
        traceback.print_exc()
        os._exit(1)
