"""Tutorial 03 — Ackermann drive on the mini banked oval (duct map).

Supports --robot f1tenth|mushr. Constant speed + small steer offset →
vehicle tracks a curve along the centerline. Captures top-down + chase.

Run:
    OMNI_KIT_ACCEPT_EULA=YES python Itutorial/03_ackermann_on_track.py --robot mushr
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
parser.add_argument("--steps", type=int, default=500)
parser.add_argument("--robot", type=str, default="f1tenth")
parser.add_argument("--track", type=str, default=None)
parser.add_argument("--speed", type=float, default=3.0)
parser.add_argument("--steer", type=float, default=0.15)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True
launcher = AppLauncher(args)
simulation_app = launcher.app

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402

from _common import DEFAULT_TRACK, apply_ackermann, get_robot_spec  # noqa: E402
from hmclab_isaac.envs.racing._base import default_track_path  # noqa: E402
from hmclab_isaac.utils.capture import ChaseCamRecorder, TopDownRecorder  # noqa: E402
from hmclab_isaac.worlds.racing import RacingTrack  # noqa: E402
from hmclab_isaac.worlds.racing.duct_track import spawn_duct_track  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")


def _yaw_to_quat_wxyz(yaw: float):
    return (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))


def _quat_to_yaw(q):
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def main() -> int:
    spec = get_robot_spec(args.robot)
    track_name = args.track or DEFAULT_TRACK
    tag_td = f"tut03_ackermann_{spec.name}_topdown"
    tag_ch = f"tut03_ackermann_{spec.name}_chase"

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01, device="cuda:0"))
    sim_utils.DomeLightCfg(intensity=2500.0).func("/World/Light", sim_utils.DomeLightCfg(intensity=2500.0))
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())

    track = RacingTrack.load(default_track_path(track_name))
    spawn_duct_track(
        track, prim_path="/World/DuctTrack",
        pipe_radius=0.15, pipe_offset=1.1, rib_spacing=0.8,
    )

    start_xyz, start_rpy = track.spawn_pose(progress=0.0)
    start_yaw = float(start_rpy[2])
    quat = _yaw_to_quat_wxyz(start_yaw)
    init_pos = (float(start_xyz[0]), float(start_xyz[1]), float(start_xyz[2]) + spec.init_z)

    ego_cfg = spec.cfg.replace(prim_path="/World/Ego")
    ego_cfg = ego_cfg.replace(init_state=ego_cfg.init_state.replace(pos=init_pos, rot=quat))
    ego_cfg.spawn.func(ego_cfg.prim_path, ego_cfg.spawn, translation=init_pos, orientation=quat)
    ego = Articulation(ego_cfg)

    xs, ys = track.positions[:, 0], track.positions[:, 1]
    cx = float((xs.min() + xs.max()) / 2)
    cy = float((ys.min() + ys.max()) / 2)
    span = float(max(xs.max() - xs.min(), ys.max() - ys.min())) * 1.2 + 2.0
    topdown = TopDownRecorder(
        center=(cx, cy), size=span, prim_path="/World/TopDownCam",
        out_dir=OUT_DIR, tag=tag_td,
    )
    chase = ChaseCamRecorder(
        out_dir=OUT_DIR, tag=tag_ch, prim_path="/World/ChaseCam",
        behind=2.5, above=1.4, look_up=0.2,
    )

    sim.reset()
    topdown.on_reset()
    chase.on_reset()

    steer_ids, drive_ids = spec.resolve_joints(ego.joint_names)
    print(
        f">>> {spec.name}: steer_ids={steer_ids} drive_ids={drive_ids} "
        f"joints={ego.joint_names}",
        flush=True,
    )

    speed_log: list[float] = []
    for i in range(args.steps):
        apply_ackermann(
            ego, spec, steer_rad=args.steer, speed_mps=args.speed,
            steer_ids=steer_ids, drive_ids=drive_ids,
        )
        ego.write_data_to_sim()
        sim.step()
        ego.update(sim.get_physics_dt())
        speed_log.append(float(ego.data.root_lin_vel_w[0, :2].norm().item()))
        if i % 2 == 0:
            topdown.capture_frame()
            pos = ego.data.root_pos_w[0].cpu().numpy()
            yaw = _quat_to_yaw(ego.data.root_quat_w[0].cpu().numpy())
            chase.update_follow((float(pos[0]), float(pos[1]), float(pos[2])), yaw)
            chase.capture_frame()

    p = ego.data.root_pos_w[0]
    mean_speed = float(np.mean(speed_log[-50:])) if speed_log else 0.0
    print(
        f">>> final pos=({p[0].item():.2f}, {p[1].item():.2f}, {p[2].item():.3f}) "
        f"tail_speed={mean_speed:.2f}",
        flush=True,
    )
    print(f"TUT03_OK robot={spec.name} track={track_name}", flush=True)
    topdown.finalize()
    chase.finalize()
    return 0


if __name__ == "__main__":
    import traceback
    code = 0
    try:
        code = main()
    except Exception:
        traceback.print_exc()
        code = 1
    os._exit(code)
