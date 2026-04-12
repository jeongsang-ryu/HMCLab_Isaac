"""Tutorial 05 — N (default 3) vehicles on the mini banked oval (duct).

Supports --robot f1tenth|mushr and --num_vehicles. Each vehicle is spawned
at a different centerline progress and runs with its own (speed, steer)
command. Top-down + chase-cam (tracking vehicle 0).

Run:
    OMNI_KIT_ACCEPT_EULA=YES python Itutorial/05_multi_vehicles.py --robot mushr --num_vehicles 4
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=400)
parser.add_argument("--robot", type=str, default="f1tenth")
parser.add_argument("--track", type=str, default=None)
parser.add_argument("--num_vehicles", type=int, default=3)
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


def _spawn(spec, prim_path, pos, yaw):
    quat = _yaw_to_quat_wxyz(yaw)
    pos3 = (float(pos[0]), float(pos[1]), float(pos[2]) + spec.init_z)
    cfg = spec.cfg.replace(prim_path=prim_path)
    cfg = cfg.replace(init_state=cfg.init_state.replace(pos=pos3, rot=quat))
    cfg.spawn.func(cfg.prim_path, cfg.spawn, translation=pos3, orientation=quat)
    return Articulation(cfg)


def main() -> int:
    spec = get_robot_spec(args.robot)
    track_name = args.track or DEFAULT_TRACK
    N = max(1, int(args.num_vehicles))
    tag_td = f"tut05_multi_{spec.name}_topdown"
    tag_ch = f"tut05_multi_{spec.name}_chase"

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01, device="cuda:0"))
    sim_utils.DomeLightCfg(intensity=2500.0).func("/World/Light", sim_utils.DomeLightCfg(intensity=2500.0))
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())

    track = RacingTrack.load(default_track_path(track_name))
    spawn_duct_track(
        track, prim_path="/World/DuctTrack",
        pipe_radius=0.15, pipe_offset=1.1, rib_spacing=0.8,
    )

    # Spawn N vehicles spaced along the first 40% of the centerline
    vehicles: list[Articulation] = []
    for i in range(N):
        prog = (i * 0.04) % 1.0
        pos, rpy = track.spawn_pose(progress=prog)
        art = _spawn(spec, f"/World/Vehicle_{i}", pos, float(rpy[2]))
        vehicles.append(art)

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

    # Pre-resolve joint ids
    joint_ids = [spec.resolve_joints(v.joint_names) for v in vehicles]

    for step in range(args.steps):
        for i, veh in enumerate(vehicles):
            speed = 2.5 + 0.25 * i
            steer = 0.10 + 0.02 * (i - (N - 1) / 2.0)
            apply_ackermann(
                veh, spec, steer_rad=steer, speed_mps=speed,
                steer_ids=joint_ids[i][0], drive_ids=joint_ids[i][1],
            )
            veh.write_data_to_sim()
        sim.step()
        for veh in vehicles:
            veh.update(sim.get_physics_dt())
        if step % 2 == 0:
            topdown.capture_frame()
            p = vehicles[0].data.root_pos_w[0].cpu().numpy()
            yaw = _quat_to_yaw(vehicles[0].data.root_quat_w[0].cpu().numpy())
            chase.update_follow((float(p[0]), float(p[1]), float(p[2])), yaw)
            chase.capture_frame()

    for i, veh in enumerate(vehicles):
        p = veh.data.root_pos_w[0]
        s = veh.data.root_lin_vel_w[0, :2].norm().item()
        print(
            f">>> v{i} pos=({p[0].item():.2f},{p[1].item():.2f}) speed={s:.2f}",
            flush=True,
        )
    print(f"TUT05_OK robot={spec.name} N={N} track={track_name}", flush=True)
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
