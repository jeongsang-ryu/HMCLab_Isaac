"""Tutorial 09 — one env, fast ego catches slow opp → collision.

Both vehicles run pure-pursuit along the mini-oval centerline. Ego has a
higher target speed, so it catches opp and collides.

Outputs:
  tut09_collision_{robot}_{topdown,chase}.png + .mp4 + .gif

Run:
    OMNI_KIT_ACCEPT_EULA=YES python Itutorial/09_collision.py --robot f1tenth
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
parser.add_argument("--ego_speed", type=float, default=5.0)
parser.add_argument("--opp_speed", type=float, default=1.0)
parser.add_argument("--lookahead", type=float, default=1.0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True
launcher = AppLauncher(args)
simulation_app = launcher.app

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402

from _common import (  # noqa: E402
    DEFAULT_TRACK,
    apply_ackermann,
    apply_high_friction_to_mesh,
    get_robot_spec,
    post_spawn_fix,
    pure_pursuit_steer,
    quat_to_yaw,
    spawn_high_friction_ground,
)
from hmclab_isaac.envs.racing._base import default_track_path  # noqa: E402
from hmclab_isaac.utils.capture import ChaseCamRecorder, TopDownRecorder  # noqa: E402
from hmclab_isaac.worlds.racing import RacingTrack, spawn_circuit  # noqa: E402
from hmclab_isaac.worlds.racing.duct_track import spawn_duct_track  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")


def _yaw_to_quat_wxyz(yaw: float):
    return (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))


def _spawn(spec, prim_path, pos, yaw):
    quat = _yaw_to_quat_wxyz(yaw)
    pos3 = (float(pos[0]), float(pos[1]), float(pos[2]) + spec.init_z)
    cfg = spec.cfg.replace(prim_path=prim_path)
    cfg = cfg.replace(init_state=cfg.init_state.replace(pos=pos3, rot=quat))
    cfg.spawn.func(cfg.prim_path, cfg.spawn, translation=pos3, orientation=quat)
    post_spawn_fix(spec, prim_path)
    return Articulation(cfg)


def _save_gif(frames, path, fps=15, width=480):
    """Downsample frames → GIF to stay under Telegram's ~50 MB file cap."""
    if len(frames) < 2:
        return False
    try:
        import numpy as np
        from PIL import Image
        # Skip frames to reduce total, resize to width px
        stride = max(1, len(frames) // 120)  # target ≤120 frames
        picked = frames[::stride]
        resized = []
        for f in picked:
            im = Image.fromarray(f)
            h, w = f.shape[:2]
            new_h = max(1, int(h * width / max(w, 1)))
            resized.append(np.asarray(im.resize((width, new_h), Image.BILINEAR)))
        import imageio.v2 as imageio
        imageio.mimsave(path, resized, duration=1.0 / max(fps, 1), loop=0)
        return True
    except Exception as exc:
        print(f"[gif] {exc}", flush=True)
        return False


def main() -> int:
    spec = get_robot_spec(args.robot)
    track_name = args.track or DEFAULT_TRACK
    tag_td = f"tut09_collision_{spec.name}_topdown"
    tag_ch = f"tut09_collision_{spec.name}_chase"

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01, device="cuda:0"))
    sim_utils.DomeLightCfg(intensity=2500.0).func("/World/Light", sim_utils.DomeLightCfg(intensity=2500.0))
    spawn_high_friction_ground("/World/ground")

    track = RacingTrack.load(default_track_path(track_name))
    # Road surface mesh that follows the (possibly banked / hilly)
    # centerline — gives the vehicles something to drive on even when the
    # track climbs above the ground plane.
    spawn_circuit(track, prim_path="/World/Road", surface_only=True,
                  color=(0.35, 0.35, 0.35))
    apply_high_friction_to_mesh("/World/Road/mesh")
    spawn_duct_track(
        track, prim_path="/World/DuctTrack",
        pipe_radius=0.15, pipe_offset=1.1, rib_spacing=0.8,
    )

    ego_pos, ego_rpy = track.spawn_pose(progress=0.0)
    opp_pos, opp_rpy = track.spawn_pose(progress=0.04)  # ~1.5 m ahead
    ego = _spawn(spec, "/World/Ego", ego_pos, float(ego_rpy[2]))
    opp = _spawn(spec, "/World/Opp", opp_pos, float(opp_rpy[2]))

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

    ego_steer_ids, ego_drive_ids = spec.resolve_joints(ego.joint_names)
    opp_steer_ids, opp_drive_ids = spec.resolve_joints(opp.joint_names)

    device = ego.device
    centerline = torch.as_tensor(
        track.positions[:, :2], dtype=torch.float32, device=device
    )

    collision_step = -1
    collision_dist = 0.5  # contact threshold for logging
    min_dist = float("inf")
    for step in range(args.steps):
        # Ego pure-pursuit (speed-adaptive lookahead)
        ego_p = ego.data.root_pos_w[:, :2]
        ego_v = ego.data.root_lin_vel_w[:, :2].norm(dim=-1)
        ego_yaw = quat_to_yaw(ego.data.root_quat_w)
        ego_steer = pure_pursuit_steer(
            ego_p, ego_yaw, centerline, args.lookahead, spec.wheelbase,
            speed=ego_v,
        )
        apply_ackermann(
            ego, spec, steer_rad=ego_steer, speed_mps=args.ego_speed,
            steer_ids=ego_steer_ids, drive_ids=ego_drive_ids,
        )
        # Opp pure-pursuit at lower speed
        opp_p = opp.data.root_pos_w[:, :2]
        opp_v = opp.data.root_lin_vel_w[:, :2].norm(dim=-1)
        opp_yaw = quat_to_yaw(opp.data.root_quat_w)
        opp_steer = pure_pursuit_steer(
            opp_p, opp_yaw, centerline, args.lookahead, spec.wheelbase,
            speed=opp_v,
        )
        apply_ackermann(
            opp, spec, steer_rad=opp_steer, speed_mps=args.opp_speed,
            steer_ids=opp_steer_ids, drive_ids=opp_drive_ids,
        )
        ego.write_data_to_sim()
        opp.write_data_to_sim()
        sim.step()
        ego.update(sim.get_physics_dt())
        opp.update(sim.get_physics_dt())

        d = float((ego_p - opp_p).norm().item())
        if d < min_dist:
            min_dist = d
        if collision_step < 0 and d < collision_dist:
            collision_step = step
            print(
                f">>> COLLISION at step={step}  dist={d:.3f}",
                flush=True,
            )

        if step % 2 == 0:
            topdown.capture_frame()
            p = ego.data.root_pos_w[0].cpu().numpy()
            yaw = quat_to_yaw(ego.data.root_quat_w[0].cpu().numpy())
            chase.update_follow((float(p[0]), float(p[1]), float(p[2])), yaw)
            chase.capture_frame()

    ep = ego.data.root_pos_w[0]
    op = opp.data.root_pos_w[0]
    es = ego.data.root_lin_vel_w[0, :2].norm().item()
    os_ = opp.data.root_lin_vel_w[0, :2].norm().item()
    print(
        f">>> ego=({ep[0].item():.2f},{ep[1].item():.2f}) v={es:.2f} | "
        f"opp=({op[0].item():.2f},{op[1].item():.2f}) v={os_:.2f} | "
        f"collision_step={collision_step} min_dist={min_dist:.3f}",
        flush=True,
    )
    topdown.finalize()
    chase.finalize()
    _save_gif(topdown._frames, os.path.join(OUT_DIR, f"{tag_td}.gif"), fps=20)
    _save_gif(chase._frames, os.path.join(OUT_DIR, f"{tag_ch}.gif"), fps=20)
    print(f"TUT09_OK robot={spec.name} collision={collision_step >= 0}", flush=True)
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
