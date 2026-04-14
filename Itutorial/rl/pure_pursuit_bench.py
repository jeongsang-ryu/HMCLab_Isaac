"""Pure-pursuit baseline — sweep target speeds, measure max stable speed.

Runs a single f1tenth with pure-pursuit steering at a series of target
speeds and reports which speeds keep the car on-track for a full loop.

Usage:
    OMNI_KIT_ACCEPT_EULA=YES python Itutorial/rl/pure_pursuit_bench.py \\
        --robot f1tenth --track mini_oval_flat --capture
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
parser.add_argument("--robot", type=str, default="f1tenth")
parser.add_argument("--track", type=str, default="mini_oval_flat")
parser.add_argument("--speeds", type=str, default="1,2,3,4,5,6,7,8",
                    help="Comma-separated target speeds to try (m/s)")
parser.add_argument("--steps_per_speed", type=int, default=1200,
                    help="Max steps to allow for each speed attempt")
parser.add_argument("--capture", action="store_true", help="Save top-down MP4")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = bool(args.capture)
launcher = AppLauncher(args)
simulation_app = launcher.app

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from centerline_env import CenterlineEnv, CenterlineEnvCfg  # noqa: E402
from _common import pure_pursuit_steer  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")


def run_one_speed(env: CenterlineEnv, target_speed: float, max_steps: int,
                  wheelbase: float, max_steer: float,
                  capture_frames=None, cam=None) -> dict:
    """Run pure-pursuit at a constant target speed. Returns stats."""
    env.reset()
    max_speed_cfg = env.cfg.max_speed
    throttle_norm = 2.0 * (target_speed / max_speed_cfg) - 1.0        # inverse of [-1,1]→[0,max]

    device = env.device
    centerline = env._centerline                                      # (N, 2)

    stats = {"target": target_speed, "steps_ok": 0, "off_track": False,
             "avg_speed": 0.0, "max_speed": 0.0, "min_cte": 1e9, "max_cte": 0.0}
    speeds = []
    ctes = []
    for step in range(max_steps):
        pos = env.ego.data.root_pos_w[:, :2]
        q = env.ego.data.root_quat_w
        w_, x_, y_, z_ = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        yaw = torch.atan2(2 * (w_ * z_ + x_ * y_), 1 - 2 * (y_ * y_ + z_ * z_))
        v = env.ego.data.root_lin_vel_w[:, :2]
        speed = v.norm(dim=-1)
        # Speed-adaptive lookahead
        steer_rad = pure_pursuit_steer(
            pos, yaw, centerline, lookahead=0.6,
            wheelbase=wheelbase, speed=speed,
            lookahead_gain=0.25, lookahead_min=0.4, lookahead_max=2.5,
        )
        steer_norm = (steer_rad / max_steer).clamp(-1.0, 1.0)
        action = torch.stack([steer_norm,
                              torch.full_like(steer_norm, throttle_norm)], dim=-1)
        env.step(action)

        # stats
        cte = (pos.unsqueeze(1) - centerline.unsqueeze(0)).norm(dim=-1).min(dim=-1).values
        cte_v = float(cte[0].item())
        spd_v = float(speed[0].item())
        speeds.append(spd_v); ctes.append(cte_v)
        stats["max_cte"] = max(stats["max_cte"], cte_v)
        stats["min_cte"] = min(stats["min_cte"], cte_v)

        # capture
        if capture_frames is not None and cam is not None and step % 2 == 0:
            try:
                cam.update(dt=0.0, force_recompute=True)
                out = getattr(cam.data, "output", None)
                rgb = out.get("rgb") if out else None
                if rgb is not None:
                    arr = rgb.detach().cpu().numpy() if hasattr(rgb, "detach") else np.asarray(rgb)
                    if arr.ndim == 4: arr = arr[0]
                    if arr.shape[-1] == 4: arr = arr[..., :3]
                    if arr.dtype != np.uint8: arr = np.clip(arr, 0, 255).astype(np.uint8)
                    capture_frames.append(arr)
            except Exception: pass

        if cte_v > env.cfg.terminate_cte:
            stats["off_track"] = True
            break
        stats["steps_ok"] = step + 1

    stats["avg_speed"] = float(np.mean(speeds)) if speeds else 0.0
    stats["max_speed"] = float(np.max(speeds)) if speeds else 0.0
    return stats


def main() -> int:
    from _common import get_robot_spec

    cfg = CenterlineEnvCfg()
    cfg.scene.num_envs = 1
    cfg.robot_name = args.robot
    cfg.track_name = args.track
    cfg.lidar_stack = 1
    cfg.lidar_downsample = 1
    cfg.max_speed = 10.0    # set high so target_speed of 8 m/s maps cleanly
    # we won't use lidar obs here, but env requires it still builds fine

    # Attach cameras if capturing
    if args.capture:
        from isaaclab.sensors import Camera
        from hmclab_isaac.utils.capture import top_down_camera_cfg
        from hmclab_isaac.envs.racing._base import default_track_path
        from hmclab_isaac.worlds.racing import RacingTrack
        tr = RacingTrack.load(default_track_path(cfg.track_name))
        xs, ys = tr.positions[:, 0], tr.positions[:, 1]
        cx = float((xs.min() + xs.max()) / 2); cy = float((ys.min() + ys.max()) / 2)
        span = float(max(xs.max() - xs.min(), ys.max() - ys.min())) * 1.3 + 2.0

        from hmclab_isaac.utils.capture import chase_camera_cfg
        class _EnvWithCam(CenterlineEnv):
            def _setup_scene(self):
                super()._setup_scene()
                self.cam_td = Camera(top_down_camera_cfg(
                    "/World/TopDownCam", center=(cx, cy), size=span,
                    width=1280, height=720))
                self.cam_ch = Camera(chase_camera_cfg(
                    "/World/ChaseCam", width=1280, height=720))
        env = _EnvWithCam(cfg)
    else:
        env = CenterlineEnv(cfg)

    spec = get_robot_spec(args.robot)
    wheelbase = spec.wheelbase
    max_steer = spec.max_steer
    print(f"[PP-bench] robot={args.robot} wheelbase={wheelbase:.3f} "
          f"max_steer={math.degrees(max_steer):.1f}° track={args.track}",
          flush=True)

    speeds = [float(s) for s in args.speeds.split(",")]
    all_stats = []
    frames_by_speed = {}
    chase_frames_by_speed = {}
    for tgt in speeds:
        capture = [] if args.capture else None
        cam = env.cam_td if args.capture and tgt == max(speeds) else None  # capture only fastest
        stats = run_one_speed(env, tgt, args.steps_per_speed,
                              wheelbase, max_steer,
                              capture_frames=capture, cam=cam)
        all_stats.append(stats)
        if capture is not None and len(capture) > 0:
            frames_by_speed[tgt] = capture
        print(f"  tgt={tgt:5.2f}  avg={stats['avg_speed']:4.2f}  max={stats['max_speed']:4.2f}  "
              f"steps_ok={stats['steps_ok']}  off_track={stats['off_track']}  "
              f"max_cte={stats['max_cte']:4.2f}", flush=True)

    # Find last stable target
    stable = [s for s in all_stats if not s["off_track"] and s["steps_ok"] >= args.steps_per_speed * 0.95]
    if stable:
        best = max(stable, key=lambda s: s["target"])
        print(f"PP_OK  max_stable_target={best['target']}  avg={best['avg_speed']:.2f}", flush=True)
    else:
        print("PP_OK  no fully-stable speed", flush=True)

    # Save capture at fastest stable speed
    if frames_by_speed:
        from PIL import Image
        import imageio.v2 as imageio
        os.makedirs(OUT_DIR, exist_ok=True)
        for tgt, frames in frames_by_speed.items():
            tag = f"pp_{args.robot}_{args.track}_v{tgt:.1f}"
            imageio.mimsave(os.path.join(OUT_DIR, f"{tag}.mp4"), frames,
                            fps=30, codec="libx264", macro_block_size=None)
            print(f"saved {tag}.mp4", flush=True)

    env.close()
    return 0


if __name__ == "__main__":
    import traceback
    code = 0
    try: code = main()
    except Exception:
        traceback.print_exc(); code = 1
    os._exit(code)
