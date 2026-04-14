"""Evaluate a trained PPO centerline policy and capture top-down + chase MP4s."""

from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=str, required=True)
parser.add_argument("--eval_steps", type=int, default=500)
parser.add_argument("--robot", type=str, default="f1tenth")
parser.add_argument("--track", type=str, default="mini_oval_flat")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--lidar_stack", type=int, default=1)
parser.add_argument("--lidar_downsample", type=int, default=1)
parser.add_argument("--optimal_line", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True
launcher = AppLauncher(args)
simulation_app = launcher.app

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from isaaclab.sensors import Camera  # noqa: E402
from model import ActorCritic  # noqa: E402
from centerline_env import CenterlineEnv, CenterlineEnvCfg, compute_obs_dim  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hmclab_isaac.utils.capture import chase_camera_cfg, top_down_camera_cfg  # noqa: E402

OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs"
)


def main() -> int:
    cfg = CenterlineEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.robot_name = args.robot
    cfg.track_name = args.track
    cfg.lidar_stack = args.lidar_stack
    cfg.lidar_downsample = args.lidar_downsample
    cfg.optimal_line_mode = args.optimal_line
    cfg.observation_space = compute_obs_dim(args.lidar_stack, args.lidar_downsample)

    from hmclab_isaac.envs.racing._base import default_track_path
    from hmclab_isaac.worlds.racing import RacingTrack
    track = RacingTrack.load(default_track_path(cfg.track_name))
    xs, ys = track.positions[:, 0], track.positions[:, 1]
    cx = float((xs.min() + xs.max()) / 2)
    cy = float((ys.min() + ys.max()) / 2)
    span = float(max(xs.max() - xs.min(), ys.max() - ys.min())) * 1.3 + 2.0

    class _EnvWithCam(CenterlineEnv):
        def _setup_scene(self):
            super()._setup_scene()
            self.cam_td = Camera(top_down_camera_cfg(
                "/World/TopDownCam", center=(cx, cy), size=span,
                width=1280, height=720))
            self.cam_ch = Camera(chase_camera_cfg(
                "/World/ChaseCam", width=1280, height=720))

    env = _EnvWithCam(cfg)
    device = env.device

    obs_dim = cfg.observation_space
    hidden = 256 if obs_dim >= 256 else 128
    model = ActorCritic(obs_dim, cfg.action_space, hidden=hidden).to(device)
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"[EVAL] iter={ck.get('iter')} mean_return={ck.get('mean_return'):.2f}",
          flush=True)

    obs = env.reset()[0]["policy"]
    frames_td, frames_ch = [], []

    def grab(cam, frames):
        try: cam.update(dt=0.0, force_recompute=True)
        except Exception: return
        out = getattr(cam.data, "output", None)
        rgb = out.get("rgb") if out else None
        if rgb is None: return
        arr = rgb.detach().cpu().numpy() if hasattr(rgb, "detach") else np.asarray(rgb)
        if arr.ndim == 4: arr = arr[0]
        if arr.shape[-1] == 4: arr = arr[..., :3]
        if arr.dtype != np.uint8: arr = np.clip(arr, 0, 255).astype(np.uint8)
        frames.append(arr)

    def update_chase(cam, ego):
        p = ego.data.root_pos_w[0]; q = ego.data.root_quat_w[0]
        w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
        yaw = math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
        px, py, pz = float(p[0]), float(p[1]), float(p[2])
        eye = (px - 2.5*math.cos(yaw), py - 2.5*math.sin(yaw), pz + 1.4)
        target = (px, py, pz + 0.2)
        eyes_t = torch.tensor([eye], device=cam._device, dtype=torch.float32)
        targets_t = torch.tensor([target], device=cam._device, dtype=torch.float32)
        try: cam.set_world_poses_from_view(eyes_t, targets_t)
        except Exception: pass

    total = 0.0
    for step in range(args.eval_steps):
        with torch.no_grad():
            a, _, _, _ = model.act(obs)
        nxt, rew, _, _, _ = env.step(a)
        total += float(rew.mean().item())
        obs = nxt["policy"]
        if step % 2 == 0:
            update_chase(env.cam_ch, env.ego)
            grab(env.cam_td, frames_td)
            grab(env.cam_ch, frames_ch)
        if step % 50 == 0:
            v = env.ego.data.root_lin_vel_w[0, :2].norm().item()
            p = env.ego.data.root_pos_w[0]
            print(f"  step={step} pos=({p[0].item():.2f},{p[1].item():.2f}) v={v:.2f}",
                  flush=True)

    from PIL import Image
    import imageio.v2 as imageio
    os.makedirs(OUT_DIR, exist_ok=True)
    tag = f"centerline_{args.robot}_{args.track}"
    if frames_td:
        Image.fromarray(frames_td[-1]).save(os.path.join(OUT_DIR, f"{tag}_topdown.png"))
        imageio.mimsave(os.path.join(OUT_DIR, f"{tag}_topdown.mp4"), frames_td,
                        fps=30, codec="libx264", macro_block_size=None)
    if frames_ch:
        Image.fromarray(frames_ch[-1]).save(os.path.join(OUT_DIR, f"{tag}_chase.png"))
        imageio.mimsave(os.path.join(OUT_DIR, f"{tag}_chase.mp4"), frames_ch,
                        fps=30, codec="libx264", macro_block_size=None)
    print(f"EVAL_OK total={total:.2f}", flush=True)
    env.close()
    return 0


if __name__ == "__main__":
    import traceback
    code = 0
    try: code = main()
    except Exception:
        traceback.print_exc(); code = 1
    os._exit(code)
