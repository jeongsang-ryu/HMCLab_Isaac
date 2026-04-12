"""Evaluate a trained PPO checkpoint on the S-curve env + capture video.

Usage:
    OMNI_KIT_ACCEPT_EULA=YES python Itutorial/rl/eval.py \
        --ckpt Itutorial/rl/ckpt_scurve.pt --steps 500
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
parser.add_argument("--ckpt", type=str, required=True)
parser.add_argument("--eval_steps", type=int, default=500)
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True
launcher = AppLauncher(args)
simulation_app = launcher.app

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sensors import Camera  # noqa: E402

from model import ActorCritic  # noqa: E402
from s_curve_env import SCurveEnv, SCurveEnvCfg  # noqa: E402

# Tutorial imports for camera helpers
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hmclab_isaac.utils.capture import (  # noqa: E402
    chase_camera_cfg,
    top_down_camera_cfg,
)

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")


def main() -> int:
    cfg = SCurveEnvCfg()
    cfg.scene.num_envs = args.num_envs

    # We need the top-down + chase cameras inside _setup_scene. Monkey-patch
    # the env class to register them before clone_environments.
    from hmclab_isaac.envs.racing._base import default_track_path
    from hmclab_isaac.worlds.racing import RacingTrack
    track = RacingTrack.load(default_track_path(cfg.track_name))
    xs = track.positions[:, 0]
    ys = track.positions[:, 1]
    cx = float((xs.min() + xs.max()) / 2)
    cy = float((ys.min() + ys.max()) / 2)
    span = float(max(xs.max() - xs.min(), ys.max() - ys.min())) * 1.3 + 2.0

    class _SCurveEnvWithCameras(SCurveEnv):
        def _setup_scene(self):
            super()._setup_scene()
            # Global top-down + chase cameras (outside env regex paths so
            # they stay single-instance across clones).
            self.cam_td = Camera(
                top_down_camera_cfg(
                    "/World/TopDownCam",
                    center=(cx, cy), size=span,
                    width=1280, height=720,
                )
            )
            self.cam_ch = Camera(
                chase_camera_cfg("/World/ChaseCam", width=1280, height=720)
            )

    env = _SCurveEnvWithCameras(cfg)
    device = env.device

    obs_dim = cfg.observation_space
    act_dim = cfg.action_space
    model = ActorCritic(obs_dim, act_dim).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(
        f"[EVAL] loaded ckpt from iter={ckpt.get('iter')} "
        f"mean_return={ckpt.get('mean_return'):.2f}",
        flush=True,
    )

    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]

    frames_td, frames_ch = [], []

    def grab(cam, frames):
        try:
            cam.update(dt=0.0, force_recompute=True)
        except Exception:
            return
        out = getattr(cam.data, "output", None)
        rgb = out.get("rgb") if out else None
        if rgb is None:
            return
        arr = rgb.detach().cpu().numpy() if hasattr(rgb, "detach") else np.asarray(rgb)
        if arr.ndim == 4:
            arr = arr[0]
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        frames.append(arr)

    def update_chase(cam, ego):
        p = ego.data.root_pos_w[0]
        q = ego.data.root_quat_w[0]
        w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        px, py, pz = float(p[0]), float(p[1]), float(p[2])
        eye = (px - 2.5 * math.cos(yaw), py - 2.5 * math.sin(yaw), pz + 1.4)
        target = (px, py, pz + 0.2)
        eyes_t = torch.tensor([eye], device=cam._device, dtype=torch.float32)
        targets_t = torch.tensor([target], device=cam._device, dtype=torch.float32)
        try:
            cam.set_world_poses_from_view(eyes_t, targets_t)
        except Exception:
            pass

    total_reward = 0.0
    for step in range(args.eval_steps):
        with torch.no_grad():
            action, _, _, _ = model.act(obs)
        next_obs, rew, dones, truncs, _ = env.step(action)
        total_reward += float(rew.mean().item())
        obs = next_obs["policy"]

        if step % 2 == 0:
            update_chase(env.cam_ch, env.ego)
            grab(env.cam_td, frames_td)
            grab(env.cam_ch, frames_ch)
        if step % 50 == 0:
            speed = env.ego.data.root_lin_vel_w[0, :2].norm().item()
            p = env.ego.data.root_pos_w[0]
            print(
                f"  step={step} pos=({p[0].item():.2f},{p[1].item():.2f},{p[2].item():.2f}) "
                f"v={speed:.2f}",
                flush=True,
            )

    from PIL import Image
    os.makedirs(OUT_DIR, exist_ok=True)
    import imageio.v2 as imageio
    td_mp4 = os.path.join(OUT_DIR, "tut12_rl_scurve_topdown.mp4")
    ch_mp4 = os.path.join(OUT_DIR, "tut12_rl_scurve_chase.mp4")
    td_png = os.path.join(OUT_DIR, "tut12_rl_scurve_topdown.png")
    ch_png = os.path.join(OUT_DIR, "tut12_rl_scurve_chase.png")

    if frames_td:
        Image.fromarray(frames_td[-1]).save(td_png)
        imageio.mimsave(td_mp4, frames_td, fps=30, codec="libx264",
                        macro_block_size=None)
        print(f"saved {td_mp4}", flush=True)
    if frames_ch:
        Image.fromarray(frames_ch[-1]).save(ch_png)
        imageio.mimsave(ch_mp4, frames_ch, fps=30, codec="libx264",
                        macro_block_size=None)
        print(f"saved {ch_mp4}", flush=True)

    print(f"TUT12_EVAL_OK total_reward={total_reward:.2f}", flush=True)
    env.close()
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
