"""Dump exactly what the policy sees — TiledCamera obs → PNG.

This script spins up a single-env UnicornRacingEnv, takes ``--steps`` random
actions, and saves each frame's actor-image obs to disk as PNG. It also
prints the proprio + privileged vectors so you can sanity-check ranges.

Useful for diagnosing "policy isn't learning" cases where the actual input
turns out to be wrong (white screen, dark frame, mis-oriented camera, …).

Usage::

    OMNI_KIT_ACCEPT_EULA=YES python scripts/inspect_camera_obs.py \\
        --track my_track --steps 20 --out /tmp/cam_check
"""
from __future__ import annotations

import argparse
import os

import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--track", type=str, default="my_track")
parser.add_argument("--steps", type=int, default=20)
parser.add_argument("--action", choices=["random", "zero", "forward", "left", "right"],
                    default="random",
                    help="Action policy. 'forward' = throttle +0.5 steer 0; "
                         "'left' = throttle +0.5 steer -1; …")
parser.add_argument("--out", type=str, default="/tmp/cam_check")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True

launcher = AppLauncher(args)
sim_app = launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402

import hmclab_isaac.envs  # noqa: F401, E402
from hmclab_isaac.envs.racing.rl.single_agent.cfg import UnicornRacingEnvCfg  # noqa: E402


def _action_for_step(name: str, num_envs: int, device) -> torch.Tensor:
    a = torch.zeros((num_envs, 2), device=device)
    if name == "random":
        a = torch.empty_like(a).uniform_(-1.0, 1.0)
    elif name == "zero":
        pass
    elif name == "forward":
        a[:, 0] = 0.5
    elif name == "left":
        a[:, 0] = 0.5
        a[:, 1] = -1.0
    elif name == "right":
        a[:, 0] = 0.5
        a[:, 1] = +1.0
    return a


def main() -> int:
    os.makedirs(args.out, exist_ok=True)
    env_cfg = UnicornRacingEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.track_name = args.track

    env = gym.make("HMCLab-Racing-Single-Visual-v0", cfg=env_cfg)
    env_raw = env.unwrapped
    device = env_raw.device

    obs, _ = env_raw.reset()
    img_t = obs["images"]
    prop = obs["policy"]
    priv = obs["privileged"]
    print(f"[inspect] obs shapes:", flush=True)
    print(f"    images:     {tuple(img_t.shape)}  dtype={img_t.dtype}  "
          f"min={float(img_t.min()):.3f}  max={float(img_t.max()):.3f}  "
          f"mean={float(img_t.mean()):.3f}", flush=True)
    print(f"    proprio:    {tuple(prop.shape)}  range=[{float(prop.min()):+.2f}, "
          f"{float(prop.max()):+.2f}]  values={prop[0].tolist()}", flush=True)
    print(f"    privileged: {tuple(priv.shape)}  range=[{float(priv.min()):+.2f}, "
          f"{float(priv.max()):+.2f}]  values={priv[0].tolist()}", flush=True)

    try:
        from PIL import Image
    except ImportError:
        Image = None
        print("[inspect] PIL not available — falling back to numpy save (.npy)",
              flush=True)

    saved = 0
    for step in range(args.steps):
        action = _action_for_step(args.action, env_raw.num_envs, device)
        obs, reward, terminated, truncated, _ = env_raw.step(action)

        img = obs["images"][0]    # (C, H, W) channel-first, normalized [0, 1]
        # channel-first → channel-last for PIL
        img_hwc = img.permute(1, 2, 0).clamp(0.0, 1.0).cpu().numpy()
        img_u8 = (img_hwc * 255.0).astype(np.uint8)
        path_png = os.path.join(args.out, f"frame_{step:03d}.png")
        if Image is not None:
            Image.fromarray(img_u8).save(path_png)
        else:
            np.save(path_png.replace(".png", ".npy"), img_u8)
        saved += 1

        prop = obs["policy"][0]
        priv = obs["privileged"][0]
        print(f"step {step:3d}  r={float(reward[0]):+.2f}  "
              f"d_signed={float(priv[1]):+.2f}  psi_err={float(priv[2]):+.2f}  "
              f"vx={float(prop[0]):+.2f}  done={bool(terminated[0] or truncated[0])}",
              flush=True)

        if terminated[0] or truncated[0]:
            print(f"[inspect] episode ended at step {step}; resetting", flush=True)
            obs, _ = env_raw.reset()

    print(f"\n[inspect] saved {saved} frames to {args.out}/frame_*.png", flush=True)
    print(f"[inspect] open with: feh {args.out}  or  eog {args.out}/frame_000.png",
          flush=True)
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
