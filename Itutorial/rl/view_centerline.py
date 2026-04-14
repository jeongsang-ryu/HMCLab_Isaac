"""Visualise 64 vehicles racing in parallel on the centerline track (GUI).

Runs the same CenterlineEnv used for training, with a visible Kit window so
you can watch the cars drive. If a trained checkpoint is given it uses that
policy; otherwise it applies a simple "full throttle + look-ahead steering"
open-loop controller so cars still move.

Usage:
    OMNI_KIT_ACCEPT_EULA=YES python Itutorial/rl/view_centerline.py
    OMNI_KIT_ACCEPT_EULA=YES python Itutorial/rl/view_centerline.py \\
        --num_envs 64 --ckpt Itutorial/rl/ckpt_centerline_4k.pt --steps 3000
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=2000)
parser.add_argument("--robot", type=str, default="f1tenth")
parser.add_argument("--track", type=str, default="mini_oval_flat")
parser.add_argument("--ckpt", type=str, default=None,
                    help="Optional PPO checkpoint to drive the cars.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = False          # GUI on
args.enable_cameras = False
launcher = AppLauncher(args)
simulation_app = launcher.app

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from centerline_env import CenterlineEnv, CenterlineEnvCfg  # noqa: E402


def main() -> int:
    cfg = CenterlineEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.robot_name = args.robot
    cfg.track_name = args.track

    env = CenterlineEnv(cfg)
    device = env.device
    N = env.num_envs
    print(
        f"[view] num_envs={N} robot={args.robot} track={args.track} "
        f"ckpt={args.ckpt or 'none (open-loop)'}",
        flush=True,
    )

    model = None
    if args.ckpt and os.path.exists(args.ckpt):
        from model import ActorCritic
        hidden = 256 if cfg.observation_space >= 256 else 128
        model = ActorCritic(cfg.observation_space, cfg.action_space, hidden=hidden).to(device)
        ck = torch.load(args.ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        model.eval()
        print(
            f"[view] loaded ckpt iter={ck.get('iter')} "
            f"mean_return={ck.get('mean_return'):.2f}",
            flush=True,
        )

    obs = env.reset()[0]["policy"]

    for step in range(args.steps):
        with torch.no_grad():
            if model is not None:
                action, _, _, _ = model.act(obs)
            else:
                # Open-loop fallback: gentle straight-ahead drive
                action = torch.zeros((N, 2), device=device)
                action[:, 1] = 0.0   # ~50% target speed

        obs = env.step(action)[0]["policy"]

        if step % 100 == 0:
            v = env.ego.data.root_lin_vel_w[:, :2].norm(dim=-1)
            print(f"  step={step} v_mean={v.mean().item():.2f} v_max={v.max().item():.2f}",
                  flush=True)

    env.close()
    print("VIEW_DONE", flush=True)
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
