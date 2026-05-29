"""Smoke test: opponent-enabled env. 4 envs, constant-throttle, 60 steps."""
from __future__ import annotations

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--track", type=str, default="my_track")
parser.add_argument("--preset", type=str, default="pp_slow")
parser.add_argument("--count", type=int, default=5)
parser.add_argument("--steps", type=int, default=60)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True

launcher = AppLauncher(args)
sim_app = launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402
import gymnasium as gym  # noqa: E402

import hmclab_isaac.envs  # noqa: F401, E402
from hmclab_isaac.envs.racing.rl.single_agent.cfg import UnicornRacingEnvCfg  # noqa: E402


def main() -> int:
    cfg = UnicornRacingEnvCfg()
    cfg.scene.num_envs = 4
    cfg.track_name = args.track
    cfg.opponent_enabled = True
    cfg.opponent_preset = args.preset
    cfg.opponent_count = args.count
    cfg.opponent_ahead_m = 8.0

    env = gym.make("HMCLab-Racing-Single-Visual-v0", cfg=cfg)
    er = env.unwrapped
    obs, _ = er.reset()
    print(f"[smoke] reset OK. obs keys: {list(obs.keys())}", flush=True)
    print(f"[smoke] opponents={len(er._opponents)} pp={er._opp_pp is not None}",
          flush=True)

    for i in range(args.steps):
        act = torch.zeros((4, 2), device=er.device)
        act[:, 0] = 0.5
        obs, r, term, trunc, _ = er.step(act)

    lp = er._robot.data.root_pos_w[0, :2].cpu().numpy()
    lv = float(er._robot.data.root_lin_vel_b[0, 0])
    print(f"[smoke] after {args.steps} steps:", flush=True)
    print(f"        learner  xy={lp}  vx={lv:+.2f} m/s", flush=True)
    for k, opp in enumerate(er._opponents):
        op = opp.data.root_pos_w[0, :2].cpu().numpy()
        ov = float(opp.data.root_lin_vel_b[0, 0])
        print(f"        opp[{k}] xy={op}  vx={ov:+.2f} m/s  "
              f"gap={np.linalg.norm(lp - op):.2f} m", flush=True)
    print(f"[smoke] image obs shape: {tuple(obs['images'].shape)}", flush=True)
    print("[smoke] SMOKE OK", flush=True)
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
