"""A1 — Train UnicornRacingEnv with the in-house Sync A2C (A3C-faithful).

Usage::

    OMNI_KIT_ACCEPT_EULA=YES \\
      /home/js/anaconda3/envs/hmclab_test/bin/python \\
      scripts/train_racing_a2c.py --num_envs 32 --headless --seed 0
"""
from __future__ import annotations

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--max_iterations", type=int, default=4096)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--no-lstm", action="store_true", help="Disable LSTM (A3C-FF variant)")
parser.add_argument("--track", type=str, default="test")
parser.add_argument("--log-dir", type=str, default="/tmp/hmclab_runs")
parser.add_argument("--resume", type=str, default="",
                    help="Path to a .pt checkpoint to warm-start from. "
                         "Loads net+opt state and continues iter numbering.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True

launcher = AppLauncher(args)
sim_app = launcher.app

# Imports after AppLauncher boot (pxr deps)
import gymnasium as gym  # noqa: E402

import hmclab_isaac.envs  # noqa: F401, E402  (triggers gym.register)
from hmclab_isaac.algos.sync_a2c import A2CAgent, A2CConfig, train  # noqa: E402
from hmclab_isaac.envs.racing.rl.single_agent.cfg import UnicornRacingEnvCfg  # noqa: E402


def main() -> int:
    env_cfg = UnicornRacingEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.track_name = args.track

    env = gym.make("HMCLab-Racing-Single-Visual-v0", cfg=env_cfg)
    # Unwrap to the raw DirectRLEnv (gym TimeLimit + OrderEnforcing wrappers obscure it)
    env_raw = env.unwrapped

    a2c_cfg = A2CConfig(
        max_iterations=args.max_iterations,
        seed=args.seed,
        use_lstm=not args.no_lstm,
        experiment_name=f"racing_a2c_{'lstm' if not args.no_lstm else 'ff'}",
        log_dir=args.log_dir,
    )
    print(f"[train_a2c] num_envs={args.num_envs}, use_lstm={a2c_cfg.use_lstm}, "
          f"max_iters={a2c_cfg.max_iterations}", flush=True)
    agent = A2CAgent(env_raw, a2c_cfg)

    start_iter = 0
    if args.resume:
        import torch as _torch
        sd = _torch.load(args.resume, map_location=agent.device)
        agent.net.load_state_dict(sd["net"])
        agent.opt.load_state_dict(sd["opt"])
        start_iter = int(sd.get("iter", 0))
        print(f"[train_a2c] loaded {args.resume} (iter={start_iter})", flush=True)

    log_dir = train(env_raw, agent, a2c_cfg, start_iter=start_iter)
    print(f"[train_a2c] DONE — checkpoints in {log_dir}", flush=True)
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
