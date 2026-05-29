"""A3 — Train UnicornRacingEnv with the in-house asymmetric SAC.

Usage::

    OMNI_KIT_ACCEPT_EULA=YES \\
      /home/js/anaconda3/envs/hmclab_test/bin/python \\
      scripts/train_racing_sac.py --num_envs 16 --headless --seed 0
"""
from __future__ import annotations

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--total_env_steps", type=int, default=4_000_000)
parser.add_argument("--buffer_capacity", type=int, default=200_000)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--track", type=str, default="test")
parser.add_argument("--log-dir", type=str, default="/tmp/hmclab_runs")
parser.add_argument("--resume", type=str, default="",
                    help="Path to a .pt checkpoint to warm-start from. "
                         "Loads actor+critic+target+alpha; replay buffer "
                         "starts empty.")
parser.add_argument("--opponent", type=str, default="",
                    help="Enable a scripted PP opponent (preset name: "
                         "pp_slow / pp_fast). Empty = single-agent.")
parser.add_argument("--opponent-ahead", type=float, default=8.0,
                    help="Arc-length gap between consecutive opponents (m).")
parser.add_argument("--opponent-count", type=int, default=5,
                    help="Number of PP opponent cars.")
parser.add_argument("--static-box-count", type=int, default=-1,
                    help="Override static obstacle box count "
                         "(-1 = keep cfg default; 0 = no boxes).")
parser.add_argument("--w-collide-wall", type=float, default=-1.0,
                    help="Override wall-collision penalty weight "
                         "(<0 = keep cfg default).")
parser.add_argument("--w-collide-opp", type=float, default=-1.0,
                    help="Override opponent-collision penalty weight "
                         "(<0 = keep cfg default).")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True

launcher = AppLauncher(args)
sim_app = launcher.app

import gymnasium as gym  # noqa: E402

import hmclab_isaac.envs  # noqa: F401, E402
from hmclab_isaac.algos.asym_sac import SACAgent, SACConfig, train  # noqa: E402
from hmclab_isaac.envs.racing.rl.single_agent.cfg import UnicornRacingEnvCfg  # noqa: E402


def main() -> int:
    env_cfg = UnicornRacingEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.track_name = args.track
    if args.opponent:
        env_cfg.opponent_enabled = True
        env_cfg.opponent_preset = args.opponent
        env_cfg.opponent_ahead_m = args.opponent_ahead
        env_cfg.opponent_count = args.opponent_count
        print(f"[train_sac] opponent={args.opponent} count={args.opponent_count} "
              f"ahead={args.opponent_ahead}m", flush=True)
    else:
        env_cfg.opponent_enabled = False
        print("[train_sac] opponents DISABLED (solo)", flush=True)
    if args.static_box_count >= 0:
        env_cfg.static_box_count = args.static_box_count
        print(f"[train_sac] static_box_count={env_cfg.static_box_count}",
              flush=True)
    if args.w_collide_wall >= 0.0:
        env_cfg.w_collide_wall = args.w_collide_wall
        print(f"[train_sac] w_collide_wall={env_cfg.w_collide_wall}",
              flush=True)
    if args.w_collide_opp >= 0.0:
        env_cfg.w_collide_opp = args.w_collide_opp
        print(f"[train_sac] w_collide_opp={env_cfg.w_collide_opp}",
              flush=True)

    env = gym.make("HMCLab-Racing-Single-Visual-v0", cfg=env_cfg)
    env_raw = env.unwrapped

    sac_cfg = SACConfig(
        total_env_steps=args.total_env_steps,
        buffer_capacity=args.buffer_capacity,
        seed=args.seed,
        experiment_name="racing_sac_asym",
        log_dir=args.log_dir,
    )
    print(f"[train_sac] num_envs={args.num_envs}, total_steps={sac_cfg.total_env_steps}, "
          f"buffer={sac_cfg.buffer_capacity}", flush=True)
    agent = SACAgent(env_raw, sac_cfg)

    start_step = 0
    if args.resume:
        import torch as _torch
        sd = _torch.load(args.resume, map_location=agent.device)
        agent.actor.load_state_dict(sd["actor"])
        agent.critic.load_state_dict(sd["critic"])
        agent.target_critic.load_state_dict(sd["target_critic"])
        agent.log_alpha.data.copy_(sd["log_alpha"].to(agent.device))
        start_step = int(sd.get("step", 0))
        print(f"[train_sac] loaded {args.resume} (step={start_step})", flush=True)

    log_dir = train(env_raw, agent, sac_cfg, start_step=start_step)
    print(f"[train_sac] DONE — checkpoints in {log_dir}", flush=True)
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
