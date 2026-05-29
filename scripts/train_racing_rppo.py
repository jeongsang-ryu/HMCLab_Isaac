"""A3 — Train UnicornRacingEnv with the in-house Recurrent CNN-PPO.

Networks mirror the asymmetric SAC (CNN + GRU-256 actor, asymmetric V-critic);
the algorithm is true PPO (clipped surrogate + GAE + KL-adaptive lr). This is
the on-policy counterpart to ``train_racing_sac.py`` with the *same* model, so
the PPO-vs-SAC comparison isolates on-policy vs off-policy.

Usage::

    OMNI_KIT_ACCEPT_EULA=YES \\
      /home/js/anaconda3/envs/hmclab_test/bin/python \\
      scripts/train_racing_rppo.py --num_envs 32 --headless --seed 0 --track my_track
"""
from __future__ import annotations

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--max_iterations", type=int, default=4096)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--track", type=str, default="test")
parser.add_argument("--log-dir", type=str, default="/tmp/hmclab_runs")
parser.add_argument("--resume", type=str, default="",
                    help="Path to a rppo_*.pt checkpoint to warm-start from "
                         "(loads actor+critic+opt and continues iter numbering).")
# Actor-head parity toggles (default = full SAC parity)
parser.add_argument("--no-gru", action="store_true",
                    help="Disable the actor GRU (memoryless CNN+MLP actor).")
parser.add_argument("--plain-gaussian", action="store_true",
                    help="Use a plain Gaussian + state-independent log_std head "
                         "(PPO-standard, more stable) instead of SAC's tanh + "
                         "state-dependent std.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True

launcher = AppLauncher(args)
sim_app = launcher.app

import gymnasium as gym  # noqa: E402

import hmclab_isaac.envs  # noqa: F401, E402  (triggers gym.register)
from hmclab_isaac.algos.recurrent_ppo import RecurrentPPO, RPPOConfig, train  # noqa: E402
from hmclab_isaac.envs.racing.rl.single_agent.cfg import UnicornRacingEnvCfg  # noqa: E402


def main() -> int:
    env_cfg = UnicornRacingEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.track_name = args.track

    env = gym.make("HMCLab-Racing-Single-Visual-v0", cfg=env_cfg)
    env_raw = env.unwrapped

    cfg = RPPOConfig(
        max_iterations=args.max_iterations,
        seed=args.seed,
        log_dir=args.log_dir,
        use_actor_gru=not args.no_gru,
        use_tanh=not args.plain_gaussian,
        state_dependent_std=not args.plain_gaussian,
    )
    print(f"[train_rppo] num_envs={args.num_envs}, max_iters={cfg.max_iterations}, "
          f"track={args.track}, gru={cfg.use_actor_gru}, tanh={cfg.use_tanh}", flush=True)

    agent = RecurrentPPO(env_raw, cfg)

    start_iter = 0
    if args.resume:
        sd = __import__("torch").load(args.resume, map_location=agent.device)
        agent.actor.load_state_dict(sd["actor"])
        agent.critic.load_state_dict(sd["critic"])
        if "opt" in sd:
            agent.opt.load_state_dict(sd["opt"])
        start_iter = int(sd.get("iter", 0))
        print(f"[train_rppo] resumed from {args.resume} (iter={start_iter})", flush=True)

    train(env_raw, agent, cfg, start_iter=start_iter)
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
