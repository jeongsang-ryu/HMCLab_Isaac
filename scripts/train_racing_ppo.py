"""A2 — Train UnicornRacingEnv with rsl_rl's CNN-PPO (asymmetric obs_groups).

Usage::

    OMNI_KIT_ACCEPT_EULA=YES \\
      /home/js/anaconda3/envs/hmclab_test/bin/python \\
      scripts/train_racing_ppo.py --num_envs 32 --headless --seed 0
"""
from __future__ import annotations

import argparse
import os
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--max_iterations", type=int, default=4096)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--track", type=str, default="test")
parser.add_argument("--log-dir", type=str, default="/tmp/hmclab_runs")
parser.add_argument("--resume", type=str, default="",
                    help="Path to a rsl_rl model_*.pt to warm-start from. "
                         "OnPolicyRunner.load() handles iter counter resume.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True

launcher = AppLauncher(args)
sim_app = launcher.app

import gymnasium as gym  # noqa: E402

import hmclab_isaac.envs  # noqa: F401, E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from hmclab_isaac.envs.racing.rl.single_agent.cfg import UnicornRacingEnvCfg  # noqa: E402
from hmclab_isaac.envs.racing.rl.single_agent.agents.rsl_rl_ppo_cfg import (  # noqa: E402
    UnicornRacingPPORunnerCfg,
)


def main() -> int:
    env_cfg = UnicornRacingEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.track_name = args.track

    env = gym.make("HMCLab-Racing-Single-Visual-v0", cfg=env_cfg)
    env_raw = env.unwrapped

    agent_cfg = UnicornRacingPPORunnerCfg()
    agent_cfg.seed = args.seed
    agent_cfg.max_iterations = args.max_iterations

    log_dir = os.path.join(
        args.log_dir, agent_cfg.experiment_name, time.strftime("%Y%m%d_%H%M%S"),
    )
    os.makedirs(log_dir, exist_ok=True)
    print(f"[train_ppo] num_envs={args.num_envs}, max_iters={agent_cfg.max_iterations}, "
          f"log_dir={log_dir}", flush=True)

    # Wrap env for rsl_rl
    wrapped_env = RslRlVecEnvWrapper(env_raw)

    # rsl_rl 4.x deprecated four MLP-cfg fields but the runtime models reject
    # them as unexpected kwargs. Strip before constructing the runner.
    agent_dict = agent_cfg.to_dict()
    DEPRECATED = {"stochastic", "init_noise_std", "noise_std_type", "state_dependent_std"}
    for k in ("actor", "critic"):
        if isinstance(agent_dict.get(k), dict):
            for f in DEPRECATED:
                agent_dict[k].pop(f, None)

    runner = OnPolicyRunner(wrapped_env, agent_dict, log_dir=log_dir, device="cuda:0")
    if args.resume:
        runner.load(args.resume)
        print(f"[train_ppo] resumed from {args.resume} "
              f"(iter={runner.current_learning_iteration})", flush=True)
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=False)
    print(f"[train_ppo] DONE — checkpoints in {log_dir}", flush=True)
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
