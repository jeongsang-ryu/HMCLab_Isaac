"""Evaluate A1/A2/A3 checkpoints on the same env and tabulate the comparison.

For each algorithm with a non-empty checkpoint path, loads the policy, runs
``--episodes`` episodes (using ``--num_envs`` parallel envs), and records:

* completion rate (fraction of episodes that reached the lap-success terminal)
* off-track rate
* mean & best episode return
* mean episode length

Result is printed as a Markdown table.

Usage::

    python scripts/eval_racing_bakeoff.py \\
        --a2c /tmp/hmclab_runs/racing_a2c_lstm/.../a2c_final.pt \\
        --ppo /tmp/hmclab_runs/racing_ppo_asym_cnn/.../model_4095.pt \\
        --sac /tmp/hmclab_runs/racing_sac_asym/.../sac_final.pt \\
        --episodes 100 --num_envs 16 --headless
"""
from __future__ import annotations

import argparse
import os
import statistics

import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--a2c", type=str, default="")
parser.add_argument("--ppo", type=str, default="")
parser.add_argument("--sac", type=str, default="")
parser.add_argument("--episodes", type=int, default=100)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--track", type=str, default="test")
parser.add_argument("--max-steps", type=int, default=1500)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True

launcher = AppLauncher(args)
sim_app = launcher.app

import gymnasium as gym  # noqa: E402

import hmclab_isaac.envs  # noqa: F401, E402
from hmclab_isaac.envs.racing.rl.single_agent.cfg import UnicornRacingEnvCfg  # noqa: E402


def _load_a2c(env_raw, ckpt):
    from hmclab_isaac.algos.sync_a2c import A3CActorCritic, A2CConfig
    obs, _ = env_raw.reset()
    img_shape = tuple(obs["images"].shape[1:])
    proprio_dim = int(obs["policy"].shape[1])
    cfg = A2CConfig()
    device = env_raw.device
    net = A3CActorCritic(img_shape, proprio_dim, 2, cfg).to(device)
    sd = torch.load(ckpt, map_location=device)
    net.load_state_dict(sd["net"])
    net.eval()
    n = env_raw.num_envs
    h = (torch.zeros(n, cfg.lstm_hidden, device=device),
         torch.zeros(n, cfg.lstm_hidden, device=device)) if cfg.use_lstm else None
    def act(o, done):
        nonlocal h
        with torch.no_grad():
            mean, _, _, h = net(o["images"], o["policy"], h)
            if h is not None:
                m = (~done).float().unsqueeze(-1)
                h = (h[0] * m, h[1] * m)
            return mean.clamp(-1, 1)
    return act


def _load_sac(env_raw, ckpt):
    from hmclab_isaac.algos.asym_sac import SACActor, SACConfig
    obs, _ = env_raw.reset()
    img_shape = tuple(obs["images"].shape[1:])
    proprio_dim = int(obs["policy"].shape[1])
    cfg = SACConfig()
    device = env_raw.device
    net = SACActor(img_shape, proprio_dim, 2, cfg).to(device)
    sd = torch.load(ckpt, map_location=device)
    net.load_state_dict(sd["actor"])
    net.eval()
    n = env_raw.num_envs
    h = torch.zeros(n, net.gru_hidden, device=device) if net.use_gru else None
    def act(o, done):
        nonlocal h
        with torch.no_grad():
            _, _, mean_action, h = net.sample(o["images"], o["policy"], h)
            if h is not None:
                h = h * (~done).float().unsqueeze(-1)
            return mean_action.clamp(-1, 1)
    return act


def _load_ppo(env_raw, ckpt):
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner
    from hmclab_isaac.envs.racing.rl.single_agent.agents.rsl_rl_ppo_cfg import (
        UnicornRacingPPORunnerCfg,
    )
    agent_cfg = UnicornRacingPPORunnerCfg()
    wrapped = RslRlVecEnvWrapper(env_raw)
    device = env_raw.device
    runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=device)
    runner.load(ckpt)
    infer = runner.get_inference_policy(device=device)
    def act(o, done):
        with torch.no_grad():
            return infer.act_inference(o).clamp(-1, 1)
    return act


def _run_eval(env_raw, act_fn, max_steps: int, target_episodes: int):
    device = env_raw.device
    obs, _ = env_raw.reset()
    n = env_raw.num_envs
    ep_returns = torch.zeros(n, device=device)
    ep_lengths = torch.zeros(n, dtype=torch.long, device=device)
    completed = []
    off_track = []
    returns = []
    lengths = []
    finished = 0
    step = 0
    while finished < target_episodes and step < max_steps * target_episodes:
        done_in = torch.zeros(n, dtype=torch.bool, device=device)
        action = act_fn(obs, done_in)
        obs, reward, terminated, truncated, _ = env_raw.step(action)
        ep_returns += reward
        ep_lengths += 1
        d = terminated | truncated
        for env_idx in d.nonzero(as_tuple=False).flatten().tolist():
            r = float(ep_returns[env_idx].item())
            L = int(ep_lengths[env_idx].item())
            # heuristic — final-step reward > 10 ⇒ lap success terminal
            completed.append(1 if r > 10.0 else 0)
            off_track.append(1 if r < -10.0 else 0)
            returns.append(r)
            lengths.append(L)
            ep_returns[env_idx] = 0.0
            ep_lengths[env_idx] = 0
            finished += 1
        step += 1
    return {
        "n": finished,
        "completion_rate": sum(completed) / max(1, len(completed)),
        "offtrack_rate": sum(off_track) / max(1, len(off_track)),
        "mean_return": statistics.fmean(returns) if returns else 0.0,
        "best_return": max(returns) if returns else 0.0,
        "mean_length": statistics.fmean(lengths) if lengths else 0.0,
    }


def main() -> int:
    env_cfg = UnicornRacingEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.track_name = args.track

    env = gym.make("HMCLab-Racing-Single-Visual-v0", cfg=env_cfg)
    env_raw = env.unwrapped

    targets = []
    if args.a2c:
        targets.append(("A1 (Sync A2C)", _load_a2c(env_raw, args.a2c)))
    if args.ppo:
        targets.append(("A2 (PPO)", _load_ppo(env_raw, args.ppo)))
    if args.sac:
        targets.append(("A3 (SAC)", _load_sac(env_raw, args.sac)))
    if not targets:
        print("[eval] no checkpoints provided — pass --a2c/--ppo/--sac", flush=True)
        return 1

    print("\n[eval] Bake-off — track={!r}, episodes={}, num_envs={}".format(
        args.track, args.episodes, args.num_envs), flush=True)
    rows = []
    for label, act in targets:
        print(f"\n[eval] {label} …", flush=True)
        stats = _run_eval(env_raw, act, args.max_steps, args.episodes)
        rows.append((label, stats))
        print(f"[eval]   completion={stats['completion_rate']*100:.1f}%  "
              f"off-track={stats['offtrack_rate']*100:.1f}%  "
              f"mean_ret={stats['mean_return']:+.2f}  "
              f"best_ret={stats['best_return']:+.2f}  "
              f"mean_len={stats['mean_length']:.0f}  "
              f"(n={stats['n']})", flush=True)

    # Markdown table
    print("\n| Method | Completion | Off-track | Mean ret | Best ret | Mean len |", flush=True)
    print("|---|---|---|---|---|---|", flush=True)
    for label, s in rows:
        print(
            f"| {label} | {s['completion_rate']*100:.1f}% | {s['offtrack_rate']*100:.1f}% | "
            f"{s['mean_return']:+.2f} | {s['best_return']:+.2f} | {s['mean_length']:.0f} |",
            flush=True,
        )
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
