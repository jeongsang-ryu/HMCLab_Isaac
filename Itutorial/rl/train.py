"""Self-contained PPO trainer for the S-curve MuSHR env.

Minimal PyTorch PPO with:
  - Shared rollout buffer over all parallel envs
  - Gaussian policy (state-independent log std, tanh-squashed action)
  - GAE advantages, clipped surrogate objective
  - Checkpoint every N iterations

Usage:
    OMNI_KIT_ACCEPT_EULA=YES python Itutorial/rl/train.py \
        --num_envs 32 --iters 300 --ckpt Itutorial/rl/ckpt_scurve.pt
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--rollout_steps", type=int, default=32)
parser.add_argument("--iters", type=int, default=300)
parser.add_argument("--robot", type=str, default="mushr")
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--gamma", type=float, default=0.98)
parser.add_argument("--lam", type=float, default=0.95)
parser.add_argument("--clip", type=float, default=0.2)
parser.add_argument("--epochs", type=int, default=6)
parser.add_argument("--minibatches", type=int, default=4)
parser.add_argument("--ent_coef", type=float, default=0.01)
parser.add_argument("--val_coef", type=float, default=0.5)
parser.add_argument("--ckpt", type=str, default=None)
parser.add_argument("--resume", type=str, default=None,
                    help="Path to a checkpoint to warm-start from")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--gui", action="store_true",
                    help="Launch a visible Kit window instead of headless")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = not args.gui
args.enable_cameras = False          # no camera during training
launcher = AppLauncher(args)
simulation_app = launcher.app

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import ActorCritic  # noqa: E402
from s_curve_env import SCurveEnv, SCurveEnvCfg  # noqa: E402


@dataclass
class RolloutBuffer:
    obs: torch.Tensor
    actions_raw: torch.Tensor
    log_probs: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor

    def as_tuple(self):
        return (
            self.obs, self.actions_raw, self.log_probs,
            self.values, self.rewards, self.dones
        )


def main() -> int:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = SCurveEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.robot_name = args.robot
    env = SCurveEnv(cfg)
    device = env.device

    obs_dim = cfg.observation_space
    act_dim = cfg.action_space
    model = ActorCritic(obs_dim, act_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    start_iter = 0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        start_iter = int(ckpt.get("iter", 0)) + 1
        print(
            f"[PPO] resumed from {args.resume} at iter {start_iter} "
            f"(best_return {ckpt.get('mean_return', 0):.2f})",
            flush=True,
        )

    print(
        f"[PPO] obs={obs_dim} act={act_dim} num_envs={args.num_envs} "
        f"rollout={args.rollout_steps} iters={args.iters} "
        f"gui={not args.headless}",
        flush=True,
    )

    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]

    N = args.num_envs
    T = args.rollout_steps
    total_steps = 0
    t0 = time.time()
    best_mean_return = -1e9

    for it in range(start_iter, start_iter + args.iters):
        buf = RolloutBuffer(
            obs=torch.zeros(T, N, obs_dim, device=device),
            actions_raw=torch.zeros(T, N, act_dim, device=device),
            log_probs=torch.zeros(T, N, device=device),
            values=torch.zeros(T, N, device=device),
            rewards=torch.zeros(T, N, device=device),
            dones=torch.zeros(T, N, device=device),
        )
        ep_rew_sum = torch.zeros(N, device=device)
        episode_returns: list[float] = []

        for t in range(T):
            with torch.no_grad():
                action, log_prob, value, raw = model.act(obs)
            next_obs, rew, dones, truncs, _info = env.step(action)
            buf.obs[t] = obs
            buf.actions_raw[t] = raw
            buf.log_probs[t] = log_prob
            buf.values[t] = value
            buf.rewards[t] = rew
            buf.dones[t] = (dones | truncs).float()

            ep_rew_sum += rew
            reset_mask = (dones | truncs)
            for idx in torch.nonzero(reset_mask, as_tuple=False).flatten().tolist():
                episode_returns.append(float(ep_rew_sum[idx].item()))
                ep_rew_sum[idx] = 0.0

            obs = next_obs["policy"]
            total_steps += N

        # GAE + returns
        with torch.no_grad():
            _, _, last_value, _ = model.act(obs)
        advantages = torch.zeros_like(buf.rewards)
        gae = torch.zeros(N, device=device)
        next_value = last_value
        for t in reversed(range(T)):
            not_done = 1.0 - buf.dones[t]
            delta = buf.rewards[t] + args.gamma * next_value * not_done - buf.values[t]
            gae = delta + args.gamma * args.lam * not_done * gae
            advantages[t] = gae
            next_value = buf.values[t]
        returns = advantages + buf.values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Flatten
        obs_flat = buf.obs.reshape(T * N, obs_dim)
        raw_flat = buf.actions_raw.reshape(T * N, act_dim)
        logp_flat = buf.log_probs.reshape(T * N)
        adv_flat = advantages.reshape(T * N)
        ret_flat = returns.reshape(T * N)
        val_flat = buf.values.reshape(T * N)

        batch_size = T * N
        mb_size = batch_size // args.minibatches
        indices = torch.randperm(batch_size, device=device)

        pg_losses, v_losses, ent_losses, kls = [], [], [], []
        for epoch in range(args.epochs):
            indices = torch.randperm(batch_size, device=device)
            for start in range(0, batch_size, mb_size):
                mb = indices[start:start + mb_size]
                new_logp, entropy, new_v = model.evaluate(obs_flat[mb], raw_flat[mb])
                ratio = torch.exp(new_logp - logp_flat[mb])
                surr1 = ratio * adv_flat[mb]
                surr2 = torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * adv_flat[mb]
                pg_loss = -torch.min(surr1, surr2).mean()
                v_clipped = val_flat[mb] + (new_v - val_flat[mb]).clamp(-args.clip, args.clip)
                v_loss = 0.5 * torch.max(
                    (new_v - ret_flat[mb]) ** 2,
                    (v_clipped - ret_flat[mb]) ** 2,
                ).mean()
                ent_loss = entropy.mean()
                loss = pg_loss + args.val_coef * v_loss - args.ent_coef * ent_loss
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                opt.step()
                pg_losses.append(pg_loss.item())
                v_losses.append(v_loss.item())
                ent_losses.append(ent_loss.item())
                with torch.no_grad():
                    kls.append(
                        (logp_flat[mb] - new_logp).mean().item()
                    )

        mean_return = float(np.mean(episode_returns)) if episode_returns else float("nan")
        if mean_return > best_mean_return and not np.isnan(mean_return):
            best_mean_return = mean_return
            if args.ckpt:
                torch.save(
                    {
                        "model": model.state_dict(),
                        "iter": it,
                        "mean_return": mean_return,
                        "obs_dim": obs_dim,
                        "act_dim": act_dim,
                    },
                    args.ckpt,
                )

        log_rew = env.extras.get("log", {})
        elapsed = time.time() - t0
        print(
            f"[it {it:3d}/{args.iters}] steps={total_steps:>7}  "
            f"ep_ret={mean_return:+7.2f} (best {best_mean_return:+6.2f})  "
            f"pg={np.mean(pg_losses):+6.3f} v={np.mean(v_losses):6.3f} "
            f"ent={np.mean(ent_losses):5.2f} kl={np.mean(kls):+.3f}  "
            f"r_prog={log_rew.get('reward/progress', 0):+.2f} "
            f"r_speed={log_rew.get('reward/speed', 0):+.2f} "
            f"spd={log_rew.get('info/speed', 0):.2f}  "
            f"t={elapsed:5.1f}s",
            flush=True,
        )

    env.close()
    print("TRAIN_DONE", flush=True)
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
