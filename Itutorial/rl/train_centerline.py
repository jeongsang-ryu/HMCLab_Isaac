"""PPO trainer for CenterlineEnv (2D LiDAR + centerline lookahead).

Mirrors train.py's minimal PPO loop, targets CenterlineEnv.

Usage:
    OMNI_KIT_ACCEPT_EULA=YES python Itutorial/rl/train_centerline.py \\
        --num_envs 32 --iters 80 --robot f1tenth \\
        --ckpt Itutorial/rl/ckpt_centerline.pt
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

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--rollout_steps", type=int, default=32)
parser.add_argument("--iters", type=int, default=80)
parser.add_argument("--robot", type=str, default="f1tenth")
parser.add_argument("--track", type=str, default="mini_oval_flat")
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--gamma", type=float, default=0.98)
parser.add_argument("--lam", type=float, default=0.95)
parser.add_argument("--clip", type=float, default=0.2)
parser.add_argument("--epochs", type=int, default=6)
parser.add_argument("--minibatches", type=int, default=4)
parser.add_argument("--ent_coef", type=float, default=0.01)
parser.add_argument("--ent_final", type=float, default=None,
                    help="If set, ent_coef decays linearly from --ent_coef to this value.")
parser.add_argument("--val_coef", type=float, default=0.5)
parser.add_argument("--ckpt", type=str, default=None)
parser.add_argument("--resume", type=str, default=None)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--gui", action="store_true")
parser.add_argument("--lidar_stack", type=int, default=1,
                    help="Number of consecutive LiDAR frames to stack (1 = no stack)")
parser.add_argument("--lidar_downsample", type=int, default=1,
                    help="LiDAR ray stride (1 = keep 1080 rays, 4 = 270 rays)")
parser.add_argument("--optimal_line", action="store_true",
                    help="Enable optimal racing line mode (wider cte, no heading bonus)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = not args.gui
args.enable_cameras = False
launcher = AppLauncher(args)
simulation_app = launcher.app

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import ActorCritic  # noqa: E402
from centerline_env import CenterlineEnv, CenterlineEnvCfg  # noqa: E402


@dataclass
class Buf:
    obs: torch.Tensor
    raw: torch.Tensor
    logp: torch.Tensor
    val: torch.Tensor
    rew: torch.Tensor
    done: torch.Tensor


def main() -> int:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    from centerline_env import compute_obs_dim
    cfg = CenterlineEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.robot_name = args.robot
    cfg.track_name = args.track
    cfg.lidar_stack = args.lidar_stack
    cfg.lidar_downsample = args.lidar_downsample
    cfg.optimal_line_mode = args.optimal_line
    cfg.observation_space = compute_obs_dim(args.lidar_stack, args.lidar_downsample)
    env = CenterlineEnv(cfg)
    device = env.device

    obs_dim = cfg.observation_space
    act_dim = cfg.action_space
    # Larger hidden for high-dim 2D-LiDAR (1080+) observations
    hidden = 256 if obs_dim >= 256 else 128
    model = ActorCritic(obs_dim, act_dim, hidden=hidden).to(device)
    print(f"[PPO] ActorCritic hidden={hidden}", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    start_iter = 0
    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        start_iter = int(ck.get("iter", 0)) + 1
        print(f"[PPO] resumed from {args.resume} at iter {start_iter}", flush=True)

    print(
        f"[PPO-centerline] obs={obs_dim} act={act_dim} "
        f"num_envs={args.num_envs} rollout={args.rollout_steps} "
        f"iters={args.iters} robot={args.robot} track={args.track}",
        flush=True,
    )

    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]

    N, T = args.num_envs, args.rollout_steps
    total_steps = 0
    t0 = time.time()
    best_ret = -1e9

    for it in range(start_iter, start_iter + args.iters):
        buf = Buf(
            obs=torch.zeros(T, N, obs_dim, device=device),
            raw=torch.zeros(T, N, act_dim, device=device),
            logp=torch.zeros(T, N, device=device),
            val=torch.zeros(T, N, device=device),
            rew=torch.zeros(T, N, device=device),
            done=torch.zeros(T, N, device=device),
        )
        ep_rew = torch.zeros(N, device=device)
        ep_rets: list[float] = []

        for t in range(T):
            with torch.no_grad():
                act, logp, val, raw = model.act(obs)
            nobs, rew, dones, truncs, _ = env.step(act)
            buf.obs[t] = obs
            buf.raw[t] = raw
            buf.logp[t] = logp
            buf.val[t] = val
            buf.rew[t] = rew
            buf.done[t] = (dones | truncs).float()

            ep_rew += rew
            mask = (dones | truncs)
            for i in torch.nonzero(mask, as_tuple=False).flatten().tolist():
                ep_rets.append(float(ep_rew[i].item()))
                ep_rew[i] = 0.0

            obs = nobs["policy"]
            total_steps += N

        with torch.no_grad():
            _, _, last_v, _ = model.act(obs)
        adv = torch.zeros_like(buf.rew)
        gae = torch.zeros(N, device=device)
        nxt_v = last_v
        for t in reversed(range(T)):
            nd = 1.0 - buf.done[t]
            delta = buf.rew[t] + args.gamma * nxt_v * nd - buf.val[t]
            gae = delta + args.gamma * args.lam * nd * gae
            adv[t] = gae
            nxt_v = buf.val[t]
        ret = adv + buf.val
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        obs_f = buf.obs.reshape(T * N, obs_dim)
        raw_f = buf.raw.reshape(T * N, act_dim)
        logp_f = buf.logp.reshape(T * N)
        adv_f = adv.reshape(T * N)
        ret_f = ret.reshape(T * N)
        val_f = buf.val.reshape(T * N)
        bs = T * N
        mb = bs // args.minibatches

        pg_l, v_l, ent_l, kls = [], [], [], []
        for _ in range(args.epochs):
            idx = torch.randperm(bs, device=device)
            for s in range(0, bs, mb):
                m = idx[s:s + mb]
                nlp, ent, nv = model.evaluate(obs_f[m], raw_f[m])
                r = torch.exp(nlp - logp_f[m])
                s1 = r * adv_f[m]
                s2 = torch.clamp(r, 1 - args.clip, 1 + args.clip) * adv_f[m]
                pg = -torch.min(s1, s2).mean()
                vc = val_f[m] + (nv - val_f[m]).clamp(-args.clip, args.clip)
                vl = 0.5 * torch.max((nv - ret_f[m]) ** 2, (vc - ret_f[m]) ** 2).mean()
                el = ent.mean()
                _ent = args.ent_coef
                if args.ent_final is not None:
                    frac = (it - start_iter) / max(args.iters - 1, 1)
                    _ent = args.ent_coef + (args.ent_final - args.ent_coef) * frac
                loss = pg + args.val_coef * vl - _ent * el
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                opt.step()
                pg_l.append(pg.item()); v_l.append(vl.item())
                ent_l.append(el.item())
                with torch.no_grad():
                    kls.append((logp_f[m] - nlp).mean().item())

        mean_ret = float(np.mean(ep_rets)) if ep_rets else float("nan")
        if mean_ret > best_ret and not np.isnan(mean_ret):
            best_ret = mean_ret
            if args.ckpt:
                torch.save(
                    {"model": model.state_dict(), "iter": it,
                     "mean_return": mean_ret,
                     "obs_dim": obs_dim, "act_dim": act_dim},
                    args.ckpt,
                )
        # Always save a "_last" ckpt every 50 iters so final policy is captured
        if args.ckpt and (it % 50 == 0 or it == start_iter + args.iters - 1):
            last_path = args.ckpt.replace(".pt", "_last.pt")
            torch.save(
                {"model": model.state_dict(), "iter": it,
                 "mean_return": mean_ret if not np.isnan(mean_ret) else 0.0,
                 "obs_dim": obs_dim, "act_dim": act_dim},
                last_path,
            )

        lg = env.extras.get("log", {})
        el = time.time() - t0
        print(
            f"[it {it:3d}] steps={total_steps:>7}  "
            f"ret={mean_ret:+7.2f} (best {best_ret:+6.2f})  "
            f"pg={np.mean(pg_l):+5.2f} v={np.mean(v_l):5.2f} "
            f"ent={np.mean(ent_l):5.2f} kl={np.mean(kls):+5.3f}  "
            f"prog={lg.get('reward/progress',0):+5.2f} "
            f"spd={lg.get('info/speed',0):4.2f} "
            f"cte={lg.get('info/cte',0):4.2f}  t={el:5.1f}s",
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
