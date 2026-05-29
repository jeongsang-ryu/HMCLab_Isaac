"""Load a trained racing policy → drive in 1-env GUI with chase-cam capture.

Supports checkpoints from all three trainers via ``--algo {a2c,ppo,sac,random}``.

Usage::

    OMNI_KIT_ACCEPT_EULA=YES python scripts/play_racing_policy.py \\
        --algo a2c --ckpt /tmp/hmclab_runs/racing_a2c_lstm/20260513_.../a2c_final.pt \\
        --max-steps 1500
"""
from __future__ import annotations

import argparse
import math
import os
import time

import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--algo", choices=["a2c", "ppo", "sac", "rppo", "ra2c", "random"], default="random")
parser.add_argument("--ckpt", type=str, default="",
                    help="Path to a checkpoint .pt — required unless --algo random.")
parser.add_argument("--track", type=str, default="test")
parser.add_argument("--opponent", type=str, default="",
                    help="Spawn a scripted PP opponent (pp_slow / pp_fast). "
                         "Empty = no opponent.")
parser.add_argument("--opponent-count", type=int, default=-1,
                    help="Override opponent count (-1 = keep cfg default).")
parser.add_argument("--static-box-count", type=int, default=-1,
                    help="Override static obstacle box count (-1 = cfg default, 0 = none).")
parser.add_argument("--opponent-ahead", type=float, default=8.0,
                    help="Opponent spawn arc-length ahead of the policy car.")
parser.add_argument("--max-steps", type=int, default=1500)
parser.add_argument("--chase-out", type=str, default="/tmp/hmclab_play",
                    help="Directory for chase-cam MP4 output.")
parser.add_argument("--no-chase", action="store_true",
                    help="Skip chase cam recording.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True

launcher = AppLauncher(args)
sim_app = launcher.app

import gymnasium as gym  # noqa: E402

import hmclab_isaac.envs  # noqa: F401, E402
from hmclab_isaac.envs.racing.rl.single_agent.cfg import UnicornRacingEnvCfg  # noqa: E402


def main() -> int:
    env_cfg = UnicornRacingEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.track_name = args.track
    if args.opponent:
        env_cfg.opponent_enabled = True
        env_cfg.opponent_preset = args.opponent
        env_cfg.opponent_ahead_m = args.opponent_ahead
    else:
        env_cfg.opponent_enabled = False
    if args.opponent_count >= 0:
        env_cfg.opponent_count = args.opponent_count
    if args.static_box_count >= 0:
        env_cfg.static_box_count = args.static_box_count

    env = gym.make("HMCLab-Racing-Single-Visual-v0", cfg=env_cfg)
    env_raw = env.unwrapped
    device = env_raw.device

    # Chase cam
    chase_rec = None
    if not args.no_chase:
        from hmclab_isaac.utils.capture import ChaseCamRecorder
        tag = f"play_{args.algo}_{time.strftime('%Y%m%d_%H%M%S')}"
        chase_rec = ChaseCamRecorder(
            out_dir=args.chase_out,
            tag=tag,
            prim_path="/World/ChaseCamera",
            behind=3.0, above=2.0, fps=60, max_frames=6000,
        )
        print(f"[play] chase → {args.chase_out}/{tag}.mp4", flush=True)
        chase_rec.on_reset()

    # Load policy
    policy = None
    if args.algo == "a2c":
        from hmclab_isaac.algos.sync_a2c import A3CActorCritic, A2CConfig
        obs, _ = env_raw.reset()
        img_shape = tuple(obs["images"].shape[1:])
        proprio_dim = int(obs["policy"].shape[1])
        action_dim = 2
        cfg = A2CConfig()
        net = A3CActorCritic(img_shape, proprio_dim, action_dim, cfg).to(device)
        sd = torch.load(args.ckpt, map_location=device)
        net.load_state_dict(sd["net"])
        net.eval()
        h = (torch.zeros(1, cfg.lstm_hidden, device=device),
             torch.zeros(1, cfg.lstm_hidden, device=device)) if cfg.use_lstm else None

        def act(o):
            nonlocal h
            with torch.no_grad():
                mean, std, _, h = net(o["images"], o["policy"], h)
                return mean.clamp(-1, 1)
        policy = act
    elif args.algo == "sac":
        from hmclab_isaac.algos.asym_sac import SACActor, SACConfig
        obs, _ = env_raw.reset()
        img_shape = tuple(obs["images"].shape[1:])
        proprio_dim = int(obs["policy"].shape[1])
        cfg = SACConfig()
        net = SACActor(img_shape, proprio_dim, 2, cfg).to(device)
        sd = torch.load(args.ckpt, map_location=device)
        net.load_state_dict(sd["actor"])
        net.eval()
        h_sac = (torch.zeros(1, net.gru_hidden, device=device)
                 if net.use_gru else None)

        def act(o):
            nonlocal h_sac
            with torch.no_grad():
                _, _, mean_action, h_sac = net.sample(o["images"], o["policy"], h_sac)
                return mean_action.clamp(-1, 1)
        policy = act
    elif args.algo in ("rppo", "ra2c"):
        # in-house recurrent PPO / A2C — identical RPPOActor network, so the
        # same loader handles both checkpoint kinds.
        from hmclab_isaac.algos.recurrent_ppo import RPPOActor, RPPOConfig
        obs, _ = env_raw.reset()
        img_shape = tuple(obs["images"].shape[1:])
        proprio_dim = int(obs["policy"].shape[1])
        cfg = RPPOConfig()
        sd = torch.load(args.ckpt, map_location=device)
        saved = sd.get("cfg", {})   # honor head toggles the ckpt was trained with
        cfg.use_actor_gru = bool(saved.get("use_actor_gru", cfg.use_actor_gru))
        cfg.use_tanh = bool(saved.get("use_tanh", cfg.use_tanh))
        cfg.state_dependent_std = bool(saved.get("state_dependent_std", cfg.state_dependent_std))
        net = RPPOActor(img_shape, proprio_dim, 2, cfg).to(device)
        net.load_state_dict(sd["actor"])
        net.eval()
        h_rppo = (torch.zeros(1, net.gru_hidden, device=device)
                  if net.use_gru else None)

        def act(o):
            nonlocal h_rppo
            with torch.no_grad():
                action, h_rppo = net.act_deterministic(o["images"], o["policy"], h_rppo)
                return action.clamp(-1, 1)
        policy = act
    elif args.algo == "ppo":
        # rsl_rl checkpoint format
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from rsl_rl.runners import OnPolicyRunner
        from hmclab_isaac.envs.racing.rl.single_agent.agents.rsl_rl_ppo_cfg import (
            UnicornRacingPPORunnerCfg,
        )
        agent_cfg = UnicornRacingPPORunnerCfg()
        wrapped = RslRlVecEnvWrapper(env_raw)
        runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=device)
        runner.load(args.ckpt)
        infer_policy = runner.get_inference_policy(device=device)

        def act(o):
            with torch.no_grad():
                return infer_policy.act_inference(o).clamp(-1, 1)
        policy = act
    else:
        # random — already covered by direct env step

        def act(o):
            return torch.empty((1, 2), device=device).uniform_(-1.0, 1.0)
        policy = act

    obs, _ = env_raw.reset()
    for step in range(args.max_steps):
        action = policy(obs)
        obs, reward, terminated, truncated, _ = env_raw.step(action)
        if chase_rec is not None and step % 4 == 0:
            art = env_raw._robot
            rp = art.data.root_pos_w[0]
            rq = art.data.root_quat_w[0]
            qw, qx, qy, qz = float(rq[0]), float(rq[1]), float(rq[2]), float(rq[3])
            yaw = math.atan2(2.0 * (qw*qz + qx*qy), 1.0 - 2.0 * (qy*qy + qz*qz))
            chase_rec.update_follow((float(rp[0]), float(rp[1]), float(rp[2])), yaw)
            chase_rec.capture_frame()
        if terminated[0] or truncated[0]:
            print(f"[play] step {step}: episode done (r={float(reward[0]):+.2f})", flush=True)
            obs, _ = env_raw.reset()
    if chase_rec is not None:
        try:
            saved = chase_rec.finalize()
            print(f"[play] chase cam saved: {saved}", flush=True)
        except Exception as exc:
            print(f"[play] chase finalize failed: {exc}", flush=True)
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
