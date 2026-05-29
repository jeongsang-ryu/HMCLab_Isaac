"""Measure lap time of trained racing policies.

Two modes:

* **Single ckpt**::

    python scripts/measure_lap_time.py --algo sac \\
        --ckpt /tmp/hmclab_runs/racing_sac_asym/<ts>/sac_step01000000.pt \\
        --track my_track --episodes 30 --headless

* **Multi ckpt** (sweep a directory)::

    python scripts/measure_lap_time.py --algo sac \\
        --multi /tmp/hmclab_runs/racing_sac_asym/<ts> \\
        --track my_track --episodes 20 --headless

How it works:

1. Build the env with ``num_envs=1``. Override ``cfg.target_laps`` so that
   a lap completion *does* terminate the episode (≥ +30 final reward).
2. Run ``--episodes`` episodes per ckpt with the loaded policy in
   deterministic mode (action = mean of policy distribution).
3. Each episode's length in env-steps × 0.02 s = ``lap_time`` *iff* the
   terminal step's reward was clearly positive (lap_success bonus +50).
   Other terminations (stuck / spin / reverse / timeout) are recorded as
   failures.
4. Aggregate over episodes per ckpt → best lap time, mean lap time, lap
   completion rate.
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
from pathlib import Path

import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--algo", choices=["a2c", "ppo", "sac"], required=True)
parser.add_argument("--ckpt", type=str, default="",
                    help="Single ckpt path. Conflicts with --multi.")
parser.add_argument("--multi", type=str, default="",
                    help="Directory of ckpts (or run-timestamp directory). "
                         "All *.pt under it are evaluated.")
parser.add_argument("--track", type=str, default="my_track")
parser.add_argument("--episodes", type=int, default=20,
                    help="Episodes per ckpt for averaging.")
parser.add_argument("--max-steps-per-ep", type=int, default=3000,
                    help="Hard cap on per-ep step length (default 60 s).")
parser.add_argument("--lap-success-reward", type=float, default=30.0,
                    help="Terminal reward threshold to count as lap_success.")
parser.add_argument("--save-json", type=str, default="",
                    help="If set, dump per-ckpt results as JSON to this path.")
parser.add_argument("--save-fig", type=str, default="",
                    help="If set, render a PNG figure (step vs best/mean lap, "
                         "completion rate) to this path. Requires matplotlib.")
parser.add_argument("--max-ckpts", type=int, default=20,
                    help="When --multi has more ckpts than this, evenly "
                         "subsample down to this count.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True

launcher = AppLauncher(args)
sim_app = launcher.app

import gymnasium as gym  # noqa: E402

import hmclab_isaac.envs  # noqa: F401, E402
from hmclab_isaac.envs.racing.rl.single_agent.cfg import UnicornRacingEnvCfg  # noqa: E402


# ────────────────────────────────────────────────────────────────────
# Policy loaders — return a callable `act(obs)` that emits actions
# ────────────────────────────────────────────────────────────────────
def _load_a2c(env_raw, ckpt_path):
    from hmclab_isaac.algos.sync_a2c import A3CActorCritic, A2CConfig
    device = env_raw.device
    obs, _ = env_raw.reset()
    img_shape = tuple(obs["images"].shape[1:])
    proprio_dim = int(obs["policy"].shape[1])
    cfg = A2CConfig()
    net = A3CActorCritic(img_shape, proprio_dim, 2, cfg).to(device)
    sd = torch.load(ckpt_path, map_location=device, weights_only=False)
    net.load_state_dict(sd["net"])
    net.eval()
    h = (torch.zeros(1, cfg.lstm_hidden, device=device),
         torch.zeros(1, cfg.lstm_hidden, device=device)) if cfg.use_lstm else None

    def act(obs, done_flag):
        nonlocal h
        with torch.no_grad():
            mean, _, _, h = net(obs["images"], obs["policy"], h)
            if h is not None and done_flag:
                h = (torch.zeros_like(h[0]), torch.zeros_like(h[1]))
            return mean.clamp(-1, 1)
    return act


def _load_sac(env_raw, ckpt_path):
    from hmclab_isaac.algos.asym_sac import SACActor, SACConfig
    device = env_raw.device
    obs, _ = env_raw.reset()
    img_shape = tuple(obs["images"].shape[1:])
    proprio_dim = int(obs["policy"].shape[1])
    # Auto-detect whether this ckpt was trained with GRU or not by inspecting
    # the saved state_dict keys. Old ckpts (pre-GRU) won't have "gru.*" weights.
    sd = torch.load(ckpt_path, map_location=device, weights_only=False)
    actor_sd = sd["actor"]
    has_gru = any(k.startswith("gru.") for k in actor_sd.keys())
    cfg = SACConfig()
    cfg.use_actor_gru = has_gru
    net = SACActor(img_shape, proprio_dim, 2, cfg).to(device)
    net.load_state_dict(actor_sd)
    net.eval()
    h = torch.zeros(1, net.gru_hidden, device=device) if net.use_gru else None

    def act(obs, done_flag):
        nonlocal h
        with torch.no_grad():
            _, _, mean_action, h = net.sample(obs["images"], obs["policy"], h)
            if h is not None and done_flag:
                h = torch.zeros_like(h)
            return mean_action.clamp(-1, 1)
    return act


def _load_ppo(env_raw, ckpt_path):
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner
    from hmclab_isaac.envs.racing.rl.single_agent.agents.rsl_rl_ppo_cfg import (
        UnicornRacingPPORunnerCfg,
    )

    agent_cfg = UnicornRacingPPORunnerCfg()
    agent_dict = agent_cfg.to_dict()
    DEPRECATED = {"stochastic", "init_noise_std", "noise_std_type", "state_dependent_std"}
    for k in ("actor", "critic"):
        if isinstance(agent_dict.get(k), dict):
            for f in DEPRECATED:
                agent_dict[k].pop(f, None)

    wrapped = RslRlVecEnvWrapper(env_raw)
    runner = OnPolicyRunner(wrapped, agent_dict, log_dir=None, device=str(env_raw.device))
    runner.load(ckpt_path)
    infer = runner.get_inference_policy(device=str(env_raw.device))

    def act(obs, done_flag):
        with torch.no_grad():
            return infer.act_inference(obs).clamp(-1, 1)
    return act


_LOADERS = {"a2c": _load_a2c, "sac": _load_sac, "ppo": _load_ppo}


# ────────────────────────────────────────────────────────────────────
# Evaluation core
# ────────────────────────────────────────────────────────────────────
def evaluate_ckpt(env_raw, ckpt_path: str, algo: str, episodes: int,
                  max_steps: int, lap_threshold: float) -> dict:
    """Run `episodes` episodes with the loaded policy. Return stats dict."""
    loader = _LOADERS[algo]
    act = loader(env_raw, ckpt_path)
    device = env_raw.device

    obs, _ = env_raw.reset()
    lap_times: list[float] = []
    failure_terminal_rewards: list[float] = []
    n_lap_success = 0
    n_failure = 0
    n_timeout = 0
    ep_step = 0
    eps_done = 0
    # Force a clean hidden state at start for recurrent policies
    while eps_done < episodes and ep_step < max_steps * episodes * 2:
        done_flag = False
        action = act(obs, done_flag)
        obs, reward, terminated, truncated, _ = env_raw.step(action)
        ep_step += 1
        r = float(reward[0].item())
        t = bool(terminated[0].item()) or bool(truncated[0].item())
        if t:
            # Classify the terminal
            if r >= lap_threshold:
                n_lap_success += 1
                lap_times.append(ep_step * 0.02)   # 50 Hz
            elif r < -1.0:
                n_failure += 1
                failure_terminal_rewards.append(r)
            else:
                n_timeout += 1
            eps_done += 1
            ep_step = 0
            obs, _ = env_raw.reset()
            # Re-init the policy's recurrent state via a dummy call
            act(obs, True)

    out = {
        "ckpt": os.path.basename(ckpt_path),
        "episodes": eps_done,
        "lap_success": n_lap_success,
        "failure": n_failure,
        "timeout": n_timeout,
        "completion_rate": n_lap_success / max(1, eps_done),
        "best_lap_s": min(lap_times) if lap_times else float("inf"),
        "mean_lap_s": (statistics.fmean(lap_times) if lap_times else float("inf")),
        "median_lap_s": (statistics.median(lap_times) if lap_times else float("inf")),
    }
    return out


def main() -> int:
    if not args.ckpt and not args.multi:
        print("[lap-time] --ckpt or --multi required", flush=True)
        return 1

    env_cfg = UnicornRacingEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.track_name = args.track
    # Ensure lap_success terminal fires after a single lap (default cfg)
    env_cfg.target_laps = 1

    env = gym.make("HMCLab-Racing-Single-Visual-v0", cfg=env_cfg)
    env_raw = env.unwrapped

    if args.ckpt:
        ckpt_list = [args.ckpt]
    else:
        root = Path(args.multi)
        if not root.exists():
            print(f"[lap-time] --multi path does not exist: {root}", flush=True)
            return 1
        ckpt_list = sorted(str(p) for p in root.rglob("*.pt"))
        if not ckpt_list:
            print(f"[lap-time] no *.pt under {root}", flush=True)
            return 1
        # Subsample evenly if too many
        if len(ckpt_list) > args.max_ckpts:
            idx = [int(round(i)) for i in
                   [k * (len(ckpt_list) - 1) / (args.max_ckpts - 1)
                    for k in range(args.max_ckpts)]]
            ckpt_list = [ckpt_list[i] for i in sorted(set(idx))]
            print(f"[lap-time] subsampled to {len(ckpt_list)} ckpts (--max-ckpts)",
                  flush=True)

    print(f"[lap-time] algo={args.algo}  track={args.track}  episodes/ckpt={args.episodes}",
          flush=True)
    print(f"[lap-time] evaluating {len(ckpt_list)} ckpt(s)", flush=True)

    rows = []
    for path in ckpt_list:
        print(f"\n[lap-time] >>> {path}", flush=True)
        stats = evaluate_ckpt(env_raw, path, args.algo, args.episodes,
                              args.max_steps_per_ep, args.lap_success_reward)
        rows.append(stats)
        print(
            f"  episodes={stats['episodes']}  lap_success={stats['lap_success']}  "
            f"completion={stats['completion_rate']*100:.0f}%  "
            f"best_lap={stats['best_lap_s']:.2f}s  "
            f"mean_lap={stats['mean_lap_s']:.2f}s  "
            f"(failures={stats['failure']}, timeouts={stats['timeout']})",
            flush=True,
        )

    # Summary table (best at top)
    print("\n" + "=" * 80, flush=True)
    print("Summary (sorted by best_lap_s ascending)", flush=True)
    print("=" * 80, flush=True)
    print(f"{'ckpt':45s}  {'completion':>10s}  {'best_lap':>9s}  {'mean_lap':>9s}",
          flush=True)
    rows_sorted = sorted(rows, key=lambda r: r["best_lap_s"])
    for r in rows_sorted:
        best = f"{r['best_lap_s']:.2f}s" if r['best_lap_s'] != float('inf') else "  —  "
        mean = f"{r['mean_lap_s']:.2f}s" if r['mean_lap_s'] != float('inf') else "  —  "
        print(f"{r['ckpt']:45s}  {r['completion_rate']*100:>9.0f}%  {best:>9s}  {mean:>9s}",
              flush=True)

    best = rows_sorted[0] if rows_sorted else None
    if best and best["best_lap_s"] != float("inf"):
        print(f"\n[lap-time] *** BEST CKPT: {best['ckpt']} — {best['best_lap_s']:.2f}s ***",
              flush=True)
    else:
        print("\n[lap-time] No ckpt produced a lap-success episode.", flush=True)

    # ----- Persist results -----
    if args.save_json:
        import json
        # `inf` is not JSON; replace
        rows_json = []
        for r in rows:
            clean = {k: (None if isinstance(v, float) and v == float("inf") else v)
                     for k, v in r.items()}
            rows_json.append(clean)
        with open(args.save_json, "w") as f:
            json.dump({"algo": args.algo, "track": args.track,
                       "episodes": args.episodes, "rows": rows_json}, f, indent=2)
        print(f"[lap-time] JSON saved: {args.save_json}", flush=True)

    if args.save_fig:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import re
            def _step_from_name(name):
                m = re.search(r"step(\d+)", name)
                return int(m.group(1)) if m else 0
            steps = []
            best_ls = []
            mean_ls = []
            comp = []
            for r in sorted(rows, key=lambda x: _step_from_name(x["ckpt"])):
                steps.append(_step_from_name(r["ckpt"]) / 1e6)  # M steps
                best_ls.append(r["best_lap_s"] if r["best_lap_s"] != float("inf") else None)
                mean_ls.append(r["mean_lap_s"] if r["mean_lap_s"] != float("inf") else None)
                comp.append(r["completion_rate"] * 100)
            fig, ax1 = plt.subplots(figsize=(10, 6))
            ax1.set_xlabel("env-steps (M)")
            ax1.set_ylabel("lap time (s)", color="tab:blue")
            # plot only valid points
            x_b = [s for s, l in zip(steps, best_ls) if l is not None]
            y_b = [l for l in best_ls if l is not None]
            x_m = [s for s, l in zip(steps, mean_ls) if l is not None]
            y_m = [l for l in mean_ls if l is not None]
            ax1.plot(x_b, y_b, "o-", color="tab:blue", label="best lap")
            ax1.plot(x_m, y_m, "s--", color="tab:cyan", label="mean lap", alpha=0.6)
            ax1.tick_params(axis="y", labelcolor="tab:blue")
            ax1.legend(loc="upper right")
            ax1.grid(alpha=0.3)
            ax2 = ax1.twinx()
            ax2.set_ylabel("completion rate (%)", color="tab:red")
            ax2.plot(steps, comp, "^:", color="tab:red", label="completion%")
            ax2.tick_params(axis="y", labelcolor="tab:red")
            ax2.set_ylim(-5, 105)
            title = f"{args.algo.upper()} on {args.track} — {args.episodes} eps/ckpt"
            if best and best["best_lap_s"] != float("inf"):
                title += f"  |  BEST {best['best_lap_s']:.2f}s ({best['ckpt']})"
            plt.title(title)
            plt.tight_layout()
            plt.savefig(args.save_fig, dpi=130)
            print(f"[lap-time] PNG saved: {args.save_fig}", flush=True)
        except ImportError as e:
            print(f"[lap-time] matplotlib missing — skipping figure: {e}", flush=True)
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
