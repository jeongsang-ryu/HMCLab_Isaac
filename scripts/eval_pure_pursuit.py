"""Headless evaluation of the centerline pure-pursuit bot.

Runs the PP controller on the racing env (no joystick, no NN) and measures
how well it tracks the centerline:

* mean / max |d_signed|   — lateral deviation from centerline (m)
* mean / max |psi_err|    — heading error vs track tangent (rad)
* lap times               — sim-time per completed lap
* completion              — laps finished vs episode terminations

Supports a parameter sweep so we can tune lookahead / steer-gain /
lat-accel automatically.

Single run::

    python scripts/eval_pure_pursuit.py --track my_track --laps 3 \\
        --la-gain 0.4 --steer-gain 1.3 --lat-accel 8 --headless

Sweep (cartesian product of the comma-lists)::

    python scripts/eval_pure_pursuit.py --track my_track --laps 2 --headless \\
        --sweep --steer-gain 1.0,1.3,1.6 --lat-accel 6,8,12
"""
from __future__ import annotations

import argparse
import itertools
import os
import statistics

import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--track", type=str, default="my_track")
parser.add_argument("--laps", type=int, default=3,
                    help="Target completed laps per config before stopping.")
parser.add_argument("--max-steps", type=int, default=6000,
                    help="Hard per-config step cap (≈120 s).")
parser.add_argument("--target-speed", type=str, default="5.0")
parser.add_argument("--la-gain", type=str, default="0.4")
parser.add_argument("--la-min", type=str, default="1.0")
parser.add_argument("--la-max", type=str, default="4.0")
parser.add_argument("--steer-gain", type=str, default="1.3")
parser.add_argument("--lat-accel", type=str, default="8.0")
parser.add_argument("--speed-preview-n", type=str, default="5")
parser.add_argument("--brake-decel", type=str, default="6.0")
parser.add_argument("--preset", type=str, default="",
                    help="Use a named PP_PRESETS entry (pp_slow / pp_fast). "
                         "Overrides individual --la-*/--steer-gain/... args.")
parser.add_argument("--sweep", action="store_true",
                    help="Treat comma-separated values as a grid to sweep.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True

launcher = AppLauncher(args)
sim_app = launcher.app

import gymnasium as gym  # noqa: E402

import hmclab_isaac.envs  # noqa: F401, E402
from hmclab_isaac.envs.racing.rl.single_agent.cfg import UnicornRacingEnvCfg  # noqa: E402
from hmclab_isaac.envs.racing.rl.single_agent.env import _yaw_from_quat  # noqa: E402
from hmclab_isaac.algos.pure_pursuit import CenterlinePurePursuit  # noqa: E402


def _floats(s: str):
    return [float(x) for x in str(s).split(",")]


def _ints(s: str):
    return [int(x) for x in str(s).split(",")]


def evaluate(env_raw, pp, target_laps: int, max_steps: int) -> dict:
    device = env_raw.device
    obs, _ = env_raw.reset()
    lap_times: list[float] = []
    d_abs: list[float] = []
    psi_abs: list[float] = []
    speeds: list[float] = []
    n_terminations = 0
    last_lap = int(env_raw._lap_count[0].item())
    lap_start_step = 0
    step = 0
    dt = env_raw.cfg.sim.dt * env_raw.cfg.decimation

    while len(lap_times) < target_laps and step < max_steps:
        xy = env_raw._robot.data.root_pos_w[0:1, :2] - env_raw.scene.env_origins[0:1, :2]
        yaw = _yaw_from_quat(env_raw._robot.data.root_quat_w[0:1])
        spd = env_raw._robot.data.root_lin_vel_b[0:1, 0]
        action = pp.step(xy, yaw, speed=spd)            # (1, 2)
        obs, reward, terminated, truncated, _ = env_raw.step(action)
        step += 1

        f = env_raw._frenet_cache
        d_abs.append(abs(float(f["d_signed"][0])))
        psi_abs.append(abs(float(f["psi_err"][0])))
        speeds.append(float(env_raw._robot.data.root_lin_vel_b[0, 0]))

        lap_now = int(env_raw._lap_count[0].item())
        if lap_now > last_lap:
            lap_times.append((step - lap_start_step) * dt)
            lap_start_step = step
            last_lap = lap_now

        if bool(terminated[0]) or bool(truncated[0]):
            n_terminations += 1
            obs, _ = env_raw.reset()
            last_lap = int(env_raw._lap_count[0].item())
            lap_start_step = step

    return {
        "laps": len(lap_times),
        "terminations": n_terminations,
        "best_lap": min(lap_times) if lap_times else float("inf"),
        "mean_lap": statistics.fmean(lap_times) if lap_times else float("inf"),
        "mean_d": statistics.fmean(d_abs) if d_abs else float("nan"),
        "max_d": max(d_abs) if d_abs else float("nan"),
        "mean_psi": statistics.fmean(psi_abs) if psi_abs else float("nan"),
        "mean_speed": statistics.fmean(speeds) if speeds else float("nan"),
        "steps": step,
    }


def main() -> int:
    env_cfg = UnicornRacingEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.track_name = args.track
    env_cfg.target_laps = max(args.laps + 5, 50)   # don't end ep on lap success
    env_cfg.init_lateral_jitter = 0.0
    env_cfg.init_yaw_jitter = 0.0

    env = gym.make("HMCLab-Racing-Single-Visual-v0", cfg=env_cfg)
    env_raw = env.unwrapped

    if args.preset:
        from hmclab_isaac.algos.pure_pursuit import PP_PRESETS
        if args.preset not in PP_PRESETS:
            print(f"[pp-eval] unknown preset {args.preset}; "
                  f"choices: {list(PP_PRESETS)}", flush=True)
            return 1
        grid = [("__preset__",)]
    elif args.sweep:
        grid = list(itertools.product(
            _floats(args.target_speed), _floats(args.la_gain),
            _floats(args.la_min), _floats(args.la_max),
            _floats(args.steer_gain), _floats(args.lat_accel),
            _ints(args.speed_preview_n), _floats(args.brake_decel),
        ))
    else:
        grid = [(
            _floats(args.target_speed)[0], _floats(args.la_gain)[0],
            _floats(args.la_min)[0], _floats(args.la_max)[0],
            _floats(args.steer_gain)[0], _floats(args.lat_accel)[0],
            _ints(args.speed_preview_n)[0], _floats(args.brake_decel)[0],
        )]

    print(f"[pp-eval] track={args.track}  configs={len(grid)}  "
          f"target_laps={args.laps}", flush=True)
    rows = []
    for combo in grid:
        if combo == ("__preset__",):
            from hmclab_isaac.algos.pure_pursuit import PP_PRESETS
            pp = CenterlinePurePursuit(
                env_raw._frenet, wheelbase=0.344,
                max_steer_rad=env_cfg.max_steer_rad,
                max_wheel_rps=env_cfg.max_wheel_rps, wheel_radius=0.0525,
                **PP_PRESETS[args.preset],
            )
            cfg_str = f"preset={args.preset}"
        else:
            (tspd, lag, lmin, lmax, sg, la, spn, bd) = combo
            pp = CenterlinePurePursuit(
                env_raw._frenet, wheelbase=0.344,
                max_steer_rad=env_cfg.max_steer_rad,
                max_wheel_rps=env_cfg.max_wheel_rps, wheel_radius=0.0525,
                target_speed_mps=tspd, lookahead_gain=lag,
                lookahead_min=lmin, lookahead_max=lmax,
                steering_gain=sg, corner_lat_accel=la, speed_preview_n=spn,
                brake_decel=bd,
            )
            cfg_str = (f"tspd={tspd} lag={lag} lmin={lmin} lmax={lmax} "
                       f"sg={sg} la={la} spn={spn} bd={bd}")
        st = evaluate(env_raw, pp, args.laps, args.max_steps)
        rows.append((cfg_str, st))
        bl = f"{st['best_lap']:.2f}s" if st['best_lap'] != float('inf') else "—"
        print(f"[pp-eval] {cfg_str}", flush=True)
        print(f"          laps={st['laps']}  term={st['terminations']}  "
              f"best={bl}  mean_d={st['mean_d']:.3f}m  max_d={st['max_d']:.3f}m  "
              f"mean_psi={st['mean_psi']:.3f}rad  mean_v={st['mean_speed']:.2f}m/s",
              flush=True)

    # Best config = most laps, then lowest mean_d, then best_lap
    def _key(r):
        s = r[1]
        return (-s["laps"], s["mean_d"] if s["mean_d"] == s["mean_d"] else 9e9,
                s["best_lap"])
    rows.sort(key=_key)
    print("\n" + "=" * 78, flush=True)
    print("Ranked (laps↑, mean_d↓, best_lap↓):", flush=True)
    for cfg_str, s in rows:
        bl = f"{s['best_lap']:.2f}s" if s['best_lap'] != float('inf') else "—"
        print(f"  laps={s['laps']} term={s['terminations']} best={bl} "
              f"mean_d={s['mean_d']:.3f} max_d={s['max_d']:.3f} | {cfg_str}",
              flush=True)
    print("=" * 78, flush=True)
    if rows:
        print(f"[pp-eval] BEST: {rows[0][0]}", flush=True)
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
