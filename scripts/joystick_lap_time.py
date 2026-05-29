"""Drive the same UnicornRacingEnv with a joystick (or keyboard) and record
lap times — apples-to-apples comparison against the RL agents.

Identical env config to ``measure_lap_time.py``: stuck termination, simplified
reward, throttle ∈ [0, 1], steer ∈ [-1, 1]. The only difference is the action
source — the user instead of a neural network.

Usage::

    OMNI_KIT_ACCEPT_EULA=YES python scripts/joystick_lap_time.py \\
        --track my_track --max-laps 5

GUI viewport opens; drive with the controller. Each completed lap prints to
stdout; at the end, best / mean / median lap times. Optional chase cam MP4.
"""
from __future__ import annotations

import argparse
import math
import os
import statistics
import time

import torch

from isaaclab.app import AppLauncher
from hmclab_isaac.utils.teleop_input import add_input_args

parser = argparse.ArgumentParser()
parser.add_argument("--track", type=str, default="my_track")
parser.add_argument("--max-laps", type=int, default=5,
                    help="Stop after this many completed laps (across resets).")
parser.add_argument("--chase-out", type=str, default="/tmp/joystick_lap",
                    help="Directory for chase cam MP4.")
parser.add_argument("--no-chase", action="store_true",
                    help="Skip chase cam recording.")
add_input_args(parser)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True

launcher = AppLauncher(args)
sim_app = launcher.app

import gymnasium as gym  # noqa: E402

import hmclab_isaac.envs  # noqa: F401, E402
from hmclab_isaac.envs.racing.rl.single_agent.cfg import UnicornRacingEnvCfg  # noqa: E402
from hmclab_isaac.utils.teleop_input import make_controller  # noqa: E402


def main() -> int:
    env_cfg = UnicornRacingEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.track_name = args.track
    # Allow many laps per episode (RL-agent env triggers lap_success terminal
    # at target_laps=1, which would force a reset after every lap and pollute
    # the lap-time measurement with respawn overhead).
    env_cfg.target_laps = max(args.max_laps + 5, 50)

    env = gym.make("HMCLab-Racing-Single-Visual-v0", cfg=env_cfg)
    env_raw = env.unwrapped
    device = env_raw.device

    # Chase cam
    chase_rec = None
    if not args.no_chase:
        from hmclab_isaac.utils.capture import ChaseCamRecorder
        tag = f"joystick_{time.strftime('%Y%m%d_%H%M%S')}"
        chase_rec = ChaseCamRecorder(
            out_dir=args.chase_out, tag=tag,
            prim_path="/World/ChaseCamera",
            behind=3.0, above=2.0, fps=60, max_frames=12000,
        )
        chase_rec.on_reset()
        print(f"[joystick] chase cam → {args.chase_out}/{tag}.mp4", flush=True)

    ctrl = make_controller(args)
    if args.joystick:
        print(f"[joystick] device: stick X = steer (axis {args.js_axis_steer}), "
              f"stick Y = throttle (axis {args.js_axis_throttle})", flush=True)
    print(f"[joystick] track={args.track}  goal={args.max_laps} laps", flush=True)
    print("[joystick] keyboard fallback: ↑↓ throttle, ←→ steer, Space brake",
          flush=True)
    print("[joystick] click on the viewport so it owns keyboard focus.\n", flush=True)

    obs, _ = env_raw.reset()
    chase_stride = max(1, int(round(1.0 / (env_raw.cfg.sim.dt * env_raw.cfg.decimation * 60))))
    # dt_policy = sim.dt × decimation = 0.005 × 4 = 0.02 s → chase_stride = 1 (60 Hz capture)

    last_lap_count = int(env_raw._lap_count[0].item())
    lap_start_step = 0
    step = 0
    total_laps_done = 0
    lap_times: list[float] = []

    try:
        while sim_app.is_running() and total_laps_done < args.max_laps:
            thr, steer = ctrl.read()
            # throttle ≥ 0 only — env will clamp anyway, but be explicit
            thr_clipped = max(0.0, thr)
            action = torch.tensor([[thr_clipped, steer]], dtype=torch.float32, device=device)
            obs, reward, terminated, truncated, _ = env_raw.step(action)
            step += 1

            # ── Lap detection — env._lap_count increments at every total_length
            # of cumulative forward arc-length since spawn.
            current_lap = int(env_raw._lap_count[0].item())
            if current_lap > last_lap_count:
                lap_t = (step - lap_start_step) * env_raw.cfg.sim.dt * env_raw.cfg.decimation
                lap_times.append(lap_t)
                total_laps_done += 1
                print(f"[joystick] LAP {total_laps_done}: {lap_t:.2f}s "
                      f"(best so far {min(lap_times):.2f}s, "
                      f"mean {statistics.fmean(lap_times):.2f}s)", flush=True)
                lap_start_step = step
                last_lap_count = current_lap

            # ── Episode termination (stuck/spin/reverse) — reset and resume
            if bool(terminated[0]) or bool(truncated[0]):
                why = "stuck/spin/reverse" if float(reward[0]) < -1 else "timeout"
                print(f"[joystick] episode terminated ({why}) at step {step}. "
                      f"Reset and continue.", flush=True)
                obs, _ = env_raw.reset()
                last_lap_count = int(env_raw._lap_count[0].item())
                lap_start_step = step

            # Chase cam follow + capture
            if chase_rec is not None and step % chase_stride == 0:
                art = env_raw._robot
                rp = art.data.root_pos_w[0]
                rq = art.data.root_quat_w[0]
                qw, qx, qy, qz = (
                    float(rq[0]), float(rq[1]), float(rq[2]), float(rq[3])
                )
                yaw = math.atan2(
                    2.0 * (qw * qz + qx * qy),
                    1.0 - 2.0 * (qy * qy + qz * qz),
                )
                chase_rec.update_follow(
                    (float(rp[0]), float(rp[1]), float(rp[2])), yaw
                )
                chase_rec.capture_frame()

    finally:
        if chase_rec is not None:
            try:
                saved = chase_rec.finalize()
                print(f"[joystick] chase cam saved: {saved}", flush=True)
            except Exception as e:
                print(f"[joystick] chase finalize failed: {e}", flush=True)

    # ── Final stats ──
    print("\n" + "=" * 60, flush=True)
    print(f"Completed {len(lap_times)} lap(s) on track {args.track}", flush=True)
    if lap_times:
        print(f"  best   : {min(lap_times):.2f}s", flush=True)
        print(f"  mean   : {statistics.fmean(lap_times):.2f}s", flush=True)
        print(f"  median : {statistics.median(lap_times):.2f}s", flush=True)
        print(f"  worst  : {max(lap_times):.2f}s", flush=True)
        print(f"  all    : {[f'{t:.2f}s' for t in lap_times]}", flush=True)
    print("=" * 60, flush=True)
    print(f"\nFor reference: SAC best at 12.25M step = 16.36s", flush=True)
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
