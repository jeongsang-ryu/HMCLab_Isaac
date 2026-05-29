"""Race against the best SAC checkpoint.

env_0 → SAC bot (loaded ckpt, deterministic mean action)
env_1 → user (joystick / keyboard)

Both envs share the same UnicornRacingEnvCfg → identical track, reward,
terminations. Because Isaac Lab clones each env onto its own origin
(``env_spacing=60 m`` apart), the two cars cannot physically collide; they
just race in parallel parallel-universe copies of the duct track.

Per-env lap times are tracked independently. Chase cam follows the user
car. On exit, a side-by-side comparison is printed.

Usage::

    OMNI_KIT_ACCEPT_EULA=YES python scripts/joystick_vs_bot.py \\
        --bot-ckpt /tmp/hmclab_runs/racing_sac_asym/20260513_223341/sac_step12250000.pt \\
        --track my_track --max-laps 5
"""
from __future__ import annotations

import argparse
import math
import os
import signal
import statistics
import time

import numpy as np
import torch


# Module-level shutdown flag. The SIGINT handler is INSTALLED INSIDE main()
# (after AppLauncher boots) because AppLauncher overrides any handler set
# before its construction.
_shutdown_requested = False


def _on_sigint(sig, frame):
    global _shutdown_requested
    _shutdown_requested = True
    print("\n[race] Ctrl+C — shutting down cleanly, please wait for MP4 save…",
          flush=True)

from isaaclab.app import AppLauncher
from hmclab_isaac.utils.teleop_input import add_input_args

parser = argparse.ArgumentParser()
parser.add_argument(
    "--bot", choices=["sac", "pp"], default="sac",
    help="Opponent type: trained SAC ckpt or centerline pure-pursuit.",
)
parser.add_argument(
    "--pp-preset", type=str, default="",
    help="Named PP preset (pp_slow / pp_fast). Overrides individual --pp-* "
         "args. pp_slow=5m/s stable, pp_fast=10m/s straights + corner braking.",
)
parser.add_argument(
    "--bot-ckpt", type=str,
    default="/tmp/hmclab_runs/racing_sac_asym/20260513_223341/sac_step12250000.pt",
    help="SAC ckpt (only used when --bot sac).",
)
parser.add_argument("--pp-lookahead", type=float, default=2.0,
                    help="Fixed fallback lookahead (m) when --pp-la-gain 0.")
parser.add_argument("--pp-target-speed", type=float, default=5.0,
                    help="Pure-pursuit constant cruise speed (m/s).")
parser.add_argument("--pp-curve-scale", type=float, default=0.0,
                    help="If >0, slow down in corners (|kappa| scale).")
parser.add_argument("--pp-la-gain", type=float, default=0.4,
                    help="Adaptive lookahead gain: L_d=clip(gain·|v|,min,max). "
                         "0 disables (use fixed --pp-lookahead).")
parser.add_argument("--pp-la-min", type=float, default=1.0,
                    help="Adaptive lookahead min (m).")
parser.add_argument("--pp-la-max", type=float, default=4.0,
                    help="Adaptive lookahead max (m).")
parser.add_argument("--pp-steer-gain", type=float, default=1.7,
                    help="Steering angle multiplier (>1 fights understeer).")
parser.add_argument("--pp-lat-accel", type=float, default=3.0,
                    help="Corner speed limit: v=min(target, sqrt(a_lat/|κ|)). "
                         "Lower ⇒ slower in corners. 0 = constant speed.")
parser.add_argument("--pp-speed-preview-n", type=int, default=5,
                    help="# of lookahead curvature samples (Δ=1,2,4,8,16 m) "
                         "to max over for corner-speed. 3 ⇒ brake 1-4 m "
                         "before the corner. Larger ⇒ earlier braking.")
parser.add_argument("--track", type=str, default="my_track")
parser.add_argument("--max-laps", type=int, default=5,
                    help="Stop when the user completes this many laps.")
parser.add_argument("--chase-out", type=str, default="/tmp/joystick_vs_bot",
                    help="Chase cam output dir.")
parser.add_argument("--no-chase", action="store_true")
parser.add_argument("--record-front", type=str, default="",
                    help="If set, write the USER car's front_cam obs frames "
                         "(128×128 RGB, what the network sees) to this MP4. "
                         "Encoded at 50 fps = sim time, so the playback shows "
                         "true vehicle speed regardless of GUI viewport FPS.")
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
from hmclab_isaac.algos.asym_sac import SACActor, SACConfig  # noqa: E402


def main() -> int:
    if args.bot == "sac" and not os.path.isfile(args.bot_ckpt):
        print(f"[race] bot ckpt not found: {args.bot_ckpt}", flush=True)
        return 1

    # Install SIGINT handler AFTER AppLauncher boot (which silently replaces
    # any handler set before it). This lets Ctrl+C exit the loop cleanly so
    # the finally block has a chance to encode the chase-cam MP4.
    signal.signal(signal.SIGINT, _on_sigint)

    env_cfg = UnicornRacingEnvCfg()
    env_cfg.scene.num_envs = 2          # 0 = bot, 1 = user
    env_cfg.track_name = args.track
    # Many laps per ep — no early lap_success terminate
    env_cfg.target_laps = max(args.max_laps + 5, 50)
    # Ghost mode — env_spacing 0 so both cars share the same visual track.
    # Different envs still have separate collision filters, so the two cars
    # pass through each other (no physical contact). Lateral / yaw jitter
    # also disabled so they start at the EXACT same pose.
    env_cfg.scene.env_spacing = 0.0
    env_cfg.init_lateral_jitter = 0.0
    env_cfg.init_yaw_jitter = 0.0
    # NOTE: env.record_cam_enabled spawn path currently breaks TiledCamera
    # init (no _timestamp). Leave it off and upscale the 128x128 policy obs
    # to 640x480 with bicubic interpolation for the MP4. Information content
    # is unchanged, only visual size grows.
    env_cfg.record_cam_enabled = False

    env = gym.make("HMCLab-Racing-Single-Visual-v0", cfg=env_cfg)
    env_raw = env.unwrapped
    device = env_raw.device

    # ── Initial spawn — force both envs to the same track progress
    # so the race starts at the same line.
    p_init = float(torch.rand((), device="cpu").item())
    env_raw._forced_spawn_progress = torch.tensor(
        [p_init, p_init], device=env_raw.device,
    )
    obs, _ = env_raw.reset()
    env_raw._forced_spawn_progress = None

    # ── Build bot ──
    bot_actor = None
    bot_h = None
    bot_pp = None
    if args.bot == "sac":
        img_shape = tuple(obs["images"].shape[1:])
        proprio_dim = int(obs["policy"].shape[1])
        sd = torch.load(args.bot_ckpt, map_location=device, weights_only=False)
        actor_sd = sd["actor"]
        has_gru = any(k.startswith("gru.") for k in actor_sd.keys())
        sac_cfg = SACConfig()
        sac_cfg.use_actor_gru = has_gru
        bot_actor = SACActor(img_shape, proprio_dim, 2, sac_cfg).to(device)
        bot_actor.load_state_dict(actor_sd)
        bot_actor.eval()
        bot_h = (
            torch.zeros(1, bot_actor.gru_hidden, device=device)
            if bot_actor.use_gru else None
        )
        print(f"[race] bot=SAC ckpt: {os.path.basename(args.bot_ckpt)} "
              f"(GRU={has_gru})", flush=True)
    else:
        from hmclab_isaac.algos.pure_pursuit import (
            CenterlinePurePursuit, PP_PRESETS,
        )
        common = dict(
            wheelbase=0.344,
            max_steer_rad=env_cfg.max_steer_rad,
            max_wheel_rps=env_cfg.max_wheel_rps,
            wheel_radius=0.0525,
        )
        if args.pp_preset:
            if args.pp_preset not in PP_PRESETS:
                print(f"[race] unknown preset {args.pp_preset}; "
                      f"choices: {list(PP_PRESETS)}", flush=True)
                return 1
            bot_pp = CenterlinePurePursuit(
                env_raw._frenet, **common, **PP_PRESETS[args.pp_preset],
            )
            print(f"[race] bot=PurePursuit preset={args.pp_preset} "
                  f"{PP_PRESETS[args.pp_preset]}", flush=True)
        else:
            bot_pp = CenterlinePurePursuit(
                env_raw._frenet, **common,
                lookahead_m=args.pp_lookahead,
                target_speed_mps=args.pp_target_speed,
                curvature_speed_scale=args.pp_curve_scale,
                lookahead_gain=args.pp_la_gain,
                lookahead_min=args.pp_la_min,
                lookahead_max=args.pp_la_max,
                steering_gain=args.pp_steer_gain,
                corner_lat_accel=args.pp_lat_accel,
                speed_preview_n=args.pp_speed_preview_n,
            )
            print(f"[race] bot=PurePursuit  L_d=clip({args.pp_la_gain}·|v|,"
                  f"{args.pp_la_min},{args.pp_la_max})m  sg={args.pp_steer_gain}  "
                  f"target={args.pp_target_speed}m/s  la={args.pp_lat_accel}", flush=True)

    # ── Chase cam following the user car (env 1) ──
    chase_rec = None
    chase_stride = 1   # capture every sim step → 50 frames per sim sec
    if not args.no_chase:
        from hmclab_isaac.utils.capture import ChaseCamRecorder
        tag = f"race_{time.strftime('%Y%m%d_%H%M%S')}"
        chase_rec = ChaseCamRecorder(
            out_dir=args.chase_out, tag=tag,
            prim_path="/World/ChaseCamera",
            behind=3.0, above=2.0,
            # fps=50 matches sim-time (1 frame per 0.02 s) so the encoded
            # MP4 plays back at real vehicle speed.
            fps=50, max_frames=12000,
        )
        chase_rec.on_reset()
        print(f"[race] chase cam (1280x720, 50 fps sim-time) → "
              f"{args.chase_out}/{tag}.mp4", flush=True)

    ctrl = make_controller(args)
    print(f"[race] track={args.track}  goal=user {args.max_laps} laps", flush=True)
    print("[race] click on viewport so it has keyboard focus.\n", flush=True)

    # ── State for per-env lap tracking ──
    last_lap = [int(env_raw._lap_count[0].item()),
                int(env_raw._lap_count[1].item())]
    lap_start_step = [0, 0]
    lap_times = {"bot": [], "user": []}
    step = 0
    bot_frozen = False   # True after the bot terminates alone → it stops in
                         # place and waits for the user to respawn

    env_ids_bot = torch.tensor([0], dtype=torch.long, device=device)
    zero_vel_bot = torch.zeros((1, 6), device=device)
    # 0.5 s = 25 policy steps @ 50 Hz — print telemetry every N steps
    vx_print_interval = max(
        1, int(round(0.5 / (env_raw.cfg.sim.dt * env_raw.cfg.decimation)))
    )
    infer_times_ms: list[float] = []   # per-step bot inference cost (ms)

    # Optional front_cam MP4 writer — encodes at 50 fps so playback time = sim time
    front_writer = None
    if args.record_front:
        try:
            import imageio.v2 as imageio
        except ImportError:
            import imageio
        os.makedirs(os.path.dirname(args.record_front) or ".", exist_ok=True)
        front_writer = imageio.get_writer(args.record_front, fps=50)
        print(f"[race] front_cam MP4 → {args.record_front} (50 fps = sim-time)",
              flush=True)

    try:
        while (sim_app.is_running()
               and not _shutdown_requested
               and len(lap_times["user"]) < args.max_laps):
            # Capture the bot's pre-step pose. If it terminates this step we
            # warp it back here so it appears to "freeze in place" rather
            # than respawn at a random track location.
            pre_step_bot_state = env_raw._robot.data.root_state_w[0:1].clone()

            # ── Bot action ──
            if bot_frozen:
                # Zero command → no torque; suppress stuck/reverse counters
                # so env-internal termination doesn't keep firing and
                # respawning the bot every second.
                bot_action = torch.zeros((1, 2), device=device)
                env_raw._stuck_counter[0] = 0
                env_raw._reverse_counter[0] = 0
            elif bot_pp is not None:
                # Pure-pursuit bot (state-based, no NN forward)
                xy = env_raw._robot.data.root_pos_w[0:1, :2] \
                     - env_raw.scene.env_origins[0:1, :2]
                from hmclab_isaac.envs.racing.rl.single_agent.env import _yaw_from_quat
                yaw = _yaw_from_quat(env_raw._robot.data.root_quat_w[0:1])
                spd = env_raw._robot.data.root_lin_vel_b[0:1, 0]   # bot vx
                bot_action = bot_pp.step(xy, yaw, speed=spd)        # (1, 2)
            else:
                # Time the bot's actor inference (CNN+GRU forward)
                torch.cuda.synchronize()
                _t0 = time.perf_counter()
                with torch.no_grad():
                    _, _, bot_mean, bot_h = bot_actor.sample(
                        obs["images"][0:1], obs["policy"][0:1], bot_h,
                    )
                torch.cuda.synchronize()
                infer_times_ms.append((time.perf_counter() - _t0) * 1000.0)
                bot_action = bot_mean.clamp(-1.0, 1.0)   # (1, 2)

            # ── User action — joystick / keyboard ──
            thr, steer = ctrl.read()
            user_action = torch.tensor(
                [[max(0.0, thr), max(-1.0, min(1.0, steer))]],
                dtype=torch.float32, device=device,
            )

            actions = torch.cat([bot_action, user_action], dim=0)   # (2, 2)
            obs, reward, terminated, truncated, _ = env_raw.step(actions)
            step += 1

            # ── 0.5 s 마다 vx + inference telemetry ──
            if step % vx_print_interval == 0:
                vx = env_raw._robot.data.root_lin_vel_b[:, 0]
                v_bot = float(vx[0])
                v_user = float(vx[1])
                frozen_tag = " (frozen)" if bot_frozen else ""
                if infer_times_ms:
                    avg_inf = sum(infer_times_ms) / len(infer_times_ms)
                    max_inf = max(infer_times_ms)
                    infer_tag = f"  infer avg={avg_inf:.2f}ms max={max_inf:.2f}ms"
                    infer_times_ms.clear()
                else:
                    infer_tag = ""
                print(
                    f"[vx] t={step * 0.02:5.1f}s  "
                    f"bot={v_bot:+5.2f} m/s{frozen_tag}  "
                    f"user={v_user:+5.2f} m/s{infer_tag}",
                    flush=True,
                )

            # ── front_cam MP4 frame (128×128 obs → upscaled to 640×480) ──
            if front_writer is not None:
                img = obs["images"][1]                # (3, 128, 128) float [0, 1]
                img_hwc = (img.permute(1, 2, 0).clamp(0, 1).cpu().numpy()
                           * 255.0).astype(np.uint8)
                from PIL import Image
                img_pil = Image.fromarray(img_hwc).resize((640, 480), Image.BICUBIC)
                front_writer.append_data(np.array(img_pil))

            # While frozen, keep nailing the bot to its freeze pose every
            # step so any stray velocity from physics or a sneaky env reset
            # doesn't drift it.
            if bot_frozen:
                env_raw._robot.write_root_pose_to_sim(
                    pre_step_bot_state[:, :7], env_ids=env_ids_bot,
                )
                env_raw._robot.write_root_velocity_to_sim(
                    zero_vel_bot, env_ids=env_ids_bot,
                )

            # ── Per-env lap detection ──
            for i, label in enumerate(["bot", "user"]):
                lap_now = int(env_raw._lap_count[i].item())
                if lap_now > last_lap[i]:
                    lap_t = (step - lap_start_step[i]) * (
                        env_raw.cfg.sim.dt * env_raw.cfg.decimation
                    )
                    lap_times[label].append(lap_t)
                    bot_best = min(lap_times["bot"]) if lap_times["bot"] else None
                    usr_best = min(lap_times["user"]) if lap_times["user"] else None
                    print(
                        f"[{label:>4s}] LAP {lap_now}: {lap_t:.2f}s   "
                        f"| bot best {bot_best if bot_best is None else f'{bot_best:.2f}s'}  "
                        f"user best {usr_best if usr_best is None else f'{usr_best:.2f}s'}",
                        flush=True,
                    )
                    lap_start_step[i] = step
                    last_lap[i] = lap_now

            # ── Termination + spawn sync ──
            # When the USER's env terminated, env-internal step auto-reset
            # env 1 already (to a random progress). We override that: pick
            # one shared progress and re-reset BOTH env 0 and env 1 to the
            # same spot so the race always restarts side-by-side.
            user_done = bool(terminated[1]) or bool(truncated[1])
            bot_done_only = (bool(terminated[0]) or bool(truncated[0])) and not user_done
            if user_done:
                why = "stuck/spin/reverse" if float(reward[1]) < -1 else "timeout"
                print(f"[ user] env terminated ({why}) at step {step} — "
                      f"syncing bot respawn to same position", flush=True)
                p_new = float(torch.rand((), device="cpu").item())
                env_raw._forced_spawn_progress = torch.tensor(
                    [p_new, p_new], device=device,
                )
                env_raw._reset_idx(
                    torch.tensor([0, 1], dtype=torch.long, device=device)
                )
                env_raw._forced_spawn_progress = None
                if bot_actor is not None and bot_actor.use_gru and bot_h is not None:
                    bot_h = torch.zeros_like(bot_h)
                last_lap = [int(env_raw._lap_count[0].item()),
                            int(env_raw._lap_count[1].item())]
                lap_start_step = [step, step]
                bot_frozen = False   # Bot drives again on the new spawn
                # Refresh obs after the forced re-reset
                obs = env_raw._get_observations()
            elif bot_done_only and not bot_frozen:
                # Bot died alone — freeze it in place (at its pre-step pose)
                # and wait for the user to terminate, which is the only
                # event that respawns it. The env auto-reset already moved
                # the bot to a random spawn; we write_root_pose_to_sim back
                # to the death location and zero its velocity.
                why = "stuck/spin/reverse" if float(reward[0]) < -1 else "timeout"
                print(f"[  bot] env terminated ({why}) at step {step} — "
                      f"freezing in place until user respawn", flush=True)
                env_raw._robot.write_root_pose_to_sim(
                    pre_step_bot_state[:, :7], env_ids=env_ids_bot,
                )
                env_raw._robot.write_root_velocity_to_sim(
                    zero_vel_bot, env_ids=env_ids_bot,
                )
                bot_frozen = True
                last_lap[0] = int(env_raw._lap_count[0].item())
                lap_start_step[0] = step
                if bot_actor is not None and bot_actor.use_gru and bot_h is not None:
                    bot_h = torch.zeros_like(bot_h)

            # ── Chase cam follow user car (env 1) ──
            if chase_rec is not None and step % chase_stride == 0:
                art = env_raw._robot
                rp = art.data.root_pos_w[1]
                rq = art.data.root_quat_w[1]
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

    except KeyboardInterrupt:
        print("\n[race] KeyboardInterrupt — finalizing recordings…", flush=True)
    finally:
        if chase_rec is not None:
            try:
                saved = chase_rec.finalize()
                print(f"[race] chase cam saved: {saved}", flush=True)
            except Exception as e:
                print(f"[race] chase finalize failed: {e}", flush=True)
        if front_writer is not None:
            try:
                front_writer.close()
                print(f"[race] front_cam MP4 saved: {args.record_front}", flush=True)
            except Exception as e:
                print(f"[race] front mp4 close failed: {e}", flush=True)

    # ── Final comparison ──
    print("\n" + "=" * 70, flush=True)
    print(f"Race result on track {args.track}", flush=True)
    print("=" * 70, flush=True)
    for label in ("bot", "user"):
        lt = lap_times[label]
        if not lt:
            print(f"  {label:>5s}: no completed laps", flush=True)
        else:
            best = min(lt); mean = statistics.fmean(lt); med = statistics.median(lt)
            print(f"  {label:>5s}: {len(lt)} laps  best={best:.2f}s  mean={mean:.2f}s  "
                  f"median={med:.2f}s  laps={[f'{x:.2f}' for x in lt]}", flush=True)
    if lap_times["bot"] and lap_times["user"]:
        diff = min(lap_times["user"]) - min(lap_times["bot"])
        winner = "user" if diff < 0 else "bot"
        print(f"\n  → {winner.upper()} wins by {abs(diff):.2f}s on best lap", flush=True)
    print("=" * 70, flush=True)
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
