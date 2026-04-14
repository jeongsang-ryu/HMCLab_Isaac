"""Run MuSHR under pure-pursuit and log per-wheel velocity targets + chassis pitch.

Output: Itutorial/outputs/wheel_log.png (3 subplots)
"""

from __future__ import annotations
import argparse, math, os, sys
import numpy as np
import torch

from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--robot", type=str, default="mushr")
parser.add_argument("--target", type=float, default=4.0)
parser.add_argument("--steps", type=int, default=300)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True; args.enable_cameras = False
launcher = AppLauncher(args); simulation_app = launcher.app

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from centerline_env import CenterlineEnv, CenterlineEnvCfg  # noqa
from _common import pure_pursuit_steer  # noqa


def main() -> int:
    cfg = CenterlineEnvCfg()
    cfg.scene.num_envs = 1
    cfg.robot_name = args.robot
    cfg.track_name = "mini_oval_flat"
    env = CenterlineEnv(cfg)
    env.reset()

    names = list(env.ego.joint_names)
    print(f"[log] joints: {names}", flush=True)
    # Find wheel joints + chassis body for pitch
    def fidx(sub): return next((i for i, n in enumerate(names) if sub in n), -1)
    fl = fidx("front_left_wheel_throttle")
    fr = fidx("front_right_wheel_throttle")
    bl = fidx("back_left_wheel_throttle")
    br = fidx("back_right_wheel_throttle")
    print(f"[log] wheel idx FL={fl} FR={fr} BL={bl} BR={br}", flush=True)

    # Logs
    T = args.steps
    t_arr = np.zeros(T)
    vel_target = np.zeros((T, 4))   # commanded wheel angular vel
    vel_actual = np.zeros((T, 4))   # actual wheel angular vel
    pitch = np.zeros(T)
    z_height = np.zeros(T)
    speed_body = np.zeros(T)

    # Closed-loop PP
    centerline = env._centerline
    wheelbase = env._spec.wheelbase
    max_steer = env._spec.max_steer
    throttle_norm = 2.0 * (args.target / env.cfg.max_speed) - 1.0

    for step in range(T):
        pos = env.ego.data.root_pos_w[:, :2]
        q = env.ego.data.root_quat_w
        w_, x_, y_, z_ = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        yaw = torch.atan2(2*(w_*z_ + x_*y_), 1 - 2*(y_*y_ + z_*z_))
        v = env.ego.data.root_lin_vel_w[:, :2]
        steer_rad = pure_pursuit_steer(
            pos, yaw, centerline, lookahead=0.6, wheelbase=wheelbase,
            speed=v.norm(dim=-1),
            lookahead_gain=0.25, lookahead_min=0.4, lookahead_max=2.5,
        )
        steer_norm = (steer_rad / max_steer).clamp(-1.0, 1.0)
        action = torch.stack([steer_norm,
                              torch.full_like(steer_norm, throttle_norm)], dim=-1)
        env.step(action)

        # Actual wheel angular vel (rad/s) — from articulation joint_vel
        jv = env.ego.data.joint_vel[0]    # (num_joints,)
        # Commanded wheel vel (our per-wheel ackermann)
        # Recompute to match _apply_action_4wd_ackermann
        speed = (steer_norm[0] * 0.0 + (throttle_norm + 1) * 0.5 * cfg.max_speed)  # target speed in m/s
        from math import tan, atan, sqrt
        L = env._spec.wheelbase; W = cfg.track_width; Rw = env._spec.wheel_radius
        s = float(steer_rad[0].item())
        if abs(s) < 1e-3:
            v_rl = v_rr = v_fl_ = v_fr_ = float(speed) / Rw
        else:
            R = L / tan(s)
            rl_arm = R - W/2; rr_arm = R + W/2
            v_rl = float(speed) * abs(rl_arm / R) / Rw
            v_rr = float(speed) * abs(rr_arm / R) / Rw
            v_fl_ = float(speed) * sqrt(rl_arm**2 + L**2) / abs(R) / Rw
            v_fr_ = float(speed) * sqrt(rr_arm**2 + L**2) / abs(R) / Rw

        t_arr[step] = step * cfg.sim.dt * cfg.decimation
        vel_target[step] = [v_fl_, v_fr_, v_rl, v_rr]
        vel_actual[step] = [float(jv[fl]), float(jv[fr]),
                            float(jv[bl]), float(jv[br])]
        # chassis pitch (rotation about body Y axis)
        wx, xx, yy, zz = float(q[0,0]), float(q[0,1]), float(q[0,2]), float(q[0,3])
        pitch[step] = math.degrees(math.asin(max(-1, min(1, 2*(wx*yy - zz*xx)))))
        z_height[step] = float(env.ego.data.root_pos_w[0, 2].item())
        speed_body[step] = float(v.norm(dim=-1)[0].item())

    # Plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(4, 1, figsize=(10, 11), sharex=True)
    names_wheel = ["FL", "FR", "BL", "BR"]
    for i in range(4):
        axs[0].plot(t_arr, vel_target[:, i], label=f"{names_wheel[i]} cmd", linestyle="--", alpha=0.7)
        axs[0].plot(t_arr, vel_actual[:, i], label=f"{names_wheel[i]} actual")
    axs[0].set_ylabel("wheel ω (rad/s)")
    axs[0].legend(ncol=4, fontsize=8); axs[0].grid(alpha=0.3)
    axs[0].set_title(f"{args.robot}  target_speed={args.target} m/s  (dashed=cmd, solid=actual)")

    axs[1].plot(t_arr, pitch, color="tab:red")
    axs[1].set_ylabel("chassis pitch (°)")
    axs[1].axhline(0, color="k", lw=0.5); axs[1].grid(alpha=0.3)

    axs[2].plot(t_arr, z_height, color="tab:purple")
    axs[2].set_ylabel("chassis z (m)")
    axs[2].grid(alpha=0.3)

    axs[3].plot(t_arr, speed_body, color="tab:green")
    axs[3].set_ylabel("body speed (m/s)"); axs[3].set_xlabel("time (s)")
    axs[3].grid(alpha=0.3)

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"wheel_log_{args.robot}.png")
    plt.tight_layout(); plt.savefig(out_path, dpi=100)
    print(f"LOG_OK saved {out_path}", flush=True)

    env.close()
    return 0


if __name__ == "__main__":
    import traceback
    try: main()
    except Exception: traceback.print_exc()
    os._exit(0)
