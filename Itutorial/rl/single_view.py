"""Single-vehicle demonstration — pure pursuit drive, chase cam capture."""
from __future__ import annotations
import argparse, math, os, sys
import numpy as np
import torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--robot", type=str, default="mushr")
parser.add_argument("--target", type=float, default=3.0)
parser.add_argument("--steps", type=int, default=500)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True; args.enable_cameras = True
launcher = AppLauncher(args); simulation_app = launcher.app

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaaclab.sensors import Camera  # noqa
from centerline_env import CenterlineEnv, CenterlineEnvCfg  # noqa
from _common import pure_pursuit_steer  # noqa
from hmclab_isaac.utils.capture import chase_camera_cfg, top_down_camera_cfg  # noqa
from hmclab_isaac.envs.racing._base import default_track_path
from hmclab_isaac.worlds.racing import RacingTrack


def main() -> int:
    cfg = CenterlineEnvCfg()
    cfg.scene.num_envs = 1
    cfg.robot_name = args.robot
    cfg.track_name = "mini_oval_flat"

    tr = RacingTrack.load(default_track_path(cfg.track_name))
    xs, ys = tr.positions[:, 0], tr.positions[:, 1]
    cx = float((xs.min()+xs.max())/2); cy = float((ys.min()+ys.max())/2)
    span = float(max(xs.max()-xs.min(), ys.max()-ys.min())) * 1.3 + 2.0

    class EnvCam(CenterlineEnv):
        def _setup_scene(self):
            super()._setup_scene()
            self.cam_td = Camera(top_down_camera_cfg("/World/TopDownCam",
                center=(cx, cy), size=span, width=1280, height=720))
            self.cam_ch = Camera(chase_camera_cfg("/World/ChaseCam",
                width=1280, height=720))
    env = EnvCam(cfg)
    env.reset()

    wheelbase = env._spec.wheelbase
    max_steer = env._spec.max_steer
    max_speed_cfg = env.cfg.max_speed
    throttle_norm = 2.0 * (args.target / max_speed_cfg) - 1.0
    centerline = env._centerline

    frames_td, frames_ch = [], []

    def grab(cam, frames):
        try: cam.update(dt=0.0, force_recompute=True)
        except Exception: return
        out = getattr(cam.data, "output", None); rgb = out.get("rgb") if out else None
        if rgb is None: return
        arr = rgb.detach().cpu().numpy() if hasattr(rgb, "detach") else np.asarray(rgb)
        if arr.ndim == 4: arr = arr[0]
        if arr.shape[-1] == 4: arr = arr[..., :3]
        if arr.dtype != np.uint8: arr = np.clip(arr, 0, 255).astype(np.uint8)
        frames.append(arr)

    def chase(cam, ego):
        p = ego.data.root_pos_w[0]; q = ego.data.root_quat_w[0]
        w_, x_, y_, z_ = float(q[0]), float(q[1]), float(q[2]), float(q[3])
        yaw_ = math.atan2(2*(w_*z_ + x_*y_), 1 - 2*(y_*y_ + z_*z_))
        px, py, pz = float(p[0]), float(p[1]), float(p[2])
        eye = (px - 2.5*math.cos(yaw_), py - 2.5*math.sin(yaw_), pz + 1.4)
        tgt = (px, py, pz + 0.2)
        try: cam.set_world_poses_from_view(
            torch.tensor([eye], device=cam._device, dtype=torch.float32),
            torch.tensor([tgt], device=cam._device, dtype=torch.float32))
        except Exception: pass

    for step in range(args.steps):
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
        action = torch.stack([steer_norm, torch.full_like(steer_norm, throttle_norm)], dim=-1)
        env.step(action)
        if step % 2 == 0:
            chase(env.cam_ch, env.ego)
            grab(env.cam_td, frames_td)
            grab(env.cam_ch, frames_ch)
        if step % 50 == 0:
            p = env.ego.data.root_pos_w[0]; sp = v.norm(dim=-1)[0].item()
            print(f"  step={step} pos=({p[0].item():.2f},{p[1].item():.2f},{p[2].item():.2f}) v={sp:.2f}", flush=True)

    import imageio.v2 as imageio
    OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
    os.makedirs(OUT, exist_ok=True)
    tag = f"single_{args.robot}_v{args.target:.1f}"
    if frames_td: imageio.mimsave(os.path.join(OUT, f"{tag}_topdown.mp4"), frames_td, fps=30, codec="libx264", macro_block_size=None)
    if frames_ch: imageio.mimsave(os.path.join(OUT, f"{tag}_chase.mp4"),   frames_ch, fps=30, codec="libx264", macro_block_size=None)
    print(f"SINGLE_OK saved tag={tag}", flush=True)
    env.close()
    return 0


if __name__ == "__main__":
    import traceback
    try: main()
    except Exception: traceback.print_exc()
    os._exit(0)
