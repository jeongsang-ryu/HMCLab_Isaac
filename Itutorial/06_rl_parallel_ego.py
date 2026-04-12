"""Tutorial 06 — parallel RL envs with ego only (mini banked oval, duct).

Supports --robot f1tenth|mushr. DirectRLEnv with N cloned envs, each
containing one vehicle. Top-down + chase cam (tracking env 0's ego).

Run:
    OMNI_KIT_ACCEPT_EULA=YES python Itutorial/06_rl_parallel_ego.py --robot f1tenth --num_envs 4
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from typing import Sequence

import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=250)
parser.add_argument("--robot", type=str, default="f1tenth")
parser.add_argument("--track", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True
launcher = AppLauncher(args)
simulation_app = launcher.app

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg  # noqa: E402
from isaaclab.scene import InteractiveSceneCfg  # noqa: E402
from isaaclab.sensors import Camera  # noqa: E402
from isaaclab.sim import SimulationCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

from _common import DEFAULT_TRACK, apply_ackermann, get_robot_spec  # noqa: E402
from hmclab_isaac.envs.racing._base import default_track_path  # noqa: E402
from hmclab_isaac.utils.capture import (  # noqa: E402
    chase_camera_cfg,
    top_down_camera_cfg,
)
from hmclab_isaac.worlds.racing import RacingTrack  # noqa: E402
from hmclab_isaac.worlds.racing.duct_track import spawn_duct_track  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
TAG_TD = "tut06_rl_parallel_{robot}_topdown"
TAG_CH = "tut06_rl_parallel_{robot}_chase"


@configclass
class TutEnvCfg(DirectRLEnvCfg):
    decimation: int = 2
    episode_length_s: float = 15.0
    sim: SimulationCfg = SimulationCfg(dt=0.005, render_interval=2, device="cuda:0")
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4, env_spacing=0.0, replicate_physics=True
    )
    action_space: int = 2
    observation_space: int = 4
    state_space: int = 0
    track_name: str = DEFAULT_TRACK
    robot_name: str = "f1tenth"
    _camera_params: tuple = (0.0, 0.0, 40.0)


class TutEnv(DirectRLEnv):
    cfg: TutEnvCfg

    def __init__(self, cfg, render_mode=None, **kwargs):
        self._spec = get_robot_spec(cfg.robot_name)
        self._track = RacingTrack.load(default_track_path(cfg.track_name))
        super().__init__(cfg, render_mode, **kwargs)
        self._steer_ids, self._drive_ids = self._spec.resolve_joints(self.ego.joint_names)

    def _setup_scene(self):
        ego_cfg = self._spec.cfg.replace(prim_path="/World/envs/env_.*/Ego")
        self.ego = Articulation(ego_cfg)

        sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
        spawn_duct_track(
            self._track, prim_path="/World/Track",
            pipe_radius=0.15, pipe_offset=1.1, rib_spacing=0.8,
        )

        cx, cy, span = self.cfg._camera_params
        self.cam_td = Camera(
            top_down_camera_cfg(
                "/World/TopDownCam", center=(cx, cy), size=span,
                width=1280, height=720,
            )
        )
        self.cam_ch = Camera(
            chase_camera_cfg("/World/ChaseCam", width=1280, height=720)
        )

        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["ego"] = self.ego

        sim_utils.DomeLightCfg(intensity=2000.0).func(
            "/World/Light", sim_utils.DomeLightCfg(intensity=2000.0)
        )

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._actions = actions

    def _apply_action(self) -> None:
        steer = self._actions[:, 0] * self._spec.max_steer
        speed = (self._actions[:, 1].clamp(-1.0, 1.0) + 1.0) * 0.5 * 4.0
        apply_ackermann(
            self.ego, self._spec, steer_rad=steer, speed_mps=speed,
            steer_ids=self._steer_ids, drive_ids=self._drive_ids,
        )

    def _get_observations(self) -> dict:
        p = self.ego.data.root_pos_w[:, :2]
        v = self.ego.data.root_lin_vel_w[:, :2]
        return {"policy": torch.cat([p, v], dim=-1)}

    def _get_rewards(self) -> torch.Tensor:
        return self.ego.data.root_lin_vel_w[:, :2].norm(dim=-1)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        term = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        trunc = self.episode_length_buf >= self.max_episode_length - 1
        return term, trunc

    def _reset_idx(self, env_ids: Sequence[int] | None) -> None:
        super()._reset_idx(env_ids)
        if env_ids is None:
            env_ids = self.ego._ALL_INDICES
        n_reset = len(env_ids)
        n_pts = self._track.num_points
        pos, rpy = self._track.spawn_pose(progress=0.0)
        yaw = float(rpy[2])
        state = self.ego.data.default_root_state[env_ids].clone()
        state[:, 0] = float(pos[0]) + self.scene.env_origins[env_ids, 0]
        state[:, 1] = float(pos[1]) + self.scene.env_origins[env_ids, 1]
        state[:, 2] = self._spec.init_z
        state[:, 3] = math.cos(yaw / 2)
        state[:, 4:6] = 0.0
        state[:, 6] = math.sin(yaw / 2)
        state[:, 7:] = 0.0
        self.ego.write_root_pose_to_sim(state[:, :7], env_ids)
        self.ego.write_root_velocity_to_sim(state[:, 7:], env_ids)
        self.ego.write_joint_state_to_sim(
            self.ego.data.default_joint_pos[env_ids],
            torch.zeros_like(self.ego.data.default_joint_pos[env_ids]),
            None, env_ids,
        )


def _grab(cam, frames):
    import numpy as np
    try:
        cam.update(dt=0.0, force_recompute=True)
    except Exception:
        return
    out = getattr(cam.data, "output", None)
    rgb = out.get("rgb") if out else None
    if rgb is None:
        return
    arr = rgb.detach().cpu().numpy() if hasattr(rgb, "detach") else np.asarray(rgb)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    frames.append(arr)


def _save(frames, tag):
    from PIL import Image
    os.makedirs(OUT_DIR, exist_ok=True)
    if not frames:
        return {}
    saved = {}
    png = os.path.join(OUT_DIR, f"{tag}.png")
    Image.fromarray(frames[-1]).save(png)
    saved["png"] = png
    if len(frames) >= 2:
        try:
            import imageio.v2 as imageio
            mp4 = os.path.join(OUT_DIR, f"{tag}.mp4")
            imageio.mimsave(mp4, frames, fps=30, codec="libx264", macro_block_size=None)
            saved["mp4"] = mp4
        except Exception as exc:
            print(f"[capture] mp4 failed: {exc}", flush=True)
    print(f"[capture] saved {tag}: {saved}", flush=True)
    return saved


def _update_chase(cam, ego, behind=2.5, above=1.4, look_up=0.2) -> None:
    # Follow env 0's ego
    p = ego.data.root_pos_w[0]
    q = ego.data.root_quat_w[0]
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    px, py, pz = float(p[0]), float(p[1]), float(p[2])
    eye = (px - behind * math.cos(yaw), py - behind * math.sin(yaw), pz + above)
    target = (px, py, pz + look_up)
    eyes_t = torch.tensor([eye], device=cam._device, dtype=torch.float32)
    targets_t = torch.tensor([target], device=cam._device, dtype=torch.float32)
    try:
        cam.set_world_poses_from_view(eyes_t, targets_t)
    except Exception as exc:
        print(f"[chase] set_world_poses failed: {exc}", flush=True)


def main() -> int:
    spec = get_robot_spec(args.robot)
    track_name = args.track or DEFAULT_TRACK
    tag_td = TAG_TD.format(robot=spec.name)
    tag_ch = TAG_CH.format(robot=spec.name)

    track = RacingTrack.load(default_track_path(track_name))
    xs, ys = track.positions[:, 0], track.positions[:, 1]
    cx = float((xs.min() + xs.max()) / 2)
    cy = float((ys.min() + ys.max()) / 2)
    span = float(max(xs.max() - xs.min(), ys.max() - ys.min())) * 1.2 + 2.0

    cfg = TutEnvCfg()
    cfg.scene.num_envs = int(args.num_envs)
    cfg.robot_name = args.robot
    cfg.track_name = track_name
    cfg._camera_params = (cx, cy, span)

    env = TutEnv(cfg)
    print(
        f">>> env created — num_envs={env.num_envs} robot={spec.name} "
        f"track={track_name}",
        flush=True,
    )
    obs, info = env.reset()
    print(f">>> reset OK — obs={tuple(obs['policy'].shape)}", flush=True)

    frames_td, frames_ch = [], []
    actions = torch.zeros(env.num_envs, 2, device=env.device)
    actions[:, 0] = 0.2  # right steer
    actions[:, 1] = 0.4  # ~70% throttle

    for step in range(args.steps):
        env.step(actions)
        if step % 2 == 0:
            _update_chase(env.cam_ch, env.ego)
            _grab(env.cam_td, frames_td)
            _grab(env.cam_ch, frames_ch)
        if step % 50 == 0:
            speed = env.ego.data.root_lin_vel_w[:, :2].norm(dim=-1).mean().item()
            print(f"    step={step} mean_speed={speed:.2f}", flush=True)

    _save(frames_td, tag_td)
    _save(frames_ch, tag_ch)
    print(f"TUT06_OK robot={spec.name}", flush=True)
    env.close()
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
