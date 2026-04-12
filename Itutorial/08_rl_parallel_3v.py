"""Tutorial 08 — parallel RL envs with 3+ vehicles per env (mini oval + duct).

Supports --robot f1tenth|mushr and --num_vehicles N. Each cloned env
contains N vehicles at distinct prim paths. All are batch-controlled per
step with slightly different (steer, speed).

Run:
    OMNI_KIT_ACCEPT_EULA=YES python Itutorial/08_rl_parallel_3v.py --robot mushr --num_envs 3 --num_vehicles 3
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
parser.add_argument("--num_envs", type=int, default=3)
parser.add_argument("--num_vehicles", type=int, default=3)
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
N_VEHICLES = int(args.num_vehicles)


@configclass
class TutEnvCfg(DirectRLEnvCfg):
    decimation: int = 2
    episode_length_s: float = 15.0
    sim: SimulationCfg = SimulationCfg(dt=0.005, render_interval=2, device="cuda:0")
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=3, env_spacing=0.0, replicate_physics=True
    )
    action_space: int = 2
    observation_space: int = 4
    state_space: int = 0
    track_name: str = DEFAULT_TRACK
    robot_name: str = "f1tenth"
    _camera_params: tuple = (0.0, 0.0, 40.0)

    def __post_init__(self):
        self.observation_space = 4 * N_VEHICLES


class TutEnv(DirectRLEnv):
    cfg: TutEnvCfg

    def __init__(self, cfg, render_mode=None, **kwargs):
        self._spec = get_robot_spec(cfg.robot_name)
        self._track = RacingTrack.load(default_track_path(cfg.track_name))
        super().__init__(cfg, render_mode, **kwargs)
        self._joint_ids = [self._spec.resolve_joints(v.joint_names) for v in self.vehicles]

    def _setup_scene(self):
        self.vehicles: list[Articulation] = []
        for i in range(N_VEHICLES):
            c = self._spec.cfg.replace(prim_path=f"/World/envs/env_.*/Vehicle_{i}")
            c = c.replace(init_state=c.init_state.replace(pos=(0.0, 0.8 * i - 0.4, self._spec.init_z)))
            self.vehicles.append(Articulation(c))

        sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
        spawn_duct_track(
            self._track, prim_path="/World/Track",
            pipe_radius=0.15, pipe_offset=1.1, rib_spacing=0.8,
        )

        cx, cy, span = self.cfg._camera_params
        self.cam_td = Camera(
            top_down_camera_cfg("/World/TopDownCam", center=(cx, cy), size=span,
                                width=1280, height=720)
        )
        self.cam_ch = Camera(chase_camera_cfg("/World/ChaseCam", width=1280, height=720))

        self.scene.clone_environments(copy_from_source=False)
        for i, art in enumerate(self.vehicles):
            self.scene.articulations[f"vehicle_{i}"] = art

        sim_utils.DomeLightCfg(intensity=2000.0).func(
            "/World/Light", sim_utils.DomeLightCfg(intensity=2000.0)
        )

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._actions = actions

    def _apply_action(self) -> None:
        for v_i, art in enumerate(self.vehicles):
            speed = 2.5 + 0.4 * v_i
            steer = 0.15 + 0.04 * (v_i - (N_VEHICLES - 1) / 2.0)
            apply_ackermann(
                art, self._spec, steer_rad=steer, speed_mps=speed,
                steer_ids=self._joint_ids[v_i][0],
                drive_ids=self._joint_ids[v_i][1],
            )

    def _get_observations(self) -> dict:
        feats = []
        for art in self.vehicles:
            p = art.data.root_pos_w[:, :2]
            v = art.data.root_lin_vel_w[:, :2]
            feats.append(torch.cat([p, v], dim=-1))
        return {"policy": torch.cat(feats, dim=-1)}

    def _get_rewards(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, device=self.device)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        term = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        trunc = self.episode_length_buf >= self.max_episode_length - 1
        return term, trunc

    def _reset_idx(self, env_ids: Sequence[int] | None) -> None:
        super()._reset_idx(env_ids)
        if env_ids is None:
            env_ids = self.vehicles[0]._ALL_INDICES
        n_pts = self._track.num_points
        for v_i, art in enumerate(self.vehicles):
            prog = (0.05 * v_i) % 1.0
            pos, rpy = self._track.spawn_pose(progress=prog)
            yaw = float(rpy[2])
            state = art.data.default_root_state[env_ids].clone()
            state[:, 0] = float(pos[0]) + self.scene.env_origins[env_ids, 0]
            state[:, 1] = float(pos[1]) + self.scene.env_origins[env_ids, 1]
            state[:, 2] = self._spec.init_z
            state[:, 3] = math.cos(yaw / 2)
            state[:, 4:6] = 0.0
            state[:, 6] = math.sin(yaw / 2)
            state[:, 7:] = 0.0
            art.write_root_pose_to_sim(state[:, :7], env_ids)
            art.write_root_velocity_to_sim(state[:, 7:], env_ids)
            art.write_joint_state_to_sim(
                art.data.default_joint_pos[env_ids],
                torch.zeros_like(art.data.default_joint_pos[env_ids]),
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
        return
    Image.fromarray(frames[-1]).save(os.path.join(OUT_DIR, f"{tag}.png"))
    if len(frames) >= 2:
        try:
            import imageio.v2 as imageio
            imageio.mimsave(
                os.path.join(OUT_DIR, f"{tag}.mp4"),
                frames, fps=30, codec="libx264", macro_block_size=None,
            )
        except Exception as exc:
            print(f"[capture] mp4 failed: {exc}", flush=True)
    print(f"[capture] saved {tag}", flush=True)


def _update_chase(cam, art):
    p = art.data.root_pos_w[0]
    q = art.data.root_quat_w[0]
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    px, py, pz = float(p[0]), float(p[1]), float(p[2])
    eye = (px - 2.5 * math.cos(yaw), py - 2.5 * math.sin(yaw), pz + 1.4)
    target = (px, py, pz + 0.2)
    eyes_t = torch.tensor([eye], device=cam._device, dtype=torch.float32)
    targets_t = torch.tensor([target], device=cam._device, dtype=torch.float32)
    try:
        cam.set_world_poses_from_view(eyes_t, targets_t)
    except Exception as exc:
        print(f"[chase] failed: {exc}", flush=True)


def main() -> int:
    spec = get_robot_spec(args.robot)
    track_name = args.track or DEFAULT_TRACK

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
        f">>> env created — num_envs={env.num_envs} "
        f"vehicles/env={N_VEHICLES} total_actors={env.num_envs*N_VEHICLES} "
        f"robot={spec.name}",
        flush=True,
    )
    env.reset()

    frames_td, frames_ch = [], []
    actions = torch.zeros(env.num_envs, 2, device=env.device)

    for step in range(args.steps):
        env.step(actions)
        if step % 2 == 0:
            _update_chase(env.cam_ch, env.vehicles[0])
            _grab(env.cam_td, frames_td)
            _grab(env.cam_ch, frames_ch)
        if step % 50 == 0:
            speeds = [
                round(v.data.root_lin_vel_w[:, :2].norm(dim=-1).mean().item(), 2)
                for v in env.vehicles
            ]
            print(f"    step={step} speeds={speeds}", flush=True)

    _save(frames_td, f"tut08_rl_3v_{spec.name}_topdown")
    _save(frames_ch, f"tut08_rl_3v_{spec.name}_chase")
    print(f"TUT08_OK robot={spec.name}", flush=True)
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
