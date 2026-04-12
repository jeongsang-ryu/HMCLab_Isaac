"""Base DirectRLEnv for racing vehicles.

Provides the minimum pipeline every racing env (sim / solo / h2h) needs:
  - spawn the robot via a config-provided `ArticulationCfg`
  - spawn a `RacingTrack` circuit in the scene
  - flat LiDAR observation
  - progress + speed reward
  - off-track / flipped termination

Subclasses can extend observations (e.g. add opponent), rewards, and add
more actors. Everything lives in `_base.py` so solo and h2h share one
well-tested pipeline.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.ray_caster import MultiMeshRayCaster, MultiMeshRayCasterCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from hmclab_isaac.robots.racing.f1tenth_mid360 import (
    F1TENTH_MID360_CFG,
    LIDAR_MOUNT_OFFSET,
    MAX_STEER,
    WHEEL_RADIUS,
)
from hmclab_isaac.worlds.racing import RacingTrack, spawn_circuit


def default_track_path(name: str = "austin") -> str:
    """Return the packaged track file path by short name."""
    here = Path(__file__).resolve().parent.parent.parent
    p = here / "worlds" / "racing" / "_tracks_data" / f"{name}.txt"
    if not p.exists():
        raise FileNotFoundError(f"track {name} not found at {p}")
    return str(p)


@configclass
class F1TenthRacingBaseEnvCfg(DirectRLEnvCfg):
    """Minimal racing env cfg. Subclasses override `num_envs` and tweaks."""

    # DirectRLEnv required fields
    decimation: int = 2
    episode_length_s: float = 15.0

    sim: SimulationCfg = SimulationCfg(
        dt=0.005,
        render_interval=2,
        device="cuda:0",
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4,
        env_spacing=0.0,
        replicate_physics=True,
    )

    # Spaces
    action_space: int = 2  # [steer, speed]
    observation_space: int = 64  # num_lidar_bins + 1 (set in __post_init__)
    state_space: int = 0

    # Robot
    ego_cfg: ArticulationCfg = F1TENTH_MID360_CFG.replace(
        prim_path="/World/envs/env_.*/Ego"
    )

    # Track
    track_name: str = "austin"
    wall_height: float = 0.5

    # LiDAR (simple built-in planar pattern, kept cheap for M4 smoke)
    num_lidar_bins: int = 64
    lidar_max_range: float = 10.0

    # Control limits
    max_speed: float = 6.0  # m/s

    # Reward weights
    progress_weight: float = 10.0
    speed_weight: float = 1.0
    slow_penalty: float = -0.2
    min_speed_threshold: float = 0.5
    off_track_threshold: float = 2.0  # meters from centerline

    def __post_init__(self):
        self.observation_space = self.num_lidar_bins + 1


class F1TenthRacingBaseEnv(DirectRLEnv):
    """Minimum racing pipeline: single robot, LiDAR obs, progress reward."""

    cfg: F1TenthRacingBaseEnvCfg

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def __init__(
        self,
        cfg: F1TenthRacingBaseEnvCfg,
        render_mode: str | None = None,
        **kwargs,
    ):
        # Load track BEFORE super().__init__ so _setup_scene has it.
        self._track: RacingTrack = RacingTrack.load(default_track_path(cfg.track_name))
        super().__init__(cfg, render_mode, **kwargs)

        self._steer_ids = [
            i for i, n in enumerate(self.ego.joint_names) if "steering" in n
        ]
        self._drive_ids = [
            i for i, n in enumerate(self.ego.joint_names)
            if "rear" in n and "wheel" in n
        ]

        self._register_track_tensors()
        self._prev_progress = torch.zeros(self.num_envs, device=self.device)
        self._prev_actions = torch.zeros(self.num_envs, 2, device=self.device)

    def _register_track_tensors(self) -> None:
        xy = self._track.positions[:, :2]
        headings = self._track.rpy[:, 2]
        import numpy as np

        diffs = np.diff(xy, axis=0, append=xy[:1] if self._track.closed else xy[-1:])
        seg_len = np.linalg.norm(diffs, axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg_len[:-1])])
        self._centerline = torch.as_tensor(xy, device=self.device, dtype=torch.float32)
        self._cum_lengths = torch.as_tensor(cum, device=self.device, dtype=torch.float32)
        self._headings_t = torch.as_tensor(headings, device=self.device, dtype=torch.float32)
        self._total_length = float(self._cum_lengths[-1].item() + seg_len[-1])

    # ------------------------------------------------------------------
    # Scene
    # ------------------------------------------------------------------
    def _setup_scene(self):
        # Robot
        self.ego = Articulation(self.cfg.ego_cfg)

        # Track (global, shared across envs)
        spawn_circuit(
            self._track,
            prim_path="/World/Track",
            wall_height=self.cfg.wall_height,
        )

        # Ground plane
        sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())

        lidar_cfg = self._build_lidar_cfg()
        self.lidar = MultiMeshRayCaster(lidar_cfg)

        # Clone envs and register
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["ego"] = self.ego
        self.scene.sensors["lidar"] = self.lidar

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _build_lidar_cfg(self) -> MultiMeshRayCasterCfg:
        """Construct the LiDAR cfg. Subclasses override to add dynamic targets."""
        from isaaclab.sensors.ray_caster.patterns import LidarPatternCfg

        pattern = LidarPatternCfg(
            channels=1,
            vertical_fov_range=(0.0, 0.0),
            horizontal_fov_range=(-180.0, 180.0),
            horizontal_res=360.0 / self.cfg.num_lidar_bins,
        )
        return MultiMeshRayCasterCfg(
            prim_path="/World/envs/env_.*/Ego/base_link",
            update_period=0.0,
            offset=MultiMeshRayCasterCfg.OffsetCfg(pos=LIDAR_MOUNT_OFFSET),
            pattern_cfg=pattern,
            max_distance=self.cfg.lidar_max_range,
            mesh_prim_paths=self._lidar_mesh_targets(),
        )

    def _lidar_mesh_targets(self) -> list:
        """Static mesh paths the LiDAR sees. Override to extend."""
        return ["/World/Track/mesh", "/World/ground"]

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # EMA smoothing prevents jerky action jumps during early training.
        alpha = 0.3
        if not hasattr(self, "_smoothed"):
            self._smoothed = actions.clone()
        self._smoothed = alpha * actions + (1 - alpha) * self._smoothed
        self.actions = self._smoothed.clone()

    def _apply_action(self) -> None:
        self._apply_ego_action()

    def _apply_ego_action(self) -> None:
        n_j = self.ego.num_joints
        steer_cmd = self.actions[:, 0].clamp(-1.0, 1.0) * MAX_STEER
        speed_norm = (self.actions[:, 1].clamp(-1.0, 1.0) + 1.0) * 0.5
        wheel_vel = speed_norm * self.cfg.max_speed / WHEEL_RADIUS

        steer_t = torch.zeros(self.num_envs, n_j, device=self.device)
        vel_t = torch.zeros(self.num_envs, n_j, device=self.device)
        for idx in self._steer_ids:
            steer_t[:, idx] = steer_cmd
        for idx in self._drive_ids:
            vel_t[:, idx] = wheel_vel
        self.ego.set_joint_position_target(steer_t)
        self.ego.set_joint_velocity_target(vel_t)

    # ------------------------------------------------------------------
    # Observations / rewards / dones
    # ------------------------------------------------------------------
    def _get_observations(self) -> dict:
        lidar_hits_w = self.lidar.data.ray_hits_w  # (B, N, 3)
        ego_pos = self.ego.data.root_pos_w  # (B, 3)
        dist = torch.norm(lidar_hits_w - ego_pos.unsqueeze(1), dim=-1)
        dist = torch.nan_to_num(dist, nan=self.cfg.lidar_max_range, posinf=self.cfg.lidar_max_range)
        dist = dist.clamp(0.0, self.cfg.lidar_max_range) / self.cfg.lidar_max_range

        # Pad/truncate to num_lidar_bins in case the pattern produced a
        # slightly different ray count (rounding on horizontal_res).
        target = self.cfg.num_lidar_bins
        if dist.shape[1] != target:
            if dist.shape[1] > target:
                dist = dist[:, :target]
            else:
                pad = torch.ones(
                    self.num_envs, target - dist.shape[1], device=self.device
                )
                dist = torch.cat([dist, pad], dim=-1)

        speed = torch.norm(self.ego.data.root_lin_vel_w[:, :2], dim=-1, keepdim=True)
        speed_n = (speed / self.cfg.max_speed).clamp(0.0, 2.0)
        policy = torch.cat([dist, speed_n], dim=-1)  # (B, bins + 1)
        return {"policy": policy}

    def _get_rewards(self) -> torch.Tensor:
        ego_xy = self.ego.data.root_pos_w[:, :2]
        diffs = ego_xy.unsqueeze(1) - self._centerline.unsqueeze(0)
        nearest = torch.norm(diffs, dim=-1).argmin(dim=-1)
        progress = self._cum_lengths[nearest]

        delta = progress - self._prev_progress
        # Wrap around for closed tracks
        half = self._total_length / 2
        delta = torch.where(delta < -half, delta + self._total_length, delta)
        delta = torch.where(delta > half, delta - self._total_length, delta)
        self._prev_progress = progress

        speed = torch.norm(self.ego.data.root_lin_vel_w[:, :2], dim=-1)
        speed_n = (speed / self.cfg.max_speed).clamp(0.0, 1.0)

        progress_r = self.cfg.progress_weight * delta.clamp(-1.0, 2.0)
        speed_r = self.cfg.speed_weight * speed_n
        slow_r = self.cfg.slow_penalty * (speed < self.cfg.min_speed_threshold).float()

        total = progress_r + speed_r + slow_r

        self.extras["log"] = {
            "reward/progress": progress_r.mean().item(),
            "reward/speed": speed_r.mean().item(),
            "reward/slow": slow_r.mean().item(),
            "reward/total": total.mean().item(),
            "info/ego_speed": speed.mean().item(),
        }
        self._prev_actions = self.actions.clone()
        return total

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        root_pos = self.ego.data.root_pos_w
        root_quat = self.ego.data.root_quat_w

        fallen = root_pos[:, 2] < -0.1

        w, x, y, z = root_quat[:, 0], root_quat[:, 1], root_quat[:, 2], root_quat[:, 3]
        up_z = 1.0 - 2.0 * (x * x + y * y)
        flipped = up_z < 0.5

        diffs = root_pos[:, :2].unsqueeze(1) - self._centerline.unsqueeze(0)
        min_dist = torch.norm(diffs, dim=-1).min(dim=-1).values
        off = min_dist > self.cfg.off_track_threshold

        terminated = fallen | flipped | off
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.ego._ALL_INDICES
        super()._reset_idx(env_ids)
        self._reset_ego(env_ids)

    # ------------------------------------------------------------------
    # Helpers (reusable by subclasses)
    # ------------------------------------------------------------------
    def _reset_ego(self, env_ids) -> None:
        n_reset = len(env_ids)
        n_pts = self._centerline.shape[0]
        rand_idx = torch.randint(0, n_pts, (n_reset,), device=self.device)
        xy = self._centerline[rand_idx]
        heading = self._headings_t[rand_idx]

        state = self.ego.data.default_root_state[env_ids].clone()
        state[:, 0] = xy[:, 0] + self.scene.env_origins[env_ids, 0]
        state[:, 1] = xy[:, 1] + self.scene.env_origins[env_ids, 1]
        state[:, 2] = 0.05
        state[:, 3] = torch.cos(heading / 2)  # qw
        state[:, 4:6] = 0.0                    # qx, qy
        state[:, 6] = torch.sin(heading / 2)   # qz
        state[:, 7:] = 0.0
        self.ego.write_root_pose_to_sim(state[:, :7], env_ids)
        self.ego.write_root_velocity_to_sim(state[:, 7:], env_ids)
        self.ego.write_joint_state_to_sim(
            self.ego.data.default_joint_pos[env_ids],
            torch.zeros_like(self.ego.data.default_joint_pos[env_ids]),
            None,
            env_ids,
        )
        self._prev_actions[env_ids] = 0.0
        self._prev_progress[env_ids] = self._cum_lengths[rand_idx]
