"""S-curve RL environment for MuSHR with 3D LiDAR (DirectRLEnv).

Self-contained, minimal RL env used by `train.py` + `eval.py`.

Design choices:
- **Robot**: MuSHR Nano v2 (RWD, chassis box collider, high-friction ground).
- **Track**: `mini_s_curve` (double-sine S-curve with modest banking).
- **Sensor**: single 3D LiDAR attached to `laser_link`, 4 vertical channels
  × 64 azimuth bins. Observation is the min-per-azimuth distance vector
  (64 dims), so the policy sees a 2.5D depth horizon without paying for
  the full 3D point cloud.
- **Observation** (67 dims):
    - 64 normalised LiDAR distances ∈ [0, 1]
    - 1 current speed / max_speed
    - 2 previous action (steer, throttle)
- **Action** (2 dims, ∈ [-1, 1]):
    - steer_norm — multiplied by spec.max_steer
    - throttle_norm — mapped to (0 .. max_speed) via ((x + 1) / 2)
- **Reward**:
    - progress_weight × centerline_progress_delta
    - speed_weight × forward_speed / max_speed
    - action_smoothness × |Δaction|
    - -off_track_penalty on excursions (cte > off_track_threshold)
    - -wall_proximity_penalty when min LiDAR dist < wall_danger_dist
    - +alive_bonus per step that isn't terminal
- **Termination**:
    - off-track: min distance to centerline > terminate_cte
    - flipped: up-axis z < 0.35
    - truncate at episode_length_s

Fit-for-purpose: this is not meant to be a general racing env — rewards
are tuned to favour "fast + smooth" driving on this single S-curve.
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import MISSING
from typing import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.ray_caster import MultiMeshRayCaster, MultiMeshRayCasterCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

# Tutorial package isn't importable; add its dir to sys.path for _common.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_THIS_DIR))  # Itutorial/

from _common import (  # noqa: E402
    apply_ackermann,
    apply_high_friction_to_mesh,
    get_robot_spec,
    post_spawn_fix,
    spawn_high_friction_ground,
)
from hmclab_isaac.envs.racing._base import default_track_path  # noqa: E402
from hmclab_isaac.worlds.racing import RacingTrack, spawn_circuit  # noqa: E402
from hmclab_isaac.worlds.racing.duct_track import spawn_duct_track  # noqa: E402


@configclass
class SCurveEnvCfg(DirectRLEnvCfg):
    # ---- DirectRLEnv required ----
    decimation: int = 2
    episode_length_s: float = 10.0
    sim: SimulationCfg = SimulationCfg(
        dt=0.005, render_interval=2, device="cuda:0"
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=32, env_spacing=0.0, replicate_physics=True
    )
    action_space: int = 2
    observation_space: int = 64 + 3
    state_space: int = 0

    # ---- Track ----
    track_name: str = "mini_oval_flat"
    # ---- Robot (f1tenth | mushr | srx8) ----
    robot_name: str = "f1tenth"

    # ---- LiDAR ----
    num_lidar_channels: int = 4
    num_lidar_bins: int = 64
    lidar_vfov_deg: tuple = (-8.0, 8.0)
    lidar_max_range: float = 8.0

    # ---- Vehicle limits ----
    max_speed: float = 8.0    # m/s

    # ---- Reward weights ----
    progress_weight: float = 80.0
    speed_weight: float = 8.0
    smoothness_weight: float = -0.02
    steer_rate_weight: float = -0.3           # relaxed for higher cornering speed
    off_track_weight: float = -3.0
    wall_penalty_weight: float = 0.0
    wall_danger_dist: float = 0.10
    centerline_weight: float = 0.5
    centerline_sigma: float = 0.8
    alive_bonus: float = 0.01
    forward_only_penalty: float = -0.3
    backward_bonus: float = 0.0

    # ---- Termination ----
    off_track_threshold: float = 1.2        # soft boundary
    terminate_cte: float = 1.6              # hard termination
    flip_up_z_threshold: float = 0.35


class SCurveEnv(DirectRLEnv):
    """S-curve racing env for MuSHR with LiDAR-only observations."""

    cfg: SCurveEnvCfg

    def __init__(self, cfg: SCurveEnvCfg, render_mode: str | None = None, **kwargs):
        self._spec = get_robot_spec(cfg.robot_name)
        self._track = RacingTrack.load(default_track_path(cfg.track_name))
        super().__init__(cfg, render_mode, **kwargs)

        # Cache joint indices
        self._steer_ids, self._drive_ids = self._spec.resolve_joints(self.ego.joint_names)

        # Track centerline cached on device
        self._centerline = torch.as_tensor(
            self._track.positions[:, :2], dtype=torch.float32, device=self.device
        )
        import numpy as np
        diffs = np.diff(self._track.positions[:, :2], axis=0, append=self._track.positions[:1, :2])
        seg_len = np.linalg.norm(diffs, axis=1)
        self._seg_len_t = torch.as_tensor(seg_len, device=self.device, dtype=torch.float32)
        cum = np.concatenate([[0.0], np.cumsum(seg_len[:-1])])
        self._cum_lengths = torch.as_tensor(cum, device=self.device, dtype=torch.float32)
        self._total_length = float(self._cum_lengths[-1].item() + float(seg_len[-1]))
        # Headings for spawn yaw
        self._headings = torch.as_tensor(
            self._track.rpy[:, 2], dtype=torch.float32, device=self.device
        )
        # Track z per centerline point
        self._track_z = torch.as_tensor(
            self._track.positions[:, 2], dtype=torch.float32, device=self.device
        )

        # Per-env state buffers
        self._prev_progress = torch.zeros(self.num_envs, device=self.device)
        self._prev_actions = torch.zeros(self.num_envs, 2, device=self.device)

    # ------------------------------------------------------------------
    # Scene setup
    # ------------------------------------------------------------------
    def _setup_scene(self):
        # Robot
        ego_cfg = self._spec.cfg.replace(prim_path="/World/envs/env_.*/Ego")
        self.ego = Articulation(ego_cfg)
        post_spawn_fix(self._spec, "/World/envs/env_.*/Ego")

        # Ground + road + duct (high friction applied after)
        spawn_high_friction_ground("/World/ground")
        spawn_circuit(
            self._track, prim_path="/World/Road", surface_only=True,
            color=(0.35, 0.35, 0.35),
        )
        apply_high_friction_to_mesh("/World/Road/mesh")
        spawn_duct_track(
            self._track, prim_path="/World/Duct",
            pipe_radius=0.15, pipe_offset=1.0, rib_spacing=0.8,
        )

        # LiDAR — 3D pattern (multi-channel) attached to laser_link.
        self.lidar = MultiMeshRayCaster(self._build_lidar_cfg())

        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["ego"] = self.ego
        self.scene.sensors["lidar"] = self.lidar

        sim_utils.DomeLightCfg(intensity=2000.0, color=(0.9, 0.9, 0.9)).func(
            "/World/Light",
            sim_utils.DomeLightCfg(intensity=2000.0, color=(0.9, 0.9, 0.9)),
        )

    def _build_lidar_cfg(self) -> MultiMeshRayCasterCfg:
        from isaaclab.sensors.ray_caster.patterns import LidarPatternCfg

        horizontal_res = 360.0 / self.cfg.num_lidar_bins
        vfov_min, vfov_max = self.cfg.lidar_vfov_deg
        pattern = LidarPatternCfg(
            channels=self.cfg.num_lidar_channels,
            vertical_fov_range=(vfov_min, vfov_max),
            horizontal_fov_range=(-180.0, 180.0),
            horizontal_res=horizontal_res,
        )
        # Robot-specific base_link path. MuSHR's USD nests an extra
        # `mushr_nano/` group; SRX8 / F1Tenth put base_link directly under
        # the Ego prim.
        if self.cfg.robot_name == "mushr":
            base_link = "/World/envs/env_.*/Ego/mushr_nano/base_link"
        else:
            base_link = "/World/envs/env_.*/Ego/base_link"
        return MultiMeshRayCasterCfg(
            prim_path=base_link,
            update_period=0.0,
            offset=MultiMeshRayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.18)),
            pattern_cfg=pattern,
            max_distance=self.cfg.lidar_max_range,
            mesh_prim_paths=[
                "/World/Duct/duct",
                "/World/Duct/ribs",
                "/World/Road/mesh",
                "/World/ground",
            ],
        )

    # ------------------------------------------------------------------
    # Step hooks
    # ------------------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._actions = actions

    def _apply_action(self) -> None:
        steer = self._actions[:, 0].clamp(-1.0, 1.0) * self._spec.max_steer
        speed_norm = (self._actions[:, 1].clamp(-1.0, 1.0) + 1.0) * 0.5
        speed = speed_norm * self.cfg.max_speed
        apply_ackermann(
            self.ego, self._spec, steer_rad=steer, speed_mps=speed,
            steer_ids=self._steer_ids, drive_ids=self._drive_ids,
            force_steer=True,
        )

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------
    def _compact_lidar_distances(self) -> torch.Tensor:
        """Return (num_envs, num_bins) — min distance per azimuth bin,
        normalised to [0, 1] by lidar_max_range."""
        hits_w = self.lidar.data.ray_hits_w  # (B, N, 3)
        ego_pos = self.ego.data.root_pos_w   # (B, 3)
        rel = hits_w - ego_pos.unsqueeze(1)
        dist = torch.norm(rel, dim=-1)       # (B, N)
        dist = torch.nan_to_num(
            dist, nan=self.cfg.lidar_max_range,
            posinf=self.cfg.lidar_max_range,
        )
        dist = dist.clamp(0.0, self.cfg.lidar_max_range)

        B = dist.shape[0]
        N = dist.shape[1]
        target = self.cfg.num_lidar_bins
        if N == target * self.cfg.num_lidar_channels:
            dist = dist.view(B, self.cfg.num_lidar_channels, target)
            dist = dist.min(dim=1).values       # (B, target)
        else:
            # Fall back: pad/truncate to the target (num_lidar_bins)
            if N >= target:
                dist = dist[:, :target]
            else:
                pad = torch.full(
                    (B, target - N), self.cfg.lidar_max_range, device=dist.device
                )
                dist = torch.cat([dist, pad], dim=-1)

        return dist / self.cfg.lidar_max_range

    def _get_observations(self) -> dict:
        lidar_norm = self._compact_lidar_distances()
        speed = torch.norm(self.ego.data.root_lin_vel_w[:, :2], dim=-1, keepdim=True)
        speed_n = (speed / self.cfg.max_speed).clamp(0.0, 2.0)
        prev = self._prev_actions  # (B, 2)
        obs = torch.cat([lidar_norm, speed_n, prev], dim=-1)
        return {"policy": obs}

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------
    def _get_rewards(self) -> torch.Tensor:
        ego_xy = self.ego.data.root_pos_w[:, :2]
        diffs = ego_xy.unsqueeze(1) - self._centerline.unsqueeze(0)
        dists = torch.norm(diffs, dim=-1)
        nearest = dists.argmin(dim=-1)

        progress = self._cum_lengths[nearest]
        delta = progress - self._prev_progress
        half = self._total_length / 2
        delta = torch.where(delta < -half, delta + self._total_length, delta)
        delta = torch.where(delta > half, delta - self._total_length, delta)
        self._prev_progress = progress

        speed = torch.norm(self.ego.data.root_lin_vel_w[:, :2], dim=-1)
        speed_n = (speed / self.cfg.max_speed).clamp(-1.0, 2.0)

        # Compute forward speed along heading
        q = self.ego.data.root_quat_w
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        forward_dir = torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=-1)
        v = self.ego.data.root_lin_vel_w[:, :2]
        forward_speed = (v * forward_dir).sum(dim=-1)
        back_penalty = (
            (forward_speed < -0.2).float() * self.cfg.forward_only_penalty
        )

        progress_r = self.cfg.progress_weight * delta.clamp(-1.0, 2.0)
        speed_r = self.cfg.speed_weight * speed_n.clamp_min(0.0)

        action_diff = (self.actions - self._prev_actions).norm(dim=-1)
        smooth_r = self.cfg.smoothness_weight * action_diff
        # Steering rate: |Δsteer| per step. This heavily penalises
        # steering flicker and encourages smooth paths.
        steer_delta = (self.actions[:, 0] - self._prev_actions[:, 0]).abs()
        steer_rate_r = self.cfg.steer_rate_weight * steer_delta

        min_dist_to_center = dists.min(dim=-1).values
        off_track = (min_dist_to_center > self.cfg.off_track_threshold).float()
        off_r = self.cfg.off_track_weight * off_track

        # Gaussian attraction toward the centerline — rewards the agent
        # for staying in the middle of the road, without penalising small
        # excursions linearly.
        sigma = self.cfg.centerline_sigma
        center_r = self.cfg.centerline_weight * torch.exp(
            -(min_dist_to_center * min_dist_to_center) / (2 * sigma * sigma)
        )

        lidar_min = self._compact_lidar_distances().min(dim=-1).values \
                    * self.cfg.lidar_max_range
        wall_danger = (lidar_min < self.cfg.wall_danger_dist).float()
        wall_r = self.cfg.wall_penalty_weight * wall_danger

        alive_r = torch.full(
            (self.num_envs,), self.cfg.alive_bonus, device=self.device
        )

        total = (progress_r + speed_r + smooth_r + steer_rate_r
                 + off_r + wall_r + center_r + back_penalty + alive_r)

        self.extras["log"] = {
            "reward/progress":  progress_r.mean().item(),
            "reward/speed":     speed_r.mean().item(),
            "reward/center":    center_r.mean().item(),
            "reward/smooth":    smooth_r.mean().item(),
            "reward/steer_rt":  steer_rate_r.mean().item(),
            "reward/off":       off_r.mean().item(),
            "reward/wall":      wall_r.mean().item(),
            "reward/alive":     alive_r.mean().item(),
            "reward/total":     total.mean().item(),
            "info/speed":       speed.mean().item(),
            "info/forward_speed": forward_speed.mean().item(),
            "info/cte":         min_dist_to_center.mean().item(),
            "info/steer_delta": steer_delta.mean().item(),
        }
        self._prev_actions = self.actions.clone()
        return total

    # ------------------------------------------------------------------
    # Dones
    # ------------------------------------------------------------------
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        root_pos = self.ego.data.root_pos_w
        root_quat = self.ego.data.root_quat_w

        diffs = root_pos[:, :2].unsqueeze(1) - self._centerline.unsqueeze(0)
        min_dist = torch.norm(diffs, dim=-1).min(dim=-1).values
        off = min_dist > self.cfg.terminate_cte

        w, x, y, z = root_quat[:, 0], root_quat[:, 1], root_quat[:, 2], root_quat[:, 3]
        up_z = 1.0 - 2.0 * (x * x + y * y)
        flipped = up_z < self.cfg.flip_up_z_threshold

        fallen = root_pos[:, 2] < -0.3

        terminated = off | flipped | fallen
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.ego._ALL_INDICES
        super()._reset_idx(env_ids)

        n_reset = len(env_ids)
        n_pts = self._centerline.shape[0]
        # Randomise progress in [0, 1) so agents see the whole track
        rand_idx = torch.randint(0, n_pts, (n_reset,), device=self.device)
        xy = self._centerline[rand_idx]
        heading = self._headings[rand_idx]
        z0 = self._track_z[rand_idx] + self._spec.init_z

        state = self.ego.data.default_root_state[env_ids].clone()
        state[:, 0] = xy[:, 0] + self.scene.env_origins[env_ids, 0]
        state[:, 1] = xy[:, 1] + self.scene.env_origins[env_ids, 1]
        state[:, 2] = z0
        state[:, 3] = torch.cos(heading / 2)
        state[:, 4:6] = 0.0
        state[:, 6] = torch.sin(heading / 2)
        state[:, 7:] = 0.0
        self.ego.write_root_pose_to_sim(state[:, :7], env_ids)
        self.ego.write_root_velocity_to_sim(state[:, 7:], env_ids)
        self.ego.write_joint_state_to_sim(
            self.ego.data.default_joint_pos[env_ids],
            torch.zeros_like(self.ego.data.default_joint_pos[env_ids]),
            None,
            env_ids,
        )

        self._prev_progress[env_ids] = self._cum_lengths[rand_idx]
        self._prev_actions[env_ids] = 0.0
