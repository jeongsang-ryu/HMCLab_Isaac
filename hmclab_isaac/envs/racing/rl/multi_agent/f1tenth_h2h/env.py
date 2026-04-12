"""Head-to-head F1Tenth racing env.

Extends `F1TenthRacingBaseEnv` with a rule-based opponent that:
  - follows the centerline at constant speed
  - steers toward the next centerline point via heading error

Observation adds opponent relative pose. Rewards gain an overtake bonus and
an opponent-collision penalty.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch

from isaaclab.assets import Articulation
from isaaclab.sensors.ray_caster import MultiMeshRayCasterCfg

from hmclab_isaac.envs.racing._base import F1TenthRacingBaseEnv
from hmclab_isaac.robots.racing.f1tenth_mid360 import MAX_STEER, WHEEL_RADIUS

from .cfg import F1TenthH2HEnvCfg


class F1TenthH2HEnv(F1TenthRacingBaseEnv):
    cfg: F1TenthH2HEnvCfg

    def __init__(self, cfg: F1TenthH2HEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._opp_steer_ids = [
            i for i, n in enumerate(self.opp.joint_names) if "steering" in n
        ]
        self._opp_drive_ids = [
            i for i, n in enumerate(self.opp.joint_names)
            if "rear" in n and "wheel" in n
        ]
        self._prev_opp_progress = torch.zeros(self.num_envs, device=self.device)
        self._ego_ahead = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        # Observation now includes opponent rel (dx, dy) → +2 dims
        self._obs_dim = self.cfg.num_lidar_bins + 1 + 2

    # ------------------------------------------------------------------
    # Scene
    # ------------------------------------------------------------------
    def _setup_scene(self):
        # Spawn opponent alongside ego, then fall through to base setup.
        # Ego's LiDAR cfg is built inside super()._setup_scene, and our
        # `_lidar_mesh_targets` override below adds the opponent as a
        # dynamic raycast target so MARL envs can see it.
        self.opp = Articulation(self.cfg.opp_cfg)
        super()._setup_scene()
        self.scene.articulations["opp"] = self.opp

    def _lidar_mesh_targets(self) -> list:
        # Base static mesh targets + dynamic opponent body.
        return [
            "/World/Track/mesh",
            "/World/ground",
            MultiMeshRayCasterCfg.RaycastTargetCfg(
                prim_expr="/World/envs/env_.*/Opp/base_link",
                track_mesh_transforms=True,
                is_shared=False,
                merge_prim_meshes=True,
            ),
        ]

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def _apply_action(self) -> None:
        self._apply_ego_action()
        self._apply_opponent_action()

    def _apply_opponent_action(self) -> None:
        n_j = self.opp.num_joints
        opp_wvel = self.cfg.opp_speed / WHEEL_RADIUS
        vel_t = torch.zeros(self.num_envs, n_j, device=self.device)
        for idx in self._opp_drive_ids:
            vel_t[:, idx] = opp_wvel

        opp_xy = self.opp.data.root_pos_w[:, :2]
        opp_quat = self.opp.data.root_quat_w

        diffs = opp_xy.unsqueeze(1) - self._centerline.unsqueeze(0)
        nearest = torch.norm(diffs, dim=-1).argmin(dim=-1)
        ahead_idx = (nearest + 5) % self._centerline.shape[0]
        target_xy = self._centerline[ahead_idx]
        to_target = target_xy - opp_xy
        target_heading = torch.atan2(to_target[:, 1], to_target[:, 0])

        w, x, y, z = opp_quat[:, 0], opp_quat[:, 1], opp_quat[:, 2], opp_quat[:, 3]
        cur_heading = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        err = target_heading - cur_heading
        err = (err + math.pi) % (2 * math.pi) - math.pi
        steer_cmd = err.clamp(-MAX_STEER, MAX_STEER)

        steer_t = torch.zeros(self.num_envs, n_j, device=self.device)
        for idx in self._opp_steer_ids:
            steer_t[:, idx] = steer_cmd

        self.opp.set_joint_velocity_target(vel_t)
        self.opp.set_joint_position_target(steer_t)

    # ------------------------------------------------------------------
    # Obs / rewards
    # ------------------------------------------------------------------
    def _get_observations(self) -> dict:
        base = super()._get_observations()["policy"]
        rel = self.opp.data.root_pos_w[:, :2] - self.ego.data.root_pos_w[:, :2]
        rel = rel / self.cfg.lidar_max_range  # same normalization scale as lidar
        rel = rel.clamp(-2.0, 2.0)
        return {"policy": torch.cat([base, rel], dim=-1)}

    def _get_rewards(self) -> torch.Tensor:
        base_r = super()._get_rewards()

        ego_xy = self.ego.data.root_pos_w[:, :2]
        opp_xy = self.opp.data.root_pos_w[:, :2]

        diffs_opp = opp_xy.unsqueeze(1) - self._centerline.unsqueeze(0)
        opp_nearest = torch.norm(diffs_opp, dim=-1).argmin(dim=-1)
        opp_progress = self._cum_lengths[opp_nearest]

        half = self._total_length / 2
        opp_delta = opp_progress - self._prev_opp_progress
        opp_delta = torch.where(opp_delta < -half, opp_delta + self._total_length, opp_delta)
        opp_delta = torch.where(opp_delta > half, opp_delta - self._total_length, opp_delta)
        self._prev_opp_progress = opp_progress

        ego_lead = self._prev_progress - opp_progress
        ego_lead = torch.where(ego_lead < -half, ego_lead + self._total_length, ego_lead)
        ego_lead = torch.where(ego_lead > half, ego_lead - self._total_length, ego_lead)

        ego_ahead_now = ego_lead > self.cfg.opp_overtake_distance
        just_overtook = (~self._ego_ahead) & ego_ahead_now
        self._ego_ahead = ego_ahead_now
        overtake_r = self.cfg.overtake_bonus * just_overtook.float()

        opp_dist = torch.norm(ego_xy - opp_xy, dim=-1)
        opp_hit = (opp_dist < self.cfg.opp_proximity_threshold).float()
        opp_hit_r = self.cfg.opp_collision_penalty * opp_hit

        extra = overtake_r + opp_hit_r
        self.extras["log"]["reward/overtake"] = overtake_r.mean().item()
        self.extras["log"]["reward/opp_hit"] = opp_hit_r.mean().item()
        self.extras["log"]["info/ego_opp_dist"] = opp_dist.mean().item()
        return base_r + extra

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def _reset_idx(self, env_ids: Sequence[int] | None):
        super()._reset_idx(env_ids)
        self._reset_opponent(env_ids if env_ids is not None else self.ego._ALL_INDICES)

    def _reset_opponent(self, env_ids) -> None:
        n_reset = len(env_ids)
        n_pts = self._centerline.shape[0]
        # Spawn opponent a few centerline points ahead of ego.
        ego_idx = torch.searchsorted(
            self._cum_lengths, self._prev_progress[env_ids].clamp_min(0.0)
        ).clamp_max(n_pts - 1)
        ahead = torch.randint(3, 10, (n_reset,), device=self.device)
        opp_idx = (ego_idx + ahead) % n_pts

        xy = self._centerline[opp_idx]
        heading = self._headings_t[opp_idx]

        state = self.opp.data.default_root_state[env_ids].clone()
        state[:, 0] = xy[:, 0] + self.scene.env_origins[env_ids, 0]
        state[:, 1] = xy[:, 1] + self.scene.env_origins[env_ids, 1]
        state[:, 2] = 0.05
        state[:, 3] = torch.cos(heading / 2)
        state[:, 4:6] = 0.0
        state[:, 6] = torch.sin(heading / 2)
        state[:, 7:] = 0.0
        self.opp.write_root_pose_to_sim(state[:, :7], env_ids)
        self.opp.write_root_velocity_to_sim(state[:, 7:], env_ids)
        self.opp.write_joint_state_to_sim(
            self.opp.data.default_joint_pos[env_ids],
            torch.zeros_like(self.opp.data.default_joint_pos[env_ids]),
            None,
            env_ids,
        )
        self._prev_opp_progress[env_ids] = self._cum_lengths[opp_idx]
        self._ego_ahead[env_ids] = False
