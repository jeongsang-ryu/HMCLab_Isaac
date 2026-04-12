"""Base DirectRLEnv for offroad rover navigation.

Pipeline:
  - Procedural rough terrain (TerrainImporter)
  - ExoMy rover spawned on a flat start pad
  - 2D continuous action: (forward_speed, yaw_rate_cmd)
  - Observation: (goal_dx_local, goal_dy_local, heading_sin, heading_cos, speed)
  - Reward: distance-to-goal shaping + alive + flip penalty
  - Termination: goal reached, flipped, timeout

Kept intentionally minimal for M6 smoke. sim/demo and rl/single_agent
subclasses only tweak `num_envs` and episode length.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporter, TerrainImporterCfg
from isaaclab.utils import configclass

from hmclab_isaac.robots.others.exomy import (
    EXOMY_CFG,
    MAX_STEER,
    WHEEL_RADIUS,
)
from hmclab_isaac.worlds.offroad import rover_rough_terrain_cfg


@configclass
class ExomyOffroadBaseEnvCfg(DirectRLEnvCfg):
    decimation: int = 4
    episode_length_s: float = 20.0

    sim: SimulationCfg = SimulationCfg(
        dt=0.01,
        render_interval=4,
        device="cuda:0",
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4,
        env_spacing=4.0,
        replicate_physics=True,
    )

    action_space: int = 2          # [forward_speed_norm, yaw_rate_norm]
    observation_space: int = 5     # [dx, dy, sin_yaw, cos_yaw, speed]
    state_space: int = 0

    rover_cfg: ArticulationCfg = EXOMY_CFG.replace(
        prim_path="/World/envs/env_.*/Rover"
    )
    terrain: TerrainImporterCfg = rover_rough_terrain_cfg(
        num_rows=2, num_cols=2, size=(8.0, 8.0), noise_range=(0.02, 0.05)
    )

    max_speed: float = 1.0
    max_yaw_rate: float = 1.5

    goal_distance: float = 3.0
    goal_radius: float = 0.3

    progress_weight: float = 5.0
    alive_weight: float = 0.0
    flip_penalty: float = -5.0
    goal_bonus: float = 50.0


class ExomyOffroadBaseEnv(DirectRLEnv):
    cfg: ExomyOffroadBaseEnvCfg

    def __init__(self, cfg: ExomyOffroadBaseEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._steer_ids = [
            i for i, n in enumerate(self.rover.joint_names) if "Steer_Joint" in n
        ]
        self._drive_ids = [
            i for i, n in enumerate(self.rover.joint_names) if "Drive_Joint" in n
        ]

        self._goal_xy = torch.zeros(self.num_envs, 2, device=self.device)
        self._prev_dist = torch.zeros(self.num_envs, device=self.device)

    # ------------------------------------------------------------------
    # Scene
    # ------------------------------------------------------------------
    def _setup_scene(self):
        # Populate runtime fields on the terrain cfg before construction.
        self.cfg.terrain.num_envs = self.cfg.scene.num_envs
        self.cfg.terrain.env_spacing = self.cfg.scene.env_spacing
        self._terrain = TerrainImporter(self.cfg.terrain)

        self.rover = Articulation(self.cfg.rover_cfg)

        self.scene.clone_environments(copy_from_source=False)

        self.scene.articulations["rover"] = self.rover

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.85, 0.85, 0.85))
        light_cfg.func("/World/Light", light_cfg)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clamp(-1.0, 1.0)

    def _apply_action(self) -> None:
        forward = self.actions[:, 0] * self.cfg.max_speed
        yaw_rate = self.actions[:, 1] * self.cfg.max_yaw_rate

        # Steering: map yaw_rate to a per-axle angle (front +, rear -).
        # Simple bicycle approximation; ExoMy can do much better but this
        # is enough for a smoke test.
        steer_angle = torch.atan2(yaw_rate * 0.3, torch.clamp(forward.abs(), min=0.2))
        steer_angle = steer_angle.clamp(-MAX_STEER, MAX_STEER)

        wheel_vel = forward / WHEEL_RADIUS

        n_j = self.rover.num_joints
        drive_t = torch.zeros(self.num_envs, n_j, device=self.device)
        steer_t = torch.zeros(self.num_envs, n_j, device=self.device)
        for idx in self._drive_ids:
            drive_t[:, idx] = wheel_vel
        for idx in self._steer_ids:
            steer_t[:, idx] = steer_angle

        self.rover.set_joint_velocity_target(drive_t)
        self.rover.set_joint_position_target(steer_t)

    # ------------------------------------------------------------------
    # Obs / reward / termination
    # ------------------------------------------------------------------
    def _get_observations(self) -> dict:
        pos = self.rover.data.root_pos_w[:, :2]
        quat = self.rover.data.root_quat_w
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

        goal_world = self._goal_xy + self.scene.env_origins[:, :2]
        to_goal_w = goal_world - pos
        cos_yaw = torch.cos(-yaw)
        sin_yaw = torch.sin(-yaw)
        dx_local = to_goal_w[:, 0] * cos_yaw - to_goal_w[:, 1] * sin_yaw
        dy_local = to_goal_w[:, 0] * sin_yaw + to_goal_w[:, 1] * cos_yaw

        speed = torch.norm(self.rover.data.root_lin_vel_w[:, :2], dim=-1)

        obs = torch.stack(
            [
                (dx_local / 10.0).clamp(-1.0, 1.0),
                (dy_local / 10.0).clamp(-1.0, 1.0),
                torch.sin(yaw),
                torch.cos(yaw),
                (speed / self.cfg.max_speed).clamp(0.0, 2.0),
            ],
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        pos = self.rover.data.root_pos_w[:, :2]
        goal_world = self._goal_xy + self.scene.env_origins[:, :2]
        dist = torch.norm(goal_world - pos, dim=-1)
        delta = self._prev_dist - dist
        self._prev_dist = dist

        progress_r = self.cfg.progress_weight * delta.clamp(-1.0, 1.0)
        alive_r = self.cfg.alive_weight * torch.ones(self.num_envs, device=self.device)
        goal_reached = (dist < self.cfg.goal_radius).float()
        goal_r = self.cfg.goal_bonus * goal_reached

        quat = self.rover.data.root_quat_w
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        up_z = 1.0 - 2.0 * (x * x + y * y)
        flipped = (up_z < 0.5).float()
        flip_r = self.cfg.flip_penalty * flipped

        total = progress_r + alive_r + goal_r + flip_r
        self.extras["log"] = {
            "reward/progress": progress_r.mean().item(),
            "reward/goal": goal_r.mean().item(),
            "reward/flip": flip_r.mean().item(),
            "reward/total": total.mean().item(),
            "info/dist_to_goal": dist.mean().item(),
        }
        return total

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        pos = self.rover.data.root_pos_w[:, :2]
        goal_world = self._goal_xy + self.scene.env_origins[:, :2]
        dist = torch.norm(goal_world - pos, dim=-1)

        reached = dist < self.cfg.goal_radius

        quat = self.rover.data.root_quat_w
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        up_z = 1.0 - 2.0 * (x * x + y * y)
        flipped = up_z < 0.3

        terminated = reached | flipped
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.rover._ALL_INDICES
        super()._reset_idx(env_ids)

        n = len(env_ids)

        # Reset rover to origin of its env
        state = self.rover.data.default_root_state[env_ids].clone()
        state[:, 0] = self.scene.env_origins[env_ids, 0]
        state[:, 1] = self.scene.env_origins[env_ids, 1]
        state[:, 2] = 0.2
        state[:, 3] = 1.0  # qw
        state[:, 4:7] = 0.0
        state[:, 7:] = 0.0
        self.rover.write_root_pose_to_sim(state[:, :7], env_ids)
        self.rover.write_root_velocity_to_sim(state[:, 7:], env_ids)
        self.rover.write_joint_state_to_sim(
            self.rover.data.default_joint_pos[env_ids],
            torch.zeros_like(self.rover.data.default_joint_pos[env_ids]),
            None,
            env_ids,
        )

        # Pick a random goal on the unit circle scaled to cfg.goal_distance
        angle = torch.rand(n, device=self.device) * 2 * math.pi
        self._goal_xy[env_ids, 0] = self.cfg.goal_distance * torch.cos(angle)
        self._goal_xy[env_ids, 1] = self.cfg.goal_distance * torch.sin(angle)
        goal_world = self._goal_xy[env_ids] + self.scene.env_origins[env_ids, :2]
        self._prev_dist[env_ids] = torch.norm(
            goal_world - state[:, :2], dim=-1
        )
