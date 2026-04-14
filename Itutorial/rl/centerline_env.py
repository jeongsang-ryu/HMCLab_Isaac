"""Centerline-following RL env — 2D LiDAR + centerline lookahead observations.

Observation layout (49 dims):
    - 36 normalised 2D LiDAR distances (single horizontal ring, 10° resolution)
    -  1 current speed / max_speed
    -  2 previous action (steer, throttle)
    - 10 next K=5 centerline points in ego frame (dx, dy), normalised by 5 m

Action (2 dims, tanh-squashed to [-1, 1]):
    - steer_norm * spec.max_steer
    - throttle_norm mapped to [0, max_speed]

Reward:
    + progress_weight  * centerline progress delta (arc length)
    + speed_weight     * forward speed / max_speed
    + center_weight    * gaussian(cte, sigma)          # stay on the middle
    - steer_rate_w     * |Δsteer|                      # smoothness
    - off_track_w      * 1[cte > off_track_threshold]
    + alive_bonus

Termination: cte > terminate_cte | flipped | fallen | truncate at T.
"""

from __future__ import annotations

import math
import os
import sys
from typing import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.ray_caster import MultiMeshRayCaster, MultiMeshRayCasterCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

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


NUM_LIDAR_BINS = 1080      # 270° / 0.25° — Hokuyo UST-10LX style
LIDAR_FOV_DEG = 270.0
SPEED_NOISE_STD = 0.1      # m/s, sensor noise on forward speed


def compute_obs_dim(lidar_stack: int = 1, lidar_downsample: int = 1) -> int:
    """Compute observation size for given lidar stack / downsample config."""
    n_bins = NUM_LIDAR_BINS // max(lidar_downsample, 1)
    return n_bins * max(lidar_stack, 1) + 1 + 2


OBS_DIM = compute_obs_dim()   # default: 1080 + 1 + 2 = 1083 (no stack, no ds)


@configclass
class CenterlineEnvCfg(DirectRLEnvCfg):
    decimation: int = 2
    episode_length_s: float = 12.0
    sim: SimulationCfg = SimulationCfg(dt=0.005, render_interval=2, device="cuda:0")
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=32, env_spacing=0.0, replicate_physics=True
    )
    action_space: int = 2
    observation_space: int = OBS_DIM
    state_space: int = 0

    track_name: str = "mini_oval_flat"
    robot_name: str = "f1tenth"

    # 2D LiDAR — single horizontal ring
    lidar_vfov_deg: tuple = (-0.5, 0.5)
    lidar_max_range: float = 8.0

    # Temporal stacking (Phase 2+) — 1 = no stack, 10 = use last 10 frames
    lidar_stack: int = 1
    lidar_downsample: int = 1   # stride; 4 → 1080→270 rays per frame

    max_speed: float = 7.0

    # Per-wheel actuator (4WD). Tuned via straight_test for fast tracking.
    drive_effort_limit: float = 500.0   # Nm per wheel — 4×500 = 2000 Nm total
    drive_damping: float = 1000.0       # velocity-mode P gain (high → tight tracking)
    drive_velocity_limit: float = 400.0 # rad/s — caps wheel angular velocity
    track_width: float = 0.20           # rear axle track (m) for Ackermann distribution

    # Chassis mass / CoM tuning (anti-wheelie)
    chassis_mass: float = 4.5           # kg
    chassis_com_z: float = -0.06        # m — CoM 6 cm below default
    chassis_link_name: str = "base_link"

    # Ground / track friction
    mu_static: float = 2.5
    mu_dynamic: float = 2.2

    # Steering actuator — realistic RC servo (fast but not instant).
    # Default tutorial uses 8000/200 → snap instant. Real servo ~5-15 ms step.
    steer_stiffness: float = 3000.0
    steer_damping: float = 150.0
    steer_effort_limit: float = 150.0

    # Reward weights — progress-dominant "go fast + smooth steering" shaping
    progress_weight: float = 100.0
    speed_weight: float = 2.0
    center_weight: float = 0.0
    center_sigma: float = 0.6
    steer_rate_weight: float = -1.5          # |Δsteer| — boosted for smoother steering
    steer_jerk_weight: float = -1.0          # |Δ²steer| — stronger anti-oscillation
    throttle_rate_weight: float = -0.05
    off_track_weight: float = -2.0
    back_penalty: float = -0.5
    stall_penalty: float = -1.0

    # Optimal-line mode — when True, policy is free to take the racing
    # line rather than hugging the centerline. Reward removes Frenet-
    # heading bonus, loosens termination, and widens off-track threshold.
    optimal_line_mode: bool = False

    # Heading alignment with track tangent (Frenet d_psi)
    heading_weight: float = 1.0          # +cos(yaw_err) term
    heading_misalign_penalty: float = -0.5  # extra penalty when |yaw_err| > 60°

    # Spawn-time domain randomisation
    spawn_yaw_noise_rad: float = 0.2     # ±0.2 rad ≈ ±11°
    spawn_lateral_noise_m: float = 0.15  # ±0.15 m offset from centerline

    # Post-spawn settle: freeze actions/reward/termination while the vehicle
    # drops from init_z and the suspension damps out. 40 × 0.01 s ≈ 0.40 s.
    settle_steps: int = 40

    # Termination (allow slightly more aggressive racing line)
    off_track_threshold: float = 1.3
    terminate_cte: float = 1.6
    flip_up_z_threshold: float = 0.1     # allow pitch/roll up to ~42° (prev 0.35 = 35°)


class CenterlineEnv(DirectRLEnv):
    cfg: CenterlineEnvCfg

    def __init__(self, cfg: CenterlineEnvCfg, render_mode: str | None = None, **kwargs):
        self._spec = get_robot_spec(cfg.robot_name)
        self._apply_unlocked_actuators(cfg)
        self._track = RacingTrack.load(default_track_path(cfg.track_name))
        super().__init__(cfg, render_mode, **kwargs)

        self._steer_ids, self._drive_ids = self._spec.resolve_joints(self.ego.joint_names)

        # Per-wheel joint indices for proper Ackermann distribution (4WD).
        # Joint-name substrings differ by robot.
        names = list(self.ego.joint_names)
        def _find(sub_l, sub_r):
            il = next((i for i, n in enumerate(names) if sub_l in n), -1)
            ir = next((i for i, n in enumerate(names) if sub_r in n), -1)
            return il, ir
        name = self.cfg.robot_name.lower()
        if name == "mushr":
            self._steer_l_idx, self._steer_r_idx = _find("front_left_wheel_steer", "front_right_wheel_steer")
            self._fl_idx, self._fr_idx = _find("front_left_wheel_throttle", "front_right_wheel_throttle")
            self._rl_idx, self._rr_idx = _find("back_left_wheel_throttle", "back_right_wheel_throttle")
        else:
            self._steer_l_idx, self._steer_r_idx = _find("front_left_steering", "front_right_steering")
            self._fl_idx, self._fr_idx = _find("front_left_wheel_joint", "front_right_wheel_joint")
            self._rl_idx, self._rr_idx = _find("rear_left_wheel_joint", "rear_right_wheel_joint")
        self._has_per_wheel = all(i >= 0 for i in [self._steer_l_idx, self._steer_r_idx,
                                                    self._fl_idx, self._fr_idx,
                                                    self._rl_idx, self._rr_idx])
        if not self._has_per_wheel:
            print(f"[env] WARN: per-wheel joints not all found "
                  f"(steer_l/r, FL/FR, RL/RR). Falling back to uniform.",
                  flush=True)

        # Anti-wheelie: increase chassis mass and lower CoM
        self._tune_chassis_mass_and_com()

        import numpy as np
        xy = self._track.positions[:, :2].astype(np.float32)
        self._centerline = torch.as_tensor(xy, device=self.device)
        diffs = np.diff(xy, axis=0, append=xy[:1])
        seg_len = np.linalg.norm(diffs, axis=1).astype(np.float32)
        cum = np.concatenate([[0.0], np.cumsum(seg_len[:-1])]).astype(np.float32)
        self._seg_len_t = torch.as_tensor(seg_len, device=self.device)
        self._cum_lengths = torch.as_tensor(cum, device=self.device)
        self._total_length = float(cum[-1] + seg_len[-1])
        self._headings = torch.as_tensor(
            self._track.rpy[:, 2].astype(np.float32), device=self.device
        )
        self._track_z = torch.as_tensor(
            self._track.positions[:, 2].astype(np.float32), device=self.device
        )

        self._prev_progress = torch.zeros(self.num_envs, device=self.device)
        self._prev_actions = torch.zeros(self.num_envs, 2, device=self.device)
        self._prev_prev_actions = torch.zeros(self.num_envs, 2, device=self.device)
        self._settle_left = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # LiDAR frame-stack ring buffer (filled newest-last on each obs call)
        self._n_bins_ds = NUM_LIDAR_BINS // max(self.cfg.lidar_downsample, 1)
        self._lidar_buf = torch.zeros(
            self.num_envs, max(self.cfg.lidar_stack, 1), self._n_bins_ds,
            device=self.device,
        )

    # ------------------------------------------------------------------ scene
    def _setup_scene(self):
        ego_cfg = self._spec.cfg.replace(prim_path="/World/envs/env_.*/Ego")
        self.ego = Articulation(ego_cfg)
        post_spawn_fix(self._spec, "/World/envs/env_.*/Ego")

        spawn_high_friction_ground("/World/ground",
                                   static_friction=self.cfg.mu_static,
                                   dynamic_friction=self.cfg.mu_dynamic)
        spawn_circuit(
            self._track, prim_path="/World/Road", surface_only=True,
            color=(0.35, 0.35, 0.35),
        )
        apply_high_friction_to_mesh("/World/Road/mesh",
                                    static_friction=self.cfg.mu_static,
                                    dynamic_friction=self.cfg.mu_dynamic)
        spawn_duct_track(
            self._track, prim_path="/World/Duct",
            pipe_radius=0.15, pipe_offset=1.0, rib_spacing=0.8,
        )

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
        # 270° centred on forward, 0.25° resolution → 1080 rays
        horizontal_res = LIDAR_FOV_DEG / NUM_LIDAR_BINS
        half_fov = LIDAR_FOV_DEG / 2.0
        vmin, vmax = self.cfg.lidar_vfov_deg
        pattern = LidarPatternCfg(
            channels=1,
            vertical_fov_range=(vmin, vmax),
            horizontal_fov_range=(-half_fov, half_fov),
            horizontal_res=horizontal_res,
        )
        if self.cfg.robot_name == "mushr":
            base_link = "/World/envs/env_.*/Ego/mushr_nano/base_link"
        else:
            base_link = "/World/envs/env_.*/Ego/base_link"
        return MultiMeshRayCasterCfg(
            prim_path=base_link,
            update_period=1.0 / 40.0,   # 40 Hz — matches real 2D LiDAR rate
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

    # ------------------------------------------------------------------ step
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._actions = actions

    def _apply_action(self) -> None:
        # Zero commands while the vehicle is still settling after spawn-drop.
        settling = (self._settle_left > 0).unsqueeze(-1).to(self._actions.dtype)
        act = self._actions * (1.0 - settling)
        steer = act[:, 0].clamp(-1.0, 1.0) * self._spec.max_steer
        speed_norm = (act[:, 1].clamp(-1.0, 1.0) + 1.0) * 0.5
        speed = speed_norm * self.cfg.max_speed
        if self._has_per_wheel:
            self._apply_action_4wd_ackermann(steer, speed)
        else:
            apply_ackermann(
                self.ego, self._spec,
                steer_rad=steer, speed_mps=speed,
                steer_ids=self._steer_ids, drive_ids=self._drive_ids,
                force_steer=True,
            )

    def _apply_action_4wd_ackermann(self, steer: torch.Tensor, speed: torch.Tensor) -> None:
        """Per-wheel angular velocity targets + Ackermann steering.

        Front wheels: inner/outer steering angles atan(L / (R ∓ W/2))
        All four wheels driven (4WD); each wheel gets its own angular
        velocity from |path_radius_at_wheel| / wheel_radius.
        """
        L = self._spec.wheelbase
        W = self.cfg.track_width
        Rw = self._spec.wheel_radius
        eps = 1e-6
        tan_s = torch.tan(steer)
        tan_s = torch.where(tan_s.abs() < eps,
                            torch.full_like(tan_s, eps).copysign(tan_s + eps),
                            tan_s)
        radius = L / tan_s                                      # signed
        straight = steer.abs() < 1e-3
        rl_arm = radius - W * 0.5
        rr_arm = radius + W * 0.5
        # Path radius at each wheel (front uses sqrt(arm² + L²))
        fl_path = torch.sqrt(rl_arm * rl_arm + L * L)
        fr_path = torch.sqrt(rr_arm * rr_arm + L * L)
        v_rl = speed * (rl_arm.abs() / radius.abs().clamp_min(eps))
        v_rr = speed * (rr_arm.abs() / radius.abs().clamp_min(eps))
        v_fl = speed * (fl_path.abs() / radius.abs().clamp_min(eps))
        v_fr = speed * (fr_path.abs() / radius.abs().clamp_min(eps))
        # Straight-line: all wheels at same speed
        v_rl = torch.where(straight, speed, v_rl)
        v_rr = torch.where(straight, speed, v_rr)
        v_fl = torch.where(straight, speed, v_fl)
        v_fr = torch.where(straight, speed, v_fr)

        # Steering: Ackermann inner/outer angles
        delta_l_pos = torch.atan(L / rl_arm)   # left wheel as inner when steer>0
        delta_r_pos = torch.atan(L / rr_arm)
        delta_l = torch.where(straight, steer, delta_l_pos)
        delta_r = torch.where(straight, steer, delta_r_pos)

        N = self.num_envs
        n_j = self.ego.num_joints
        vel_t = torch.zeros(N, n_j, device=self.device)
        vel_t[:, self._fl_idx] = v_fl / Rw
        vel_t[:, self._fr_idx] = v_fr / Rw
        vel_t[:, self._rl_idx] = v_rl / Rw
        vel_t[:, self._rr_idx] = v_rr / Rw
        self.ego.set_joint_velocity_target(vel_t)

        # Steering — use PD target (realistic servo lag), NOT direct write
        N = self.num_envs
        n_j = self.ego.num_joints
        pos_t = torch.zeros(N, n_j, device=self.device)
        pos_t[:, self._steer_l_idx] = delta_l
        pos_t[:, self._steer_r_idx] = delta_r
        self.ego.set_joint_position_target(pos_t)

    # ------------------------------------------------------------------ obs helpers
    def _lidar_2d(self) -> torch.Tensor:
        hits_w = self.lidar.data.ray_hits_w            # (B, N, 3)
        ego_pos = self.ego.data.root_pos_w             # (B, 3)
        dist = torch.norm(hits_w - ego_pos.unsqueeze(1), dim=-1)
        dist = torch.nan_to_num(
            dist, nan=self.cfg.lidar_max_range, posinf=self.cfg.lidar_max_range,
        ).clamp(0.0, self.cfg.lidar_max_range)
        if dist.shape[1] != NUM_LIDAR_BINS:
            # pattern may have produced slightly different count — truncate/pad
            B = dist.shape[0]
            if dist.shape[1] >= NUM_LIDAR_BINS:
                dist = dist[:, :NUM_LIDAR_BINS]
            else:
                pad = torch.full(
                    (B, NUM_LIDAR_BINS - dist.shape[1]), self.cfg.lidar_max_range,
                    device=dist.device,
                )
                dist = torch.cat([dist, pad], dim=-1)
        return dist / self.cfg.lidar_max_range          # (B, 36)

    def _nearest_idx(self, ego_xy: torch.Tensor) -> torch.Tensor:
        diffs = ego_xy.unsqueeze(1) - self._centerline.unsqueeze(0)   # (B, N, 2)
        return torch.norm(diffs, dim=-1).argmin(dim=-1)               # (B,)

    def _ego_yaw(self) -> torch.Tensor:
        q = self.ego.data.root_quat_w
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    def _get_observations(self) -> dict:
        lidar = self._lidar_2d()                                    # (B, 1080)
        # Downsample per-frame if requested
        ds = max(self.cfg.lidar_downsample, 1)
        if ds > 1:
            lidar = lidar[:, ::ds][:, :self._n_bins_ds]             # (B, n_ds)
        # Ring-shift buffer and insert newest
        K = max(self.cfg.lidar_stack, 1)
        if K > 1:
            self._lidar_buf = torch.roll(self._lidar_buf, shifts=-1, dims=1)
            self._lidar_buf[:, -1, :] = lidar
            lidar_obs = self._lidar_buf.reshape(self.num_envs, K * self._n_bins_ds)
        else:
            lidar_obs = lidar

        # Forward speed in ego frame + Gaussian sensor noise
        v = self.ego.data.root_lin_vel_w[:, :2]
        yaw = self._ego_yaw()
        fwd = torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=-1)
        forward_speed = (v * fwd).sum(dim=-1, keepdim=True)         # (B, 1)
        noise = torch.randn_like(forward_speed) * SPEED_NOISE_STD
        fwd_speed_n = ((forward_speed + noise) / self.cfg.max_speed).clamp(-1.0, 2.0)
        prev = self._prev_actions                                   # (B, 2)
        obs = torch.cat([lidar_obs, fwd_speed_n, prev], dim=-1)
        return {"policy": obs}

    # ------------------------------------------------------------------ reward
    def _get_rewards(self) -> torch.Tensor:
        ego_xy = self.ego.data.root_pos_w[:, :2]
        diffs = ego_xy.unsqueeze(1) - self._centerline.unsqueeze(0)
        dists = torch.norm(diffs, dim=-1)
        nearest = dists.argmin(dim=-1)
        cte = dists.min(dim=-1).values

        progress = self._cum_lengths[nearest]
        delta = progress - self._prev_progress
        half = self._total_length / 2
        delta = torch.where(delta < -half, delta + self._total_length, delta)
        delta = torch.where(delta >  half, delta - self._total_length, delta)
        self._prev_progress = progress

        v = self.ego.data.root_lin_vel_w[:, :2]
        speed = torch.norm(v, dim=-1)
        speed_n = (speed / self.cfg.max_speed).clamp(0.0, 2.0)

        yaw = self._ego_yaw()
        fwd = torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=-1)
        forward_speed = (v * fwd).sum(dim=-1)
        back_r = (forward_speed < -0.2).float() * self.cfg.back_penalty

        progress_r = self.cfg.progress_weight * delta.clamp(-1.0, 2.0)
        # reward forward speed only; standing still earns 0
        fwd_n = (forward_speed / self.cfg.max_speed).clamp(0.0, 1.5)
        speed_r = self.cfg.speed_weight * fwd_n

        # Frenet-style heading alignment with track tangent
        track_yaw = self._headings[nearest]                         # (B,)
        yaw_err = torch.atan2(torch.sin(yaw - track_yaw),
                              torch.cos(yaw - track_yaw))           # wrap to [-π, π]
        # In optimal-line mode the policy chooses its own heading (no
        # forced alignment). We zero the cosine bonus but keep the hard-
        # misalignment penalty so the car can't drive backwards.
        if self.cfg.optimal_line_mode:
            heading_r = torch.zeros(self.num_envs, device=self.device)
        else:
            heading_r = self.cfg.heading_weight * torch.cos(yaw_err)
        heading_pen = (yaw_err.abs() > (math.pi / 3.0)).float() * self.cfg.heading_misalign_penalty

        sigma = self.cfg.center_sigma
        # center bonus gated by forward motion — prevents "sit on centerline" cheat
        center_gauss = torch.exp(-(cte * cte) / (2 * sigma * sigma))
        center_r = self.cfg.center_weight * center_gauss * fwd_n.clamp(0.0, 1.0)

        stall = (forward_speed < 0.5).float() * self.cfg.stall_penalty

        steer_delta = (self.actions[:, 0] - self._prev_actions[:, 0]).abs()
        steer_r = self.cfg.steer_rate_weight * steer_delta
        # Second-derivative (steering jerk) — penalises oscillation even if
        # single-step Δsteer is small.
        steer_jerk = (self.actions[:, 0] - 2 * self._prev_actions[:, 0]
                      + self._prev_prev_actions[:, 0]).abs()
        jerk_r = self.cfg.steer_jerk_weight * steer_jerk
        throttle_delta = (self.actions[:, 1] - self._prev_actions[:, 1]).abs()
        throttle_r = self.cfg.throttle_rate_weight * throttle_delta

        # Optimal-line mode widens the allowable lateral range and halves
        # the off-track penalty so the policy can ride the inside/outside.
        off_thresh = (self.cfg.off_track_threshold
                      * (2.0 if self.cfg.optimal_line_mode else 1.0))
        off_weight = (self.cfg.off_track_weight
                      * (0.3 if self.cfg.optimal_line_mode else 1.0))
        off = (cte > off_thresh).float()
        off_r = off_weight * off

        total = (progress_r + speed_r + center_r
                 + heading_r + heading_pen
                 + steer_r + jerk_r + throttle_r
                 + off_r + back_r + stall)

        # Settle phase: zero out reward and hold progress/length counters
        # steady until the vehicle has damped out after spawn-drop.
        settling = self._settle_left > 0
        if settling.any():
            total = torch.where(settling, torch.zeros_like(total), total)
            # don't count settle frames toward progress delta or episode len
            self._prev_progress = torch.where(settling, progress, self._prev_progress)
            self.episode_length_buf = torch.where(
                settling,
                torch.zeros_like(self.episode_length_buf),
                self.episode_length_buf,
            )
            self._settle_left = torch.clamp(self._settle_left - 1, min=0)

        self.extras["log"] = {
            "reward/progress": progress_r.mean().item(),
            "reward/speed":    speed_r.mean().item(),
            "reward/center":   center_r.mean().item(),
            "reward/heading":  heading_r.mean().item(),
            "reward/h_pen":    heading_pen.mean().item(),
            "reward/steer":    steer_r.mean().item(),
            "reward/jerk":     jerk_r.mean().item(),
            "reward/thr":      throttle_r.mean().item(),
            "reward/off":      off_r.mean().item(),
            "reward/stall":    stall.mean().item(),
            "reward/total":    total.mean().item(),
            "info/steer_delta":steer_delta.mean().item(),
            "info/steer_jerk": steer_jerk.mean().item(),
            "info/speed":      speed.mean().item(),
            "info/fwd_speed":  forward_speed.mean().item(),
            "info/cte":        cte.mean().item(),
        }
        self._prev_prev_actions = self._prev_actions.clone()
        self._prev_actions = self.actions.clone()
        return total

    # ------------------------------------------------------------------ dones
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        root_pos = self.ego.data.root_pos_w
        root_quat = self.ego.data.root_quat_w

        diffs = root_pos[:, :2].unsqueeze(1) - self._centerline.unsqueeze(0)
        cte = torch.norm(diffs, dim=-1).min(dim=-1).values
        term_cte = (self.cfg.terminate_cte
                    * (2.0 if self.cfg.optimal_line_mode else 1.0))
        off = cte > term_cte

        w, x, y, z = root_quat[:, 0], root_quat[:, 1], root_quat[:, 2], root_quat[:, 3]
        up_z = 1.0 - 2.0 * (x * x + y * y)
        flipped = up_z < self.cfg.flip_up_z_threshold
        fallen = root_pos[:, 2] < -0.3

        terminated = off | flipped | fallen
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        # Suppress termination while settling (the drop + roll could otherwise
        # register as off-track or flipped before the car even starts).
        settling = self._settle_left > 0
        terminated = terminated & ~settling
        truncated = truncated & ~settling
        return terminated, truncated

    # ------------------------------------------------------------------ reset
    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.ego._ALL_INDICES
        super()._reset_idx(env_ids)

        n_reset = len(env_ids)
        n_pts = self._centerline.shape[0]
        rand_idx = torch.randint(0, n_pts, (n_reset,), device=self.device)
        xy = self._centerline[rand_idx]
        track_yaw = self._headings[rand_idx]
        z0 = self._track_z[rand_idx] + self._spec.init_z

        # Domain randomisation: yaw noise + lateral offset perpendicular to track
        yaw_noise = (torch.rand(n_reset, device=self.device) * 2.0 - 1.0) \
                    * self.cfg.spawn_yaw_noise_rad
        heading = track_yaw + yaw_noise
        lat_noise = (torch.rand(n_reset, device=self.device) * 2.0 - 1.0) \
                    * self.cfg.spawn_lateral_noise_m
        # Lateral offset in world frame: perpendicular to track tangent
        nx = -torch.sin(track_yaw)
        ny =  torch.cos(track_yaw)
        x_world = xy[:, 0] + lat_noise * nx + self.scene.env_origins[env_ids, 0]
        y_world = xy[:, 1] + lat_noise * ny + self.scene.env_origins[env_ids, 1]

        state = self.ego.data.default_root_state[env_ids].clone()
        state[:, 0] = x_world
        state[:, 1] = y_world
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
            None, env_ids,
        )

        self._prev_progress[env_ids] = self._cum_lengths[rand_idx]
        self._prev_actions[env_ids] = 0.0
        self._prev_prev_actions[env_ids] = 0.0
        # Clear lidar buffer so freshly-reset envs don't remember old frames
        self._lidar_buf[env_ids] = 0.0
        # Freeze learning for the first few steps so the vehicle can drop
        # from init_z and settle on the ground before actions/reward count.
        self._settle_left[env_ids] = int(self.cfg.settle_steps)

    # ------------------------------------------------------------------ actuator override
    def _tune_chassis_mass_and_com(self) -> None:
        """Override chassis link mass + CoM Z to suppress wheelie tendency.

        Uses the underlying physx ArticulationView to set per-link masses and
        per-link inertia properties at the start of training (after spawn).
        """
        try:
            body_names = list(self.ego.body_names)
            if self.cfg.chassis_link_name not in body_names:
                print(f"[env] chassis link '{self.cfg.chassis_link_name}' not found "
                      f"among {body_names}", flush=True)
                return
            link_idx = body_names.index(self.cfg.chassis_link_name)
            view = self.ego.root_physx_view
            # Per-robot chassis mass target
            target_mass = self.cfg.chassis_mass
            if self.cfg.robot_name.lower() == "mushr":
                target_mass = 4.0        # slightly heavier for stability
            # Set chassis mass (all envs)
            masses = view.get_masses().clone()                # (N, num_links)
            masses[:, link_idx] = target_mass
            view.set_masses(masses, indices=torch.arange(self.num_envs, device="cpu"))
            # Lower CoM z (relative to link frame). get_coms returns (N, num_links, 7)
            # where the first 3 are xyz, last 4 are quaternion.
            try:
                coms = view.get_coms().clone()
                coms[:, link_idx, 2] = self.cfg.chassis_com_z
                view.set_coms(coms, indices=torch.arange(self.num_envs, device="cpu"))
                print(f"[env] chassis '{self.cfg.chassis_link_name}': "
                      f"mass={target_mass} kg, com_z={self.cfg.chassis_com_z} m",
                      flush=True)
            except Exception as exc:
                print(f"[env] CoM override skipped: {exc}", flush=True)
        except Exception as exc:
            print(f"[env] mass/com tune failed: {exc}", flush=True)

    def _apply_unlocked_actuators(self, cfg: "CenterlineEnvCfg") -> None:
        """4WD with bounded per-wheel effort. Joint-name regex differs by
        robot: f1tenth uses `*_wheel_joint` / `front_*_steering_joint`,
        MuSHR uses `*_wheel_throttle` / `front_*_wheel_steer`.
        """
        try:
            from isaaclab.actuators import ImplicitActuatorCfg
        except Exception:
            return
        acts = dict(getattr(self._spec.cfg, "actuators", {}) or {})

        name = cfg.robot_name.lower()
        # Per-robot torque scale (mushr is lighter ⇒ lower effort to avoid wheelie)
        if name == "mushr":
            steer_expr = ["front_.*_wheel_steer"]
            rear_expr  = ["back_.*_wheel_throttle"]
            front_expr = ["front_.*_wheel_throttle"]
            eff = 5.0                        # 4×5 = 20 Nm total
            dmp = 15.0
        else:   # f1tenth (default)
            steer_expr = ["front_.*_steering_joint"]
            rear_expr  = ["rear_.*_wheel_joint"]
            front_expr = ["front_.*_wheel_joint"]
            eff = cfg.drive_effort_limit
            dmp = cfg.drive_damping

        acts["steering"] = ImplicitActuatorCfg(
            joint_names_expr=steer_expr,
            effort_limit_sim=cfg.steer_effort_limit,
            stiffness=cfg.steer_stiffness,
            damping=cfg.steer_damping,
        )
        # MuSHR USD has 4 passive suspension joints that bounce under
        # acceleration. Lock them near 0 with stiff spring+damper so the
        # chassis doesn't visibly hop on torque impulses.
        if name == "mushr":
            acts["suspension"] = ImplicitActuatorCfg(
                joint_names_expr=[".*_wheel_suspension"],
                effort_limit_sim=40.0,
                stiffness=400.0,
                damping=40.0,
            )
        acts["drive"] = ImplicitActuatorCfg(
            joint_names_expr=rear_expr,
            effort_limit_sim=eff,
            velocity_limit_sim=cfg.drive_velocity_limit,
            stiffness=0.0,
            damping=dmp,
        )
        acts["front_wheels"] = ImplicitActuatorCfg(
            joint_names_expr=front_expr,
            effort_limit_sim=eff,
            velocity_limit_sim=cfg.drive_velocity_limit,
            stiffness=0.0,
            damping=dmp,
        )
        self._spec.cfg = self._spec.cfg.replace(actuators=acts)
