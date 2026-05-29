"""Centerline pure-pursuit controller.

Given a :class:`FrenetField`, the controller projects the vehicle onto the
track centerline, picks a target point ``lookahead_m`` meters ahead of the
projection, transforms that point into the vehicle frame, and computes the
steering angle that an Ackermann bicycle of length ``wheelbase`` would need
to drive through the target point on an arc:

    δ = atan2(2 · L · y_local, L_d²)

where ``L`` is the wheelbase, ``L_d`` the lookahead distance, and
``y_local`` the target's lateral offset in the vehicle frame.

Throttle is a constant fraction of ``max_wheel_speed`` corresponding to
``target_speed_mps`` (defaults to ~half top speed). For tighter pacing the
caller can scale throttle by curvature in a wrapper.

All inputs/outputs are batched over ``num_envs``.
"""
from __future__ import annotations

import math

import torch


# Named tuned presets — pass into CenterlinePurePursuit(**PP_PRESETS[name]).
# Tuned headless via scripts/eval_pure_pursuit.py on my_track.
PP_PRESETS = {
    # Stable cruiser: top 5 m/s. ~34 s/lap, mean centerline dev ~0.22 m.
    "pp_slow": dict(
        target_speed_mps=5.0, steering_gain=1.7, corner_lat_accel=3.0,
        speed_preview_n=5, lookahead_gain=0.4,
        lookahead_min=1.0, lookahead_max=4.0, brake_decel=6.0,
    ),
    # Top 10 m/s on straights with a brake-distance-aware speed profile.
    # Verified: 3 laps, term≈2/9000 steps, mean dev 0.10 m, max 0.61 m
    # (wall margin OK), ~39 s/lap. Tuned via eval_pure_pursuit.py.
    "pp_fast": dict(
        target_speed_mps=10.0, steering_gain=1.7, corner_lat_accel=7.0,
        speed_preview_n=5, lookahead_gain=0.4,
        lookahead_min=1.5, lookahead_max=6.0, brake_decel=13.0,
    ),
}


class CenterlinePurePursuit:
    """Pure-pursuit controller for a single track shared by all envs."""

    def __init__(
        self,
        frenet,
        wheelbase: float = 0.344,
        max_steer_rad: float = 0.785,
        max_wheel_rps: float = 300.0,
        wheel_radius: float = 0.0525,
        lookahead_m: float = 2.0,
        target_speed_mps: float = 5.0,
        curvature_speed_scale: float = 0.0,
        lookahead_gain: float = 0.4,
        lookahead_min: float = 1.0,
        lookahead_max: float = 4.0,
        # Defaults tuned via scripts/eval_pure_pursuit.py on my_track:
        # 3 laps, term≈0, mean centerline deviation ~0.22 m, ~34 s/lap.
        steering_gain: float = 1.7,
        corner_lat_accel: float = 3.0,
        speed_preview_n: int = 5,
        brake_decel: float = 6.0,
        # ── Behavioural diversity (per-opponent) ──
        lateral_offset: float = 0.0,
        weave_amp: float = 0.0,
        weave_wavelength: float = 18.0,
        weave_phase: float = 0.0,
        speed_scale: float = 1.0,
    ):
        """
        Args:
            frenet:                FrenetField for the active track.
            wheelbase:             Vehicle wheelbase (m). 0.344 for UNICORN_3.
            max_steer_rad:         Steering joint limit (rad). Action is
                                   normalized so action_steer × max_steer_rad
                                   = commanded angle.
            max_wheel_rps:         Wheel angular velocity at action_throttle=1.
            wheel_radius:          Tire radius (m).
            lookahead_m:           Fixed fallback lookahead used only when
                                   lookahead_gain == 0.
            target_speed_mps:      Constant cruise speed.
            curvature_speed_scale: If >0, reduce target speed in proportion
                                   to |kappa| at lookahead (slows in corners).
            lookahead_gain:        Adaptive lookahead — L_d = clip(gain·|v|,
                                   min, max). Larger gain ⇒ smoother at speed,
                                   shorter L_d at low speed ⇒ tighter corners.
                                   Set 0 to disable (use fixed lookahead_m).
            lookahead_min/max:     Clip range for the adaptive lookahead (m).
            steering_gain:         Multiplier on the pure-pursuit steering
                                   angle. >1 fights understeer (the
                                   "drives into the wall" symptom).
            corner_lat_accel:      Physically-motivated corner speed —
                                   v_corner = min(target, sqrt(a_lat/|κ|)).
                                   Lower ⇒ slower through corners. Set 0 to
                                   disable (constant target speed).
            speed_preview_n:       (legacy heuristic — superseded by the
                                   precomputed speed profile when
                                   brake_decel > 0.)
            brake_decel:           Longitudinal braking capability (m/s²).
                                   Drives a backward-pass speed profile over
                                   the whole centerline: each point's allowed
                                   speed is min(corner limit, what you can
                                   bleed off from the next point given this
                                   decel). This makes the bot brake LONG
                                   before a corner instead of slamming on at
                                   the apex. Set 0 to fall back to the simple
                                   preview-max heuristic.
        """
        self.frenet = frenet
        self.device = frenet.device
        self.wheelbase = float(wheelbase)
        self.max_steer = float(max_steer_rad)
        self.max_wheel_v = float(max_wheel_rps) * float(wheel_radius)  # m/s
        self.lookahead = float(lookahead_m)
        self.target_speed = float(target_speed_mps)
        self.curvature_speed_scale = float(curvature_speed_scale)
        self.corner_lat_accel = float(corner_lat_accel)
        self.speed_preview_n = max(1, min(int(speed_preview_n), 5))
        self.brake_decel = float(brake_decel)
        self.lookahead_gain = float(lookahead_gain)
        self.lookahead_min = float(lookahead_min)
        self.lookahead_max = float(lookahead_max)
        self.steering_gain = float(steering_gain)
        # Behavioural diversity — pursue a line offset from the centerline
        # (+left) with an optional slow sinusoidal weave, and scale the
        # cruise speed. Lets a fleet of bots take varied lines/paces so the
        # learner can't just memorise "avoid the exact centerline".
        self.lateral_offset = float(lateral_offset)
        self.weave_amp = float(weave_amp)
        self.weave_wavelength = max(float(weave_wavelength), 1e-3)
        self.weave_phase = float(weave_phase)
        self.speed_scale = float(speed_scale)
        # Precompute total_length-aware arclen for searchsorted
        self._arclen = frenet.cl_arclen
        self._cl_xy = frenet.cl_xy
        self._cl_normal = frenet.cl_normal          # (N, 2) unit, +left
        self._cl_widths = frenet.cl_widths          # (N, 2) [d_left,d_right]
        self._cl_kappa = frenet.cl_kappa
        self._total = float(frenet.total_length)

        # ── Precomputed brake-distance-aware speed profile ──
        # v_profile[i] = max safe speed AT centerline point i such that the
        # car can still decelerate (at brake_decel) down to every upcoming
        # corner's lateral-accel limit. Two passes around the closed loop.
        self._speed_profile = None
        if self.brake_decel > 0.0 and self.corner_lat_accel > 0.0:
            kappa = self._cl_kappa.abs().clamp_min(1e-4)        # (N,)
            v_corner = torch.sqrt(self.corner_lat_accel / kappa)
            v_corner = v_corner.clamp_max(self.target_speed)
            n = v_corner.shape[0]
            # per-segment arc length (closed loop)
            seg = torch.diff(
                self._arclen,
                append=self._arclen[-1:] + (self._total - self._arclen[-1]),
            ).abs().clamp_min(1e-3)                             # (N,)
            v = v_corner.clone()
            # Backward pass (2 loops around for closed-loop convergence)
            for _ in range(2):
                for i in range(n - 1, -1, -1):
                    nxt = (i + 1) % n
                    # v_i ≤ sqrt(v_{i+1}² + 2·a·ds)  (can bleed this much)
                    allowed = torch.sqrt(
                        v[nxt] * v[nxt] + 2.0 * self.brake_decel * seg[i]
                    )
                    v[i] = torch.minimum(v[i], allowed)
            self._speed_profile = v                              # (N,)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _idx_at_s(self, s: torch.Tensor) -> torch.Tensor:
        """Find the centerline index whose arclen is closest to s (wrap)."""
        s_wrapped = s.remainder(self._total)
        # searchsorted on a monotonically-increasing arclen
        idx = torch.searchsorted(self._arclen, s_wrapped).clamp(
            0, self.frenet.num_points - 1
        )
        return idx

    # ------------------------------------------------------------------
    # Main step
    # ------------------------------------------------------------------
    @torch.no_grad()
    def step(self, xy: torch.Tensor, yaw: torch.Tensor,
             speed: torch.Tensor | None = None) -> torch.Tensor:
        """Compute (throttle, steer) action for each env.

        Args:
            xy:    (E, 2) vehicle XY in track frame (env-local).
            yaw:   (E,) vehicle yaw (rad), world frame.
            speed: (E,) current forward speed (m/s) for adaptive lookahead.
                   If None, falls back to the fixed lookahead_m.

        Returns:
            action: (E, 2) tensor with throttle ∈ [0, 1], steer ∈ [-1, 1].
        """
        E = xy.shape[0]
        # 1. Project to centerline → current arc-length s
        f = self.frenet.query(xy, yaw)
        s = f["s"]                                  # (E,)
        # 2. Adaptive lookahead: L_d = clip(gain·|v|, min, max)
        if self.lookahead_gain > 0.0 and speed is not None:
            L_d = (self.lookahead_gain * speed.abs()).clamp(
                self.lookahead_min, self.lookahead_max
            )                                       # (E,)
        else:
            L_d = torch.full((E,), self.lookahead, device=self.device)
        # 3. Target arc-length = s + L_d, wrapped
        s_target = s + L_d
        target_idx = self._idx_at_s(s_target)
        target_xy = self._cl_xy[target_idx]         # (E, 2)
        # Behavioural diversity: pursue a point offset off the centerline.
        # offset = base + weave·sin(2π·s/λ + φ), then SAFETY-clamped to keep
        # the bot inside the duct (±60 % of the local half-width) so it
        # never gets driven into a wall regardless of cfg values.
        if self.lateral_offset != 0.0 or self.weave_amp != 0.0:
            off = self.lateral_offset + self.weave_amp * torch.sin(
                2.0 * math.pi * s / self.weave_wavelength + self.weave_phase
            )                                       # (E,)
            w = self._cl_widths[target_idx]         # (E, 2) [d_left,d_right]
            half = torch.minimum(w[:, 0], w[:, 1])
            off = off.clamp(min=-0.6 * half, max=0.6 * half)
            target_xy = target_xy + off.unsqueeze(-1) * self._cl_normal[target_idx]
        # 4. Vehicle-frame coords of the target point
        dx = target_xy[:, 0] - xy[:, 0]
        dy = target_xy[:, 1] - xy[:, 1]
        cy = torch.cos(yaw)
        sy = torch.sin(yaw)
        local_y = -dx * sy + dy * cy                # lateral offset (+left)
        # 5. Pure-pursuit steering: δ = atan2(2 L y, L_d²), with gain
        L_d_sq = L_d * L_d
        delta = torch.atan2(2.0 * self.wheelbase * local_y, L_d_sq)
        delta = delta * self.steering_gain
        steer_norm = (delta / self.max_steer).clamp(-1.0, 1.0)
        # 6. Throttle — corner speed limited by a lateral-accel budget.
        #    Use the MAX |kappa| over the first `speed_preview_n` of
        #    FrenetField's lookahead samples (Δ = 1,2,4,8,16 m) so the bot
        #    brakes BEFORE the corner, not at the apex.
        v_target = torch.full((E,), self.target_speed, device=self.device)
        if self._speed_profile is not None:
            # Brake-distance-aware: look up the precomputed safe speed at the
            # current centerline index (already accounts for upcoming corners
            # and how far ahead braking must start).
            cur_idx = f["idx"]
            v_target = torch.minimum(v_target, self._speed_profile[cur_idx])
        else:
            kappa_preview = (
                f["kappa_lookahead"][:, : self.speed_preview_n].abs().max(dim=1).values
            )
            if self.corner_lat_accel > 0.0:
                v_corner = torch.sqrt(
                    self.corner_lat_accel / kappa_preview.clamp_min(1e-4)
                )
                v_target = torch.minimum(v_target, v_corner)
            elif self.curvature_speed_scale > 0.0:
                v_target = v_target / (1.0 + self.curvature_speed_scale * kappa_preview)
        # Per-opponent pace variation.
        v_target = v_target * self.speed_scale
        throttle_norm = (v_target / self.max_wheel_v).clamp(0.0, 1.0)
        return torch.stack([throttle_norm, steer_norm], dim=-1)
