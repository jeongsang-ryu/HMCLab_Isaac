"""GPU-batched Ackermann wheel control for 4-wheel articulations.

Two control modes:
    apply_speed_steer(target_speed_mps, steer_rad)     — physical units
    apply_throttle_steer(throttle_norm, steer_norm)    — both in [-1, 1]

Tensor shapes:
    Per-env scalars are accepted as Python floats, 0-D tensors, or [N_envs]
    tensors. Internal buffers are [N_envs, 4] (per-wheel) and
    [N_envs, len(steer_corners)] (per-steer-joint). All on `art.device`.

Joint resolution (`WheelDriveController`):
    Pass a regex (e.g. ".*_wheel_joint") or an explicit list. Each matched
    joint name must start with one of the corner tokens "fl_", "fr_",
    "rl_", "rr_" so the controller can sort matches into a stable
    (fl, fr, rl, rr) order. This matches the naming convention produced by
    `scripts/restructure_wishbone_vehicle.py`.

Pure-function math (`ackermann_split`, `wheel_velocities`) is exposed
separately so an Isaac Lab `ActionTerm` subclass can call it directly with
its own joint-id resolution and tensor handling.

Standalone usage gotcha: between `set_joint_*_target(...)` and `sim.step()`
you must call `art.write_data_to_sim()` (Isaac Lab does this automatically
inside RL `ManagerBasedEnv` / `DirectRLEnv` step pipelines, but not in
hand-written sim loops).

Sign conventions:
    + steer_angle = left turn (yaw axis = world +Z, right-handed).
    + speed       = forward.
If your USD's steering joint axis is mirrored, flip the sign of the input
`steer_angle` (or apply a per-corner sign in your own wrapper).

Example:
    from hmclab_isaac.utils.wheel_control import (
        AckermannParams, WheelDriveController,
    )

    params = AckermannParams(
        wheelbase=0.32, track_width=0.20, wheel_radius=0.05,
        max_steer=0.45, max_speed=5.0,
        use_ackermann=True,
    )
    ctrl = WheelDriveController(articulation, params)
    ctrl.apply_speed_steer(target_speed=3.0, steer_angle=0.15)
    articulation.write_data_to_sim()
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch
from torch import Tensor


CORNER_KEYS: tuple[str, ...] = ("fl", "fr", "rl", "rr")


@dataclass
class AckermannParams:
    wheelbase: float
    """L: distance from front axle to rear axle [m]."""

    track_width: float
    """W: lateral distance between left- and right-wheel centers [m]."""

    wheel_radius: float
    """r: rolling radius of the driven wheel [m]."""

    max_steer: float
    """|steer_angle| clamp [rad] — applies to the *input* command before
    the Ackermann split, i.e. the equivalent bicycle-model steering."""

    max_speed: float
    """Forward speed at throttle = 1.0 [m/s]. Reverse at -1.0 unless
    `no_reverse=True`."""

    use_ackermann: bool = True
    """If False, both front wheels receive the same angle and all driven
    wheels receive the same angular velocity (simple symmetric steering)."""

    no_reverse: bool = False
    """If True, target speed is clamped to >= 0 (no reverse)."""

    drive_corners: tuple[str, ...] = ("fl", "fr", "rl", "rr")
    """Which wheels are driven (velocity-controlled). Others are not
    written each step — they freewheel under their actuator's natural
    damping. Use ("rl", "rr") for RWD, etc."""

    steer_corners: tuple[str, ...] = ("fl", "fr")
    """Which corners have a steering joint. For non-front-only steering
    (e.g. 4WS), include the rear corners — but the same Ackermann split is
    applied with rear corners getting the negated angle."""

    steer_sign: tuple[float, float] = (1.0, 1.0)
    """Per-side multiplier (fl, fr) applied to the Ackermann split
    targets. Use -1 on either side if its `physics:axis` direction
    (after `physics:localRot0`) is opposite from what you expect — left
    and right wishbone knuckles are typically MIRRORED, so one side
    often needs flipping. Diagnostic: at zero command, drive forward
    briefly. If one wheel hardly moves under steer commands while the
    other swings wildly, the unbalanced side is the one with a
    mismatched joint axis — flip it."""

    steer_offset: tuple[float, float] = (0.0, 0.0)
    """Per-side (fl, fr) calibration added AFTER sign flip. Use to
    cancel toe-in/toe-out built into the USD's rest pose. Set both
    sides individually because the USD may have asymmetric authoring
    on left vs right knuckles."""

    drive_sign: float = 1.0
    """Multiplier applied to all driven wheel angular velocities. Use
    -1.0 if positive `target_speed` makes the vehicle reverse (wheel
    joint axis flipped)."""


# ===================================================================
# Pure functions (stateless, GPU-friendly, no articulation dependency)
# ===================================================================
def ackermann_split(
    steer: Tensor, params: AckermannParams
) -> tuple[Tensor, Tensor]:
    """Per-wheel front-axle steering angles given a bicycle-model command.

    Returns (delta_left, delta_right), same shape and device as `steer`.
    For the inside wheel of a turn, |delta| is larger than `steer`; for
    the outside wheel, smaller. At zero steer both outputs are exactly 0.
    """
    if not params.use_ackermann:
        return steer, steer
    L = params.wheelbase
    W = params.track_width
    eps = 1e-6
    tan_s = torch.tan(steer)
    is_straight = torch.abs(tan_s) < eps
    safe_tan = torch.where(is_straight, torch.full_like(tan_s, eps), tan_s)
    R = L / safe_tan  # signed turning radius at vehicle center
    delta_l = torch.atan(L / (R - W * 0.5))
    delta_r = torch.atan(L / (R + W * 0.5))
    zeros = torch.zeros_like(steer)
    delta_l = torch.where(is_straight, zeros, delta_l)
    delta_r = torch.where(is_straight, zeros, delta_r)
    return delta_l, delta_r


def wheel_velocities(
    speed: Tensor, steer: Tensor, params: AckermannParams
) -> Tensor:
    """Per-wheel angular velocity targets [rad/s], shape [..., 4] in
    (fl, fr, rl, rr) order.

    On a turn, the wheel farther from the instantaneous center spins
    faster (longer arc per unit time). Without this correction the
    inside/outside wheels scrub the ground.
    """
    L = params.wheelbase
    W = params.track_width
    r = params.wheel_radius
    eps = 1e-6

    if not params.use_ackermann:
        omega = speed / r
        return omega.unsqueeze(-1).expand(*omega.shape, 4)

    tan_s = torch.tan(steer)
    is_straight = torch.abs(tan_s) < eps
    safe_tan = torch.where(is_straight, torch.full_like(tan_s, eps), tan_s)
    R = L / safe_tan
    abs_R = torch.abs(R)

    R_fl = torch.sqrt((R - W * 0.5) ** 2 + L ** 2)
    R_fr = torch.sqrt((R + W * 0.5) ** 2 + L ** 2)
    R_rl = torch.abs(R - W * 0.5)
    R_rr = torch.abs(R + W * 0.5)

    v_fl = speed * R_fl / abs_R / r
    v_fr = speed * R_fr / abs_R / r
    v_rl = speed * R_rl / abs_R / r
    v_rr = speed * R_rr / abs_R / r

    v_straight = speed / r
    v_fl = torch.where(is_straight, v_straight, v_fl)
    v_fr = torch.where(is_straight, v_straight, v_fr)
    v_rl = torch.where(is_straight, v_straight, v_rl)
    v_rr = torch.where(is_straight, v_straight, v_rr)

    return torch.stack([v_fl, v_fr, v_rl, v_rr], dim=-1)


# ===================================================================
# Articulation helper
# ===================================================================
class WheelDriveController:
    """Resolves wheel/steer joint indices on an Articulation and applies
    Ackermann targets each sim step.

    Joint resolution requires names that start with a corner token —
    e.g. `fl_wheel_joint`, `rr_steering_joint`. Multiple matches per
    corner or zero matches will raise.
    """

    def __init__(
        self,
        articulation,
        params: AckermannParams,
        wheel_joint_pattern: str | Sequence[str] = ".*_wheel_joint",
        steering_joint_pattern: str | Sequence[str] = ".*_steering_joint",
    ):
        self.params = params
        self.art = articulation
        self.device = articulation.device
        self.num_envs = articulation.num_instances

        self._wheel_id_by_corner = self._resolve_corner_joints(
            wheel_joint_pattern, CORNER_KEYS
        )
        self._steer_id_by_corner = self._resolve_corner_joints(
            steering_joint_pattern, params.steer_corners
        )

        # Driven-wheel ids in (fl, fr, rl, rr) sub-order, matching the
        # wheel_velocities() output we will index out of.
        self._drive_corners = tuple(
            c for c in CORNER_KEYS if c in params.drive_corners
        )
        self._drive_ids = [self._wheel_id_by_corner[c] for c in self._drive_corners]
        self._drive_idx_in_4 = torch.tensor(
            [CORNER_KEYS.index(c) for c in self._drive_corners],
            device=self.device,
            dtype=torch.long,
        )

        # Steer ids in `params.steer_corners` order.
        self._steer_ids = [self._steer_id_by_corner[c] for c in params.steer_corners]

    # -----------------------------------------------------------------
    def _resolve_corner_joints(
        self, pattern: str | Sequence[str], corners: Sequence[str]
    ) -> dict[str, int]:
        ids, names = self.art.find_joints(pattern)
        out: dict[str, int] = {}
        for c in corners:
            matches = [(i, n) for i, n in zip(ids, names) if n.startswith(c + "_")]
            if not matches:
                raise ValueError(
                    f"WheelDriveController: no joint with corner prefix "
                    f"'{c}_' matched pattern {pattern!r}. "
                    f"Joints found by pattern: {names}"
                )
            if len(matches) > 1:
                raise ValueError(
                    f"WheelDriveController: multiple joints matched corner "
                    f"prefix '{c}_': {[n for _, n in matches]}"
                )
            out[c] = matches[0][0]
        return out

    # -----------------------------------------------------------------
    def _as_batch(self, x) -> Tensor:
        if not isinstance(x, Tensor):
            return torch.full((self.num_envs,), float(x), device=self.device)
        if x.device != self.device:
            x = x.to(self.device)
        if x.ndim == 0:
            x = x.expand(self.num_envs)
        elif x.ndim > 1:
            x = x.reshape(-1)
        return x

    # -----------------------------------------------------------------
    def apply_speed_steer(
        self, target_speed, steer_angle
    ) -> None:
        """Write joint targets for a (speed [m/s], steer [rad]) command.
        Caller is responsible for `articulation.write_data_to_sim()` and
        `sim.step()` afterwards."""
        v = self._as_batch(target_speed)
        s = self._as_batch(steer_angle).clamp(
            -self.params.max_steer, self.params.max_steer
        )
        if self.params.no_reverse:
            v = v.clamp(min=0.0)

        omegas_4 = wheel_velocities(v, s, self.params)  # [N, 4] in (fl,fr,rl,rr)
        # Pick out only the driven corners' columns; apply optional sign flip.
        drive_targets = omegas_4.index_select(-1, self._drive_idx_in_4)
        drive_targets = drive_targets * self.params.drive_sign

        delta_l, delta_r = ackermann_split(s, self.params)
        # Apply per-side sign flip and calibration offset.
        delta_l = self.params.steer_sign[0] * delta_l + self.params.steer_offset[0]
        delta_r = self.params.steer_sign[1] * delta_r + self.params.steer_offset[1]

        # Build the steer target tensor in `params.steer_corners` order.
        per_corner_steer = {
            "fl": delta_l, "fr": delta_r,
            # Rear-steer convention: opposite phase of front (e.g. crab steer).
            "rl": -delta_l, "rr": -delta_r,
        }
        steer_targets = torch.stack(
            [per_corner_steer[c] for c in self.params.steer_corners], dim=-1
        )

        self.art.set_joint_velocity_target(drive_targets, joint_ids=self._drive_ids)
        self.art.set_joint_position_target(steer_targets, joint_ids=self._steer_ids)

    # -----------------------------------------------------------------
    def apply_throttle_steer(self, throttle, steer) -> None:
        """Write joint targets for a normalized (throttle, steer) command,
        each in [-1, 1]. Internally scales by `max_speed`/`max_steer`."""
        target_speed = self._as_batch(throttle) * self.params.max_speed
        steer_angle = self._as_batch(steer) * self.params.max_steer
        self.apply_speed_steer(target_speed, steer_angle)

    # -----------------------------------------------------------------
    @property
    def drive_joint_ids(self) -> list[int]:
        """Articulation joint indices of the *driven* wheels, in
        (fl, fr, rl, rr) sub-order filtered by `params.drive_corners`."""
        return list(self._drive_ids)

    @property
    def steer_joint_ids(self) -> list[int]:
        """Articulation joint indices of the steering joints in
        `params.steer_corners` order."""
        return list(self._steer_ids)

    @property
    def wheel_joint_ids(self) -> dict[str, int]:
        """Per-corner wheel joint indices (all 4)."""
        return dict(self._wheel_id_by_corner)
