"""Shared helpers for the Itutorial scripts.

Centralises three things:
  - per-robot spec (USD cfg + joint regexes + vehicle constants)
  - default track selection (mini_oval)
  - a tiny Ackermann application helper that works for both F1Tenth and
    MuSHR (which has 4WD throttle, different steer regex).

Kept intentionally small — each tutorial still owns its own `main()`
and camera setup.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Callable

import torch

__all__ = [
    "RobotSpec",
    "get_robot_spec",
    "DEFAULT_TRACK",
    "apply_ackermann",
    "pure_pursuit_steer",
    "quat_to_yaw",
    "post_spawn_fix",
    "spawn_high_friction_ground",
    "apply_high_friction_to_mesh",
]


DEFAULT_TRACK = "mini_oval"


def spawn_high_friction_ground(
    prim_path: str = "/World/ground",
    static_friction: float = 1.5,
    dynamic_friction: float = 1.3,
) -> None:
    """Spawn a ground plane with a grippy physics material.

    Default PhysX friction is ~0.5 which is too low for RC tyres on the
    3D banked oval — vehicles slip on descents and stall on climbs
    (see `_analyze_slip.py`). Using 1.3–1.5 gives realistic rubber-on-
    asphalt traction.
    """
    import isaaclab.sim as sim_utils
    cfg = sim_utils.GroundPlaneCfg(
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=static_friction,
            dynamic_friction=dynamic_friction,
            restitution=0.0,
        ),
    )
    cfg.func(prim_path, cfg)


def apply_high_friction_to_mesh(
    mesh_prim_path: str,
    static_friction: float = 1.5,
    dynamic_friction: float = 1.3,
) -> None:
    """Bind a high-friction PhysicsMaterial to an existing mesh prim.

    Use after spawning the road surface or duct to give the vehicle grip
    on the 3D track. Creates a fresh material under `<prim>/physicsMaterial`
    and binds it via `MaterialBindingAPI`.
    """
    import omni.usd
    from pxr import Sdf, UsdPhysics, UsdShade
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(mesh_prim_path)
    if not prim.IsValid():
        return
    mat_path = f"{mesh_prim_path}_PhysMat"
    if not stage.GetPrimAtPath(mat_path).IsValid():
        mat = UsdShade.Material.Define(stage, mat_path)
        api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
        api.CreateStaticFrictionAttr().Set(float(static_friction))
        api.CreateDynamicFrictionAttr().Set(float(dynamic_friction))
        api.CreateRestitutionAttr().Set(0.0)
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(
        UsdShade.Material(stage.GetPrimAtPath(mat_path)),
        bindingStrength=UsdShade.Tokens.weakerThanDescendants,
        materialPurpose="physics",
    )


def post_spawn_fix(spec: "RobotSpec", prim_path: str) -> None:
    """Robot-specific tweaks to run right after the articulation prim has
    been spawned (before `Articulation(cfg)` or before `sim.reset()`).

    For MuSHR this attaches a chassis box collider so full-body contacts
    work; the stock USD only has wheel + sensor colliders.
    """
    if spec.name == "mushr":
        from hmclab_isaac.robots.racing.mushr_nano_v2.mushr_nano_v2_cfg import (
            add_chassis_collider,
        )
        add_chassis_collider(prim_path)


@dataclass
class RobotSpec:
    name: str
    cfg: object                      # ArticulationCfg (template)
    wheelbase: float
    wheel_radius: float
    max_steer: float
    steer_regex: str
    drive_regex: str
    init_z: float                    # spawn z above ground

    def resolve_joints(self, joint_names: list[str]) -> tuple[list[int], list[int]]:
        steer_re = re.compile(self.steer_regex)
        drive_re = re.compile(self.drive_regex)
        steer_ids = [i for i, n in enumerate(joint_names) if steer_re.search(n)]
        drive_ids = [i for i, n in enumerate(joint_names) if drive_re.search(n)]
        return steer_ids, drive_ids


def get_robot_spec(name: str) -> RobotSpec:
    name = name.lower()
    if name in ("f1tenth", "f1"):
        from isaaclab.actuators import ImplicitActuatorCfg
        from hmclab_isaac.robots.racing.f1tenth_mid360 import (
            F1TENTH_MID360_CFG,
            MAX_STEER,
            WHEEL_RADIUS,
            WHEELBASE,
        )
        # The shared F1Tenth cfg uses stiffness=200/effort=50 for steering,
        # which is too soft for pure-pursuit — `actual_steer` lags the
        # command by ~2 s in tutorials. Override with a stiffer actuator
        # so steering reaches the commanded angle inside one physics step.
        f1_cfg = F1TENTH_MID360_CFG.replace(
            actuators={
                "steering": ImplicitActuatorCfg(
                    joint_names_expr=["front_.*_steering_joint"],
                    effort_limit_sim=500.0,
                    stiffness=8000.0,
                    damping=200.0,
                ),
                # Lower rear-wheel effort so the tires don't break traction
                # on cold start — the original 500 Nm limit caused wheel
                # spin at stand-still, which made the vehicle crawl despite
                # a high velocity target.
                "drive": ImplicitActuatorCfg(
                    joint_names_expr=["rear_.*_wheel_joint"],
                    effort_limit_sim=15.0,
                    stiffness=0.0,
                    damping=30.0,
                ),
                "front_wheels": ImplicitActuatorCfg(
                    joint_names_expr=["front_.*_wheel_joint"],
                    effort_limit_sim=0.0,
                    stiffness=0.0,
                    damping=0.5,
                ),
            }
        )
        return RobotSpec(
            name="f1tenth",
            cfg=f1_cfg,
            wheelbase=WHEELBASE,
            wheel_radius=WHEEL_RADIUS,
            max_steer=MAX_STEER,
            steer_regex=r"steering",
            drive_regex=r"rear.*wheel_joint",
            init_z=0.05,
        )
    if name in ("srx8", "srx_8", "unicorn_srx8"):
        from hmclab_isaac.robots.racing.srx8 import (
            MAX_STEER,
            SRX8_CFG,
            WHEEL_RADIUS,
            WHEELBASE,
        )
        return RobotSpec(
            name="srx8",
            cfg=SRX8_CFG,
            wheelbase=WHEELBASE,
            wheel_radius=WHEEL_RADIUS,
            max_steer=MAX_STEER,
            steer_regex=r"wheel_steer$",
            drive_regex=r"wheel_throttle$",   # 4WD
            init_z=0.08,
        )
    if name in ("mushr", "mushr_nano_v2"):
        from hmclab_isaac.robots.racing.mushr_nano_v2.mushr_nano_v2_cfg import (
            MUSHR_NANO_V2_CFG,
            MAX_STEER,
            WHEEL_RADIUS,
            WHEELBASE,
        )
        return RobotSpec(
            name="mushr",
            cfg=MUSHR_NANO_V2_CFG,
            wheelbase=WHEELBASE,
            wheel_radius=WHEEL_RADIUS,
            max_steer=MAX_STEER,
            steer_regex=r"wheel_steer$",
            # RWD only — 4WD with wheel scrub on tight corners lifted the
            # nose and flipped the car. Driving only the rear pair keeps
            # the chassis planted.
            drive_regex=r"back.*wheel_throttle$",
            init_z=0.08,
        )
    raise ValueError(f"unknown robot: {name}")


def quat_to_yaw(q) -> torch.Tensor | float:
    """Convert a (wxyz) quaternion (scalar or (B,4) tensor) to yaw (rad)."""
    if isinstance(q, torch.Tensor):
        w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
        return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    import math
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def pure_pursuit_steer(
    pos_xy: torch.Tensor,
    yaw: torch.Tensor,
    centerline: torch.Tensor,
    lookahead,
    wheelbase: float,
    *,
    speed: torch.Tensor | None = None,
    lookahead_gain: float = 0.25,
    lookahead_min: float = 0.5,
    lookahead_max: float = 3.0,
) -> torch.Tensor:
    """Pure-pursuit steering from a centerline (with optional speed-adaptive lookahead).

    Args:
        pos_xy:    (B, 2) current vehicle positions.
        yaw:       (B,) current yaw.
        centerline: (N, 2) centerline points (world xy).
        lookahead: base lookahead distance (m). If `speed` is given the
            actual lookahead becomes `clamp(lookahead + gain*speed, min, max)`
            per-vehicle, which is the standard speed-adaptive pure-pursuit.
        wheelbase: robot wheelbase (m).
        speed:     optional (B,) current speed (m/s). If provided, the
            lookahead becomes speed-adaptive.
        lookahead_gain, lookahead_min, lookahead_max: adaptive coeffs.

    Returns (B,) steering angle in radians.

    Algorithm: classic `delta = atan2(2 L sin(alpha), Ld)` with a forward
    walk from the nearest centerline point to the first point at least
    Ld m away. Vectorised over all B vehicles.
    """
    B = pos_xy.shape[0]
    N = centerline.shape[0]
    device = pos_xy.device

    # Per-vehicle lookahead
    if speed is not None:
        ld = (lookahead + lookahead_gain * speed).clamp(lookahead_min, lookahead_max)
    else:
        ld = torch.full((B,), float(lookahead), device=device)

    # Nearest centerline index per vehicle
    diffs = pos_xy.unsqueeze(1) - centerline.unsqueeze(0)
    dists = torch.norm(diffs, dim=-1)
    nearest = dists.argmin(dim=-1)  # (B,)

    # Walk forward along the centerline until the point is at least Ld m
    # away (per-vehicle threshold — so the scan must respect each vehicle's ld).
    max_hops = min(N, max(8, int(float(ld.max().item()) * 6) + 4))
    target_idx = nearest.clone()
    found = torch.zeros(B, dtype=torch.bool, device=device)
    for h in range(1, max_hops + 1):
        probe = (nearest + h) % N
        probe_pts = centerline[probe]
        d = torch.norm(probe_pts - pos_xy, dim=-1)
        take = (~found) & (d >= ld)
        target_idx = torch.where(take, probe, target_idx)
        found = found | take
        if found.all():
            break
    target = centerline[target_idx]  # (B, 2)

    dx = target[:, 0] - pos_xy[:, 0]
    dy = target[:, 1] - pos_xy[:, 1]
    alpha = torch.atan2(dy, dx) - yaw
    alpha = torch.atan2(torch.sin(alpha), torch.cos(alpha))  # wrap to [-pi, pi]
    ld_eff = torch.norm(target - pos_xy, dim=-1).clamp_min(1e-3)
    return torch.atan2(2.0 * wheelbase * torch.sin(alpha), ld_eff)


def apply_ackermann(
    art,
    spec: RobotSpec,
    steer_rad: float | torch.Tensor,
    speed_mps: float | torch.Tensor,
    *,
    steer_ids: list[int] | None = None,
    drive_ids: list[int] | None = None,
    force_steer: bool = True,
) -> None:
    """Apply a batch Ackermann command (steer_rad, speed_mps) to one articulation.

    steer_rad / speed_mps can be scalars or (B,) tensors. If index lists are
    not passed, they're resolved from `art.joint_names`.

    `force_steer=True` directly writes the steering joint positions via
    `write_joint_position_to_sim`, bypassing actuator dynamics. This is a
    teaching-demo shortcut — the shared F1Tenth USD's steering drives lag
    pure-pursuit by ~2 s under any reasonable stiffness, so tutorials
    force-set the angle instead. Pass `force_steer=False` to use the
    normal `set_joint_position_target` path.
    """
    if steer_ids is None or drive_ids is None:
        steer_ids, drive_ids = spec.resolve_joints(art.joint_names)
    num_envs = art.num_instances
    device = art.device
    n_j = art.num_joints

    if not isinstance(steer_rad, torch.Tensor):
        steer_rad = torch.full((num_envs,), float(steer_rad), device=device)
    if not isinstance(speed_mps, torch.Tensor):
        speed_mps = torch.full((num_envs,), float(speed_mps), device=device)

    steer_rad = steer_rad.clamp(-spec.max_steer, spec.max_steer)
    wheel_vel = speed_mps / spec.wheel_radius

    # Drive (velocity targets on rear wheels)
    vel_t = torch.zeros(num_envs, n_j, device=device)
    for idx in drive_ids:
        vel_t[:, idx] = wheel_vel
    art.set_joint_velocity_target(vel_t)

    # Steering (position) — either force-write or PD-target
    if force_steer and len(steer_ids) > 0:
        steer_col = steer_rad.unsqueeze(-1).expand(num_envs, len(steer_ids))
        art.write_joint_position_to_sim(
            steer_col, joint_ids=list(steer_ids)
        )
    else:
        pos_t = torch.zeros(num_envs, n_j, device=device)
        for idx in steer_ids:
            pos_t[:, idx] = steer_rad
        art.set_joint_position_target(pos_t)
