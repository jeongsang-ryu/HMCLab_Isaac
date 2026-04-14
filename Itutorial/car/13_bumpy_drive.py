"""Car Tutorial 13 — 울퉁불퉁한 지형에서 임의 제어 주행 데모.

목적: MuSHR의 prismatic suspension joint가 실제로 작동하는 것을 확인.

  - 작은 박스/실린더를 무작위로 흩뿌려 bumpy terrain 생성
  - 차량을 위에 스폰
  - 시간에 따라 변하는 sinusoidal throttle/steer 입력 (랜덤 시드 기반)
  - top-down + chase-cam 동시 녹화 → MP4

Run:
    python Itutorial/car/13_bumpy_drive.py --robot MUSHR_01 --headless
    python Itutorial/car/13_bumpy_drive.py --robot MUSHR_01 --duration 10
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys

import numpy as np
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--robot", type=str, default="MUSHR_01")
parser.add_argument("--duration", type=float, default=8.0,
                    help="Recording duration in seconds")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--n_bumps", type=int, default=80)
parser.add_argument("--bump_area", type=float, default=8.0,
                    help="Bumps scattered in ±area square (m)")
parser.add_argument("--out_dir", type=str,
                    default="/home/js/hmcl_issac_project/HMCLab_Isaac/Itutorial/outputs")
parser.add_argument("--tag", type=str, default="tut13_bumpy")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
simulation_app = launcher.app

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402
from pxr import Gf, UsdGeom, UsdPhysics  # noqa: E402
import omni.usd  # noqa: E402

from _common import get_robot_spec, quat_to_yaw  # noqa: E402
from hmclab_isaac.utils.capture import (  # noqa: E402
    TopDownRecorder,
    ChaseCamRecorder,
)


def spawn_bumps(stage, n: int, area: float, seed: int) -> None:
    """Scatter small static cuboid bumps over a square area."""
    rng = random.Random(seed)
    parent = "/World/Bumps"
    UsdGeom.Xform.Define(stage, parent)
    for i in range(n):
        x = rng.uniform(-area, area)
        y = rng.uniform(-area, area)
        # Skip the spawn zone so we don't drop the car onto a bump
        if abs(x) < 0.6 and abs(y) < 0.6:
            continue
        sx = rng.uniform(0.10, 0.30)
        sy = rng.uniform(0.10, 0.30)
        sz = rng.uniform(0.015, 0.045)
        path = f"{parent}/bump_{i:03d}"
        cube = UsdGeom.Cube.Define(stage, path)
        cube.CreateSizeAttr(1.0)
        xf = UsdGeom.Xformable(cube)
        xf.ClearXformOpOrder()
        xf.AddTranslateOp().Set(Gf.Vec3d(x, y, sz / 2))
        xf.AddScaleOp().Set(Gf.Vec3f(sx, sy, sz))
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        # color the bumps a bit so they're visible
        cube.CreateDisplayColorAttr([Gf.Vec3f(0.55, 0.35, 0.20)])


def main():
    spec = get_robot_spec(args.robot)

    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=0.01, device="cuda:0")
    )

    # ── Ground + lighting ──
    sim_utils.GroundPlaneCfg(
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.5, dynamic_friction=1.3,
        ),
    ).func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=3000.0).func(
        "/World/Light", sim_utils.DomeLightCfg(intensity=3000.0)
    )

    # ── Bumps ──
    stage = omni.usd.get_context().get_stage()
    spawn_bumps(stage, n=args.n_bumps, area=args.bump_area, seed=args.seed)

    # ── Robot — spawn higher so it lands and settles ──
    cfg = spec.cfg.replace(prim_path="/World/Robot")
    init_z = spec.init_z + 0.10
    cfg = cfg.replace(init_state=cfg.init_state.replace(pos=(0.0, 0.0, init_z)))
    cfg.spawn.func("/World/Robot", cfg.spawn, translation=(0.0, 0.0, init_z))
    robot = Articulation(cfg)

    # ── Recorders ──
    top = TopDownRecorder(
        center=(0.0, 0.0), size=args.bump_area * 1.2,
        out_dir=args.out_dir, tag=f"{args.tag}_{args.robot}_topdown",
        max_frames=int(args.duration * 30) + 30,
    )
    chase = ChaseCamRecorder(
        out_dir=args.out_dir, tag=f"{args.tag}_{args.robot}_chase",
        behind=2.0, above=1.2, max_frames=int(args.duration * 30) + 30,
    )

    sim.reset()
    top.on_reset()
    chase.on_reset()

    steer_ids, drive_ids = spec.resolve_joints(robot.joint_names)
    # Force 4WD-ish: apply throttle to ALL drive joints in the regex result
    n_j = robot.num_joints
    device = robot.device
    dt = sim.get_physics_dt()

    print(f"""
{'='*60}
  Bumpy Drive Test — {spec.name}
  joints: {robot.joint_names}
  steer_ids: {steer_ids}  drive_ids: {drive_ids}
  bumps: {args.n_bumps} over ±{args.bump_area} m
  duration: {args.duration} s
{'='*60}
""")

    # ── Settle (let suspension compress under gravity) ──
    settle_steps = 80
    for _ in range(settle_steps):
        vel_t = torch.zeros(1, n_j, device=device)
        robot.set_joint_velocity_target(vel_t)
        robot.write_data_to_sim()
        sim.step()
        robot.update(dt)

    # ── Drive loop with sinusoidal random control ──
    rng = np.random.default_rng(args.seed)
    # Random freq/phase so each run looks different but smooth
    f_throttle = rng.uniform(0.15, 0.30)
    f_steer1 = rng.uniform(0.20, 0.40)
    f_steer2 = rng.uniform(0.05, 0.12)
    phase_t = rng.uniform(0, math.tau)
    phase_s = rng.uniform(0, math.tau)

    n_steps = int(args.duration / dt)
    capture_every = max(1, int(round(1.0 / (30 * dt))))  # ~30 fps

    max_wheel_vel = 2.0 / spec.wheel_radius  # ~2 m/s top speed
    print(f"  control: throttle freq={f_throttle:.2f}Hz, "
          f"steer freqs=({f_steer1:.2f}, {f_steer2:.2f})Hz")

    for step in range(n_steps):
        t = step * dt

        # Throttle: oscillates 0.4 ~ 1.0 (no reverse — keeps it simple)
        throttle_norm = 0.7 + 0.3 * math.sin(2 * math.pi * f_throttle * t + phase_t)
        # Steering: combination of two sines for pseudo-random S-curve
        steer_norm = (
            0.7 * math.sin(2 * math.pi * f_steer1 * t + phase_s)
            + 0.3 * math.sin(2 * math.pi * f_steer2 * t)
        )
        steer_norm = max(-1.0, min(1.0, steer_norm))

        steer_val = steer_norm * spec.max_steer
        wheel_vel = throttle_norm * max_wheel_vel

        # Apply steering (force write so it tracks immediately)
        if steer_ids:
            steer_t = torch.full((1, len(steer_ids)), steer_val, device=device)
            robot.write_joint_position_to_sim(steer_t, joint_ids=list(steer_ids))

        # Apply throttle to ALL drive joints (4WD if MuSHR has 4 drives)
        if drive_ids:
            vel_t = torch.zeros(1, n_j, device=device)
            for di in drive_ids:
                vel_t[0, di] = wheel_vel
            robot.set_joint_velocity_target(vel_t)

        robot.write_data_to_sim()
        sim.step()
        robot.update(dt)

        # ── Capture ──
        if step % capture_every == 0:
            pos = robot.data.root_pos_w[0].cpu().numpy()
            quat = robot.data.root_quat_w[0].cpu().numpy()
            yaw = float(quat_to_yaw(quat))
            top.capture_frame()
            chase.update_follow(vehicle_pos=pos, yaw_rad=yaw)
            chase.capture_frame()

            if step % (capture_every * 10) == 0:
                # Read suspension joint positions if any exist
                sus_status = ""
                sus_ids = [i for i, n in enumerate(robot.joint_names)
                           if "suspension" in n]
                if sus_ids:
                    sus_pos = robot.data.joint_pos[0, sus_ids].cpu().numpy()
                    sus_status = f"  sus=[{', '.join(f'{p*1000:+.1f}mm' for p in sus_pos)}]"
                print(
                    f"  t={t:5.2f}s pos=({pos[0]:+5.2f},{pos[1]:+5.2f},{pos[2]:.3f}) "
                    f"yaw={math.degrees(yaw):+6.1f}° "
                    f"thr={throttle_norm:.2f} str={steer_norm:+.2f}{sus_status}",
                    flush=True,
                )

    # ── Save ──
    top_paths = top.finalize()
    chase_paths = chase.finalize()
    print(f"\n  Top-down: {top_paths}")
    print(f"  Chase   : {chase_paths}")

    simulation_app.close()


if __name__ == "__main__":
    main()
