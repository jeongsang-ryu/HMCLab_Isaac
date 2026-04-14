"""Car Tutorial 14 — Suspension rig test (body fixed, lift wheels).

실제 shock absorber 테스트 벤치처럼:
  - 차량 body는 공중에 고정 (kinematic)
  - 각 휠 아래에 lift platform을 두고 천천히 위로 밀어올림
  - 서스펜션이 압축되며 휠 위치가 변하는 거동을 관측

화면 + 콘솔에 prismatic joint position을 mm 단위로 실시간 표시.

Run:
    python Itutorial/car/14_suspension_rig_test.py --robot TOY_01_4WD --duration 8
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--robot", type=str, default="TOY_01_4WD")
parser.add_argument("--duration", type=float, default=10.0)
parser.add_argument("--lift_height", type=float, default=0.025,
                    help="Max platform travel (m)")
parser.add_argument("--cycle_period", type=float, default=4.0,
                    help="Up-down cycle period (s)")
parser.add_argument("--out_dir", type=str,
                    default="/home/js/hmcl_issac_project/HMCLab_Isaac/Itutorial/outputs")
parser.add_argument("--tag", type=str, default="tut14_susrig")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
simulation_app = launcher.app

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402
from pxr import Gf, UsdGeom, UsdPhysics  # noqa: E402
import omni.usd  # noqa: E402

from _common import get_robot_spec  # noqa: E402
from hmclab_isaac.utils.capture import (  # noqa: E402
    TopDownRecorder,
    ChaseCamRecorder,
)


# Wheel ground-contact positions in chassis frame (rough — TOY_01 wheelbase
# 0.2255 m, track 0.20 m, wheel_radius 0.0365 m). Used to place the lift
# platforms directly under each wheel.
WHEEL_OFFSETS = {
    "front_left":  (+0.113, +0.10, 0.0),
    "front_right": (+0.113, -0.10, 0.0),
    "rear_left":   (-0.113, +0.10, 0.0),
    "rear_right":  (-0.113, -0.10, 0.0),
}


def spawn_lift_platforms(stage, hold_z: float):
    """Create 4 thin platforms (kinematic) under each wheel.

    Returns dict[name → prim_path] for later motion control.
    """
    UsdGeom.Xform.Define(stage, "/World/Lifts")
    paths = {}
    for name, (x, y, _) in WHEEL_OFFSETS.items():
        path = f"/World/Lifts/lift_{name}"
        cube = UsdGeom.Cube.Define(stage, path)
        cube.CreateSizeAttr(1.0)
        xf = UsdGeom.Xformable(cube)
        xf.ClearXformOpOrder()
        # Scale to 0.10 × 0.08 × 0.02 m platform
        xf.AddTranslateOp().Set(Gf.Vec3d(x, y, hold_z))
        xf.AddScaleOp().Set(Gf.Vec3f(0.10, 0.08, 0.02))
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        # color: red(FL) green(FR) blue(RL) yellow(RR)
        colors = {
            "front_left": (0.9, 0.2, 0.2),
            "front_right": (0.2, 0.8, 0.2),
            "rear_left": (0.2, 0.4, 0.9),
            "rear_right": (0.9, 0.8, 0.1),
        }
        cube.CreateDisplayColorAttr([Gf.Vec3f(*colors[name])])
        paths[name] = path
    return paths


def fix_body(stage, body_path: str):
    """Make a RigidBody kinematic (body stays anchored)."""
    prim = stage.GetPrimAtPath(body_path)
    if not prim:
        print(f"  ⚠ body not found: {body_path}")
        return
    rb = UsdPhysics.RigidBodyAPI(prim)
    if rb:
        rb.CreateKinematicEnabledAttr(True)
        print(f"  ✓ body fixed (kinematic): {body_path}")


def main():
    spec = get_robot_spec(args.robot)

    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=0.005, device="cuda:0")
    )
    sim_utils.GroundPlaneCfg(
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.5, dynamic_friction=1.3,
        ),
    ).func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=3000.0).func(
        "/World/Light", sim_utils.DomeLightCfg(intensity=3000.0)
    )

    # Spawn the vehicle suspended ~7 cm above ground so wheels initially
    # hang in air; lifts will rise to touch them.
    body_z = 0.10
    cfg = spec.cfg.replace(prim_path="/World/Robot")
    cfg = cfg.replace(init_state=cfg.init_state.replace(pos=(0.0, 0.0, body_z)))
    cfg.spawn.func("/World/Robot", cfg.spawn, translation=(0.0, 0.0, body_z))
    robot = Articulation(cfg)

    # Lift platforms — start at z = 0.005 (near floor)
    stage = omni.usd.get_context().get_stage()
    initial_lift_z = 0.005
    lift_paths = spawn_lift_platforms(stage, hold_z=initial_lift_z)
    # Make each lift a kinematic RigidBody so we can move it programmatically
    for name, p in lift_paths.items():
        prim = stage.GetPrimAtPath(p)
        UsdPhysics.RigidBodyAPI.Apply(prim)
        UsdPhysics.RigidBodyAPI(prim).CreateKinematicEnabledAttr(True)

    # Recorders
    chase = ChaseCamRecorder(
        out_dir=args.out_dir, tag=f"{args.tag}_{args.robot}_chase",
        behind=1.2, above=0.6, max_frames=int(args.duration * 30) + 30,
    )
    top = TopDownRecorder(
        center=(0.0, 0.0), size=1.5,
        out_dir=args.out_dir, tag=f"{args.tag}_{args.robot}_topdown",
        max_frames=int(args.duration * 30) + 30,
        height_above=1.5,
    )

    sim.reset()

    # Now anchor the chassis body
    fix_body(stage, "/World/Robot/base_link")

    chase.on_reset()
    top.on_reset()

    # Joint indices
    sus_ids = [i for i, n in enumerate(robot.joint_names) if "sus_joint" in n]
    print(f"\n  Suspension joints ({len(sus_ids)}): "
          f"{[robot.joint_names[i] for i in sus_ids]}")

    n_j = robot.num_joints
    device = robot.device
    dt = sim.get_physics_dt()

    print(f"""
{'='*65}
  Suspension Rig Test — {spec.name}
  body fixed at z={body_z}m, lifts oscillate ±{args.lift_height*1000:.0f}mm
  cycle period = {args.cycle_period}s, duration = {args.duration}s
{'='*65}
""")

    n_steps = int(args.duration / dt)
    capture_every = max(1, int(round(1.0 / (30 * dt))))

    # Order matches actual joint name order for clean prints
    for step in range(n_steps):
        t = step * dt

        # Sinusoidal lift profile — start at 0 and oscillate up to lift_height
        # All four lifts move in phase (uniform compression)
        # phase = -pi/2 so we start at the lowest point and rise smoothly
        norm = 0.5 * (1 - math.cos(2 * math.pi * t / args.cycle_period))
        lift_z = initial_lift_z + args.lift_height * norm

        # Update each lift platform z
        for name, path in lift_paths.items():
            prim = stage.GetPrimAtPath(path)
            xf = UsdGeom.Xformable(prim)
            for op in xf.GetOrderedXformOps():
                if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                    cur = op.Get()
                    op.Set(Gf.Vec3d(float(cur[0]), float(cur[1]), lift_z))
                    break

        # Zero all joint targets — let gravity + spring do the work
        vel_t = torch.zeros(1, n_j, device=device)
        robot.set_joint_velocity_target(vel_t)
        robot.write_data_to_sim()

        sim.step()
        robot.update(dt)

        if step % capture_every == 0:
            chase.update_follow(vehicle_pos=(0.0, 0.0, body_z), yaw_rad=0.0)
            chase.capture_frame()
            top.capture_frame()

            if step % (capture_every * 5) == 0:
                if sus_ids:
                    pos_mm = robot.data.joint_pos[0, sus_ids].cpu().numpy() * 1000
                    sus_str = "  ".join(
                        f"{robot.joint_names[sus_ids[i]].split('_wheel')[0]}={p:+5.1f}mm"
                        for i, p in enumerate(pos_mm)
                    )
                    print(f"  t={t:5.2f}s lift={lift_z*1000:5.1f}mm  {sus_str}",
                          flush=True)

    top_paths = top.finalize()
    chase_paths = chase.finalize()
    print(f"\n  Top   : {top_paths}")
    print(f"  Chase : {chase_paths}")

    simulation_app.close()


if __name__ == "__main__":
    main()
