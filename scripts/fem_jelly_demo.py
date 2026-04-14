"""Minimal FEM deformable-body demo — jelly cube drops onto ground.

Purpose: validate that Isaac Sim 5.1's deformable-body pipeline works in this
env before attempting the heavier MuSHR-tire integration.

Run:
    OMNI_KIT_ACCEPT_EULA=YES \
      /home/js/anaconda3/envs/hmclab_test/bin/python \
      scripts/fem_jelly_demo.py
"""

from __future__ import annotations

import os
import sys
import time

from isaaclab.app import AppLauncher

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=900)
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--youngs", type=float, default=5e5)
parser.add_argument("--label", type=str, default="jelly")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = False
args.enable_cameras = True
launcher = AppLauncher(args)
sim_app = launcher.app

import isaaclab.sim as sim_utils  # noqa: E402

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Itutorial")
)
from hmclab_isaac.utils.capture import ChaseCamRecorder  # noqa: E402

OUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "Itutorial", "outputs"
)


def make_deformable_cube(prim_path: str, size: float, youngs: float, pos):
    """Create a tet-mesh cube as a FEM deformable body at `pos`."""
    import omni.usd
    from pxr import Gf, UsdGeom, UsdPhysics, PhysxSchema, Sdf

    stage = omni.usd.get_context().get_stage()

    # Create mesh prim — use a UsdGeom.Mesh cube by authoring geometry.
    mesh = UsdGeom.Mesh.Define(stage, prim_path)

    # Unit cube vertices (8 corners), scaled by `size`
    s = size / 2.0
    pts = [
        (-s, -s, -s), (s, -s, -s), (s, s, -s), (-s, s, -s),
        (-s, -s, s), (s, -s, s), (s, s, s), (-s, s, s),
    ]
    mesh.CreatePointsAttr([Gf.Vec3f(*p) for p in pts])
    # 12 triangles — PhysX cooking requires triangle meshes
    tris = [
        # bottom (-z): 0,1,2 / 0,2,3
        0, 1, 2,   0, 2, 3,
        # top (+z): 4,6,5 / 4,7,6
        4, 6, 5,   4, 7, 6,
        # -y side: 0,4,5 / 0,5,1
        0, 4, 5,   0, 5, 1,
        # +x side: 1,5,6 / 1,6,2
        1, 5, 6,   1, 6, 2,
        # +y side: 2,6,7 / 2,7,3
        2, 6, 7,   2, 7, 3,
        # -x side: 3,7,4 / 3,4,0
        3, 7, 4,   3, 4, 0,
    ]
    mesh.CreateFaceVertexCountsAttr([3] * 12)
    mesh.CreateFaceVertexIndicesAttr(tris)

    # Transform
    xform = UsdGeom.Xformable(mesh)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*pos))

    prim = mesh.GetPrim()

    # Use the high-level helper from omni.physx.scripts
    from omni.physx.scripts import deformableUtils as du
    from pxr import UsdShade

    try:
        ok = du.add_physx_deformable_body(
            stage,
            prim_path,
            simulation_hexahedral_resolution=8,
            solver_position_iteration_count=32,
            self_collision=False,
            collision_simplification=True,
        )
        if not ok:
            print(f"[fem] add_physx_deformable_body returned False", flush=True)
            return False

        # Create deformable material and bind
        mat_path = f"{prim_path}_deform_mat"
        du.add_deformable_body_material(
            stage,
            mat_path,
            youngs_modulus=float(youngs),
            poissons_ratio=0.45,
            damping_scale=0.0,
            dynamic_friction=0.5,
            elasticity_damping=0.005,
            density=1000.0,
        )
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            UsdShade.Material(stage.GetPrimAtPath(mat_path)),
            bindingStrength=UsdShade.Tokens.strongerThanDescendants,
            materialPurpose="physics",
        )
        print(f"[fem] deformable body created at {prim_path} E={youngs}", flush=True)
        return True
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f"[fem] failed: {exc}", flush=True)
        return False


def main() -> int:
    tag = f"fem_jelly_{args.label}"

    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device="cuda:0")
    sim = sim_utils.SimulationContext(sim_cfg)
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=2500.0, color=(0.95, 0.95, 0.95)).func(
        "/World/Light",
        sim_utils.DomeLightCfg(intensity=2500.0, color=(0.95, 0.95, 0.95)),
    )

    ok = make_deformable_cube(
        "/World/Jelly", size=0.2, youngs=args.youngs, pos=(0.0, 0.0, 0.6)
    )
    if not ok:
        print("[fem] aborting — deformable creation failed", flush=True)
        return 1

    side = ChaseCamRecorder(
        out_dir=OUT_DIR,
        tag=tag,
        prim_path="/World/SideCam",
        behind=0.0, above=0.0, look_up=0.0,
        width=960, height=540,
    )

    sim.reset()
    side.on_reset()

    import torch
    dev = sim.device
    eyes = torch.tensor([[0.0, 0.8, 0.35]], device=dev, dtype=torch.float32)
    tgts = torch.tensor([[0.0, 0.0, 0.1]], device=dev, dtype=torch.float32)
    side._camera.set_world_poses_from_view(eyes, tgts)

    dt = sim.get_physics_dt()
    frame_step = max(1, int(round(1.0 / (args.fps * dt))))
    step_times = []

    for i in range(args.steps):
        t0 = time.perf_counter()
        sim.step()
        step_times.append(time.perf_counter() - t0)
        if i % frame_step == 0:
            side._camera.set_world_poses_from_view(eyes, tgts)
            side.capture_frame()

    out = side.finalize()
    print(f"JELLY_DONE outputs={out}", flush=True)

    import statistics
    warm = step_times[30:]
    if warm:
        print(
            f"STEP_STATS label={args.label} median={statistics.median(warm)*1000:.3f}ms "
            f"mean={statistics.mean(warm)*1000:.3f}ms "
            f"p95={sorted(warm)[int(len(warm)*0.95)]*1000:.3f}ms n={len(warm)}",
            flush=True,
        )

    return 0


if __name__ == "__main__":
    import traceback
    rc = 0
    try:
        rc = main()
    except Exception:
        traceback.print_exc()
        rc = 1
    os._exit(rc)
