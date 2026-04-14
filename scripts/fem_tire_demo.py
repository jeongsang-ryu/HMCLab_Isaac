"""FEM deformable tire demo.

Drops (or presses) a tire-shaped deformable body onto the ground to show
the jelly-physics applied to a wheel shape, not a cube.

Run:
    OMNI_KIT_ACCEPT_EULA=YES \
      /home/js/anaconda3/envs/hmclab_test/bin/python \
      scripts/fem_tire_demo.py --steps 800 --youngs 3e5 --radius 0.08 --label tire
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=800)
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--youngs", type=float, default=3e5)
parser.add_argument("--radius", type=float, default=0.08)
parser.add_argument("--tire-width", type=float, default=0.05, dest="tire_width")
parser.add_argument("--segments", type=int, default=24)
parser.add_argument("--drop-height", type=float, default=0.35)
parser.add_argument("--label", type=str, default="tire")
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


def make_cylinder_mesh(prim_path: str, radius: float, width: float, segments: int, pos):
    """Create a closed triangle-mesh cylinder (tire-shaped) and return its prim."""
    import omni.usd
    from pxr import Gf, UsdGeom

    stage = omni.usd.get_context().get_stage()
    mesh = UsdGeom.Mesh.Define(stage, prim_path)

    # Cylinder axis along Y (so the "tire" rolls in the X direction when laid flat).
    # Here we just make a standing cylinder with axis along Z so it sits on the
    # ground on its side: actually simplest is axis along Y and drop it.
    # We'll generate with axis along Y and spawn lying with axis horizontal.
    hw = width / 2.0

    pts: list = []
    # Side vertices: 2 rings at y=-hw and y=+hw
    for ring_y in (-hw, +hw):
        for i in range(segments):
            theta = 2 * math.pi * i / segments
            pts.append(Gf.Vec3f(radius * math.cos(theta), ring_y, radius * math.sin(theta)))
    # Cap centers
    pts.append(Gf.Vec3f(0.0, -hw, 0.0))   # index = 2*segments
    pts.append(Gf.Vec3f(0.0, +hw, 0.0))   # index = 2*segments + 1

    mesh.CreatePointsAttr(pts)

    tris: list = []
    # Side: quad strip between two rings -> 2 triangles each segment
    for i in range(segments):
        a = i
        b = (i + 1) % segments
        c = segments + b
        d = segments + a
        # quad a-b-c-d -> triangles a,b,c and a,c,d (outward normals)
        tris += [a, b, c,  a, c, d]
    # Bottom cap (ring y=-hw): center = 2s, fan
    cb = 2 * segments
    for i in range(segments):
        a = i
        b = (i + 1) % segments
        tris += [cb, b, a]  # inward to bottom so normal points -Y
    # Top cap (ring y=+hw): center = 2s+1, fan opposite winding
    ct = 2 * segments + 1
    for i in range(segments):
        a = segments + i
        b = segments + ((i + 1) % segments)
        tris += [ct, a, b]

    mesh.CreateFaceVertexCountsAttr([3] * (len(tris) // 3))
    mesh.CreateFaceVertexIndicesAttr(tris)

    xform = UsdGeom.Xformable(mesh)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*pos))
    # Lie the cylinder on its side — rotate +90° around X so axis that was Y now aligns with world Z? Actually we want axis horizontal (parallel to ground).
    # Our cylinder axis is Y already. To make it roll like a tire, keep axis along Y (also a horizontal dir). That's fine — it already "lies" with circular face vertical.
    return mesh.GetPrim()


def make_deformable(prim, youngs: float):
    import omni.usd
    from omni.physx.scripts import deformableUtils as du
    from pxr import UsdShade

    stage = omni.usd.get_context().get_stage()
    prim_path = prim.GetPath().pathString
    ok = du.add_physx_deformable_body(
        stage,
        prim_path,
        simulation_hexahedral_resolution=10,
        solver_position_iteration_count=32,
        self_collision=False,
        collision_simplification=True,
    )
    if not ok:
        return False
    # Tighten collision offsets so the tire visibly touches the ground
    from pxr import PhysxSchema
    col_api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    col_api.CreateContactOffsetAttr(0.002)
    col_api.CreateRestOffsetAttr(0.0)
    mat_path = f"{prim_path}_mat"
    du.add_deformable_body_material(
        stage,
        mat_path,
        youngs_modulus=float(youngs),
        poissons_ratio=0.45,
        damping_scale=0.0,
        dynamic_friction=1.2,
        elasticity_damping=0.005,
        density=1200.0,
    )
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(
        UsdShade.Material(stage.GetPrimAtPath(mat_path)),
        bindingStrength=UsdShade.Tokens.strongerThanDescendants,
        materialPurpose="physics",
    )
    return True


def main() -> int:
    tag = f"fem_tire_{args.label}"
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=0.005, device="cuda:0")
    )
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=2500.0, color=(0.95, 0.95, 0.95)).func(
        "/World/Light",
        sim_utils.DomeLightCfg(intensity=2500.0, color=(0.95, 0.95, 0.95)),
    )

    prim = make_cylinder_mesh(
        "/World/Tire",
        radius=args.radius,
        width=args.tire_width,
        segments=args.segments,
        pos=(0.0, 0.0, args.drop_height),
    )
    if not make_deformable(prim, args.youngs):
        print("[fem_tire] failed to cook deformable", flush=True)
        return 1
    print(f"[fem_tire] OK r={args.radius} w={args.tire_width} E={args.youngs}", flush=True)

    side = ChaseCamRecorder(
        out_dir=OUT_DIR, tag=tag, prim_path="/World/SideCam",
        behind=0.0, above=0.0, look_up=0.0, width=960, height=540,
    )
    sim.reset()
    side.on_reset()

    import torch
    dev = sim.device
    # Low side-view so we can see the ground contact cleanly
    eyes = torch.tensor([[0.0, 0.50, 0.06]], device=dev, dtype=torch.float32)
    tgts = torch.tensor([[0.0, 0.0, 0.04]], device=dev, dtype=torch.float32)
    side._camera.set_world_poses_from_view(eyes, tgts)

    dt = sim.get_physics_dt()
    frame_step = max(1, int(round(1.0 / (args.fps * dt))))
    step_times: list = []

    # Also read the tire's simulation points z to confirm contact
    import omni.usd
    from pxr import PhysxSchema

    for i in range(args.steps):
        t0 = time.perf_counter()
        sim.step()
        step_times.append(time.perf_counter() - t0)
        if i % frame_step == 0:
            side._camera.set_world_poses_from_view(eyes, tgts)
            side.capture_frame()
        if i == args.steps - 1:
            stage = omni.usd.get_context().get_stage()
            prim = stage.GetPrimAtPath("/World/Tire")
            attr = prim.GetAttribute("physxDeformable:simulationPoints")
            if attr.IsValid():
                pts = attr.Get()
                if pts:
                    zs = sorted(p[2] for p in pts)
                    print(f"FINAL_TIRE_Z min={zs[0]:.4f} max={zs[-1]:.4f} avg={sum(zs)/len(zs):.4f}", flush=True)

    out = side.finalize()
    print(f"TIRE_DONE outputs={out}", flush=True)

    import statistics
    warm = step_times[30:]
    if warm:
        print(
            f"STEP_STATS label={args.label} E={args.youngs} "
            f"median={statistics.median(warm)*1000:.3f}ms "
            f"mean={statistics.mean(warm)*1000:.3f}ms "
            f"p95={sorted(warm)[int(len(warm)*0.95)]*1000:.3f}ms",
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
