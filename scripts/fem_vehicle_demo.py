"""FEM deformable-tire MuSHR driving demo.

Spawns a MuSHR Nano v2, replaces each rigid wheel's collider with a
FEM-deformable cylinder tire attached to the rigid wheel_link via
PhysX auto-attachment. Then drives forward in a straight line (with a
small steer oscillation) and records chase cam.

Run:
    OMNI_KIT_ACCEPT_EULA=YES \
      /home/js/anaconda3/envs/hmclab_test/bin/python \
      scripts/fem_vehicle_demo.py --steps 1200 --youngs 3e5
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=1200)
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--youngs", type=float, default=3e5)
parser.add_argument("--tire-radius", type=float, default=0.055, dest="tire_radius")
parser.add_argument("--tire-width", type=float, default=0.05, dest="tire_width")
parser.add_argument("--disable-rim-collider", action="store_true",
                    help="Disable rigid wheel colliders (tire-only contacts). If off, "
                         "rigid wheel supports the body and tire only deforms visibly.")
parser.add_argument("--segments", type=int, default=20)
parser.add_argument("--throttle", type=float, default=3.0, help="wheel angular vel target")
parser.add_argument("--label", type=str, default="fem")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = False
args.enable_cameras = True
launcher = AppLauncher(args)
sim_app = launcher.app

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Itutorial")
)
from _common import get_robot_spec  # noqa: E402
from hmclab_isaac.utils.capture import ChaseCamRecorder  # noqa: E402

OUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "Itutorial", "outputs"
)


def make_tire_cylinder_mesh(stage, prim_path: str, radius: float, width: float, segments: int, pos, axis_dir="y"):
    """Triangulated cylinder with axis along `axis_dir` at world `pos`."""
    from pxr import Gf, UsdGeom

    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    hw = width / 2.0
    pts = []
    # Rings along axis
    for ring_a in (-hw, +hw):
        for i in range(segments):
            theta = 2 * math.pi * i / segments
            x = radius * math.cos(theta)
            z = radius * math.sin(theta)
            if axis_dir == "y":
                pts.append(Gf.Vec3f(x, ring_a, z))
            elif axis_dir == "x":
                pts.append(Gf.Vec3f(ring_a, x, z))
            else:  # z
                pts.append(Gf.Vec3f(x, z, ring_a))
    # Caps
    if axis_dir == "y":
        pts.append(Gf.Vec3f(0.0, -hw, 0.0))
        pts.append(Gf.Vec3f(0.0, +hw, 0.0))
    elif axis_dir == "x":
        pts.append(Gf.Vec3f(-hw, 0.0, 0.0))
        pts.append(Gf.Vec3f(+hw, 0.0, 0.0))
    else:
        pts.append(Gf.Vec3f(0.0, 0.0, -hw))
        pts.append(Gf.Vec3f(0.0, 0.0, +hw))

    mesh.CreatePointsAttr(pts)
    tris = []
    for i in range(segments):
        a = i
        b = (i + 1) % segments
        c = segments + b
        d = segments + a
        tris += [a, b, c,  a, c, d]
    cb = 2 * segments
    for i in range(segments):
        a = i
        b = (i + 1) % segments
        tris += [cb, b, a]
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
    return mesh.GetPrim()


def make_deformable(stage, prim, youngs: float, hex_res: int = 8):
    from omni.physx.scripts import deformableUtils as du
    from pxr import PhysxSchema, UsdShade

    prim_path = prim.GetPath().pathString
    ok = du.add_physx_deformable_body(
        stage,
        prim_path,
        simulation_hexahedral_resolution=hex_res,
        solver_position_iteration_count=32,
        self_collision=False,
        collision_simplification=True,
    )
    if not ok:
        return False
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
        dynamic_friction=1.3,
        elasticity_damping=0.005,
        density=1200.0,
    )
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(
        UsdShade.Material(stage.GetPrimAtPath(mat_path)),
        bindingStrength=UsdShade.Tokens.strongerThanDescendants,
        materialPurpose="physics",
    )
    return True


def disable_rigid_wheel_colliders(stage, root_prim_path: str) -> int:
    """Disable colliders on each wheel_link so the deformable tires take contacts."""
    from pxr import UsdPhysics
    n = 0
    for prim in stage.Traverse():
        p = prim.GetPath().pathString
        if not p.startswith(root_prim_path):
            continue
        if "_wheel_link" not in p:
            continue
        # Disable collision on this prim and all collider children
        for sub in [prim, *prim.GetChildren()]:
            if sub.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(sub).GetCollisionEnabledAttr().Set(False)
                n += 1
    print(f"[fem_veh] disabled collision on {n} wheel prims", flush=True)
    return n


def attach_tire_to_wheel(stage, tire_path: str, wheel_link_path: str, attach_name: str):
    """Manually create a PhysxPhysicsAttachment prim with AutoAttachmentAPI
    between the deformable tire and the rigid wheel link.
    """
    from pxr import PhysxSchema, Sdf
    att_path = Sdf.Path(f"/World/Attachments/{attach_name}")
    att = PhysxSchema.PhysxPhysicsAttachment.Define(stage, att_path)
    att.CreateActor0Rel().AddTarget(Sdf.Path(tire_path))
    att.CreateActor1Rel().AddTarget(Sdf.Path(wheel_link_path))
    att.CreateAttachmentEnabledAttr(True)
    # Auto-attachment generates constraint points where deformable overlaps actor1
    auto_api = PhysxSchema.PhysxAutoAttachmentAPI.Apply(att.GetPrim())
    try:
        auto_api.CreateEnableDeformableVertexAttachmentsAttr(True)
    except Exception:
        pass
    try:
        auto_api.CreateEnableRigidSurfaceAttachmentsAttr(True)
    except Exception:
        pass
    print(f"[fem_veh] attach {attach_name}: created {att_path}", flush=True)
    return True


def main() -> int:
    tag = f"fem_vehicle_{args.label}"
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device="cuda:0")
    sim = sim_utils.SimulationContext(sim_cfg)
    sim_utils.GroundPlaneCfg(
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.5, dynamic_friction=1.3, restitution=0.0
        )
    ).func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=2500.0, color=(0.95, 0.95, 0.95)).func(
        "/World/Light",
        sim_utils.DomeLightCfg(intensity=2500.0, color=(0.95, 0.95, 0.95)),
    )

    spec = get_robot_spec("mushr")
    init_pos = (0.0, 0.0, spec.init_z)
    ego_cfg = spec.cfg.replace(prim_path="/World/Ego")
    ego_cfg = ego_cfg.replace(init_state=ego_cfg.init_state.replace(pos=init_pos))
    ego_cfg.spawn.func(ego_cfg.prim_path, ego_cfg.spawn, translation=init_pos)

    import omni.usd
    stage = omni.usd.get_context().get_stage()

    # Make the attachment anchor xform
    stage.DefinePrim("/World/Attachments", "Xform")

    # Wheel positions relative to spawn (world frame)
    # Based on MuSHR Nano v2 geometry (WHEELBASE=0.32, TRACK_WIDTH=0.24, wheel_radius≈0.05)
    wheelbase = 0.32
    track = 0.24
    wheels = {
        "front_left":  (+wheelbase / 2, +track / 2),
        "front_right": (+wheelbase / 2, -track / 2),
        "back_left":   (-wheelbase / 2, +track / 2),
        "back_right":  (-wheelbase / 2, -track / 2),
    }
    # Find the correct wheel_link prim paths
    wheel_link_paths = {}
    for name in wheels:
        for prim in stage.Traverse():
            p = prim.GetPath().pathString
            if p.endswith(f"/{name}_wheel_link"):
                wheel_link_paths[name] = p
                break

    print(f"[fem_veh] wheel_link paths: {wheel_link_paths}", flush=True)

    if args.disable_rim_collider:
        disable_rigid_wheel_colliders(stage, ego_cfg.prim_path)
    else:
        print("[fem_veh] keeping rigid wheel colliders enabled (body support)", flush=True)

    # Spawn a deformable tire at each wheel center and attach
    tire_axis_dir = "y"  # car's lateral axis is +Y
    for name, (ox, oy) in wheels.items():
        tire_pos = (init_pos[0] + ox, init_pos[1] + oy, args.tire_radius)
        tire_path = f"/World/Tires/{name}"
        stage.DefinePrim("/World/Tires", "Xform")
        prim = make_tire_cylinder_mesh(
            stage,
            tire_path,
            radius=args.tire_radius,
            width=args.tire_width,
            segments=args.segments,
            pos=tire_pos,
            axis_dir=tire_axis_dir,
        )
        if not make_deformable(stage, prim, args.youngs, hex_res=6):
            print(f"[fem_veh] deform failed for {name}", flush=True)
            return 1
        if name not in wheel_link_paths:
            print(f"[fem_veh] missing wheel_link for {name}", flush=True)
            continue
        attach_tire_to_wheel(stage, tire_path, wheel_link_paths[name], f"{name}_att")

    ego = Articulation(ego_cfg)

    chase = ChaseCamRecorder(
        out_dir=OUT_DIR,
        tag=tag,
        prim_path="/World/ChaseCam",
        behind=1.2, above=0.5, look_up=0.1,
        width=960, height=540,
    )

    sim.reset()
    chase.on_reset()

    # Resolve joints we want to drive/steer
    joint_names = ego.joint_names
    drive_ids = [i for i, n in enumerate(joint_names) if "back" in n and "throttle" in n]
    steer_ids = [i for i, n in enumerate(joint_names) if "steer" in n]
    print(f"[fem_veh] drive_ids={drive_ids} steer_ids={steer_ids}", flush=True)

    import torch
    dev = sim.device
    dt = sim.get_physics_dt()
    frame_step = max(1, int(round(1.0 / (args.fps * dt))))
    step_times = []

    throttle_vel = torch.full((1, len(drive_ids)), float(args.throttle), device=dev)
    steer_cmd = torch.zeros((1, len(steer_ids)), device=dev)

    for i in range(args.steps):
        # Simple drive: set velocity target for rear wheels, small steer oscillation
        t = i * dt
        steer_ang = 0.15 * math.sin(2 * math.pi * 0.2 * t)
        steer_cmd[:] = steer_ang
        ego.set_joint_velocity_target(throttle_vel, joint_ids=drive_ids)
        ego.set_joint_position_target(steer_cmd, joint_ids=steer_ids)
        ego.write_data_to_sim()

        t0 = time.perf_counter()
        sim.step()
        step_times.append(time.perf_counter() - t0)
        ego.update(dt)

        if i % frame_step == 0:
            pos = ego.data.root_pos_w[0].cpu().numpy()
            q = ego.data.root_quat_w[0].cpu().numpy()
            w, xq, yq, zq = float(q[0]), float(q[1]), float(q[2]), float(q[3])
            yaw = math.atan2(2 * (w * zq + xq * yq), 1 - 2 * (yq * yq + zq * zq))
            chase.update_follow(
                (float(pos[0]), float(pos[1]), float(pos[2])), yaw
            )
            chase.capture_frame()

    out = chase.finalize()
    print(f"FEM_VEH_DONE outputs={out}", flush=True)

    import statistics
    warm = step_times[30:]
    if warm:
        print(
            f"STEP_STATS label={args.label} "
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
