"""Unmanned Surface Vehicle (USV) demo on a virtual water plane.

There is no built-in Isaac Sim water fluid solver, so this demo models
buoyancy as an analytical force: for each hull sampling point below the
water surface, an upward force proportional to (water_level - z) is
applied. The hull is a simple rigid box. Thrust is applied as an
external force at the stern.

Run:
    OMNI_KIT_ACCEPT_EULA=YES \
      /home/js/anaconda3/envs/hmclab_test/bin/python \
      scripts/usv_demo.py --steps 2000
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=2000)
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--water-level", type=float, default=0.0, dest="water_level")
parser.add_argument("--label", type=str, default="usv")
parser.add_argument("--wave-style", type=str, default="flat",
                    choices=["flat", "gerstner", "pbr"],
                    help="Water visualization: flat (blue plane), gerstner (animated waves), pbr (reflective)")
parser.add_argument("--grid-res", type=int, default=40, dest="grid_res",
                    help="Water mesh grid resolution (per axis) for animated waves")
parser.add_argument("--wave-coupling", action="store_true",
                    help="One-way wave→hull coupling for buoyancy (physics feels the waves)")
parser.add_argument("--no-thrust", action="store_true",
                    help="Disable forward thrust and yaw commands (just let the boat sit)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = False
args.enable_cameras = True
launcher = AppLauncher(args)
sim_app = launcher.app

import torch  # noqa: E402
import numpy as np  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import RigidObject, RigidObjectCfg  # noqa: E402

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Itutorial")
)
from hmclab_isaac.utils.capture import ChaseCamRecorder  # noqa: E402

OUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "Itutorial", "outputs"
)


def spawn_water_plane(prim_path: str = "/World/Water", size: float = 10.0, level: float = 0.0,
                      style: str = "flat", grid_res: int = 2):
    """Spawn a water surface plane. Returns (mesh, base_pts_np) for animation.

    style='flat' — single quad, cheapest
    style='gerstner' — grid mesh you can animate per-vertex
    style='pbr' — single quad with high-gloss PBR material
    """
    import omni.usd
    import numpy as np
    from pxr import Gf, UsdGeom, UsdShade, Sdf

    stage = omni.usd.get_context().get_stage()
    plane = UsdGeom.Mesh.Define(stage, prim_path)

    s = size / 2.0
    if style == "gerstner":
        n = max(2, grid_res)
        pts = []
        xs = np.linspace(-s, s, n)
        ys = np.linspace(-s, s, n)
        for yy in ys:
            for xx in xs:
                pts.append((float(xx), float(yy), float(level)))
        # build quads
        face_verts = []
        for j in range(n - 1):
            for i in range(n - 1):
                a = j * n + i
                b = j * n + i + 1
                c = (j + 1) * n + i + 1
                d = (j + 1) * n + i
                face_verts += [a, b, c, d]
        plane.CreatePointsAttr([Gf.Vec3f(*p) for p in pts])
        plane.CreateFaceVertexCountsAttr([4] * ((n - 1) * (n - 1)))
        plane.CreateFaceVertexIndicesAttr(face_verts)
        base_pts = np.array(pts, dtype=np.float32)
    else:
        pts_list = [Gf.Vec3f(-s, -s, level), Gf.Vec3f(s, -s, level),
                    Gf.Vec3f(s, s, level), Gf.Vec3f(-s, s, level)]
        plane.CreatePointsAttr(pts_list)
        plane.CreateFaceVertexCountsAttr([4])
        plane.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
        base_pts = np.array([(p[0], p[1], p[2]) for p in pts_list], dtype=np.float32)

    # Material
    mat_path = f"{prim_path}_mat"
    mat = UsdShade.Material.Define(stage, mat_path)
    shader = UsdShade.Shader.Define(stage, f"{mat_path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    if style == "pbr":
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.02, 0.10, 0.20))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.05)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.15)
        shader.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(1.33)
        shader.CreateInput("clearcoat", Sdf.ValueTypeNames.Float).Set(0.6)
    elif style == "gerstner":
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.05, 0.25, 0.45))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.12)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.08)
    else:
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.1, 0.4, 0.8))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.2)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(plane.GetPrim()).Bind(mat)
    return plane, base_pts


def gerstner_wave_z(base_pts, t, waves=None):
    """Sum of Gerstner-like sinusoidal waves over (x, y) at time t.
    waves: list of dicts with amplitude A, wavevector (kx, ky), omega, phase.
    Returns array of z values.
    """
    import numpy as np
    if waves is None:
        # two cross-waves + a small swell
        waves = [
            dict(A=0.08, k=(0.8, 0.3), w=1.2, phi=0.0),
            dict(A=0.05, k=(0.4, -0.6), w=1.7, phi=1.5),
            dict(A=0.03, k=(1.5, 0.9), w=2.4, phi=0.7),
        ]
    z = np.zeros(base_pts.shape[0], dtype=np.float32)
    x = base_pts[:, 0]
    y = base_pts[:, 1]
    for w in waves:
        kx, ky = w["k"]
        z += w["A"] * np.sin(kx * x + ky * y - w["w"] * t + w["phi"])
    return z


def main() -> int:
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=0.005, device="cuda:0")
    )
    # Sea-bed ground (far below water)
    sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.92, 1.0)).func(
        "/World/Light",
        sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.92, 1.0)),
    )

    water_mesh, water_base_pts = spawn_water_plane(
        "/World/Water", size=16.0, level=args.water_level,
        style=args.wave_style, grid_res=args.grid_res,
    )

    # USV hull: 0.6 × 0.3 × 0.2 m box (length × width × height) with ~4 kg mass
    hull_cfg = RigidObjectCfg(
        prim_path="/World/Hull",
        spawn=sim_utils.CuboidCfg(
            size=(0.6, 0.3, 0.2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                disable_gravity=False,
                max_linear_velocity=20.0,
                max_angular_velocity=20.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=4.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.8, 0.8, 0.2), roughness=0.4
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, args.water_level + 0.08)),
    )
    hull = RigidObject(hull_cfg)

    chase = ChaseCamRecorder(
        out_dir=OUT_DIR,
        tag=f"usv_{args.label}",
        prim_path="/World/ChaseCam",
        behind=2.5, above=1.2, look_up=0.1,
        width=960, height=540,
    )

    sim.reset()
    chase.on_reset()

    # Sample points on the hull (in local frame) for buoyancy integration
    # 6×3×2 grid covering the hull volume
    hull_L, hull_W, hull_H = 0.6, 0.3, 0.2
    nL, nW, nH = 6, 3, 2
    local_pts = []
    for i in range(nL):
        for j in range(nW):
            for k in range(nH):
                x = -hull_L / 2 + (i + 0.5) * hull_L / nL
                y = -hull_W / 2 + (j + 0.5) * hull_W / nW
                z = -hull_H / 2 + (k + 0.5) * hull_H / nH
                local_pts.append((x, y, z))
    local_pts_np = np.array(local_pts, dtype=np.float32)
    n_samples = len(local_pts_np)

    # Effective submerged-volume per sample (hull volume / n_samples)
    vol_per = (hull_L * hull_W * hull_H) / n_samples  # m³
    rho_water = 1000.0  # kg/m³
    g = 9.81

    # Buoyancy at fully submerged sample: F_buoy = rho * g * V_per = 1000 * 9.81 * 0.006/36 ≈ 0.01635 N per sample
    print(
        f"[usv] hull mass=4.0, n_samples={n_samples}, vol_per={vol_per:.5f} m³, "
        f"F_buoy_per_sample={rho_water*g*vol_per:.4f} N",
        flush=True,
    )

    dt = sim.get_physics_dt()
    frame_step = max(1, int(round(1.0 / (args.fps * dt))))
    step_times = []

    device = sim.device

    # Gerstner wave animation helpers
    animate_water = args.wave_style == "gerstner"
    water_pts_attr = water_mesh.GetPointsAttr()

    for step in range(args.steps):
        t = step * dt

        if animate_water and step % 2 == 0:  # update at 100 Hz to save time
            from pxr import Gf, Vt
            zs = gerstner_wave_z(water_base_pts, t)
            new_pts = water_base_pts.copy()
            new_pts[:, 2] = args.water_level + zs
            water_pts_attr.Set(Vt.Vec3fArray([Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in new_pts]))

        # Current hull pose
        pos = hull.data.root_pos_w[0].cpu().numpy()
        quat = hull.data.root_quat_w[0].cpu().numpy()  # w,x,y,z
        w_, x_, y_, z_ = quat
        # Build rotation matrix from quaternion
        xx, yy, zz = x_ * x_, y_ * y_, z_ * z_
        xy, xz, yz = x_ * y_, x_ * z_, y_ * z_
        wx, wy, wz = w_ * x_, w_ * y_, w_ * z_
        R = np.array(
            [
                [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
                [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
                [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
            ],
            dtype=np.float32,
        )

        # Transform each sample point to world
        world_pts = local_pts_np @ R.T + pos  # (n, 3)

        if args.wave_coupling:
            # Sample the wave field at each hull (x, y) and use that as the
            # local water level. This is the one-way wave->hull coupling.
            wave_z = gerstner_wave_z(world_pts[:, :3].copy(), t)
            local_level = args.water_level + wave_z
        else:
            local_level = np.full(world_pts.shape[0], args.water_level, dtype=np.float32)
        depth = local_level - world_pts[:, 2]  # positive if below water

        # Buoyancy force per sample (upward, world +Z), saturated by available depth
        sub_frac = np.clip(depth / (hull_H / nH), 0.0, 1.0)  # 0..1 submerged fraction per sample
        forces_per = rho_water * g * vol_per * sub_frac  # N per sample, upward

        # Drag: linear in velocity, per submerged sample (rough water drag)
        root_vel = hull.data.root_lin_vel_w[0].cpu().numpy()
        root_avel = hull.data.root_ang_vel_w[0].cpu().numpy()

        # Accumulate total force + torque on the root body
        total_F = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        total_T = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        for i in range(n_samples):
            if sub_frac[i] <= 0:
                continue
            F_world = np.array([0.0, 0.0, forces_per[i]], dtype=np.float32)
            # Moment arm from CoG (=root_pos) to the sample
            r = world_pts[i] - pos
            total_F += F_world
            total_T += np.cross(r, F_world)

        # Simple forward thrust + yaw oscillation
        phase = t * 0.3
        thrust_mag = 0.0 if args.no_thrust else 20.0  # N forward
        # Forward direction = local +X rotated to world
        fwd_local = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        fwd_world = R @ fwd_local
        # Thrust applied at stern (back of hull), which creates little yaw naturally
        thrust_world = thrust_mag * fwd_world
        stern_local = np.array([-hull_L / 2, 0.0, 0.0], dtype=np.float32)
        r_stern = R @ stern_local
        total_F += thrust_world
        total_T += np.cross(r_stern, thrust_world)

        # Yaw moment to make it weave (serpentine path)
        if not args.no_thrust:
            yaw_cmd = 3.0 * math.sin(0.5 * t)  # N·m
            total_T += np.array([0.0, 0.0, yaw_cmd], dtype=np.float32)

        # Water drag — Fossen-style linear + quadratic mix
        # F_drag = -c_lin * v   - c_quad * v * |v|
        submerged_frac = float(sub_frac.mean())
        c_lin_lin, c_lin_quad = 30.0, 10.0   # translational
        c_ang_lin, c_ang_quad = 6.0, 3.0     # rotational
        v_mag = float(np.linalg.norm(root_vel))
        w_mag = float(np.linalg.norm(root_avel))
        drag_lin = -submerged_frac * (c_lin_lin * root_vel + c_lin_quad * root_vel * v_mag)
        drag_ang = -submerged_frac * (c_ang_lin * root_avel + c_ang_quad * root_avel * w_mag)
        total_F += drag_lin
        total_T += drag_ang

        forces_t = torch.tensor([[total_F]], device=device, dtype=torch.float32)
        torques_t = torch.tensor([[total_T]], device=device, dtype=torch.float32)
        hull.set_external_force_and_torque(
            forces=forces_t, torques=torques_t, is_global=True
        )
        hull.write_data_to_sim()

        t0 = time.perf_counter()
        sim.step()
        step_times.append(time.perf_counter() - t0)
        hull.update(dt)

        if step % frame_step == 0:
            p = hull.data.root_pos_w[0]
            q = hull.data.root_quat_w[0]
            w, xq, yq, zq = q[0].item(), q[1].item(), q[2].item(), q[3].item()
            yaw = math.atan2(2 * (w * zq + xq * yq), 1 - 2 * (yq * yq + zq * zq))
            chase.update_follow((p[0].item(), p[1].item(), p[2].item()), yaw)
            chase.capture_frame()

    out = chase.finalize()
    print(f"USV_DONE outputs={out}", flush=True)

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
