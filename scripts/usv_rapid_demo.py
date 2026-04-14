"""Valley-rapid USV demo.

Approximates a canyon rapid using:
  - a winding water channel with two sloped banks (rigid boxes as canyon walls)
  - a handful of boulder rigid bodies the USV can crash into
  - an analytical downstream current force:  F_cur = c * (v_cur(x,y) - v_boat)
  - chaotic high-frequency Gerstner waves + noise jitter
  - randomized yaw perturbation in narrow sections (fake turbulence)

Run:
    OMNI_KIT_ACCEPT_EULA=YES \
      /home/js/anaconda3/envs/hmclab_test/bin/python \
      scripts/usv_rapid_demo.py --steps 2000
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=2000)
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--water-level", type=float, default=0.0, dest="water_level")
parser.add_argument("--current", type=float, default=0.0, help="Downstream current speed [m/s] (0 = gravity-only)")
parser.add_argument("--slope", type=float, default=0.08, help="Water surface slope (z-drop per x)")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--label", type=str, default="rapid")
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

random.seed(args.seed)
np.random.seed(args.seed)


# ---------------------------------------------------------------------
# Channel geometry — a centreline path that winds in Y as we go +X
# ---------------------------------------------------------------------
CHANNEL_LENGTH = 24.0  # meters along X
CHANNEL_HALF_WIDTH = 1.2  # meters Y half-width at the centre


def water_level_at(x: float) -> float:
    """Water surface height as a function of longitudinal position (sloped downstream)."""
    return args.water_level - args.slope * x


def channel_centreline_y(x: float) -> float:
    """Y offset of the channel centreline at longitudinal position x."""
    return 1.6 * math.sin(0.35 * x) + 0.6 * math.sin(0.9 * x + 0.7)


def channel_half_width_at(x: float) -> float:
    """Half-width of the navigable channel at x (constricts in the middle)."""
    # narrow in the middle (rapids), wider at ends
    middle_factor = 1.0 - 0.5 * math.exp(-((x - CHANNEL_LENGTH / 2) ** 2) / 18.0)
    return CHANNEL_HALF_WIDTH * middle_factor


def current_velocity_at(x: float, y: float) -> tuple[float, float]:
    """Analytical current velocity at (x, y). Mostly +X, a bit of Y to follow the
    centreline so a floating boat is naturally swept along the path."""
    cy = channel_centreline_y(x)
    # strength ramps up in the narrow middle
    hw = channel_half_width_at(x)
    base = args.current * (CHANNEL_HALF_WIDTH / max(hw, 0.4))
    vx = base
    # transverse component so flow follows the centreline curvature
    dy_dx = 1.6 * 0.35 * math.cos(0.35 * x) + 0.6 * 0.9 * math.cos(0.9 * x + 0.7)
    vy_centreline = base * dy_dx
    # correction pushing boat toward centreline (y -> cy)
    vy_restore = -1.2 * (y - cy)
    return vx, vy_centreline + vy_restore


# ---------------------------------------------------------------------
# Water mesh (winding grid + chaotic Gerstner animation)
# ---------------------------------------------------------------------
def spawn_rapid_water(prim_path: str, level: float, nx: int = 96, ny: int = 24):
    import omni.usd
    from pxr import Gf, UsdGeom, UsdShade, Sdf

    stage = omni.usd.get_context().get_stage()
    mesh = UsdGeom.Mesh.Define(stage, prim_path)

    pts = []
    for j in range(ny):
        for i in range(nx):
            x = (i / (nx - 1)) * CHANNEL_LENGTH
            # extend mesh a bit beyond banks so walls sit in water
            y = (-CHANNEL_HALF_WIDTH - 1.5) + (j / (ny - 1)) * (2 * (CHANNEL_HALF_WIDTH + 1.5))
            z_here = water_level_at(x)
            pts.append((float(x), float(y) + channel_centreline_y(x), float(z_here)))
    face_verts = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            a = j * nx + i
            b = j * nx + i + 1
            c = (j + 1) * nx + i + 1
            d = (j + 1) * nx + i
            face_verts += [a, b, c, d]
    mesh.CreatePointsAttr([Gf.Vec3f(*p) for p in pts])
    mesh.CreateFaceVertexCountsAttr([4] * ((nx - 1) * (ny - 1)))
    mesh.CreateFaceVertexIndicesAttr(face_verts)

    # Turbulent-looking material
    mat_path = f"{prim_path}_mat"
    mat = UsdShade.Material.Define(stage, mat_path)
    shader = UsdShade.Shader.Define(stage, f"{mat_path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.45, 0.62, 0.78))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.25)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)

    return mesh, np.array(pts, dtype=np.float32)


def chaotic_waves(base_pts: np.ndarray, t: float) -> np.ndarray:
    """Sum of several high-k Gerstner + noise jitter -> whitewater-ish surface."""
    x = base_pts[:, 0]
    y = base_pts[:, 1]
    z = np.zeros_like(x)
    waves = [
        (0.06, 4.5, 1.2, 6.0, 0.0),
        (0.04, 2.2, 3.1, 4.5, 1.1),
        (0.03, 6.8, -0.5, 7.5, 0.4),
        (0.025, 1.1, 5.2, 3.8, 2.0),
    ]
    for A, kx, ky, w, phi in waves:
        z += A * np.sin(kx * x + ky * y - w * t + phi)
    # add position-dependent jitter emphasis in the fast middle
    xmid = CHANNEL_LENGTH / 2
    turb = np.exp(-((x - xmid) ** 2) / 18.0)
    z += 0.04 * turb * np.sin(12.0 * x + 8.0 * y - 9.0 * t)
    return z


# ---------------------------------------------------------------------
# Canyon walls + boulders (static rigid bodies with colliders)
# ---------------------------------------------------------------------
def spawn_static_box(prim_path: str, pos, size, color):
    cfg = RigidObjectCfg(
        prim_path=prim_path,
        spawn=sim_utils.CuboidCfg(
            size=size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=True,  # static wall
                disable_gravity=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=1000.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=color, roughness=0.8
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos),
    )
    return RigidObject(cfg)


def build_canyon_walls(water_level: float):
    """Place stepped box segments along both banks of the winding channel."""
    walls = []
    n_segs = 40
    for i in range(n_segs):
        x = (i + 0.5) * (CHANNEL_LENGTH / n_segs)
        cy = channel_centreline_y(x)
        hw = channel_half_width_at(x)
        wall_len = CHANNEL_LENGTH / n_segs * 1.1
        # left bank: y = cy + hw + 0.4
        z_local = water_level_at(x)
        walls.append(
            spawn_static_box(
                f"/World/Walls/LB_{i}",
                (x, cy + hw + 0.4, z_local + 0.2),
                (wall_len, 0.8, 0.8),
                (0.35, 0.25, 0.15),
            )
        )
        walls.append(
            spawn_static_box(
                f"/World/Walls/RB_{i}",
                (x, cy - hw - 0.4, z_local + 0.2),
                (wall_len, 0.8, 0.8),
                (0.35, 0.25, 0.15),
            )
        )
    return walls


def build_boulders(water_level: float, n: int = 8):
    rocks = []
    # place a few rocks partially in the channel
    for k in range(n):
        x = 4.0 + k * (CHANNEL_LENGTH / (n + 2))
        cy = channel_centreline_y(x)
        y = cy + (random.random() - 0.5) * channel_half_width_at(x) * 0.8
        size = 0.35 + 0.25 * random.random()
        z_local = water_level_at(x)
        rocks.append(
            spawn_static_box(
                f"/World/Rocks/R_{k}",
                (x, y, z_local + size * 0.3),
                (size, size, size),
                (0.35, 0.33, 0.30),
            )
        )
    return rocks


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> int:
    tag = f"usv_{args.label}"
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=0.005, device="cuda:0")
    )
    sim_utils.DomeLightCfg(intensity=2500.0, color=(1.0, 0.96, 0.90)).func(
        "/World/Light",
        sim_utils.DomeLightCfg(intensity=2500.0, color=(1.0, 0.96, 0.90)),
    )

    water_mesh, water_base_pts = spawn_rapid_water(
        "/World/Water", args.water_level, nx=96, ny=24
    )

    # Canyon + boulders
    build_canyon_walls(args.water_level)
    build_boulders(args.water_level, n=6)

    # USV hull
    hull_L, hull_W, hull_H = 0.6, 0.3, 0.2
    start_x = 0.5
    start_y = channel_centreline_y(start_x)
    hull_cfg = RigidObjectCfg(
        prim_path="/World/Hull",
        spawn=sim_utils.CuboidCfg(
            size=(hull_L, hull_W, hull_H),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                disable_gravity=False,
                max_linear_velocity=30.0,
                max_angular_velocity=30.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=4.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.95, 0.85, 0.25), roughness=0.4
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(start_x, start_y, water_level_at(start_x) + hull_H / 2)
        ),
    )
    hull = RigidObject(hull_cfg)

    chase = ChaseCamRecorder(
        out_dir=OUT_DIR, tag=tag, prim_path="/World/ChaseCam",
        behind=3.2, above=1.8, look_up=0.2, width=960, height=540,
    )

    sim.reset()
    chase.on_reset()

    # Pre-compute hull sample grid (local frame)
    nL, nW, nH = 6, 3, 2
    local_pts = []
    for i in range(nL):
        for j in range(nW):
            for k in range(nH):
                local_pts.append((
                    -hull_L / 2 + (i + 0.5) * hull_L / nL,
                    -hull_W / 2 + (j + 0.5) * hull_W / nW,
                    -hull_H / 2 + (k + 0.5) * hull_H / nH,
                ))
    local_pts_np = np.array(local_pts, dtype=np.float32)
    n_samples = len(local_pts_np)
    vol_per = (hull_L * hull_W * hull_H) / n_samples
    rho_w = 1000.0
    g = 9.81

    dt = sim.get_physics_dt()
    frame_step = max(1, int(round(1.0 / (args.fps * dt))))
    step_times = []
    water_pts_attr = water_mesh.GetPointsAttr()

    from pxr import Gf, Vt

    for step in range(args.steps):
        t = step * dt

        # Animate water surface every 2 steps
        if step % 2 == 0:
            zs = chaotic_waves(water_base_pts, t)
            new_pts = water_base_pts.copy()
            # base_pts already carries the slope z at spawn; just add ripple
            new_pts[:, 2] = water_base_pts[:, 2] + zs
            water_pts_attr.Set(
                Vt.Vec3fArray([Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in new_pts])
            )

        pos = hull.data.root_pos_w[0].cpu().numpy()
        quat = hull.data.root_quat_w[0].cpu().numpy()
        w_, x_, y_, z_ = quat
        xx, yy, zz = x_ * x_, y_ * y_, z_ * z_
        xy, xz, yz = x_ * y_, x_ * z_, y_ * z_
        wx, wy, wz = w_ * x_, w_ * y_, w_ * z_
        R = np.array([
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ], dtype=np.float32)

        # Buoyancy integration — water level varies with x (sloped downstream)
        world_pts = local_pts_np @ R.T + pos
        local_levels = args.water_level - args.slope * world_pts[:, 0]
        depth = local_levels - world_pts[:, 2]
        sub_frac = np.clip(depth / (hull_H / nH), 0.0, 1.0)

        root_vel = hull.data.root_lin_vel_w[0].cpu().numpy()
        root_avel = hull.data.root_ang_vel_w[0].cpu().numpy()

        total_F = np.zeros(3, dtype=np.float32)
        total_T = np.zeros(3, dtype=np.float32)
        for i in range(n_samples):
            if sub_frac[i] <= 0:
                continue
            F_world = np.array([0.0, 0.0, rho_w * g * vol_per * sub_frac[i]], dtype=np.float32)
            r = world_pts[i] - pos
            total_F += F_world
            total_T += np.cross(r, F_world)

        # --- Current drag: push the boat along flow ---
        vcx, vcy = current_velocity_at(float(pos[0]), float(pos[1]))
        v_cur = np.array([vcx, vcy, 0.0], dtype=np.float32)
        dv = v_cur - root_vel
        c_cur = 6.0  # drag coefficient to the current
        submerged_frac = float(sub_frac.mean())
        F_current = c_cur * submerged_frac * dv
        total_F += F_current

        # --- Extra water drag (beyond current coupling) on angular motion ---
        total_T += -1.8 * submerged_frac * root_avel

        # --- Turbulence perturbation in narrow sections ---
        xmid = CHANNEL_LENGTH / 2
        turb = math.exp(-((float(pos[0]) - xmid) ** 2) / 18.0)
        total_T[2] += 2.2 * turb * math.sin(7.0 * t + 3.1 * float(pos[0]))

        total_F[0] += 0.6 * turb * math.sin(5.7 * t)
        total_F[1] += 0.6 * turb * math.cos(4.3 * t + 1.7)

        # Stop if we've reached the end
        if pos[0] > CHANNEL_LENGTH - 0.5:
            break

        forces_t = torch.tensor([[total_F]], device=sim.device, dtype=torch.float32)
        torques_t = torch.tensor([[total_T]], device=sim.device, dtype=torch.float32)
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
    print(f"RAPID_DONE outputs={out}", flush=True)

    import statistics
    warm = step_times[30:]
    if warm:
        print(
            f"STEP_STATS label={args.label} "
            f"median={statistics.median(warm)*1000:.3f}ms "
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
