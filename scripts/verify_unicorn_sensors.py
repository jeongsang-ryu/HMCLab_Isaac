"""Smoke-test UNICORN_3 / UNICORN_4 configs.

Spawns one vehicle from the chosen config, attaches its LiDAR (Mid-360
or Hokuyo) and camera sensors, runs ~30 sim steps, and reports:
  - articulation joint count + joint names
  - lidar frame: ray count, hit count, range stats
  - camera frame: image shape, non-zero pixel ratio
  - mount Xform world transforms

Usage:
  OMNI_KIT_ACCEPT_EULA=YES python scripts/verify_unicorn_sensors.py --variant 3
  OMNI_KIT_ACCEPT_EULA=YES python scripts/verify_unicorn_sensors.py --variant 4
"""
from __future__ import annotations

import argparse
import os
import sys

# Make `hmclab_isaac` importable when this script is launched from anywhere.
_PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--variant", type=int, choices=[3, 4], required=True,
                    help="3 = UNICORN_3 (Hokuyo 2D), 4 = UNICORN_4 (Mid-360 3D)")
parser.add_argument("--steps", type=int, default=30)
parser.add_argument("--dt", type=float, default=0.005)
parser.add_argument("--no-camera", action="store_true",
                    help="Skip camera (debug-friendly: lidar-only run)")
parser.add_argument("--dump-stage", action="store_true",
                    help="After spawn, print all sensor-relevant prim paths")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True   # required for camera sensor to render

launcher = AppLauncher(args)
sim_app = launcher.app

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.assets import AssetBaseCfg  # noqa: E402


# Load chosen variant
if args.variant == 3:
    from hmclab_isaac.robots.vehicles.UNICORN import UNICORN_3 as variant
    LIDAR_FACTORY = variant.make_hokuyo
    LIDAR_LABEL = "Hokuyo UST 2D"
else:
    from hmclab_isaac.robots.vehicles.UNICORN import UNICORN_4 as variant
    LIDAR_FACTORY = variant.make_mid360
    LIDAR_LABEL = "Livox Mid-360 3D"


def main() -> int:
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=args.dt, device="cuda:0")
    )
    sim.set_camera_view(eye=(2.0, 2.0, 1.5), target=(0.0, 0.0, 0.2))

    sim_utils.DomeLightCfg(intensity=2000.0).func(
        "/World/Light", sim_utils.DomeLightCfg(intensity=2000.0))
    gp = sim_utils.GroundPlaneCfg()
    gp.func("/World/ground", gp)

    # spawn obstacles so lidar has something to hit
    for i, (x, y) in enumerate([(2.0, 0.0), (-2.0, 1.5), (0.0, -2.0)]):
        cube_cfg = sim_utils.MeshCuboidCfg(size=(0.5, 0.5, 1.0))
        cube_cfg.func(f"/World/Obstacle_{i}", cube_cfg, translation=(x, y, 0.5))

    chassis_cfg = variant.CFG.replace(prim_path="/World/envs/env_0/Robot")

    # Build sensor cfgs (attached as scene-level children of the robot)
    robot_path_pattern = "/World/envs/env_0/Robot"
    lidar_cfg = LIDAR_FACTORY(
        robot_prim_path=robot_path_pattern,
        mesh_targets=["/World/ground", "/World/Obstacle_0",
                      "/World/Obstacle_1", "/World/Obstacle_2"],
    )

    if args.no_camera:
        class _SceneCfg(InteractiveSceneCfg):
            robot = chassis_cfg
            lidar = lidar_cfg
    else:
        camera_cfg = variant.make_camera(robot_prim_path=robot_path_pattern)
        class _SceneCfg(InteractiveSceneCfg):
            robot = chassis_cfg
            lidar = lidar_cfg
            camera = camera_cfg

    # First spawn the robot only so we can inspect the composed stage,
    # then add sensors. Build scene normally — if camera path is wrong,
    # error message tells us. With --dump-stage, we show prim tree first.
    if args.dump_stage:
        class _DumpSceneCfg(InteractiveSceneCfg):
            robot = chassis_cfg
        dump_scene = InteractiveScene(_DumpSceneCfg(
            num_envs=1, env_spacing=2.5, replicate_physics=True))
        sim.reset()
        import omni.usd
        stage = omni.usd.get_context().get_stage()
        with open("/tmp/spawn_tree.txt", "w") as f:
            f.write("STAGE PRIM TREE (under /World/envs/env_0/Robot/base_link)\n")
            for prim in stage.Traverse():
                path = str(prim.GetPath())
                if "/World/envs/env_0/Robot/base_link" not in path:
                    continue
                t = str(prim.GetTypeName())
                f.write(f"  {path}  ({t})\n")
        return 0

    scene = InteractiveScene(_SceneCfg(num_envs=1, env_spacing=2.5,
                                       replicate_physics=True))
    sim.reset()

    art = scene["robot"]
    lidar = scene["lidar"]
    camera = scene["camera"] if not args.no_camera else None

    print(f"\n========= UNICORN_{args.variant} verification =========")
    print(f"  USD: {variant.USD_PATH.split('/')[-1]}")
    print(f"  Lidar: {LIDAR_LABEL}")
    print(f"  Joints (articulation): {len(art.joint_names)}")
    print(f"    {art.joint_names}")

    # warm up: a few steps so sensors emit data
    for i in range(args.steps):
        art.write_data_to_sim()
        sim.step()
        scene.update(args.dt)

    # --- LiDAR check ---
    print("\n  --- LiDAR ---")
    try:
        ray_hits = lidar.data.ray_hits_w[0]   # [num_rays, 3]
        ray_dirs = lidar.data.ray_directions_w[0] if hasattr(lidar.data, 'ray_directions_w') else None
        n_rays = ray_hits.shape[0]
        # ray_hits is (x, y, z) of hit points; non-hits often = inf
        valid = torch.isfinite(ray_hits).all(dim=-1)
        n_valid = int(valid.sum().item())
        if n_valid > 0:
            v = ray_hits[valid]
            r = torch.linalg.norm(v - lidar.data.pos_w[0], dim=-1)
            print(f"    rays = {n_rays}   valid hits = {n_valid}")
            print(f"    range: min={r.min():.2f}m  max={r.max():.2f}m  mean={r.mean():.2f}m")
        else:
            print(f"    rays = {n_rays}   valid hits = 0  (no hits)")
        print(f"    sensor pos_w = {tuple(lidar.data.pos_w[0].tolist())}")
        print(f"    sensor quat_w = {tuple(lidar.data.quat_w[0].tolist())}")
    except Exception as e:
        print(f"    ERROR reading lidar data: {e}")

    if args.no_camera:
        print("\n  --- Camera (skipped: --no-camera) ---")
        print(f"\n========= verification done (variant {args.variant}) =========\n")
        return 0

    # --- Camera check ---
    print("\n  --- Camera ---")
    try:
        for dt_name, dt_data in camera.data.output.items():
            if dt_data is None:
                print(f"    {dt_name}: <None>")
                continue
            shape = tuple(dt_data.shape)
            if dt_name == "rgb":
                non_zero = float((dt_data > 0).any(dim=-1).float().mean().item())
                print(f"    {dt_name}: shape={shape}  non-zero pixel ratio={non_zero:.3f}")
            elif "distance" in dt_name or "depth" in dt_name:
                d = dt_data.squeeze()
                finite = torch.isfinite(d)
                nfin = int(finite.sum().item())
                if nfin > 0:
                    df = d[finite]
                    print(f"    {dt_name}: shape={shape}  finite={nfin}  "
                          f"min={df.min():.2f}m  max={df.max():.2f}m")
                else:
                    print(f"    {dt_name}: shape={shape}  no finite values")
            else:
                print(f"    {dt_name}: shape={shape}")
    except Exception as e:
        print(f"    ERROR reading camera data: {e}")

    # --- Mount transforms ---
    print("\n  --- Transforms ---")
    art_pos = art.data.root_pos_w[0]
    print(f"    chassis world pos: ({art_pos[0]:.3f}, {art_pos[1]:.3f}, {art_pos[2]:.3f})")
    print(f"    lidar  world pos: ({lidar.data.pos_w[0,0]:.3f}, "
          f"{lidar.data.pos_w[0,1]:.3f}, {lidar.data.pos_w[0,2]:.3f})")
    print(f"    camera world pos: ({camera.data.pos_w[0,0]:.3f}, "
          f"{camera.data.pos_w[0,1]:.3f}, {camera.data.pos_w[0,2]:.3f})")

    print(f"\n========= verification done (variant {args.variant}) =========\n")
    return 0


if __name__ == "__main__":
    import os
    import traceback
    rc = 0
    try:
        rc = main()
    except Exception:
        traceback.print_exc()
        rc = 1
    os._exit(rc)
