"""Live matplotlib viewer for UNICORN sensors with joystick teleop.

Spawns one vehicle from UNICORN_3.py (Hokuyo) or UNICORN_4.py (Mid-360)
on the chosen oval track (default: mini_oval as duct walls), drives it
with the joystick (keyboard fallback), and shows two live data panels:

  Left panel:  LiDAR points (top-down XY scatter, color = z height).
  Right panel: Vehicle-mounted camera RGB (the SG2 camera authored
               on /Robot/base_link/cam in the device USD).

The vehicle camera is attached AFTER ``sim.reset()`` because the device
USD payload only resolves at reset time — at scene-init time the
deeply-nested camera prim is not yet in the stage.

Joystick (default Xbox-style):
    Left stick Y    → throttle (up = forward)
    Right stick X   → steering
    Button 0 (A)    → brake

Keyboard:
    ↑ / ↓   throttle
    ← / →   steer
    Space   brake

Usage:
    OMNI_KIT_ACCEPT_EULA=YES python scripts/view_unicorn_sensors.py --variant 4
    OMNI_KIT_ACCEPT_EULA=YES python scripts/view_unicorn_sensors.py --variant 3 \\
        --track mini_oval --max-wheel-rps 600
"""
from __future__ import annotations

import argparse
import math
import os
import sys

# Make `hmclab_isaac` importable when launched from anywhere
_PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

from isaaclab.app import AppLauncher

from hmclab_isaac.utils.teleop_input import add_input_args, make_controller


TRACKS_DIR = (
    "/home/js/hmcl_issac_project/HMCLab_Isaac/"
    "hmclab_isaac/worlds/racing/_tracks_data"
)
TRACK_CHOICES = sorted(
    f.removesuffix(".txt") for f in os.listdir(TRACKS_DIR) if f.endswith(".txt")
)


parser = argparse.ArgumentParser()
parser.add_argument("--variant", type=int, choices=[3, 4], required=True,
                    help="3 = UNICORN_3 (Hokuyo 2D), 4 = UNICORN_4 (Mid-360 3D)")
parser.add_argument("--track", choices=TRACK_CHOICES, default="my_track",
                    help="Track centerline file (rendered as duct).")
parser.add_argument("--no-track", action="store_true",
                    help="Skip duct track — use flat ground + a few cubes.")
parser.add_argument("--track-z-offset", type=float, default=0.20,
                    help="Spawn the vehicle this far above the first track point.")
parser.add_argument("--track-subdivide", type=int, default=10)
parser.add_argument("--pipe-radius", type=float, default=0.20)
parser.add_argument("--pipe-offset", type=float, default=0.50)
parser.add_argument("--rib-spacing", type=float, default=0.5)
parser.add_argument("--max-wheel-rps", type=float, default=600.0,
                    help="Commanded wheel angular velocity at full throttle.")
parser.add_argument("--dt", type=float, default=0.005)
parser.add_argument("--ground-static-friction", type=float, default=1.5)
parser.add_argument("--ground-dynamic-friction", type=float, default=1.3)
parser.add_argument("--friction-combine-mode", default="multiply",
                    choices=["average", "min", "max", "multiply"])
parser.add_argument("--cam-rot", nargs=4, type=float,
                    default=[0.5, -0.5, 0.5, -0.5],
                    metavar=("W", "X", "Y", "Z"),
                    help="Camera mount orientation in parent frame (ROS "
                         "convention). Default rotates body→ROS optical so "
                         "camera looks along vehicle +X (forward).")
add_input_args(parser)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = False
args.enable_cameras = True

launcher = AppLauncher(args)
sim_app = launcher.app


import numpy as np  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sensors import Camera, CameraCfg  # noqa: E402


if args.variant == 3:
    from hmclab_isaac.robots.vehicles.UNICORN import UNICORN_3 as variant
    LIDAR_FACTORY = variant.make_hokuyo
    LIDAR_LABEL = "Hokuyo UST-10/20LX (2D, 270deg)"
else:
    from hmclab_isaac.robots.vehicles.UNICORN import UNICORN_4 as variant
    LIDAR_FACTORY = variant.make_mid360
    LIDAR_LABEL = "Livox Mid-360 (3D)"


def _make_camera_post_reset(robot_prim_path: str) -> Camera:
    """Spawn a fresh Camera prim under the `/cam` Xform mount.

    Called AFTER ``sim.reset()`` so the chassis payload has resolved
    and `<robot>/base_link/cam` exists. We don't reuse the SG2 device
    USD's Camera prim because its xformOp stack uses `rotateXYZ` with
    Float precision baked into the payload layer (read-only from this
    session) — Isaac Lab's XformPrimView refuses that and even
    standardize_xform_ops can't override the precision. Instead we
    spawn a new UsdGeom.Camera with PinholeCameraCfg under the
    `/cam` Xform mount, so the USD-authored mount transform still
    applies as parent and we get clean canonical xform ops on the
    sensor itself.
    """
    import omni.usd

    cam_mount_path = f"{robot_prim_path}/base_link/cam"
    stage = omni.usd.get_context().get_stage()
    if not stage.GetPrimAtPath(cam_mount_path):
        raise RuntimeError(f"Cam mount Xform missing post-reset: {cam_mount_path}")

    sensor_path = f"{cam_mount_path}/sensor"
    cfg = CameraCfg(
        prim_path=sensor_path,
        update_period=1.0 / variant.CAMERA_RATE_HZ,
        height=variant.CAMERA_HEIGHT,
        width=variant.CAMERA_WIDTH,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=variant.CAMERA_FOCAL_LENGTH,
            horizontal_aperture=variant.CAMERA_HORIZONTAL_APERTURE,
            clipping_range=variant.CAMERA_CLIPPING_RANGE,
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0),
            rot=tuple(args.cam_rot),
            convention="ros",
        ),
    )
    return Camera(cfg)


def main() -> int:
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=args.dt, device="cuda:0")
    )

    sim_utils.DomeLightCfg(intensity=2000.0, color=(0.95, 0.95, 0.95)).func(
        "/World/Light",
        sim_utils.DomeLightCfg(intensity=2000.0, color=(0.95, 0.95, 0.95)),
    )

    spawn_pos = (0.0, 0.0, variant.INIT_HEIGHT)
    spawn_yaw = 0.0
    mesh_targets: list[str] = ["/World/ground"]

    if args.no_track:
        gp_mat = sim_utils.RigidBodyMaterialCfg(
            static_friction=args.ground_static_friction,
            dynamic_friction=args.ground_dynamic_friction,
            restitution=0.0,
            friction_combine_mode=args.friction_combine_mode,
        )
        gp = sim_utils.GroundPlaneCfg(physics_material=gp_mat)
        gp.func("/World/ground", gp)
        for i, (x, y) in enumerate([(2.5, 0.0), (-2.5, 1.5), (0.0, -2.5),
                                    (1.5, 2.0), (-1.5, -1.5)]):
            cube_cfg = sim_utils.MeshCuboidCfg(
                size=(0.6, 0.6, 1.2),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.7, 0.3, 0.3)
                ),
            )
            path = f"/World/Obstacle_{i}"
            cube_cfg.func(path, cube_cfg, translation=(x, y, 0.6))
            mesh_targets.append(path)
    else:
        from hmclab_isaac.worlds.racing import RacingTrack, spawn_duct_track

        track = RacingTrack.load(os.path.join(TRACKS_DIR, f"{args.track}.txt"))
        # Densify so the duct mesh has more facets
        sub = max(1, int(args.track_subdivide))
        if sub > 1:
            n_old = len(track.positions)
            new_pos, new_rpy, new_w = [], [], []
            last = n_old if track.closed else n_old - 1
            for i in range(last):
                j = (i + 1) % n_old
                for k in range(sub):
                    t = k / sub
                    new_pos.append(track.positions[i] * (1 - t) + track.positions[j] * t)
                    new_rpy.append(track.rpy[i] * (1 - t) + track.rpy[j] * t)
                    new_w.append(track.widths[i] * (1 - t) + track.widths[j] * t)
            if not track.closed:
                new_pos.append(track.positions[-1])
                new_rpy.append(track.rpy[-1])
                new_w.append(track.widths[-1])
            track = type(track)(
                name=track.name,
                positions=np.array(new_pos),
                rpy=np.array(new_rpy),
                widths=np.array(new_w),
                closed=track.closed,
            )

        # Ground plane underneath
        gp_mat = sim_utils.RigidBodyMaterialCfg(
            static_friction=args.ground_static_friction,
            dynamic_friction=args.ground_dynamic_friction,
            restitution=0.0,
            friction_combine_mode=args.friction_combine_mode,
        )
        gp = sim_utils.GroundPlaneCfg(physics_material=gp_mat)
        gp.func("/World/ground", gp)

        # Duct track — orange pipes + dark ribs
        spawn_duct_track(
            track, "/World/track",
            pipe_radius=args.pipe_radius,
            pipe_offset=args.pipe_offset,
            rib_spacing=args.rib_spacing,
            duct_color=(1.0, 0.5, 0.0),
            rib_color=(0.05, 0.05, 0.05),
        )
        mesh_targets += ["/World/track/duct", "/World/track/ribs"]

        spawn_pos = (
            float(track.positions[0, 0]),
            float(track.positions[0, 1]),
            float(track.positions[0, 2]) + args.track_z_offset,
        )
        spawn_yaw = float(track.rpy[0, 2])
        sim.set_camera_view(
            eye=(spawn_pos[0] + 2.0, spawn_pos[1] + 2.0, spawn_pos[2] + 1.5),
            target=spawn_pos,
        )
        print(f"[viewer] track {args.track!r} spawned: {len(track.positions)} pts, "
              f"start=({spawn_pos[0]:.2f},{spawn_pos[1]:.2f},{spawn_pos[2]:.2f}), "
              f"yaw={spawn_yaw:.3f}", flush=True)

    cy, sy = math.cos(spawn_yaw / 2.0), math.sin(spawn_yaw / 2.0)
    chassis_cfg = variant.CFG.replace(prim_path="/World/envs/env_0/Robot")
    chassis_cfg.init_state.pos = spawn_pos
    chassis_cfg.init_state.rot = (cy, 0.0, 0.0, sy)

    lidar_cfg = LIDAR_FACTORY(
        robot_prim_path="/World/envs/env_0/Robot",
        mesh_targets=mesh_targets,
    )

    class _SceneCfg(InteractiveSceneCfg):
        robot = chassis_cfg
        lidar = lidar_cfg

    scene = InteractiveScene(_SceneCfg(num_envs=1, env_spacing=2.5,
                                       replicate_physics=True))
    sim.reset()

    # Camera AFTER reset (payload now resolved, SG2 prim exists)
    camera = _make_camera_post_reset("/World/envs/env_0/Robot")
    scene._sensors["camera"] = camera   # let scene.update() drive it
    sim.reset()                         # re-init so callbacks register
    print("[viewer] vehicle camera attached", flush=True)

    art = scene["robot"]
    lidar = scene["lidar"]

    print(f"\n=== UNICORN_{args.variant} viewer ===")
    print(f"  USD: {variant.USD_PATH.split('/')[-1]}")
    print(f"  Lidar: {LIDAR_LABEL}")
    print(f"  Joints: {len(art.joint_names)}")

    import re as _re
    steer_ids = [i for i, n in enumerate(art.joint_names)
                 if _re.match(r'^(fl|fr)_steering$', n)]
    drive_ids = [i for i, n in enumerate(art.joint_names)
                 if _re.match(r'^(fl|fr|rl|rr)_wheel$', n)]
    print(f"  steer_ids={steer_ids}, drive_ids={drive_ids}", flush=True)

    ctrl = make_controller(args)
    if args.joystick:
        print(f"[viewer] joystick: stick X = steer (axis {args.js_axis_steer}), "
              f"stick Y = throttle (axis {args.js_axis_throttle})  "
              f"button {args.js_button_brake} = brake", flush=True)
    else:
        print("[viewer] joystick disabled — keyboard only", flush=True)

    dev = sim.device
    drive_target = torch.zeros((1, len(drive_ids)), device=dev)
    steer_target = torch.zeros((1, len(steer_ids)), device=dev)

    # Matplotlib setup
    plt.ion()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax_lidar, ax_cam = axes

    ax_lidar.set_aspect("equal")
    ax_lidar.set_xlim(-10, 10)
    ax_lidar.set_ylim(-10, 10)
    ax_lidar.set_xlabel("x (m)")
    ax_lidar.set_ylabel("y (m)")
    ax_lidar.set_title(f"LiDAR — {LIDAR_LABEL}")
    ax_lidar.grid(True, alpha=0.3)
    lidar_scatter = ax_lidar.scatter([], [], s=2, c=[], cmap="viridis")
    sensor_marker, = ax_lidar.plot([], [], "r*", markersize=12, label="sensor")
    ax_lidar.legend(loc="upper right")
    fig.colorbar(lidar_scatter, ax=ax_lidar, fraction=0.04, label="height z (m)")

    ax_cam.set_title("Vehicle Camera (SG2 RGB)")
    ax_cam.axis("off")
    cam_im = ax_cam.imshow(np.zeros((variant.CAMERA_HEIGHT,
                                     variant.CAMERA_WIDTH, 3),
                                    dtype=np.uint8))

    fig.canvas.manager.set_window_title(f"UNICORN_{args.variant} sensors")
    plt.tight_layout()
    print("\nViewer running — close window or Ctrl+C to stop.\n", flush=True)

    step_count = 0
    try:
        while sim_app.is_running():
            thr, steer_in = ctrl.read()
            wheel_speed = thr * args.max_wheel_rps
            steer_angle = steer_in * variant.MAX_STEER
            drive_target.fill_(wheel_speed)
            steer_target.fill_(steer_angle)
            art.set_joint_velocity_target(drive_target, joint_ids=drive_ids)
            art.set_joint_position_target(steer_target, joint_ids=steer_ids)
            art.write_data_to_sim()
            sim.step()
            scene.update(args.dt)

            if step_count % 4 == 0:   # ~50 Hz GUI at 200 Hz sim
                # --- Lidar ---
                try:
                    hits = lidar.data.ray_hits_w[0]
                    valid = torch.isfinite(hits).all(dim=-1)
                    if valid.any():
                        v = hits[valid].cpu().numpy()
                        z = v[:, 2]
                        lidar_scatter.set_offsets(v[:, :2])
                        lidar_scatter.set_array(z)
                        if z.size:
                            lidar_scatter.set_clim(z.min(), z.max())
                        sp = lidar.data.pos_w[0].cpu().numpy()
                        sensor_marker.set_data([sp[0]], [sp[1]])
                        ax_lidar.set_xlim(sp[0] - 8, sp[0] + 8)
                        ax_lidar.set_ylim(sp[1] - 8, sp[1] + 8)
                except Exception as e:
                    if step_count == 0:
                        print(f"  [lidar warn] {e}", flush=True)

                # --- Camera ---
                try:
                    rgb = camera.data.output.get("rgb")
                    if rgb is not None:
                        img = rgb[0].cpu().numpy()
                        if img.dtype != np.uint8:
                            img = (img.clip(0, 1) * 255).astype(np.uint8)
                        if img.shape[-1] == 4:
                            img = img[..., :3]
                        cam_im.set_data(img)
                except Exception as e:
                    if step_count == 0:
                        print(f"  [cam warn] {e}", flush=True)

                fig.canvas.draw_idle()
                plt.pause(0.001)
                if not plt.fignum_exists(fig.number):
                    print("Window closed, exiting.")
                    break

            if step_count % int(1.0 / args.dt) == 0:    # 1 Hz status
                v = art.data.root_lin_vel_w[0]
                speed = float((v[0] ** 2 + v[1] ** 2) ** 0.5)
                print(f"  src={ctrl.last_source}  thr={thr:+.2f}  "
                      f"steer={steer_in:+.2f}  v={speed:5.2f} m/s",
                      flush=True)

            step_count += 1
    except KeyboardInterrupt:
        print("\nCtrl+C, exiting.")

    plt.close(fig)
    ctrl.close()
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
