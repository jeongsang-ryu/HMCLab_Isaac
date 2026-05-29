"""Keyboard teleop for one of the lab vehicles + live LiDAR debug viz.

Pick any of HAMA_1 / HAMA_2 / UNICORN_1 / UNICORN_2. Spawns the vehicle
in a single-env GUI scene with a few obstacles, attaches the variant's
LiDAR via its ``make_<sensor>()`` factory, turns on Isaac Lab's built-in
ray-hit debug visualisation, and lets you drive with the keyboard.

Realism options (default ON):

  * ``--mid360-cycle``  Advance the Mid-360 ``frame_index`` per LiDAR
                         update so the non-repetitive scan trajectory
                         actually shows up in the viz (different rays
                         each frame). Real-Mid-360-like behaviour.

  * ``--range-noise``    Apply per-ray Gaussian range noise (default
                         2 cm 1-σ) on top of the idealized hits before
                         the debug viz reads them.

Keys:
    Up / Down     forward / reverse throttle
    Left / Right  steer left / right
    Space         brake (zero throttle)
    L             reset all commands

Run:
    OMNI_KIT_ACCEPT_EULA=YES \\
      /home/js/anaconda3/envs/hmclab_test/bin/python \\
      scripts/keyboard_teleop_lidar.py --vehicle HAMA_1
"""
from __future__ import annotations

import argparse
import os
import re
import weakref

from isaaclab.app import AppLauncher

VEHICLE_CHOICES = ["HAMA_1", "HAMA_2",
                   "UNICORN_1", "UNICORN_2", "UNICORN_3", "UNICORN_4"]
TRACKS_DIR = (
    "/home/js/hmcl_issac_project/HMCLab_Isaac/"
    "hmclab_isaac/worlds/racing/_tracks_data"
)
TRACK_CHOICES = sorted(
    f.removesuffix(".txt") for f in os.listdir(TRACKS_DIR) if f.endswith(".txt")
)

parser = argparse.ArgumentParser()
parser.add_argument("--vehicle", choices=VEHICLE_CHOICES, default="HAMA_1")
parser.add_argument("--track", choices=["", *TRACK_CHOICES], default="",
                    help="If set, spawn a 3D racing track instead of a flat "
                         "ground plane. Vehicle is placed at the track's "
                         "starting position with proper yaw.")
parser.add_argument("--track-type", choices=["duct", "circuit"], default="duct",
                    help="duct = orange dual-pipe walls + dark ribs (default). "
                         "circuit = thin walls + flat ground strip.")
parser.add_argument("--track-wall-height", type=float, default=0.5,
                    help="(circuit only) Wall height of the spawned circuit (m).")
parser.add_argument("--track-surface-only", action="store_true",
                    help="(circuit only) Skip walls — just the ground strip.")
parser.add_argument("--track-z-offset", type=float, default=0.30,
                    help="Lift vehicle this much above the track surface (m). "
                         "Must be > base_link → wheel offset (~0.04) + wheel "
                         "radius (~0.05) so the wheels don't spawn embedded.")
parser.add_argument("--track-subdivide", type=int, default=10,
                    help="Subdivide each centerline segment into N pieces. "
                         "Higher = smoother surface, more triangles.")
parser.add_argument("--duct-pipe-radius", type=float, default=0.20,
                    help="(duct only) Pipe radius (m).")
parser.add_argument("--duct-pipe-offset", type=float, default=0.0,
                    help="(duct only) Distance from centerline to pipe center "
                         "(m). 0 = auto from track widths.")
parser.add_argument("--duct-rib-spacing", type=float, default=0.5,
                    help="(duct only) Rib ring spacing along the path (m). "
                         "Set 0 to disable ribs.")
parser.add_argument("--max-wheel-rps", type=float, default=300.0,
                    help="max commanded wheel rotational velocity (rad/s). "
                         "100 rad/s × 0.0525 m wheel ≈ 5.25 m/s top speed.")
parser.add_argument("--dt", type=float, default=0.005)
parser.add_argument("--render-interval", type=int, default=4,
                    help="Render viewport every N physics steps. Decouples "
                         "GPU vsync from PhysX rate. dt=0.005, "
                         "render_interval=4 → 50Hz render @ 200Hz physics. "
                         "Lower = smoother viewport (heavier on RTF), "
                         "higher = better RTF (choppier viewport).")
parser.add_argument("--realtime", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="Throttle the loop so wall-clock matches sim time "
                         "(RTF ≈ 1.0). If sim cannot keep up, RTF naturally "
                         "drops below 1 (no forcing). Pass --no-realtime to "
                         "let the loop free-run.")
parser.add_argument("--mid360-cycle", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="cycle Mid-360 frame_index per update for "
                         "non-repetitive scan behavior (default: on)")
parser.add_argument("--lidar-viz", action=argparse.BooleanOptionalAction,
                    default=False,
                    help="show LiDAR ray-hit debug markers in viewport "
                         "(default: off — heavy on RTF; pass --lidar-viz to enable)")
parser.add_argument("--range-noise", type=float, default=0.0,
                    help="Gaussian range noise std-dev (m). 0 = idealized. "
                         "Per-vehicle sensor params (rate, throughput, "
                         "range, mount, ...) live in the variant's .py — "
                         "edit those, not this script.")
parser.add_argument("--chase-cam", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="Record a chase-cam MP4 of the run (default: on).")
parser.add_argument("--chase-out", type=str, default="/tmp/hmclab_teleop",
                    help="Output directory for chase cam png/mp4.")
parser.add_argument("--chase-tag", type=str, default="",
                    help="Filename tag (default: chase_<vehicle>_<timestamp>).")
parser.add_argument("--chase-fps", type=int, default=60,
                    help="Chase cam capture/encode FPS (default 60).")
parser.add_argument("--chase-max-frames", type=int, default=12000,
                    help="Cap total captured frames (12000 ≈ 200s @ 60 fps).")
parser.add_argument("--chase-behind", type=float, default=3.0,
                    help="Chase cam distance behind vehicle (m).")
parser.add_argument("--chase-above", type=float, default=2.0,
                    help="Chase cam height above vehicle (m).")
parser.add_argument("--disable-vehicle-cameras",
                    action=argparse.BooleanOptionalAction, default=True,
                    help="Deactivate any Camera prims baked into the vehicle "
                         "USD (e.g. SG2_AR0233 on UNICORN) so they cannot be "
                         "picked up as render products. Chase cam under "
                         "/World/ChaseCamera is unaffected.")
# Joystick + keyboard input args (shared)
from hmclab_isaac.utils.teleop_input import add_input_args, make_controller
add_input_args(parser)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = False
args.enable_cameras = True   # required for any Camera sensor (UNICORN_1.cam etc.)
launcher = AppLauncher(args)
sim_app = launcher.app

# Suppress benign PhysX `setMassSpaceInertiaTensor: components must be > 0
# for articulations` spam emitted in GUI mode (probed: every body has
# positive inertia at runtime — this is a benign GUI-extension warning).
# Two prongs:
#   1) Lift /physics/maxNumberOfPhysXErrors so physxui's auto-stop never
#      fires (default 1000 → stops sim after ~5s of GUI mode).
#   2) Filter the log channel + replace physxui's error subscription so
#      individual PHYSX_ERROR notifications also get dropped.
import carb  # noqa: E402
import carb.settings  # noqa: E402

_settings = carb.settings.get_settings()
_settings.set("/physics/maxNumberOfPhysXErrors", 1_000_000_000)

# Silence the omni.physx.plugin log channel via the omni.log API
# (carb.settings on /log/channels/... had no effect — the channel is
# already registered by the time settings is updated, and the level
# was being read at message-emit time from the channel-local cache).
try:
    import omni.log
    _log = omni.log.get_log()
    _log.set_channel_level(
        "omni.physx.plugin",
        omni.log.Level.FATAL,
        omni.log.SettingBehavior.OVERRIDE,
    )
    # The notification_manager re-broadcasts the same message at
    # Warning level via "omni.kit.notification_manager.manager".
    _log.set_channel_level(
        "omni.kit.notification_manager.manager",
        omni.log.Level.ERROR,
        omni.log.SettingBehavior.OVERRIDE,
    )
except Exception:
    pass

import carb.input  # noqa: E402
import numpy as np  # noqa: E402
import omni.appwindow  # noqa: E402
import torch  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402


# ─────────────────────────────────────────────────────────────────────
# Custom keyboard: car-style mapping (←/→ = steer, ↑/↓ = throttle).
# ─────────────────────────────────────────────────────────────────────
class CarKeyboard:
    """Tracks held keys and exposes (throttle, steer) in [-1, 1]."""

    def __init__(self):
        self._held: set[str] = set()
        self._app = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._kb = self._app.get_keyboard()
        self._sub = self._input.subscribe_to_keyboard_events(
            self._kb,
            lambda e, *a, obj=weakref.proxy(self): obj._on_event(e, *a),
        )

    def __del__(self):
        try:
            self._input.unsubscribe_to_keyboard_events(self._kb, self._sub)
        except Exception:
            pass

    def _on_event(self, event, *args, **kwargs):
        name = event.input.name
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if name == "L":
                self._held.clear()
            else:
                self._held.add(name)
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            self._held.discard(name)
        return True

    @property
    def throttle(self) -> float:
        if "SPACE" in self._held:
            return 0.0
        return (1.0 if "UP" in self._held else 0.0) - (1.0 if "DOWN" in self._held else 0.0)

    @property
    def steer(self) -> float:
        return (1.0 if "LEFT" in self._held else 0.0) - (1.0 if "RIGHT" in self._held else 0.0)


def _resolve_variant(name: str):
    """Return (variant_module, sensor_factory, max_steer_rad)."""
    if name == "HAMA_1":
        from hmclab_isaac.robots.vehicles.HAMA import HAMA_1 as v
        return v, v.make_mid360, v.MAX_STEER, "mid360"
    if name == "HAMA_2":
        from hmclab_isaac.robots.vehicles.HAMA import HAMA_2 as v
        return v, v.make_mid360, v.MAX_STEER, "mid360"
    if name == "UNICORN_1":
        from hmclab_isaac.robots.vehicles.UNICORN import UNICORN_1 as v
        return v, v.make_mid360, v.MAX_STEER, "mid360"
    if name == "UNICORN_2":
        from hmclab_isaac.robots.vehicles.UNICORN import UNICORN_2 as v
        return v, v.make_hokuyo, v.MAX_STEER, "hokuyo"
    if name == "UNICORN_3":
        from hmclab_isaac.robots.vehicles.UNICORN import UNICORN_3 as v
        return v, v.make_hokuyo, v.MAX_STEER, "hokuyo"
    if name == "UNICORN_4":
        from hmclab_isaac.robots.vehicles.UNICORN import UNICORN_4 as v
        return v, v.make_mid360, v.MAX_STEER, "mid360"
    raise ValueError(name)


def _load_full_mid360_directions(device: str) -> torch.Tensor:
    """Pre-compute all 800K Mid-360 ray directions (sensor frame) once."""
    from hmclab_isaac.robots.devices.mid360.patterns.mid360_pattern import (
        mid360_full_pattern,
    )
    from hmclab_isaac.robots.devices.mid360.patterns.mid360_pattern_cfg import (
        Mid360PatternCfg,
    )
    cfg = Mid360PatternCfg()
    _, dirs = mid360_full_pattern(cfg, device)
    return dirs   # (800000, 3)


def main():
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(
            dt=args.dt,
            render_interval=max(1, int(args.render_interval)),
            device="cuda:0",
        )
    )

    sim_utils.DomeLightCfg(intensity=2000.0, color=(0.95, 0.95, 0.95)).func(
        "/World/Light",
        sim_utils.DomeLightCfg(intensity=2000.0, color=(0.95, 0.95, 0.95)),
    )

    # Resolve variant FIRST so we can use its INIT_HEIGHT as default spawn z.
    variant, sensor_factory, max_steer, sensor_kind = _resolve_variant(args.vehicle)
    spawn_pos = tuple(variant.CFG.init_state.pos)
    spawn_yaw = 0.0
    lidar_mesh_targets: list[str] = []

    if args.track:
        # Spawn 3D racing track from RacingTrack data.
        from hmclab_isaac.worlds.racing import (
            RacingTrack, spawn_circuit, spawn_duct_track,
        )
        track_path = os.path.join(TRACKS_DIR, f"{args.track}.txt")
        track = RacingTrack.load(track_path)

        # Densify: linearly interpolate centerline N× for a smoother
        # mesh (positions + widths). Yaw is recomputed from forward-diff
        # of the densified positions — linearly interpolating yaw fails
        # at the ±π wrap (e.g. yaw[i]=+pi, yaw[i+1]=-pi: both face the
        # same direction but lerp midway gives 0, flipping the frame and
        # producing ring artifacts in the duct/circuit mesh).
        sub = max(1, int(args.track_subdivide))
        if sub > 1:
            n_old = len(track.positions)
            track = track.densify(sub)
            print(f"[teleop] densified track {n_old} → {len(track.positions)} points "
                  f"(subdivide={sub})", flush=True)
        # Also add a flat ground plane under the track for the vehicle to
        # roll on — duct and circuit only describe the walls.
        gp = sim_utils.GroundPlaneCfg(
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.5, dynamic_friction=1.3, restitution=0.0,
            ),
        )
        gp.func("/World/ground", gp)

        if args.track_type == "duct":
            pipe_offset = args.duct_pipe_offset
            if pipe_offset <= 0:
                pipe_offset = float(track.widths.mean())
            spawn_duct_track(
                track,
                "/World/track",
                pipe_radius=args.duct_pipe_radius,
                pipe_offset=pipe_offset,
                rib_spacing=args.duct_rib_spacing,
                duct_color=(1.0, 0.5, 0.0),
                rib_color=(0.05, 0.05, 0.05),
            )
            print(f"[teleop] duct track: pipe_radius={args.duct_pipe_radius:.2f} "
                  f"offset={pipe_offset:.2f} rib_spacing={args.duct_rib_spacing:.2f}",
                  flush=True)
        else:
            spawn_circuit(
                track,
                "/World/track",
                wall_height=args.track_wall_height,
                surface_only=args.track_surface_only,
                color=(0.4, 0.4, 0.4),
            )
        # Place vehicle at track's first row position with that row's yaw.
        spawn_pos = (
            float(track.positions[0, 0]),
            float(track.positions[0, 1]),
            float(track.positions[0, 2]) + args.track_z_offset,
        )
        spawn_yaw = float(track.rpy[0, 2])
        if args.track_type == "duct":
            lidar_mesh_targets = [
                "/World/ground", "/World/track/duct", "/World/track/ribs",
            ]
        else:
            lidar_mesh_targets = ["/World/track/mesh"]
        sim.set_camera_view(
            eye=(spawn_pos[0] + 3.0, spawn_pos[1] + 3.0, spawn_pos[2] + 2.0),
            target=spawn_pos,
        )
        print(f"[teleop] spawned track {args.track!r}: "
              f"start=({spawn_pos[0]:.2f}, {spawn_pos[1]:.2f}, {spawn_pos[2]:.2f}), "
              f"yaw={spawn_yaw:.3f} rad", flush=True)
    else:
        # Plain flat ground + a few obstacles.
        sim.set_camera_view(eye=(2.5, 2.5, 1.5), target=(0.0, 0.0, 0.2))
        gp = sim_utils.GroundPlaneCfg(
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.5, dynamic_friction=1.3, restitution=0.0,
            ),
        )
        gp.func("/World/ground", gp)
        for i, (x, y) in enumerate([(3.0, 0.0), (0.0, 3.0), (-3.0, -1.5)]):
            cube_cfg = sim_utils.MeshCuboidCfg(
                size=(0.6, 0.6, 1.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.7, 0.2, 0.2)),
            )
            cube_cfg.func(f"/World/Obstacle_{i}", cube_cfg, translation=(x, y, 0.5))
        lidar_mesh_targets = [
            "/World/ground",
            "/World/Obstacle_0/geometry/mesh",
            "/World/Obstacle_1/geometry/mesh",
            "/World/Obstacle_2/geometry/mesh",
        ]

    # Override init_state with track start position + yaw.
    import math
    init_state = variant.CFG.init_state.replace(pos=spawn_pos)
    if args.track:
        # Quaternion from yaw (around Z): (cos(yaw/2), 0, 0, sin(yaw/2)) in (w, x, y, z)
        cy, sy = math.cos(spawn_yaw / 2.0), math.sin(spawn_yaw / 2.0)
        init_state = init_state.replace(rot=(cy, 0.0, 0.0, sy))
    robot_cfg = variant.CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        init_state=init_state,
    )

    # Sensor cfg is constructed from the variant's own defaults — rate,
    # throughput, range, mount offset all live in the variant's .py.
    lidar_cfg = sensor_factory(
        robot_prim_path="/World/envs/env_.*/Robot",
        mesh_targets=lidar_mesh_targets,
    )

    if sensor_kind == "mid360":
        period = 1.0 / variant.LIDAR_RATE_HZ
        thru = variant.LIDAR_THROUGHPUT_RAYS_PER_SEC
        pts = max(1, int(round(thru / variant.LIDAR_RATE_HZ)))
        print(f"[teleop] Mid-360: {variant.LIDAR_RATE_HZ:.0f} Hz × "
              f"{pts} rays/frame = {thru} rays/s "
              f"(800K cycle ≈ {800_000 / thru:.2f} s, "
              f"max_dist={variant.LIDAR_MAX_DISTANCE} m)", flush=True)
    elif sensor_kind == "hokuyo":
        from hmclab_isaac.robots.devices.hokuyo.spec import SPEC as HK_SPEC
        rng = (variant.LIDAR_MAX_DISTANCE if variant.LIDAR_MAX_DISTANCE
               is not None else float(HK_SPEC["far_range_m"]))
        res = (variant.LIDAR_HORIZONTAL_RES_DEG
               if variant.LIDAR_HORIZONTAL_RES_DEG is not None
               else float(HK_SPEC["azimuth_resolution_deg"]))
        n_rays = int(round(float(HK_SPEC["fov_h_deg"]) / res))
        print(f"[teleop] Hokuyo UST-20LX: {variant.LIDAR_RATE_HZ:.0f} Hz "
              f"(FOV={HK_SPEC['fov_h_deg']:.0f}°, res={res}°, "
              f"{n_rays} rays/frame, max_dist={rng} m)", flush=True)

    class _SceneCfg(InteractiveSceneCfg):
        robot = robot_cfg
        lidar = lidar_cfg

    scene = InteractiveScene(_SceneCfg(num_envs=1, env_spacing=2.5,
                                       replicate_physics=True))

    # Deactivate vehicle-baked camera prims so Hydra/replicator never sets up
    # a render product for them. Cheaper than MakeInvisible; the prim is
    # removed from composition until reactivated.
    if args.disable_vehicle_cameras:
        import omni.usd
        stage = omni.usd.get_context().get_stage()
        disabled = []
        for prim in stage.Traverse():
            if prim.GetTypeName() == "Camera":
                p = prim.GetPath().pathString
                if p.startswith("/World/envs/") and "/Robot/" in p:
                    prim.SetActive(False)
                    disabled.append(p)
        if disabled:
            print(f"[teleop] disabled {len(disabled)} vehicle camera(s):",
                  flush=True)
            for p in disabled:
                print(f"           - {p}", flush=True)
        else:
            print("[teleop] no vehicle-baked cameras found", flush=True)

    chase_rec = None
    if args.chase_cam:
        import time
        from hmclab_isaac.utils.capture import ChaseCamRecorder
        tag = args.chase_tag or (
            f"chase_{args.vehicle}_{time.strftime('%Y%m%d_%H%M%S')}"
        )
        chase_rec = ChaseCamRecorder(
            out_dir=args.chase_out,
            tag=tag,
            prim_path="/World/ChaseCamera",
            behind=args.chase_behind,
            above=args.chase_above,
            fps=args.chase_fps,
            max_frames=args.chase_max_frames,
        )
        print(f"[teleop] chase cam → {args.chase_out}/{tag}.mp4", flush=True)

    sim.reset()
    # Step a few frames before enabling debug viz so the first raycast
    # has hits to display (otherwise IsaacLab errors with "Number of
    # markers cannot be zero" when all hits are inf).
    for _ in range(5):
        sim.step()
        scene.update(args.dt)
    scene["lidar"].set_debug_vis(args.lidar_viz)
    if chase_rec is not None:
        chase_rec.on_reset()

    art = scene["robot"]
    joint_names = art.joint_names
    # New SRC_dw chassis uses bare names (`fl_wheel`, `fl_steering`) without
    # the `_joint` suffix. Anchor on `^` to avoid `_shock_upper` etc.
    steer_re = re.compile(r"^(fl|fr|rl|rr)_steering$")
    drive_re = re.compile(r"^(fl|fr|rl|rr)_wheel$")
    steer_ids = [i for i, n in enumerate(joint_names) if steer_re.search(n)]
    drive_ids = [i for i, n in enumerate(joint_names) if drive_re.search(n)]

    # Pre-load full 800K Mid-360 ray bank for runtime cycling.
    full_mid360_dirs = None
    slice_size = 0
    total_pts = 0
    if sensor_kind == "mid360" and args.mid360_cycle:
        full_mid360_dirs = _load_full_mid360_directions(str(sim.device))
        total_pts = full_mid360_dirs.shape[0]
        slice_size = scene["lidar"].num_rays
        print(f"[teleop] Mid-360 non-repetitive cycling: total_pts={total_pts}, "
              f"per-frame={slice_size}", flush=True)

    print(f"\n[teleop] vehicle = {args.vehicle}", flush=True)
    print(f"[teleop] joints  = {len(joint_names)}, steer={steer_ids}, "
          f"drive={drive_ids}", flush=True)
    print(f"[teleop] LiDAR   = {sensor_factory.__name__} → "
          f"{type(scene['lidar']).__name__} ({scene['lidar'].num_rays} rays)",
          flush=True)
    print(f"[teleop] noise   = {args.range_noise:.3f} m σ", flush=True)
    print("\n[teleop] Click on the GUI window first so it owns keyboard focus.",
          flush=True)
    print("[teleop] Keys: ↑/↓ throttle, ←/→ steer, Space brake, L reset.\n",
          flush=True)

    # Replace local CarKeyboard with shared controller (keyboard + joystick)
    ctrl = make_controller(args)
    if args.joystick:
        print(f"[teleop] Joystick: stick X = steer (axis {args.js_axis_steer}), "
              f"stick Y = throttle (axis {args.js_axis_throttle}), "
              f"button {args.js_button_brake} = brake.\n",
              flush=True)
    else:
        print("[teleop] Joystick disabled.\n", flush=True)

    dev = sim.device
    steer_target = torch.zeros((1, len(steer_ids)), device=dev)
    drive_target = torch.zeros((1, len(drive_ids)), device=dev)

    # LiDAR update period in sim steps (e.g. 0.1s / 0.005s = 20 steps).
    lidar_update_period = max(1, int(round(scene["lidar"].cfg.update_period / args.dt)))
    step_count = 0

    # Chase cam capture stride: capture every Nth sim step so encoded fps ≈ chase_fps
    chase_stride = (
        max(1, int(round(1.0 / (args.dt * args.chase_fps))))
        if chase_rec is not None
        else 0
    )

    import math as _math
    import time as _time
    last_diag_step = 0
    last_diag_wall = _time.perf_counter()
    rtf = 0.0
    # Wall-clock anchor for the realtime throttle.
    loop_start_wall = _time.perf_counter()
    loop_start_step = 0
    try:
        while sim_app.is_running():
            # --- Action ---
            thr_in, steer_in = ctrl.read()
            wheel_speed = thr_in * args.max_wheel_rps
            steer_angle = steer_in * max_steer
            drive_target.fill_(wheel_speed)
            steer_target.fill_(steer_angle)
            art.set_joint_velocity_target(drive_target, joint_ids=drive_ids)
            art.set_joint_position_target(steer_target, joint_ids=steer_ids)
            art.write_data_to_sim()

            # Diagnostic: every 0.5s print steer target vs actual joint pos
            if step_count % int(0.5 / args.dt) == 0:
                # RTF over the last diag window
                now = _time.perf_counter()
                wall_dt = now - last_diag_wall
                sim_dt = (step_count - last_diag_step) * args.dt
                rtf = sim_dt / wall_dt if wall_dt > 1e-6 else 0.0
                last_diag_wall = now
                last_diag_step = step_count

                actual = art.data.joint_pos[0, steer_ids]
                # Forward velocity in body frame (signed: + forward, - reverse)
                vx_body = float(art.data.root_lin_vel_b[0, 0])
                # Linear surface speed at the contact = mean(ω) * r over drive wheels
                wheel_omegas = art.data.joint_vel[0, drive_ids]
                omega_mean = float(wheel_omegas.mean())
                v_wheel = omega_mean * variant.WHEEL_RADIUS
                denom = max(abs(vx_body), abs(v_wheel), 0.05)
                slip = (v_wheel - vx_body) / denom        # [-1, +1]
                print(f"[diag] [{ctrl.last_source}] thr={thr_in:+.2f}  "
                      f"steer={steer_in:+.2f}  "
                      f"v_x={vx_body:+.2f} m/s  v_wh={v_wheel:+.2f} m/s  "
                      f"slip={slip*100:+.1f}%  rtf={rtf:.2f}x  "
                      f"steer_actual=[{','.join(f'{v:+.3f}' for v in actual.tolist())}]",
                      flush=True)

            # --- Mid-360 non-repetitive: rotate ray bank just before each update.
            if full_mid360_dirs is not None and step_count % lidar_update_period == 0:
                frame = (step_count // lidar_update_period) % (total_pts // slice_size)
                start = frame * slice_size
                sl = full_mid360_dirs[start:start + slice_size]
                # broadcast over envs (here num_envs=1).
                scene["lidar"].ray_directions[:] = sl.unsqueeze(0)

            sim.step()
            scene.update(args.dt)

            # --- Add Gaussian range noise on top of idealized hits ---
            if args.range_noise > 0.0:
                data = scene["lidar"].data
                hits = data.ray_hits_w           # (N, R, 3)
                origin = data.pos_w.unsqueeze(1) # (N, 1, 3)
                vec = hits - origin
                dist = vec.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                unit = vec / dist
                noisy = dist + torch.randn_like(dist) * args.range_noise
                data.ray_hits_w = origin + unit * noisy

            # --- Chase cam follow + capture
            if chase_rec is not None and step_count % chase_stride == 0:
                rp = art.data.root_pos_w[0]
                rq = art.data.root_quat_w[0]
                qw, qx, qy, qz = (
                    float(rq[0]), float(rq[1]), float(rq[2]), float(rq[3])
                )
                yaw = _math.atan2(
                    2.0 * (qw * qz + qx * qy),
                    1.0 - 2.0 * (qy * qy + qz * qz),
                )
                chase_rec.update_follow(
                    (float(rp[0]), float(rp[1]), float(rp[2])), yaw
                )
                chase_rec.capture_frame()

            step_count += 1

            # Wall-clock throttle: pin RTF to ≤ 1.0 so the user perceives
            # real-time motion. Sleeps only when sim is ahead of wall.
            if args.realtime:
                target_wall = (step_count - loop_start_step) * args.dt
                elapsed_wall = _time.perf_counter() - loop_start_wall
                delay = target_wall - elapsed_wall
                if delay > 0.0:
                    _time.sleep(delay)
    finally:
        if chase_rec is not None:
            try:
                saved = chase_rec.finalize()
                print(f"[teleop] chase cam saved: {saved}", flush=True)
            except Exception as exc:
                print(f"[teleop] chase finalize failed: {exc}", flush=True)

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
