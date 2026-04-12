"""Tutorial 11 — showcase a single 3D track (top-down + isometric views).

Spawns the duct + road for the given track name, sets up a top-down and
an angled isometric camera, takes a still of each. Designed to be called
in a loop over `scripts/generate_3d_tracks.py`'s catalogue so we can
produce a complete showcase.

Run:
    OMNI_KIT_ACCEPT_EULA=YES python Itutorial/11_showcase_tracks.py \\
        --track mini_figure8_over
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--track", type=str, default="mini_oval_banked")
parser.add_argument("--steps", type=int, default=30)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True
launcher = AppLauncher(args)
simulation_app = launcher.app

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sensors import Camera, CameraCfg  # noqa: E402

from _common import (  # noqa: E402
    apply_high_friction_to_mesh,
    spawn_high_friction_ground,
)
from hmclab_isaac.envs.racing._base import default_track_path  # noqa: E402
from hmclab_isaac.utils.capture import top_down_camera_cfg  # noqa: E402
from hmclab_isaac.worlds.racing import RacingTrack, spawn_circuit  # noqa: E402
from hmclab_isaac.worlds.racing.duct_track import spawn_duct_track  # noqa: E402


OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")


def _isometric_camera_cfg(prim_path, center, span, height):
    """Angled side view — camera placed up-and-back, looking at track center."""
    import math
    eye = (center[0] + span * 0.9, center[1] - span * 0.9, height + span * 0.7)
    # We'll set pose via set_world_poses_from_view after init. Start with
    # an identity pose.
    return CameraCfg(
        prim_path=prim_path,
        update_period=0.0,
        width=1280, height=720,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 500.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=eye,
            rot=(1.0, 0.0, 0.0, 0.0),
            convention="opengl",
        ),
    )


def main() -> int:
    track_name = args.track
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=0.01, device="cuda:0")
    )
    sim_utils.DomeLightCfg(intensity=2500.0, color=(0.95, 0.95, 0.95)).func(
        "/World/Light",
        sim_utils.DomeLightCfg(intensity=2500.0, color=(0.95, 0.95, 0.95)),
    )
    spawn_high_friction_ground("/World/ground")

    track = RacingTrack.load(default_track_path(track_name))
    spawn_circuit(track, prim_path="/World/Road", surface_only=True,
                  color=(0.35, 0.35, 0.35))
    apply_high_friction_to_mesh("/World/Road/mesh")
    spawn_duct_track(
        track, prim_path="/World/Duct",
        pipe_radius=0.15, pipe_offset=1.0, rib_spacing=0.8,
    )

    xs = track.positions[:, 0]
    ys = track.positions[:, 1]
    zs = track.positions[:, 2]
    cx = float((xs.min() + xs.max()) / 2)
    cy = float((ys.min() + ys.max()) / 2)
    cz = float((zs.min() + zs.max()) / 2)
    span = float(max(xs.max() - xs.min(), ys.max() - ys.min())) * 1.2 + 2.0

    # Top-down camera
    cam_td = Camera(
        top_down_camera_cfg(
            "/World/TopDownCam",
            center=(cx, cy), size=span,
            width=1280, height=720,
        )
    )
    # Isometric camera (just a pinhole; we'll reposition via set_world_poses_from_view)
    cam_iso = Camera(_isometric_camera_cfg(
        "/World/IsoCam",
        center=(cx, cy), span=span, height=cz,
    ))

    sim.reset()
    # Warm up both cameras
    for cam in (cam_td, cam_iso):
        if not getattr(cam, "_is_initialized", True):
            cam._initialize_impl()
            cam._is_initialized = True

    # Point the iso cam at the track center
    eye = torch.tensor(
        [[cx + span * 0.8, cy - span * 0.8, cz + span * 0.7]],
        device=cam_iso._device, dtype=torch.float32,
    )
    target = torch.tensor(
        [[cx, cy, cz]],
        device=cam_iso._device, dtype=torch.float32,
    )
    try:
        cam_iso.set_world_poses_from_view(eye, target)
    except Exception as exc:
        print(f"[iso] {exc}", flush=True)

    for _ in range(args.steps):
        sim.step()

    def grab(cam):
        try:
            cam.update(dt=0.0, force_recompute=True)
        except Exception as exc:
            print(f"[grab] {exc}", flush=True)
            return None
        out = getattr(cam.data, "output", None)
        rgb = out.get("rgb") if out else None
        if rgb is None:
            return None
        arr = rgb.detach().cpu().numpy() if hasattr(rgb, "detach") else np.asarray(rgb)
        if arr.ndim == 4:
            arr = arr[0]
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr

    from PIL import Image
    os.makedirs(OUT_DIR, exist_ok=True)
    for tag, cam in (("topdown", cam_td), ("iso", cam_iso)):
        frame = grab(cam)
        if frame is not None:
            out_path = os.path.join(OUT_DIR, f"tut11_{track_name}_{tag}.png")
            Image.fromarray(frame).save(out_path)
            print(f"SAVED {out_path}", flush=True)

    print(f"TUT11_OK track={track_name}", flush=True)
    return 0


if __name__ == "__main__":
    import traceback
    code = 0
    try:
        code = main()
    except Exception:
        traceback.print_exc()
        code = 1
    os._exit(code)
