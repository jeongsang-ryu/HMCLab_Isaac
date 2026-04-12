"""Top-down screenshot/video helpers for tutorials.

Creates a standalone top-down RGB Camera sensor (not attached to any robot),
steps the sim while collecting frames, and writes them as a PNG screenshot
and an MP4 recording. Designed to be dropped into any tutorial script that
already has AppLauncher initialized.

Usage (inside a script, after `AppLauncher(...)`):

    from hmclab_isaac.utils.capture import TopDownRecorder

    rec = TopDownRecorder(
        center=(10.0, 0.0),
        size=20.0,                 # world-space width the camera should cover
        prim_path="/World/TopDownCamera",
        out_dir="/tmp/tut01",
        tag="tut01_ground",
    )
    # After `sim.reset()`:
    rec.on_reset()
    # Each physics step / render:
    rec.capture_frame()
    # When done:
    rec.finalize()

The recorder writes:
    <out_dir>/<tag>.png   — final frame (used as the "screenshot")
    <out_dir>/<tag>.mp4   — accumulated frames at `fps`

The class is intentionally tolerant: if rgb is unavailable for a frame it
skips it, and `finalize()` with zero frames only writes the latest PNG.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

__all__ = [
    "TopDownRecorder",
    "ChaseCamRecorder",
    "top_down_camera_cfg",
    "chase_camera_cfg",
]


def top_down_camera_cfg(
    prim_path: str,
    center: tuple[float, float],
    size: float,
    *,
    width: int = 1280,
    height: int = 720,
    height_above: Optional[float] = None,
):
    """Build a CameraCfg for a top-down orthographic-like view.

    We use a pinhole camera placed high above `center` looking straight down.
    The camera height is chosen so the horizontal field-of-view spans `size`
    meters (width side). This mimics an orthographic top-down for the area
    of interest.
    """
    import isaaclab.sim as sim_utils  # lazy — needs Kit running
    from isaaclab.sensors.camera import CameraCfg

    # Horizontal aperture and focal length at Isaac Lab defaults
    focal_length = 24.0
    horizontal_aperture = 20.955
    # tan(fov/2) = (aperture / 2) / focal_length  -> fov angle is fixed by these
    half_fov = math.atan((horizontal_aperture / 2.0) / focal_length)
    if height_above is None:
        # height so camera covers `size` meters horizontally, with 20% margin
        height_above = (size / 2.0) / max(math.tan(half_fov), 1e-6) * 1.2
        height_above = max(height_above, 5.0)

    # opengl convention: forward = -Z, up = +Y. Identity quaternion already
    # points the camera straight down with world +Y as image-up.
    cfg = CameraCfg(
        prim_path=prim_path,
        update_period=0.0,
        height=height,
        width=width,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=focal_length,
            focus_distance=400.0,
            horizontal_aperture=horizontal_aperture,
            clipping_range=(0.1, height_above * 4.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(float(center[0]), float(center[1]), float(height_above)),
            rot=(1.0, 0.0, 0.0, 0.0),
            convention="opengl",
        ),
    )
    return cfg


@dataclass
class TopDownRecorder:
    center: tuple[float, float]
    size: float
    prim_path: str = "/World/TopDownCamera"
    out_dir: str = "/tmp/hmclab_tutorial"
    tag: str = "capture"
    width: int = 1280
    height: int = 720
    height_above: Optional[float] = None
    fps: int = 30
    max_frames: int = 600  # safety cap (~20 s @30 fps)

    _camera: object = field(init=False, default=None)
    _frames: list = field(init=False, default_factory=list)
    _latest: object = field(init=False, default=None)

    def __post_init__(self) -> None:
        from isaaclab.sensors.camera import Camera
        cfg = top_down_camera_cfg(
            self.prim_path,
            self.center,
            self.size,
            width=self.width,
            height=self.height,
            height_above=self.height_above,
        )
        self._camera = Camera(cfg)
        os.makedirs(self.out_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def on_reset(self) -> None:
        """Call once after `sim.reset()` (or env.reset()) to warm up the sensor.

        Sensors auto-initialize on the timeline PLAY event. If this recorder
        is created *after* the sim has already been played (e.g. after
        `DirectRLEnv.__init__`), the PLAY callback was missed — fall back to
        invoking `_initialize_impl()` directly so `_timestamp` and friends
        get set up.
        """
        cam = self._camera
        if not getattr(cam, "_is_initialized", True):
            try:
                cam._initialize_impl()
                cam._is_initialized = True
            except Exception as exc:
                print(f"[capture] lazy init failed: {exc}", flush=True)
        # Some callback paths still leave _timestamp unset
        if not hasattr(cam, "_timestamp"):
            try:
                cam._initialize_impl()
                cam._is_initialized = True
            except Exception as exc:
                print(f"[capture] retry init failed: {exc}", flush=True)
        try:
            cam.update(dt=0.0, force_recompute=True)
        except TypeError:
            cam.update(0.0)

    def capture_frame(self) -> None:
        """Grab the current RGB output and append to the frame buffer."""
        if len(self._frames) >= self.max_frames:
            return
        try:
            self._camera.update(dt=0.0, force_recompute=True)
        except TypeError:
            self._camera.update(0.0)

        data = self._camera.data
        out = getattr(data, "output", None)
        if out is None or "rgb" not in out:
            return
        rgb = out["rgb"]
        if rgb is None:
            return
        # rgb is a torch tensor shape (N, H, W, C) or similar. Take first.
        try:
            import torch
            if isinstance(rgb, torch.Tensor):
                rgb_np = rgb.detach().cpu().numpy()
            else:
                rgb_np = np.asarray(rgb)
        except Exception:
            rgb_np = np.asarray(rgb)

        if rgb_np.ndim == 4:
            rgb_np = rgb_np[0]
        if rgb_np.shape[-1] == 4:
            rgb_np = rgb_np[..., :3]
        if rgb_np.dtype != np.uint8:
            rgb_np = np.clip(rgb_np, 0, 255).astype(np.uint8)
        self._latest = rgb_np
        self._frames.append(rgb_np)

    def finalize(self) -> dict:
        return _save_frames(self._frames, self._latest, self.out_dir, self.tag, self.fps)


def _save_frames(frames, latest, out_dir, tag, fps) -> dict:
    """Shared PNG + MP4 writer used by all recorder types."""
    os.makedirs(out_dir, exist_ok=True)
    png_path = os.path.join(out_dir, f"{tag}.png")
    mp4_path = os.path.join(out_dir, f"{tag}.mp4")
    saved: dict = {}
    if latest is not None:
        try:
            from PIL import Image
            Image.fromarray(latest).save(png_path)
            saved["png"] = png_path
        except Exception as exc:
            print(f"[capture] PNG write failed: {exc}", flush=True)
    if len(frames) >= 2:
        try:
            import imageio.v2 as imageio
            imageio.mimsave(
                mp4_path, frames, fps=fps, codec="libx264", macro_block_size=None
            )
            saved["mp4"] = mp4_path
        except Exception as exc:
            try:
                import imageio.v2 as imageio
                gif_path = os.path.join(out_dir, f"{tag}.gif")
                imageio.mimsave(gif_path, frames, duration=1.0 / max(fps, 1))
                saved["gif"] = gif_path
            except Exception as exc2:
                print(f"[capture] MP4/GIF write failed: {exc} / {exc2}", flush=True)
    print(f"[capture] saved: {saved}", flush=True)
    return saved


# ----------------------------------------------------------------------
# Chase camera
# ----------------------------------------------------------------------
def chase_camera_cfg(
    prim_path: str,
    *,
    width: int = 1280,
    height: int = 720,
):
    """Build a CameraCfg for a chase-cam that will be re-posed each step.

    Initial pose is arbitrary — the caller calls
    `recorder.update_follow(vehicle_pos, vehicle_yaw)` each frame to move
    the camera behind the vehicle.
    """
    import isaaclab.sim as sim_utils  # lazy
    from isaaclab.sensors.camera import CameraCfg
    return CameraCfg(
        prim_path=prim_path,
        update_period=0.0,
        width=width,
        height=height,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0,          # a bit wider than default
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 100.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 2.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            convention="opengl",
        ),
    )


@dataclass
class ChaseCamRecorder:
    """Rear-top third-person camera that follows a moving vehicle.

    Usage:
        rec = ChaseCamRecorder(out_dir=..., tag="chase01")
        # after sim.reset():
        rec.on_reset()
        # each step:
        rec.update_follow(vehicle_pos=(x,y,z), yaw_rad=...)
        rec.capture_frame()
        # at end:
        rec.finalize()

    Default offset: 3 m behind the vehicle, 2 m above, camera looks at the
    vehicle's current position (+0.2 m up). Uses
    `Camera.set_world_poses_from_view` each frame, which auto-computes the
    look-at quaternion.
    """
    out_dir: str = "/tmp/hmclab_tutorial"
    tag: str = "chase"
    prim_path: str = "/World/ChaseCamera"
    width: int = 1280
    height: int = 720
    behind: float = 3.0
    above: float = 2.0
    look_up: float = 0.2
    fps: int = 30
    max_frames: int = 600

    _camera: object = field(init=False, default=None)
    _frames: list = field(init=False, default_factory=list)
    _latest: object = field(init=False, default=None)

    def __post_init__(self) -> None:
        from isaaclab.sensors.camera import Camera
        cfg = chase_camera_cfg(self.prim_path, width=self.width, height=self.height)
        self._camera = Camera(cfg)
        os.makedirs(self.out_dir, exist_ok=True)

    def on_reset(self) -> None:
        cam = self._camera
        if not getattr(cam, "_is_initialized", True):
            try:
                cam._initialize_impl()
                cam._is_initialized = True
            except Exception as exc:
                print(f"[chase] lazy init failed: {exc}", flush=True)
        if not hasattr(cam, "_timestamp"):
            try:
                cam._initialize_impl()
                cam._is_initialized = True
            except Exception as exc:
                print(f"[chase] retry init failed: {exc}", flush=True)

    def update_follow(self, vehicle_pos, yaw_rad: float) -> None:
        """Move the camera to behind-and-above the vehicle, looking at it."""
        import torch
        px, py, pz = float(vehicle_pos[0]), float(vehicle_pos[1]), float(vehicle_pos[2])
        # eye: behind the car along -heading, lifted up
        cos_y = math.cos(yaw_rad)
        sin_y = math.sin(yaw_rad)
        eye = (
            px - self.behind * cos_y,
            py - self.behind * sin_y,
            pz + self.above,
        )
        target = (px, py, pz + self.look_up)
        eyes_t = torch.tensor([eye], device=self._camera._device, dtype=torch.float32)
        targets_t = torch.tensor([target], device=self._camera._device, dtype=torch.float32)
        try:
            self._camera.set_world_poses_from_view(eyes_t, targets_t)
        except Exception as exc:
            print(f"[chase] set_world_poses_from_view failed: {exc}", flush=True)

    def capture_frame(self) -> None:
        if len(self._frames) >= self.max_frames:
            return
        try:
            self._camera.update(dt=0.0, force_recompute=True)
        except TypeError:
            self._camera.update(0.0)
        except Exception:
            return
        out = getattr(self._camera.data, "output", None)
        rgb = out.get("rgb") if out else None
        if rgb is None:
            return
        try:
            import torch
            if isinstance(rgb, torch.Tensor):
                arr = rgb.detach().cpu().numpy()
            else:
                arr = np.asarray(rgb)
        except Exception:
            arr = np.asarray(rgb)
        if arr.ndim == 4:
            arr = arr[0]
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        self._latest = arr
        self._frames.append(arr)

    def finalize(self) -> dict:
        return _save_frames(self._frames, self._latest, self.out_dir, self.tag, self.fps)
