"""Camera sensor factories (TiledCamera based)."""

from __future__ import annotations

from typing import Sequence


def make_tiled_camera_cfg(
    prim_path: str,
    width: int = 640,
    height: int = 480,
    mount_pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
    mount_rot: tuple[float, float, float, float] = (0.5, -0.5, 0.5, -0.5),
    focal_length: float = 24.0,
    horizontal_aperture: float = 44.73,
    clipping_range: tuple[float, float] = (0.01, 300.0),
    update_period: float = 0.033,
    data_types: Sequence[str] = ("rgb",),
):
    """Front/rear camera factory.

    Default rotation is ROS convention (Z-forward → X-forward).
    Default aperture matches RealSense D435 HFOV (~86°) at width=640.
    """
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import TiledCameraCfg

    return TiledCameraCfg(
        prim_path=prim_path,
        update_period=update_period,
        offset=TiledCameraCfg.OffsetCfg(
            pos=mount_pos,
            rot=mount_rot,
            convention="ros",
        ),
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=focal_length,
            focus_distance=400.0,
            horizontal_aperture=horizontal_aperture,
            clipping_range=clipping_range,
        ),
        data_types=list(data_types),
        width=width,
        height=height,
    )
