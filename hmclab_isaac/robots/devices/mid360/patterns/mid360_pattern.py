"""Livox Mid-360 non-repetitive scan pattern for Isaac Lab ray_caster.

Loads the pre-recorded scan pattern from CSV (time, azimuth_deg, zenith_deg)
and converts it to ray directions compatible with Isaac Lab's PatternBaseCfg interface.

The Mid-360 uses a prism-based non-repetitive scanning pattern:
  - 4 channels, 360° horizontal FOV, -7° to +52° vertical FOV
  - 800,000 points per full cycle (40 frames at 20K pts/frame, 10Hz)
  - Pattern repeats after 4 seconds

Reference: https://github.com/fratopa/Mid360_simulation_plugin
"""

from __future__ import annotations

import math
import numpy as np
import os
import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .mid360_pattern_cfg import Mid360PatternCfg


# Default path to the CSV scan pattern file
_DEFAULT_CSV_PATH = os.path.join(os.path.dirname(__file__), "mid360-real-centr.csv")


def mid360_pattern(cfg: Mid360PatternCfg, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate ray directions from the Livox Mid-360 scan pattern CSV.

    The CSV contains columns: time, azimuth(deg), zenith(deg).
    Zenith is measured from the pole (0=up), so elevation = 90 - zenith.

    The function selects a subset of rays based on ``cfg.points_per_frame`` and
    ``cfg.frame_index``, simulating the sliding-window approach of the real sensor.

    Args:
        cfg: Configuration for the Mid-360 pattern.
        device: The device to create the pattern on.

    Returns:
        Tuple of (ray_starts, ray_directions), both shape (N, 3).
    """
    csv_path = cfg.csv_path if cfg.csv_path else _DEFAULT_CSV_PATH

    # Load CSV data: columns are [time, azimuth_deg, zenith_deg]
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    total_points = len(data)

    azimuth_deg = data[:, 1]    # horizontal angle in degrees
    zenith_deg = data[:, 2]     # angle from pole in degrees

    # Convert zenith to elevation: elevation = 90 - zenith
    elevation_deg = 90.0 - zenith_deg

    # Select frame subset (sliding window)
    start_idx = (cfg.frame_index * cfg.points_per_frame) % total_points
    end_idx = start_idx + cfg.points_per_frame

    if end_idx <= total_points:
        indices = np.arange(start_idx, end_idx)
    else:
        # Wrap around
        indices = np.concatenate([
            np.arange(start_idx, total_points),
            np.arange(0, end_idx - total_points),
        ])

    # Apply downsampling
    if cfg.downsample > 1:
        indices = indices[::cfg.downsample]

    az = np.deg2rad(azimuth_deg[indices])
    el = np.deg2rad(elevation_deg[indices])

    # Spherical to Cartesian (x=forward, y=left, z=up in Isaac convention)
    x = np.cos(el) * np.cos(az)
    y = np.cos(el) * np.sin(az)
    z = np.sin(el)

    ray_directions = torch.tensor(
        np.stack([x, y, z], axis=-1), dtype=torch.float32, device=device
    )
    ray_starts = torch.zeros_like(ray_directions)

    return ray_starts, ray_directions


def mid360_full_pattern(cfg: Mid360PatternCfg, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Load ALL rays from the Mid-360 CSV (all 800K points).

    Useful for pre-computing the full pattern and indexing per-frame at runtime.

    Args:
        cfg: Configuration for the Mid-360 pattern.
        device: The device to create the pattern on.

    Returns:
        Tuple of (ray_starts, ray_directions), both shape (800000, 3).
    """
    csv_path = cfg.csv_path if cfg.csv_path else _DEFAULT_CSV_PATH

    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)

    azimuth_deg = data[:, 1]
    elevation_deg = 90.0 - data[:, 2]

    az = np.deg2rad(azimuth_deg)
    el = np.deg2rad(elevation_deg)

    x = np.cos(el) * np.cos(az)
    y = np.cos(el) * np.sin(az)
    z = np.sin(el)

    ray_directions = torch.tensor(
        np.stack([x, y, z], axis=-1), dtype=torch.float32, device=device
    )
    ray_starts = torch.zeros_like(ray_directions)

    return ray_starts, ray_directions
