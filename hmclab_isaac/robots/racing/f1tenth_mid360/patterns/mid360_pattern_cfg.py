"""Configuration for the Livox Mid-360 scan pattern."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import MISSING

from isaaclab.utils import configclass
from isaaclab.sensors.ray_caster.patterns.patterns_cfg import PatternBaseCfg

from .mid360_pattern import mid360_pattern as _mid360_pattern_func


@configclass
class Mid360PatternCfg(PatternBaseCfg):
    """Configuration for the Livox Mid-360 non-repetitive scan pattern.

    The Mid-360 has the following specifications:
      - 4 laser channels
      - Horizontal FOV: 360°
      - Vertical FOV: -7° to +52° (59° total)
      - ~200,000 points/sec (20,000 pts/frame at 10Hz)
      - Non-repetitive prism-based scanning
      - Detection range: 0.1m to 40m

    The scan pattern is loaded from a pre-recorded CSV file containing
    (time, azimuth_deg, zenith_deg) for 800,000 points (40 frames).
    """

    func: Callable = _mid360_pattern_func
    """Pattern generation function. Defaults to single-frame pattern."""

    csv_path: str = os.path.join(os.path.dirname(__file__), "mid360-real-centr.csv")
    """Path to the Mid-360 scan pattern CSV file."""

    points_per_frame: int = 20000
    """Number of ray points per frame (at 10Hz update rate). Defaults to 20000."""

    frame_index: int = 0
    """Which frame to use from the scan cycle (0-39). Defaults to 0.

    The Mid-360 has a 40-frame cycle (4 seconds at 10Hz).
    Different frame indices give different ray directions due to
    the non-repetitive scanning pattern.
    """

    downsample: int = 1
    """Downsampling factor for rays. 1 = no downsampling. Defaults to 1.

    Set to higher values (e.g., 4 or 10) to reduce computation at the
    cost of sparser point clouds.
    """

    max_range: float = 40.0
    """Maximum detection range in meters. Defaults to 40.0m (Mid-360 spec)."""

    min_range: float = 0.1
    """Minimum detection range in meters. Defaults to 0.1m."""
