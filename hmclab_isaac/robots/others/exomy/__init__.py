"""ExoMy 6-wheel Mars rover.

Codename: `exomy`. 4-steer / 6-drive / 2-bogie articulation.
Asset sourced from RLRoverLab.

Exports:
  EXOMY_CFG — ArticulationCfg template
  WHEELBASE, TRACK_WIDTH, WHEEL_RADIUS, MAX_STEER — vehicle constants
"""

from .exomy_cfg import (
    EXOMY_CFG,
    MAX_STEER,
    TRACK_WIDTH,
    USD_PATH,
    WHEELBASE,
    WHEEL_RADIUS,
)

__all__ = [
    "EXOMY_CFG",
    "USD_PATH",
    "WHEELBASE",
    "TRACK_WIDTH",
    "WHEEL_RADIUS",
    "MAX_STEER",
]
