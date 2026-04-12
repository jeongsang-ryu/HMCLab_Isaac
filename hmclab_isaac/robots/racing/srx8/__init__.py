"""1/8-scale buggy (Serpent SRX8-inspired) ArticulationCfg.

Generated from primitives by `scripts/build_srx8_usd.py`. Joint naming
mirrors the MuSHR layout so our tutorial `_common.get_robot_spec` regexes
work unchanged (2 front steer, 4 throttle, 4 passive suspension).
"""

from .srx8_cfg import (
    MAX_STEER,
    SRX8_CFG,
    TRACK_WIDTH,
    USD_PATH,
    WHEEL_RADIUS,
    WHEELBASE,
)

__all__ = [
    "SRX8_CFG",
    "USD_PATH",
    "WHEELBASE",
    "TRACK_WIDTH",
    "WHEEL_RADIUS",
    "MAX_STEER",
]
