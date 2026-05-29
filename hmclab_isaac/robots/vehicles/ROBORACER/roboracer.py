"""ROBORACER vehicle config — placeholder.

Fill in USD paths, geometry constants, and the actuator dict once the
ROBORACER chassis lands under ``chassis/`` and the assembly USDs are
built (see ``scripts/build_vehicles.py`` for the pattern).
"""
from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# Populate once the assembly USDs exist:
#   USD_PATHS = {
#       1: os.path.join(_THIS_DIR, "ROBORACER_1.usd"),
#       2: os.path.join(_THIS_DIR, "ROBORACER_2.usd"),
#   }
USD_PATHS: dict[int, str] = {}


def make_cfg(variant: int = 1) -> ArticulationCfg:
    """Stub — raises until USD_PATHS and actuators are filled in."""
    if variant not in USD_PATHS:
        raise NotImplementedError(
            f"ROBORACER variant {variant} not configured yet. "
            "Add the USD path and actuator dict in roboracer.py."
        )
    return ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(usd_path=USD_PATHS[variant]),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.10)),
        actuators={},
    )


__all__ = ["USD_PATHS", "make_cfg"]
