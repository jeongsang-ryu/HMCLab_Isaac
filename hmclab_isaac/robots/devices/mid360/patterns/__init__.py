"""Livox Mid-360 non-repetitive scan pattern for Isaac Lab ray_caster.

The pattern function and config live here so the robot package owns its
sensor pattern. `robots/_sensors/lidar.py` is pattern-agnostic and accepts
whatever `PatternBaseCfg` instance you pass in.
"""

from .mid360_pattern_cfg import Mid360PatternCfg
from .mid360_pattern import mid360_pattern

__all__ = ["Mid360PatternCfg", "mid360_pattern"]
