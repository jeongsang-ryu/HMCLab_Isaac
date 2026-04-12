"""Offroad worlds: rough terrain, slopes, obstacles.

Primary entry point: `rough_terrain.rover_rough_terrain_cfg()` — returns a
`TerrainImporterCfg` that a DirectRLEnv drops into its scene. `pxr` is only
touched when `TerrainImporterCfg.func()` fires, so this module is safe to
import before `AppLauncher` only if `isaaclab.terrains` stays lazy (it does).
"""

from .rough_terrain import ROVER_ROUGH_TERRAIN_CFG, rover_rough_terrain_cfg

__all__ = ["ROVER_ROUGH_TERRAIN_CFG", "rover_rough_terrain_cfg"]
