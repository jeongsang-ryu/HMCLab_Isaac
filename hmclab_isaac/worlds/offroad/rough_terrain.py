"""Procedural rough terrain for offroad rover environments.

Wraps Isaac Lab's `TerrainImporterCfg` with a single-cell rover-friendly
preset: random uniform bumps at small vertical scale, sized for rover-scale
navigation. Heavier terrain curricula (stairs, slopes) live in the built-in
`isaaclab.terrains.config.rough.ROUGH_TERRAINS_CFG` — use that directly if
you want the full mix.

Two entry points:

    rover_rough_terrain_cfg(...)
        Returns a `TerrainImporterCfg`. Drop into a `DirectRLEnvCfg` (or
        call its `.func()` from `_setup_scene`) to spawn the terrain.

    RoverRoughTerrainCfg
        Module-level default instance with sensible parameters for M6 smoke.
"""

from __future__ import annotations

import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg


def rover_rough_terrain_cfg(
    *,
    num_rows: int = 4,
    num_cols: int = 4,
    size: tuple[float, float] = (10.0, 10.0),
    noise_range: tuple[float, float] = (0.02, 0.08),
    border_width: float = 10.0,
) -> TerrainImporterCfg:
    """Single-cell rover-scale rough terrain.

    Args:
        num_rows, num_cols: tiling of sub-terrain cells.
        size: per-cell size in meters.
        noise_range: (min, max) vertical noise amplitude in meters.
        border_width: flat border around the terrain grid.
    """
    generator = TerrainGeneratorCfg(
        size=size,
        border_width=border_width,
        num_rows=num_rows,
        num_cols=num_cols,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        use_cache=False,
        sub_terrains={
            "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
                proportion=1.0,
                noise_range=noise_range,
                noise_step=0.02,
                border_width=0.25,
            ),
        },
    )
    return TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=generator,
        max_init_terrain_level=None,
        collision_group=-1,
        physics_material=None,
        debug_vis=False,
    )


ROVER_ROUGH_TERRAIN_CFG = rover_rough_terrain_cfg()
