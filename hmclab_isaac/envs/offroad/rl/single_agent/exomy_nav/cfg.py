"""Parallel ExoMy nav training cfg."""

from __future__ import annotations

from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from hmclab_isaac.envs.offroad._base import (
    ExomyOffroadBaseEnv,
    ExomyOffroadBaseEnvCfg,
)


@configclass
class ExomyNavEnvCfg(ExomyOffroadBaseEnvCfg):
    episode_length_s: float = 20.0
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=8,
        env_spacing=4.0,
        replicate_physics=True,
    )


ExomyNavEnv = ExomyOffroadBaseEnv
