"""Single-agent F1Tenth racing, N parallel envs. Same logic as base."""

from __future__ import annotations

from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from hmclab_isaac.envs.racing._base import F1TenthRacingBaseEnv, F1TenthRacingBaseEnvCfg


@configclass
class F1TenthSoloEnvCfg(F1TenthRacingBaseEnvCfg):
    episode_length_s: float = 15.0
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=16,
        env_spacing=0.0,
        replicate_physics=True,
    )


F1TenthSoloEnv = F1TenthRacingBaseEnv
