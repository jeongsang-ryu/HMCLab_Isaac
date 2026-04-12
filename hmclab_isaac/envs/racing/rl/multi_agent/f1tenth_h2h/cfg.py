"""Head-to-head F1Tenth racing env cfg."""

from __future__ import annotations

from isaaclab.assets import ArticulationCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from hmclab_isaac.envs.racing._base import F1TenthRacingBaseEnvCfg
from hmclab_isaac.robots.racing.f1tenth_mid360 import F1TENTH_MID360_CFG


@configclass
class F1TenthH2HEnvCfg(F1TenthRacingBaseEnvCfg):
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=8,
        env_spacing=0.0,
        replicate_physics=True,
    )
    episode_length_s: float = 20.0

    # Spawn opponent next to ego. Same USD, different prim regex.
    opp_cfg: ArticulationCfg = F1TENTH_MID360_CFG.replace(
        prim_path="/World/envs/env_.*/Opp"
    )

    # Rule-based opponent speed (constant)
    opp_speed: float = 3.0

    # Reward tweaks on top of base
    overtake_bonus: float = 10.0
    opp_collision_penalty: float = -30.0
    opp_proximity_threshold: float = 0.3
    opp_overtake_distance: float = 1.0
