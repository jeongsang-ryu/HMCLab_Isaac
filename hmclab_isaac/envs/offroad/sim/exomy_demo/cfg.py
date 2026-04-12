"""Single-env ExoMy demo on rough terrain. Shares the base class."""

from __future__ import annotations

from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from hmclab_isaac.envs.offroad._base import (
    ExomyOffroadBaseEnv,
    ExomyOffroadBaseEnvCfg,
)


@configclass
class ExomyDemoEnvCfg(ExomyOffroadBaseEnvCfg):
    episode_length_s: float = 60.0
    sim: SimulationCfg = SimulationCfg(dt=0.01, render_interval=2, device="cuda:0")
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1,
        env_spacing=4.0,
        replicate_physics=True,
    )


ExomyDemoEnv = ExomyOffroadBaseEnv
