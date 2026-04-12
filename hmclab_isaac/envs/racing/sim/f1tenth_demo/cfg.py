"""F1Tenth demo env cfg: 1 env, longer episode, higher render rate.

Use this for GUI playback, manual control tests, or USD inspection. Shares
the base env class so there's no separate `env.py` — the logic is identical.
"""

from __future__ import annotations

from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from hmclab_isaac.envs.racing._base import F1TenthRacingBaseEnv, F1TenthRacingBaseEnvCfg


@configclass
class F1TenthDemoEnvCfg(F1TenthRacingBaseEnvCfg):
    episode_length_s: float = 60.0
    sim: SimulationCfg = SimulationCfg(dt=0.005, render_interval=1, device="cuda:0")
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1,
        env_spacing=0.0,
        replicate_physics=True,
    )


# Demo env reuses the base class — alias for clarity.
F1TenthDemoEnv = F1TenthRacingBaseEnv
