"""Single-agent vision-based racing env (UNICORN_3 + duct track + SG2 camera).

Registers the gym id ``HMCLab-Racing-Single-Visual-v0``. The actual env class
and configs are loaded lazily on ``gym.make`` so this module can be imported
before AppLauncher boots — the env itself depends on pxr/USD.
"""
from __future__ import annotations

import gymnasium as gym

gym.register(
    id="HMCLab-Racing-Single-Visual-v0",
    entry_point="hmclab_isaac.envs.racing.rl.single_agent.env:UnicornRacingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "hmclab_isaac.envs.racing.rl.single_agent.cfg:UnicornRacingEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "hmclab_isaac.envs.racing.rl.single_agent.agents."
            "rsl_rl_ppo_cfg:UnicornRacingPPORunnerCfg"
        ),
        "sac_cfg_entry_point": (
            "hmclab_isaac.algos.asym_sac:SKRL_SAC_CONFIG"
        ),
        "a2c_cfg_entry_point": (
            "hmclab_isaac.algos.sync_a2c:DEFAULT_A2C_CONFIG"
        ),
    },
)
