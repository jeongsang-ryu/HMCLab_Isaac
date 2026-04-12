"""Launch one env with the Isaac Sim GUI for visual inspection.

Intended usage: user watches the Kit window on their desktop, Ctrl+C to quit.
Random actions are applied each step so the vehicle moves and sensors fire.

Usage:
    python scripts/gui_launch.py --which f1tenth_demo
    python scripts/gui_launch.py --which exomy_demo --action zero
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument(
    "--which",
    choices=[
        "f1tenth_demo",
        "f1tenth_solo",
        "f1tenth_h2h",
        "exomy_demo",
        "exomy_nav",
    ],
    required=True,
)
parser.add_argument(
    "--action",
    choices=["random", "zero", "forward"],
    default="random",
    help="action policy while looping",
)
parser.add_argument(
    "--max-steps",
    type=int,
    default=0,
    help="auto-exit after N steps (0 = loop until Ctrl+C)",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Force GUI on regardless of other env vars. enable_cameras=True lets
# TiledCameras render during GUI; rover env doesn't use cameras but the
# racing ones do.
args.headless = False
args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch  # noqa: E402


def build_env(which: str):
    from hmclab_isaac.envs.racing._base import F1TenthRacingBaseEnv
    from hmclab_isaac.envs.offroad._base import ExomyOffroadBaseEnv

    if which == "f1tenth_demo":
        from hmclab_isaac.envs.racing.sim.f1tenth_demo import F1TenthDemoEnvCfg
        return "f1tenth_demo", F1TenthRacingBaseEnv(F1TenthDemoEnvCfg())
    if which == "f1tenth_solo":
        from hmclab_isaac.envs.racing.rl.single_agent.f1tenth_solo import (
            F1TenthSoloEnvCfg,
        )
        return "f1tenth_solo", F1TenthRacingBaseEnv(F1TenthSoloEnvCfg())
    if which == "f1tenth_h2h":
        from hmclab_isaac.envs.racing.rl.multi_agent.f1tenth_h2h import (
            F1TenthH2HEnv,
            F1TenthH2HEnvCfg,
        )
        return "f1tenth_h2h", F1TenthH2HEnv(F1TenthH2HEnvCfg())
    if which == "exomy_demo":
        from hmclab_isaac.envs.offroad.sim.exomy_demo import ExomyDemoEnvCfg
        return "exomy_demo", ExomyOffroadBaseEnv(ExomyDemoEnvCfg())
    if which == "exomy_nav":
        from hmclab_isaac.envs.offroad.rl.single_agent.exomy_nav import (
            ExomyNavEnvCfg,
        )
        return "exomy_nav", ExomyOffroadBaseEnv(ExomyNavEnvCfg())
    raise ValueError(which)


def pick_action(which: str, env) -> torch.Tensor:
    shape = (env.num_envs, 2)
    if args.action == "zero":
        return torch.zeros(shape, device=env.device)
    if args.action == "forward":
        return torch.tensor([[0.0, 1.0]] * env.num_envs, device=env.device)
    return torch.rand(shape, device=env.device) * 2 - 1


def main() -> None:
    print(f">>> launching {args.which} with GUI", flush=True)
    name, env = build_env(args.which)
    print(
        f">>> env ready — num_envs={env.num_envs} "
        f"action_dim={env.cfg.action_space} obs_dim={env.cfg.observation_space}",
        flush=True,
    )
    obs, _ = env.reset()
    print(f">>> reset OK — obs shape={tuple(obs['policy'].shape)}", flush=True)
    print(">>> stepping... press Ctrl+C in this terminal to stop", flush=True)

    step = 0
    try:
        while args.max_steps == 0 or step < args.max_steps:
            action = pick_action(args.which, env)
            obs, reward, term, trunc, _info = env.step(action)
            step += 1
            if step % 100 == 0:
                print(
                    f"    step {step}  mean_reward={reward.mean().item():.2f}",
                    flush=True,
                )
    except KeyboardInterrupt:
        print("\n>>> Ctrl+C received, closing", flush=True)
    print(f">>> total steps: {step}", flush=True)


if __name__ == "__main__":
    import os
    import traceback

    try:
        main()
    except Exception:
        traceback.print_exc()
    finally:
        # SimulationApp.close() sometimes hangs with ExoMy; force-exit.
        os._exit(0)
