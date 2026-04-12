"""M6 smoke test: boot one offroad env kind per invocation.

Usage:
    python scripts/smoke_m6.py --which demo --steps 20
    python scripts/smoke_m6.py --which nav  --steps 20
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--which", choices=["demo", "nav"], required=True)
parser.add_argument("--steps", type=int, default=20)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch  # noqa: E402


def build_env(which: str):
    from hmclab_isaac.envs.offroad._base import ExomyOffroadBaseEnv

    if which == "demo":
        from hmclab_isaac.envs.offroad.sim.exomy_demo import ExomyDemoEnvCfg

        return ExomyOffroadBaseEnv(ExomyDemoEnvCfg())
    if which == "nav":
        from hmclab_isaac.envs.offroad.rl.single_agent.exomy_nav import ExomyNavEnvCfg

        return ExomyOffroadBaseEnv(ExomyNavEnvCfg())
    raise ValueError(which)


def smoke(name: str, env, steps: int) -> None:
    obs, _ = env.reset()
    print(
        f"[{name}] reset OK — num_envs={env.num_envs} obs={tuple(obs['policy'].shape)}",
        flush=True,
    )
    total = torch.zeros(env.num_envs, device=env.device)
    for i in range(steps):
        act = torch.rand(env.num_envs, 2, device=env.device) * 2 - 1
        obs, rew, term, trunc, _info = env.step(act)
        total += rew
        if torch.isnan(obs["policy"]).any():
            raise RuntimeError(f"{name}: NaN obs at step {i}")
        if torch.isnan(rew).any():
            raise RuntimeError(f"{name}: NaN reward at step {i}")
    print(
        f"[{name}] stepped {steps}x OK — mean_total_reward={total.mean().item():.2f}",
        flush=True,
    )
    env.close()


def main() -> None:
    print(f">>> M6_SMOKE which={args.which}", flush=True)
    env = build_env(args.which)
    print(f">>> {args.which} constructed", flush=True)
    smoke(args.which, env, args.steps)
    print(f"M6_SMOKE_{args.which.upper()}_OK", flush=True)


if __name__ == "__main__":
    import traceback

    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
