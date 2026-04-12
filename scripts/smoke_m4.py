"""M4 smoke test: boot one racing env and step a few times.

Run once per env kind (demo / solo / h2h). Creating multiple envs inside the
same Kit session doesn't work cleanly, so the wrapper script (or CI) invokes
this three times — one Kit boot per env.

Usage:
    python scripts/smoke_m4.py --which demo --steps 20
    python scripts/smoke_m4.py --which solo --steps 20
    python scripts/smoke_m4.py --which h2h  --steps 20
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--which", choices=["demo", "solo", "h2h"], required=True)
parser.add_argument("--steps", type=int, default=20)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch  # noqa: E402


def build_env(which: str):
    if which == "demo":
        from hmclab_isaac.envs.racing._base import F1TenthRacingBaseEnv
        from hmclab_isaac.envs.racing.sim.f1tenth_demo import F1TenthDemoEnvCfg

        return F1TenthRacingBaseEnv(F1TenthDemoEnvCfg())
    if which == "solo":
        from hmclab_isaac.envs.racing._base import F1TenthRacingBaseEnv
        from hmclab_isaac.envs.racing.rl.single_agent.f1tenth_solo import (
            F1TenthSoloEnvCfg,
        )

        return F1TenthRacingBaseEnv(F1TenthSoloEnvCfg())
    if which == "h2h":
        from hmclab_isaac.envs.racing.rl.multi_agent.f1tenth_h2h import (
            F1TenthH2HEnv,
            F1TenthH2HEnvCfg,
        )

        return F1TenthH2HEnv(F1TenthH2HEnvCfg())
    raise ValueError(which)


def smoke(name: str, env, steps: int) -> None:
    obs, _ = env.reset()
    assert "policy" in obs, f"{name}: missing 'policy' obs"
    policy = obs["policy"]
    print(
        f"[{name}] reset OK — num_envs={env.num_envs} obs={tuple(policy.shape)}",
        flush=True,
    )

    total_reward = torch.zeros(env.num_envs, device=env.device)
    for i in range(steps):
        actions = torch.rand(env.num_envs, 2, device=env.device) * 2 - 1
        obs, reward, term, trunc, info = env.step(actions)
        total_reward += reward
        policy = obs["policy"]
        if torch.isnan(policy).any() or torch.isinf(policy).any():
            raise RuntimeError(f"{name}: NaN/Inf in obs at step {i}")
        if torch.isnan(reward).any():
            raise RuntimeError(f"{name}: NaN reward at step {i}")

    print(
        f"[{name}] stepped {steps}x OK — mean_total_reward={total_reward.mean().item():.2f}",
        flush=True,
    )
    env.close()


def main() -> None:
    print(f">>> M4_SMOKE which={args.which} steps={args.steps}", flush=True)
    env = build_env(args.which)
    print(f">>> {args.which} constructed", flush=True)
    smoke(args.which, env, args.steps)
    print(f"M4_SMOKE_{args.which.upper()}_OK", flush=True)


if __name__ == "__main__":
    import traceback

    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
