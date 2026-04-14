"""World Tutorial 01 — Ground plane + lighting.

The absolute minimum world: a flat ground and a dome light.
No robot, no track — just the empty stage.

Run:
    python Itutorial/world/01_ground_and_light.py
"""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = False  # GUI so you can see it
launcher = AppLauncher(args)
simulation_app = launcher.app

import isaaclab.sim as sim_utils  # noqa: E402


def main():
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=0.01, device="cuda:0")
    )

    # ── Ground ──
    # GroundPlaneCfg creates an infinite flat plane at z=0.
    # physics_material controls friction (default ~0.5).
    ground_cfg = sim_utils.GroundPlaneCfg(
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=0.8,
        ),
    )
    ground_cfg.func("/World/ground", ground_cfg)
    print(">>> ground created at /World/ground")

    # ── Light ──
    # DomeLightCfg creates ambient lighting from all directions.
    # Without it, the scene is completely black.
    light_cfg = sim_utils.DomeLightCfg(
        intensity=2500.0,
        color=(0.95, 0.95, 0.95),  # slightly warm white
    )
    light_cfg.func("/World/Light", light_cfg)
    print(">>> dome light created at /World/Light")

    sim.reset()
    print(">>> sim started — close the Kit window to exit")

    while simulation_app.is_running():
        sim.step()

    os._exit(0)


if __name__ == "__main__":
    main()
