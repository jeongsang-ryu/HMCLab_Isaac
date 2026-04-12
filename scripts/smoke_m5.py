"""M5 smoke: instantiate ExoMy cfg and verify USD + joints parse OK.

Does NOT run a full DirectRLEnv (that's M6). Just confirms:
  - USD file exists
  - Articulation can be spawned on an empty stage
  - Joint names match the regex groups in the cfg
"""

from __future__ import annotations

from isaaclab.app import AppLauncher

launcher = AppLauncher(headless=True)
simulation_app = launcher.app

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402

from hmclab_isaac.robots.others.exomy import (  # noqa: E402
    EXOMY_CFG,
    USD_PATH,
    WHEELBASE,
    WHEEL_RADIUS,
)


def main() -> None:
    print(f">>> USD_PATH = {USD_PATH}", flush=True)
    import os
    assert os.path.exists(USD_PATH), "USD file missing"
    print(">>> USD file exists", flush=True)

    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=0.01, device="cuda:0")
    )
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0)
    light_cfg.func("/World/Light", light_cfg)

    # Spawn one rover — let Articulation create the prim from USD
    rover_cfg = EXOMY_CFG.replace(prim_path="/World/Rover")
    rover_cfg.spawn.func(
        rover_cfg.prim_path, rover_cfg.spawn, translation=rover_cfg.init_state.pos
    )
    rover = Articulation(rover_cfg)

    sim.reset()
    print(
        f">>> spawned OK — num_joints={rover.num_joints} "
        f"bodies={rover.num_bodies}",
        flush=True,
    )
    print(f">>> joint_names: {rover.joint_names}", flush=True)

    # Group joints by type
    steer = [n for n in rover.joint_names if "Steer_Joint" in n]
    drive = [n for n in rover.joint_names if "Drive_Joint" in n]
    bogie = [n for n in rover.joint_names if "Bogie_Joint" in n]
    print(
        f">>> groups: steer={len(steer)} drive={len(drive)} bogie={len(bogie)}",
        flush=True,
    )
    assert len(steer) == 6, f"expected 6 steer joints, got {len(steer)}"
    assert len(drive) == 6, f"expected 6 drive joints, got {len(drive)}"
    assert len(bogie) == 3, f"expected 3 bogie joints, got {len(bogie)}"

    # Step a few times to verify physics doesn't explode
    for _ in range(10):
        sim.step()
        rover.update(sim.get_physics_dt())

    print(
        f">>> after 10 steps: root_z={rover.data.root_pos_w[0, 2].item():.3f} "
        f"speed={rover.data.root_lin_vel_w[0].norm().item():.3f}",
        flush=True,
    )
    print(f">>> WHEELBASE={WHEELBASE} WHEEL_RADIUS={WHEEL_RADIUS}", flush=True)
    print("M5_SMOKE_OK", flush=True)


if __name__ == "__main__":
    import os
    import traceback

    exit_code = 0
    try:
        main()
    except Exception:
        traceback.print_exc()
        exit_code = 1

    # `simulation_app.close()` hangs for this script after the marker prints
    # (likely ExoMy-specific; other racing envs close cleanly). Skip close
    # and let the OS reap the process.
    os._exit(exit_code)
