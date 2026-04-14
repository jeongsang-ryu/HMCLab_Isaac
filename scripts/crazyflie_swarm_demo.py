"""Four Crazyflie quadcopters hovering + flying a circle.

Uses Isaac Lab's built-in CRAZYFLIE_CFG asset. Each drone is spawned at
a corner of a 2x2 grid; altitude PD + circular trajectory in XY.

Run:
    OMNI_KIT_ACCEPT_EULA=YES \
      /home/js/anaconda3/envs/hmclab_test/bin/python \
      scripts/crazyflie_swarm_demo.py --steps 1800
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=1800)
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--label", type=str, default="swarm")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = False
args.enable_cameras = True
launcher = AppLauncher(args)
sim_app = launcher.app

import torch  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402
from isaaclab_assets import CRAZYFLIE_CFG  # noqa: E402

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Itutorial")
)
from hmclab_isaac.utils.capture import TopDownRecorder  # noqa: E402

OUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "Itutorial", "outputs"
)


def main() -> int:
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=0.005, device="cuda:0")
    )
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=2500.0, color=(0.95, 0.95, 0.95)).func(
        "/World/Light",
        sim_utils.DomeLightCfg(intensity=2500.0, color=(0.95, 0.95, 0.95)),
    )

    # Spawn 4 crazyflies at corners of a 2×2 grid
    N = 4
    spawn_xy = [(+0.8, +0.8), (+0.8, -0.8), (-0.8, +0.8), (-0.8, -0.8)]
    hover_z = 1.2
    drones = []
    for i, (x, y) in enumerate(spawn_xy):
        cfg = CRAZYFLIE_CFG.replace(prim_path=f"/World/Drone{i}")
        cfg = cfg.replace(init_state=cfg.init_state.replace(pos=(x, y, hover_z)))
        cfg.spawn.func(cfg.prim_path, cfg.spawn, translation=cfg.init_state.pos)
        drones.append(Articulation(cfg))

    # Top-down camera covering the formation
    topdown = TopDownRecorder(
        center=(0.0, 0.0),
        size=4.0,
        prim_path="/World/TopDownCam",
        out_dir=OUT_DIR,
        tag=f"crazyflie_{args.label}",
        width=1024,
        height=768,
    )

    sim.reset()
    topdown.on_reset()

    # Cache per-drone info
    info = []
    for d in drones:
        prop_ids = d.find_bodies("m.*_prop")[0]
        mass = d.root_physx_view.get_masses().sum()
        gravity = torch.tensor(sim.cfg.gravity, device=sim.device).norm()
        info.append({"drone": d, "prop_ids": prop_ids, "mass": mass.item(), "g": gravity.item()})

    print(
        "[cf] " + ", ".join(f"d{i}: mass={info[i]['mass']:.4f} props={info[i]['prop_ids']}" for i in range(N)),
        flush=True,
    )

    dt = sim.get_physics_dt()
    frame_step = max(1, int(round(1.0 / (args.fps * dt))))
    step_times = []

    # Simple altitude + XY PD hover controller.
    # Each drone follows a circular path offset from its spawn.
    for step in range(args.steps):
        t = step * dt

        for i, meta in enumerate(info):
            drone = meta["drone"]
            mass = meta["mass"]
            g = meta["g"]
            # Desired XY: drift in a circle around spawn
            cx, cy = spawn_xy[i]
            desired_x = cx + 0.3 * math.cos(0.5 * t + i * math.pi / 2)
            desired_y = cy + 0.3 * math.sin(0.5 * t + i * math.pi / 2)
            desired_z = hover_z + 0.15 * math.sin(0.7 * t)

            pos = drone.data.root_pos_w[0]
            vel = drone.data.root_lin_vel_w[0]

            # PD errors
            ex = desired_x - pos[0].item()
            ey = desired_y - pos[1].item()
            ez = desired_z - pos[2].item()
            vx, vy, vz = vel[0].item(), vel[1].item(), vel[2].item()

            # Translational forces (on the body) — applied at CoG via rotor positions
            # Use rotors as force applicators. Sum must give desired body force.
            # Body-frame +Z is up for level hover. We add lateral components via
            # tilting the thrust vector slightly (naive, no attitude control).
            Fx = 1.0 * (0.8 * ex - 0.4 * vx) * mass  # lateral PD -> translate
            Fy = 1.0 * (0.8 * ey - 0.4 * vy) * mass
            Fz = (mass * g) + (8.0 * ez - 3.5 * vz) * mass  # altitude PD + gravity

            # Split total force across 4 props equally
            forces = torch.zeros(drone.num_instances, 4, 3, device=sim.device)
            torques = torch.zeros_like(forces)
            forces[..., 0] = Fx / 4.0
            forces[..., 1] = Fy / 4.0
            forces[..., 2] = Fz / 4.0
            drone.permanent_wrench_composer.set_forces_and_torques(
                forces=forces, torques=torques, body_ids=meta["prop_ids"],
            )
            drone.write_data_to_sim()

        t0 = time.perf_counter()
        sim.step()
        step_times.append(time.perf_counter() - t0)
        for meta in info:
            meta["drone"].update(dt)

        if step % frame_step == 0:
            topdown.capture_frame()

    out = topdown.finalize()
    print(f"CF_DONE outputs={out}", flush=True)

    import statistics
    warm = step_times[30:]
    if warm:
        print(
            f"STEP_STATS label={args.label} "
            f"median={statistics.median(warm)*1000:.3f}ms "
            f"mean={statistics.mean(warm)*1000:.3f}ms "
            f"p95={sorted(warm)[int(len(warm)*0.95)]*1000:.3f}ms",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    import traceback
    rc = 0
    try:
        rc = main()
    except Exception:
        traceback.print_exc()
        rc = 1
    os._exit(rc)
