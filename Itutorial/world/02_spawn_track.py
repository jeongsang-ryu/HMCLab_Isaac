"""World Tutorial 02 — Load and spawn a racing track.

Shows how to:
  1. Load a RacingTrack from the 8-column text file
  2. Spawn a road surface (circuit mesh)
  3. Spawn duct walls (pipes + ribs)
  4. Query track properties (length, spawn pose, boundaries)

Run:
    python Itutorial/world/02_spawn_track.py --track mini_oval_flat
    python Itutorial/world/02_spawn_track.py --track mini_helix
"""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--track", type=str, default="mini_oval_flat")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = False
launcher = AppLauncher(args)
simulation_app = launcher.app

import isaaclab.sim as sim_utils  # noqa: E402

from hmclab_isaac.envs.racing._base import default_track_path  # noqa: E402
from hmclab_isaac.worlds.racing import RacingTrack, spawn_circuit  # noqa: E402
from hmclab_isaac.worlds.racing.duct_track import spawn_duct_track  # noqa: E402


def main():
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=0.01, device="cuda:0")
    )
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=2500.0).func(
        "/World/Light", sim_utils.DomeLightCfg(intensity=2500.0)
    )

    # ── 1. Load track data ──
    track = RacingTrack.load(default_track_path(args.track))
    print(f"\n>>> Track: {track.name}")
    print(f"    Points:  {track.num_points}")
    print(f"    Length:  {track.length():.2f} m")
    print(f"    Closed:  {track.closed}")
    print(f"    X range: [{track.positions[:,0].min():.1f}, {track.positions[:,0].max():.1f}]")
    print(f"    Y range: [{track.positions[:,1].min():.1f}, {track.positions[:,1].max():.1f}]")
    print(f"    Z range: [{track.positions[:,2].min():.2f}, {track.positions[:,2].max():.2f}]")

    # ── 2. Spawn road surface ──
    spawn_circuit(
        track,
        prim_path="/World/Road",
        surface_only=True,       # ground mesh between boundaries (no walls)
        color=(0.35, 0.35, 0.35),
    )
    print(">>> Road surface spawned at /World/Road")

    # ── 3. Spawn duct walls ──
    spawn_duct_track(
        track,
        prim_path="/World/Duct",
        pipe_radius=0.15,
        pipe_offset=1.0,         # distance from centerline to each pipe
        rib_spacing=0.8,
        duct_color=(1.0, 0.5, 0.0),  # orange pipes
        rib_color=(0.05, 0.05, 0.05), # dark ribs
    )
    print(">>> Duct walls spawned at /World/Duct")

    # ── 4. Query spawn pose ──
    pos, rpy = track.spawn_pose(progress=0.0)
    print(f"\n    Spawn pose at progress=0.0:")
    print(f"      position: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
    print(f"      yaw:      {rpy[2]:.3f} rad ({rpy[2]*57.3:.1f} deg)")

    pos2, rpy2 = track.spawn_pose(progress=0.5)
    print(f"    Spawn pose at progress=0.5:")
    print(f"      position: ({pos2[0]:.2f}, {pos2[1]:.2f}, {pos2[2]:.2f})")

    sim.reset()
    print("\n>>> Scene ready — close Kit window to exit")

    while simulation_app.is_running():
        sim.step()

    os._exit(0)


if __name__ == "__main__":
    main()
