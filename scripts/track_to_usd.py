"""Convert a RacingTrack .txt to a standalone map .usd file.

Spawns the duct mesh (orange pipes + dark ribs) and optionally a ground
plane in a fresh USD stage, then exports the layer. The resulting .usd
can be opened directly in Isaac Sim or referenced/sublayered into a
vehicle scene.

Usage:
  OMNI_KIT_ACCEPT_EULA=YES python scripts/track_to_usd.py \\
      --input hmclab_isaac/worlds/racing/_tracks_data/my_loop.txt \\
      --output hmclab_isaac/worlds/racing/maps/my_loop.usd \\
      --with-ground
"""
from __future__ import annotations

import argparse
import os
import sys

_PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True, help="Path to track .txt")
parser.add_argument("--output", required=True, help="Path to write .usd")
parser.add_argument("--prim-path", default="/World/track",
                    help="USD prim path of the track in the output stage.")
parser.add_argument("--pipe-radius", type=float, default=0.20)
parser.add_argument("--pipe-offset", type=float, default=0.50,
                    help="Distance from centerline to pipe center (m). "
                         "If 0, the track widths from the .txt are used.")
parser.add_argument("--pipe-resolution", type=int, default=16)
parser.add_argument("--rib-spacing", type=float, default=0.5)
parser.add_argument("--rib-thickness", type=float, default=0.02)
parser.add_argument("--rib-height", type=float, default=0.03)
parser.add_argument("--rib-resolution", type=int, default=12)
parser.add_argument("--track-subdivide", type=int, default=10,
                    help="Densify centerline (1 = no subdivide)")
parser.add_argument("--with-ground", action="store_true",
                    help="Include a ground plane")
parser.add_argument("--duct-color", nargs=3, type=float,
                    default=[1.0, 0.5, 0.0], metavar=("R", "G", "B"))
parser.add_argument("--rib-color", nargs=3, type=float,
                    default=[0.05, 0.05, 0.05], metavar=("R", "G", "B"))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = False

launcher = AppLauncher(args)
sim_app = launcher.app


import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Usd, UsdGeom  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from hmclab_isaac.worlds.racing import RacingTrack, spawn_duct_track  # noqa: E402


def _densify(track: RacingTrack, sub: int) -> RacingTrack:
    """Delegates to RacingTrack.densify() which recomputes yaw via
    forward-diff (linear interp of yaw breaks at ±π wrap → ring
    artifacts in the duct/circuit mesh)."""
    return track.densify(sub)


def main() -> int:
    track = RacingTrack.load(args.input)
    track = _densify(track, int(args.track_subdivide))

    # Fresh sim context (just to get a stage)
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01))

    stage = omni.usd.get_context().get_stage()
    # set defaultPrim and metric metadata
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    if args.with_ground:
        gp = sim_utils.GroundPlaneCfg()
        gp.func("/World/ground", gp)

    # Pipe offset: if 0, use the track's per-point widths (averaged)
    pipe_offset = args.pipe_offset
    if pipe_offset <= 0:
        pipe_offset = float(track.widths.mean())
        print(f"[info] pipe_offset auto = {pipe_offset:.3f} m (from track widths)")

    spawn_duct_track(
        track, args.prim_path,
        pipe_radius=args.pipe_radius,
        pipe_offset=pipe_offset,
        pipe_resolution=args.pipe_resolution,
        rib_spacing=args.rib_spacing,
        rib_thickness=args.rib_thickness,
        rib_height=args.rib_height,
        rib_resolution=args.rib_resolution,
        duct_color=tuple(args.duct_color),
        rib_color=tuple(args.rib_color),
    )

    # Set a defaultPrim so reference/sublayer works without explicit primPath
    root = stage.GetPrimAtPath("/World")
    if root and root.IsValid():
        stage.SetDefaultPrim(root)

    # Export the in-memory stage to the output USD path
    out = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    stage.GetRootLayer().Export(out)
    print(f"[saved] {out}")
    print(f"        track length: {track.length():.2f} m")
    print(f"        defaultPrim:  /World")
    print(f"        prim path:    {args.prim_path}")

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
