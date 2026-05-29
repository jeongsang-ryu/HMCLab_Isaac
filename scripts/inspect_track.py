"""Standalone track inspector.

Loads a RacingTrack .txt and:

  * Spawns the duct mesh (or circuit) as it would appear in sim,
  * Overlays the *source* polylines (centerline blue, left green, right red)
    as floating colored lines above the surface,
  * Prints geometry stats (length, min radius, tangent-change, width stats),
  * Saves a matplotlib top-down PNG for offline review.

No vehicle, no lidar, no teleop — purely for diagnosing why a duct mesh
looks twisted vs. what the .txt actually contains.

Usage:
  OMNI_KIT_ACCEPT_EULA=YES python scripts/inspect_track.py --track test
  OMNI_KIT_ACCEPT_EULA=YES python scripts/inspect_track.py --track test \\
      --track-type duct --duct-rib-spacing 0 --line-z 1.2
"""
from __future__ import annotations

import argparse
import os
import sys

_PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

from isaaclab.app import AppLauncher

TRACKS_DIR = os.path.join(
    _PROJ_ROOT, "hmclab_isaac/worlds/racing/_tracks_data"
)
TRACK_CHOICES = sorted(
    f.removesuffix(".txt") for f in os.listdir(TRACKS_DIR) if f.endswith(".txt")
)

parser = argparse.ArgumentParser()
parser.add_argument("--track", required=True, choices=TRACK_CHOICES)
parser.add_argument("--track-type", choices=["duct", "circuit", "none"],
                    default="duct",
                    help="duct = orange pipes + ribs (default); circuit = "
                         "thin walls + strip; none = lines only.")
parser.add_argument("--track-subdivide", type=int, default=10)
parser.add_argument("--duct-pipe-radius", type=float, default=0.20)
parser.add_argument("--duct-pipe-offset", type=float, default=0.0,
                    help="0 = auto (track widths mean).")
parser.add_argument("--duct-rib-spacing", type=float, default=0.5,
                    help="Set 0 to disable ribs (easier to see pipe shape).")
parser.add_argument("--circuit-wall-height", type=float, default=0.5)
parser.add_argument("--line-z", type=float, default=1.0,
                    help="Lift debug polylines this much above the surface.")
parser.add_argument("--line-thickness", type=float, default=0.05)
parser.add_argument("--no-lines", action="store_true",
                    help="Skip debug polylines (just the mesh).")
parser.add_argument("--no-plot", action="store_true",
                    help="Skip the matplotlib 2D top-down PNG.")
parser.add_argument("--plot-out", type=str, default="/tmp/track_inspect.png")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = False
args.enable_cameras = False

launcher = AppLauncher(args)
sim_app = launcher.app


import numpy as np    # noqa: E402
import isaaclab.sim as sim_utils    # noqa: E402
from hmclab_isaac.worlds.racing import (    # noqa: E402
    RacingTrack, spawn_circuit, spawn_duct_track,
)
from hmclab_isaac.worlds.racing.duct_track import _track_frames    # noqa: E402


def _densify(track: RacingTrack, sub: int) -> RacingTrack:
    """Delegates to RacingTrack.densify() which re-derives yaw via
    forward-diff (not linear interp) to avoid the ±π wrap-around bug."""
    return track.densify(sub)


def _spawn_debug_lines(track: RacingTrack, lift_z: float, thickness: float):
    """Spawn three colored polylines (centerline / left / right)."""
    import omni.usd
    from pxr import Gf, Sdf, UsdGeom, UsdShade, Vt

    positions, left_dir, _up, _tan = _track_frames(track)
    L = positions + left_dir * track.widths[:, 0:1]
    R = positions - left_dir * track.widths[:, 1:2]

    stage = omni.usd.get_context().get_stage()
    UsdGeom.Xform.Define(stage, "/World/track_debug")

    def _draw(name, pts, color):
        path = f"/World/track_debug/{name}"
        c = UsdGeom.BasisCurves.Define(stage, path)
        p2 = pts.copy()
        p2[:, 2] += lift_z
        c.GetPointsAttr().Set(Vt.Vec3fArray(
            [Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in p2]
        ))
        c.CreateCurveVertexCountsAttr().Set([len(p2)])
        c.CreateTypeAttr().Set(UsdGeom.Tokens.linear)
        c.CreateWidthsAttr().Set(Vt.FloatArray([thickness] * len(p2)))
        c.SetWidthsInterpolation(UsdGeom.Tokens.vertex)
        mp = path + "_Mat"
        m = UsdShade.Material.Define(stage, mp)
        sh = UsdShade.Shader.Define(stage, mp + "/Shader")
        sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(*color)
        )
        sh.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(*[v * 0.6 for v in color])
        )
        sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.7)
        m.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(c.GetPrim()).Bind(m)

    _draw("centerline", positions, (0.10, 0.40, 1.00))
    _draw("left",       L,         (0.10, 1.00, 0.10))
    _draw("right",      R,         (1.00, 0.10, 0.10))


def _print_stats(track: RacingTrack):
    pos = track.positions
    n = len(pos)
    xy = pos[:, :2]
    diffs = np.diff(pos, axis=0)
    seglen = np.linalg.norm(diffs, axis=1)

    # turn radius (3-pt circle)
    radii = []
    for i in range(n):
        ip = (i - 1) % n if track.closed else max(0, i - 1)
        ix = (i + 1) % n if track.closed else min(n - 1, i + 1)
        A, B, C = xy[ip], xy[i], xy[ix]
        a = np.linalg.norm(B - C); b = np.linalg.norm(A - C); c = np.linalg.norm(A - B)
        s = (a + b + c) / 2
        area = max(s * (s - a) * (s - b) * (s - c), 1e-12) ** 0.5
        radii.append((a * b * c) / (4 * area + 1e-12))
    radii = np.array(radii)

    # tangent-angle change
    pos_x, left_dir, _u, tan = _track_frames(track)
    angles_deg = []
    for i in range(n - (0 if track.closed else 1)):
        j = (i + 1) % n
        c = np.clip(tan[i] @ tan[j], -1, 1)
        angles_deg.append(np.degrees(np.arccos(c)))
    angles_deg = np.array(angles_deg)

    print(f"\n=== Track stats: {track.name} ===")
    print(f"  points       : {n}  closed={track.closed}")
    print(f"  length       : {track.length():.2f} m")
    print(f"  segment len  : mean={seglen.mean():.3f}  "
          f"min={seglen.min():.3f}  max={seglen.max():.3f}")
    print(f"  z range      : [{pos[:, 2].min():.3f}, {pos[:, 2].max():.3f}]")
    print(f"  widths L     : mean={track.widths[:, 0].mean():.3f}  "
          f"min={track.widths[:, 0].min():.3f}  max={track.widths[:, 0].max():.3f}")
    print(f"  widths R     : mean={track.widths[:, 1].mean():.3f}  "
          f"min={track.widths[:, 1].min():.3f}  max={track.widths[:, 1].max():.3f}")
    print(f"  turn radius  : min={radii.min():.3f} m  median={np.median(radii):.3f}")
    print(f"  tangent flip : max={angles_deg.max():.2f}°  median={np.median(angles_deg):.2f}°")


def _plot_2d(track: RacingTrack, out_path: str):
    """Top-down PNG of centerline + boundaries + curvature highlights."""
    import matplotlib
    matplotlib.use("Agg")    # non-interactive — script runs headless-safe
    import matplotlib.pyplot as plt

    pos, left_dir, _up, _tan = _track_frames(track)
    L = pos + left_dir * track.widths[:, 0:1]
    R = pos - left_dir * track.widths[:, 1:2]

    # Curvature (color map)
    n = len(pos)
    xy = pos[:, :2]
    radii = []
    for i in range(n):
        ip = (i - 1) % n if track.closed else max(0, i - 1)
        ix = (i + 1) % n if track.closed else min(n - 1, i + 1)
        A, B, C = xy[ip], xy[i], xy[ix]
        a = np.linalg.norm(B - C); b = np.linalg.norm(A - C); c = np.linalg.norm(A - B)
        s = (a + b + c) / 2
        area = max(s * (s - a) * (s - b) * (s - c), 1e-12) ** 0.5
        radii.append((a * b * c) / (4 * area + 1e-12))
    radii = np.array(radii)

    fig, ax = plt.subplots(figsize=(13, 10))
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.plot(pos[:, 0], pos[:, 1], "b-", linewidth=1.5, label="centerline")
    ax.plot(L[:, 0], L[:, 1], "g-", linewidth=1.2, label="left wall (from widths)")
    ax.plot(R[:, 0], R[:, 1], "r-", linewidth=1.2, label="right wall (from widths)")
    # high-curvature dots
    sc = ax.scatter(pos[:, 0], pos[:, 1], c=np.log10(radii + 1e-3),
                    s=12, cmap="plasma", zorder=3)
    cb = fig.colorbar(sc, ax=ax, fraction=0.04)
    cb.set_label("log10(turn radius [m])  — dark = sharp turn")
    # index labels every 1/8th
    stride = max(1, n // 8)
    for i in range(0, n, stride):
        ax.annotate(str(i), (pos[i, 0], pos[i, 1]),
                    fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.legend(loc="upper right")
    ax.set_title(f"{track.name} — {n} pts, length={track.length():.2f}m, "
                 f"min R={radii.min():.2f}m")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[plot] saved {out_path}")


def main() -> int:
    track = RacingTrack.load(os.path.join(TRACKS_DIR, f"{args.track}.txt"))
    _print_stats(track)

    if not args.no_plot:
        _plot_2d(track, args.plot_out)

    # Sim stage
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=0.01, device="cuda:0")
    )
    sim_utils.DomeLightCfg(intensity=2000.0, color=(0.95, 0.95, 0.95)).func(
        "/World/Light",
        sim_utils.DomeLightCfg(intensity=2000.0, color=(0.95, 0.95, 0.95)),
    )
    gp = sim_utils.GroundPlaneCfg()
    gp.func("/World/ground", gp)

    track_dense = _densify(track, args.track_subdivide)

    if args.track_type == "duct":
        pipe_offset = args.duct_pipe_offset
        if pipe_offset <= 0:
            pipe_offset = float(track_dense.widths.mean())
        spawn_duct_track(
            track_dense, "/World/track",
            pipe_radius=args.duct_pipe_radius,
            pipe_offset=pipe_offset,
            rib_spacing=args.duct_rib_spacing,
            duct_color=(1.0, 0.5, 0.0),
            rib_color=(0.05, 0.05, 0.05),
        )
        print(f"[sim] duct: pipe_radius={args.duct_pipe_radius:.2f}  "
              f"offset={pipe_offset:.2f}  rib_spacing={args.duct_rib_spacing:.2f}")
    elif args.track_type == "circuit":
        spawn_circuit(track_dense, "/World/track",
                      wall_height=args.circuit_wall_height,
                      color=(0.4, 0.4, 0.4))
        print(f"[sim] circuit: wall_height={args.circuit_wall_height:.2f}")
    else:
        print("[sim] track_type=none — only lines.")

    if not args.no_lines:
        _spawn_debug_lines(track, args.line_z, args.line_thickness)
        print(f"[sim] debug lines drawn at z+{args.line_z:.2f} (blue/green/red)")

    # Aim viewport at track center
    center = track.positions.mean(axis=0)
    extent = float(np.linalg.norm(
        track.positions.max(axis=0) - track.positions.min(axis=0)
    )) / 2.0
    eye_h = max(extent, 10.0)
    sim.set_camera_view(
        eye=(float(center[0]) + extent * 0.5,
             float(center[1]) - extent * 0.5,
             float(center[2]) + eye_h),
        target=(float(center[0]), float(center[1]), float(center[2])),
    )

    sim.reset()

    print("\n[inspect_track] viewer running. Orbit/zoom in Isaac Sim viewport.")
    print("                close the window or Ctrl+C to exit.\n", flush=True)
    try:
        while sim_app.is_running():
            sim.step()
    except KeyboardInterrupt:
        pass
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
