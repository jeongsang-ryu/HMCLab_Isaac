"""Inspect mass distribution of a vehicle assembly USD.

For each rigid body in the composed stage:
  - mass (authored or auto-from-collision)
  - body-frame CoM offset
  - body's translate from the vehicle root

Then report:
  - total mass
  - per-body share (descending)
  - whole-vehicle CG (mass-weighted body position) in vehicle frame
  - CG height vs wheelbase → estimated wheelie threshold acceleration

Run:
    OMNI_KIT_ACCEPT_EULA=YES \\
      /home/js/anaconda3/envs/hmclab_test/bin/python \\
      scripts/inspect_vehicle_mass.py --vehicle UNICORN_1
"""
from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

VEHICLE_CHOICES = ["HAMA_1", "HAMA_2", "UNICORN_1", "UNICORN_2"]

parser = argparse.ArgumentParser()
parser.add_argument("--vehicle", choices=VEHICLE_CHOICES, default="UNICORN_1")
parser.add_argument("--show-zero-mass", action="store_true",
                    help="also list bodies with no authored mass / auto mass")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
launcher = AppLauncher(args)
sim_app = launcher.app

from pxr import Gf, Usd, UsdGeom, UsdPhysics  # noqa: E402

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _world_pos(prim: Usd.Prim) -> Gf.Vec3d:
    """Position of the prim's local frame in world coords."""
    xform = UsdGeom.Xformable(prim)
    if not xform:
        return Gf.Vec3d(0, 0, 0)
    m = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    return m.ExtractTranslation()


def _resolve_usd(name: str) -> str:
    fam = "HAMA" if name.startswith("HAMA") else "UNICORN"
    return os.path.join(_REPO, "hmclab_isaac", "robots", "vehicles",
                        fam, f"{name}.usd")


def main():
    usd_path = _resolve_usd(args.vehicle)
    print(f"\n=== {os.path.basename(usd_path)} ===\n", flush=True)
    stage = Usd.Stage.Open(usd_path)

    rows = []
    total_mass = 0.0
    weighted_pos = Gf.Vec3d(0, 0, 0)

    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        massapi = UsdPhysics.MassAPI(prim) if prim.HasAPI(UsdPhysics.MassAPI) else None
        m_attr = massapi.GetMassAttr() if massapi else None
        m = float(m_attr.Get()) if (m_attr and m_attr.HasAuthoredValue()) else None

        # Local CoM offset (in body frame); often 0 if not authored.
        com_attr = massapi.GetCenterOfMassAttr() if massapi else None
        com = (com_attr.Get() if (com_attr and com_attr.HasAuthoredValue())
               else Gf.Vec3f(0, 0, 0))

        # Diagonal inertia (kg·m²); only authored if non-default.
        inertia_attr = (massapi.GetDiagonalInertiaAttr() if massapi else None)
        inertia = (inertia_attr.Get()
                   if (inertia_attr and inertia_attr.HasAuthoredValue())
                   else None)

        wpos = _world_pos(prim)
        rows.append({
            "path": prim.GetPath().pathString,
            "name": prim.GetName(),
            "mass_kg": m,
            "local_com": tuple(com),
            "inertia": tuple(inertia) if inertia else None,
            "world_pos": (wpos[0], wpos[1], wpos[2]),
        })
        if m is not None:
            total_mass += m
            # Approximate body CG position (ignore local com offset for the
            # whole-vehicle CG estimate — usually small relative to chassis)
            weighted_pos += Gf.Vec3d(wpos[0], wpos[1], wpos[2]) * m

    # Sort by descending mass, None last.
    rows.sort(key=lambda r: -1e18 if r["mass_kg"] is None else -r["mass_kg"])

    # ---- Per-body listing
    print(f"{'#':>3}  {'name':<28} {'mass [kg]':>10}  "
          f"{'world_pos (x,y,z) [m]':<26}  {'local CoM':<22}  inertia", flush=True)
    print("-" * 120, flush=True)
    n_authored = 0
    for i, r in enumerate(rows):
        if r["mass_kg"] is None and not args.show_zero_mass:
            continue
        n_authored += int(r["mass_kg"] is not None)
        m_str = f"{r['mass_kg']:>10.4f}" if r["mass_kg"] is not None else f"{'(auto)':>10}"
        wp = r["world_pos"]
        lc = r["local_com"]
        wp_s = f"({wp[0]:6.3f}, {wp[1]:6.3f}, {wp[2]:6.3f})"
        lc_s = f"({lc[0]:5.3f},{lc[1]:5.3f},{lc[2]:5.3f})" if any(lc) else "(0,0,0)"
        in_s = "[" + ",".join(f"{v:.3g}" for v in r["inertia"]) + "]" if r["inertia"] else "-"
        print(f"{i:>3}  {r['name']:<28} {m_str}  {wp_s:<26}  {lc_s:<22}  {in_s}",
              flush=True)
    if not args.show_zero_mass:
        n_unauthored = len([r for r in rows if r['mass_kg'] is None])
        if n_unauthored:
            print(f"\n  ({n_unauthored} bodies have NO authored mass — "
                  f"PhysX auto-computes from collision volume × density. "
                  f"Use --show-zero-mass to list them.)", flush=True)

    # ---- Aggregate
    if total_mass > 0:
        cg = weighted_pos / total_mass
    else:
        cg = Gf.Vec3d(0, 0, 0)
    print(f"\n--- aggregate (authored only) ---", flush=True)
    print(f"  total authored mass = {total_mass:.3f} kg "
          f"(over {n_authored} bodies)", flush=True)
    print(f"  CG (weighted by body world pos) = "
          f"({cg[0]:.3f}, {cg[1]:.3f}, {cg[2]:.3f}) m", flush=True)

    # ---- Wheelbase / wheelie estimate
    fl = next((r for r in rows if r["name"] == "fl_wheel"), None)
    rl = next((r for r in rows if r["name"] == "rl_wheel"), None)
    if fl and rl:
        wb = abs(fl["world_pos"][0] - rl["world_pos"][0])
        # CG x-axis distance from front and rear axles
        front_x = (fl["world_pos"][0])
        rear_x = (rl["world_pos"][0])
        forward_axle = max(front_x, rear_x)   # +x assumed forward
        rear_axle = min(front_x, rear_x)
        d_front = forward_axle - cg[0]
        d_rear = cg[0] - rear_axle
        cg_h = cg[2]
        if cg_h > 0:
            a_thresh = 9.81 * d_rear / cg_h
            print(f"\n  wheelbase = {wb:.3f} m  (fl_wheel.x={front_x:.3f}, "
                  f"rl_wheel.x={rear_x:.3f})", flush=True)
            print(f"  CG → front axle = {d_front:.3f} m  "
                  f"(should be ≈ {wb*0.4:.3f} for slight rearward bias)",
                  flush=True)
            print(f"  CG → rear  axle = {d_rear:.3f} m", flush=True)
            print(f"  CG height       = {cg_h:.3f} m", flush=True)
            print(f"  → wheelie threshold accel ≈ "
                  f"{a_thresh:.2f} m/s² ({a_thresh/9.81:.2f} g)", flush=True)
    print("\nDONE", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
