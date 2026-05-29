"""Build the lab's per-vehicle assembly USDs.

Each output USD references a chassis from ``chassis/<FAMILY>/`` and a
sensor visual from ``devices/<DEVICE>/``. No geometry is copied —
chassis or sensor tweaks flow through automatically via USD references.

Output paths:
    hmclab_isaac/robots/vehicles/HAMA/HAMA_1.usd      DBXL_dw  + Mid-360
    hmclab_isaac/robots/vehicles/HAMA/HAMA_2.usd      DBXL_simple + Mid-360
    hmclab_isaac/robots/vehicles/UNICORN/UNICORN_1.usd SRC_dw    + Mid-360
    hmclab_isaac/robots/vehicles/UNICORN/UNICORN_2.usd SRC_simple + Hokuyo UST-20LX

The chassis × sensor combo per variant is encoded in :data:`VEHICLES`.

Run:
    OMNI_KIT_ACCEPT_EULA=YES \\
      /home/js/anaconda3/envs/hmclab_test/bin/python \\
      scripts/build_vehicles.py --vehicle all
"""
from __future__ import annotations

import argparse
import os

from isaaclab.app import AppLauncher

# Per-variant: (output USD name, chassis USD relpath, device USD relpath, mount offset)
VEHICLES = {
    "HAMA": [
        ("HAMA_1.usd",   "chassis/DBXL/DBXL_dw.usd",     "devices/mid360/Mid360.usd",        (0.05, 0.0, 0.30)),
        ("HAMA_2.usd",   "chassis/DBXL/DBXL_simple.usd", "devices/mid360/Mid360.usd",        (0.05, 0.0, 0.30)),
    ],
    "UNICORN": [
        ("UNICORN_1.usd", "chassis/SRX/SRC_dw.usd",     "devices/mid360/Mid360.usd",        (0.05, 0.0, 0.30)),
        ("UNICORN_2.usd", "chassis/SRX/SRC_simple.usd", "devices/hokuyo/UST_10_20LX.usd",   (0.05, 0.0, 0.20)),
    ],
}

parser = argparse.ArgumentParser()
parser.add_argument("--vehicle", choices=["all", "HAMA", "UNICORN"], default="all")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
launcher = AppLauncher(args)
sim_app = launcher.app

from pxr import Gf, Usd, UsdGeom, UsdPhysics  # noqa: E402

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROBOTS = os.path.join(_REPO, "hmclab_isaac", "robots")


def _rel(target_abs: str, base_layer_abs: str) -> str:
    return os.path.relpath(target_abs, os.path.dirname(base_layer_abs))


def _add_translate(prim: Usd.Prim, offset: tuple[float, float, float]) -> None:
    UsdGeom.Xformable(prim).AddTranslateOp(
        precision=UsdGeom.XformOp.PrecisionDouble
    ).Set(Gf.Vec3d(*offset))


def build_one(vehicle: str, out_name: str, chassis_relpath: str,
              device_relpath: str, lidar_offset: tuple[float, float, float]) -> str:
    chassis_usd = os.path.join(ROBOTS, chassis_relpath)
    device_usd = os.path.join(ROBOTS, device_relpath)
    out_dir = os.path.join(ROBOTS, "vehicles", vehicle)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, out_name)
    if os.path.exists(out_path):
        os.remove(out_path)

    stage = Usd.Stage.CreateNew(out_path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    veh = stage.DefinePrim(f"/{vehicle}", "Xform")
    stage.SetDefaultPrim(veh)
    veh.GetReferences().AddReference(_rel(chassis_usd, out_path))

    # Sensor visual + transform mounted under base_link.
    lidar = stage.DefinePrim(f"/{vehicle}/base_link/lidar", "Xform")
    _add_translate(lidar, lidar_offset)
    lidar_mesh = stage.DefinePrim(f"/{vehicle}/base_link/lidar/mesh", "Xform")
    lidar_mesh.GetReferences().AddReference(_rel(device_usd, out_path))

    if not veh.HasAPI(UsdPhysics.ArticulationRootAPI):
        UsdPhysics.ArticulationRootAPI.Apply(veh)
        print(f"  warn: ArticulationRootAPI not inherited; applied to /{vehicle}")

    stage.GetRootLayer().Save()
    print(f"wrote {out_path}", flush=True)
    print(f"  chassis ref: {_rel(chassis_usd, out_path)}", flush=True)
    print(f"  device  ref: {_rel(device_usd, out_path)} @ {lidar_offset}", flush=True)
    return out_path


def main():
    targets = VEHICLES.keys() if args.vehicle == "all" else [args.vehicle]
    for v in targets:
        for out_name, chassis, device, offset in VEHICLES[v]:
            build_one(v, out_name, chassis, device, offset)
    os._exit(0)


if __name__ == "__main__":
    main()
