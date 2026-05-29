"""Bake Isaac Sim 5.1 RTX-LiDAR sensor schema into device USDs.

For each device:
  1. (mid360 only) regenerate ``profiles/Livox_Mid360.json`` from the
     real Livox non-repetitive scan CSV at
     ``patterns/mid360-real-centr.csv``. Use ``--mid360-stride`` to
     subsample (default 10 → ~80K rays). Stride 1 keeps all 800K.
  2. Author a ``Camera`` prim under the USD's defaultPrim with the
     Ouster-style attributes (``cameraSensorType="lidar"``,
     ``sensorModelPluginName``, ``sensorModelConfig``).

Run:
    OMNI_KIT_ACCEPT_EULA=YES \\
      /home/js/anaconda3/envs/hmclab_test/bin/python \\
      scripts/author_lidar_schemas.py --target all
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--target", choices=["all", "hokuyo", "mid360"], default="all",
                    help="which device USD to author")
parser.add_argument("--sensor-prim-name", default="sensor",
                    help="name of the Camera-as-LiDAR prim under defaultPrim")
parser.add_argument("--mid360-stride", type=int, default=10,
                    help="stride into the Mid-360 CSV (1=all 800K, 10=~80K, 100=8K)")
parser.add_argument("--skip-profile-regen", action="store_true",
                    help="skip rebuilding profiles/ JSONs from CSV")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
launcher = AppLauncher(args)
sim_app = launcher.app

from pxr import Sdf, Usd, UsdGeom  # noqa: E402

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _REPO)

from hmclab_isaac.robots.devices.hokuyo.spec import (  # noqa: E402
    USD_PATH as HOKUYO_USD,
    PROFILE_JSON as HOKUYO_JSON,
    RTX_CONFIG_NAME as HOKUYO_CFG,
    SPEC as HOKUYO_SPEC,
)
from hmclab_isaac.robots.devices.mid360.spec import (  # noqa: E402
    USD_PATH as MID360_USD,
    PROFILE_JSON as MID360_JSON,
    PATTERN_CSV as MID360_CSV,
    RTX_CONFIG_NAME as MID360_CFG,
    SPEC as MID360_SPEC,
)

PLUGIN = "omni.sensors.nv.lidar.lidar_core.plugin"


def regenerate_mid360_profile(stride: int) -> None:
    """Read the Livox CSV (Time/s, Azimuth/deg, Zenith/deg) and dump
    the non-repetitive pattern into the RTX LiDAR JSON as a single
    emitter state with ``len(rays)`` arrays."""
    azimuth, elevation, fire_ns = [], [], []
    cycle_ns = int(MID360_SPEC["points_per_second"])  # =200_000 → 1 ns per slot
    # The CSV's "Time/s" column is just a sample index. Mid-360 fires
    # at 200_000 samples/s → 5_000 ns per sample.
    fire_step_ns = round(1e9 / float(MID360_SPEC["points_per_second"]))
    with open(MID360_CSV, newline="") as f:
        rdr = csv.DictReader(f)
        for i, row in enumerate(rdr):
            if i % stride != 0:
                continue
            az = float(row["Azimuth/deg"])
            zen = float(row["Zenith/deg"])
            azimuth.append(round(az, 4))
            elevation.append(round(90.0 - zen, 4))  # zenith → elevation
            fire_ns.append(i * fire_step_ns)
    n = len(azimuth)

    profile = {
        "class": "sensor",
        "type": "lidar",
        "name": "Livox Mid-360",
        "driveWorksId": "GENERIC",
        "profile": {
            "_comment": (
                f"Generated from patterns/mid360-real-centr.csv with stride={stride} "
                f"({n} of 800000 rays). Single emitter state holds the full "
                f"non-repetitive 4 s cycle as parallel arrays."
            ),
            "scanType": "solid_state",
            "intensityProcessing": "normalization",
            "rotationDirection": "CW",
            "rayType": "IDEALIZED",
            "nearRangeM": float(MID360_SPEC["near_range_m"]),
            "farRangeM": float(MID360_SPEC["far_range_m"]),
            "rangeResolutionM": 0.02,
            "rangeAccuracyM": 0.03,
            "reflectionPowerFraction": 0.5,
            "avgPowerW": 0.0005,
            "minReflectance": 0.1,
            "minReflectanceRange": float(MID360_SPEC["far_range_m"]),
            "wavelengthNm": float(MID360_SPEC["wavelength_nm"]),
            "pulseTimeNs": 6,
            "azimuthErrorMean": 0.0,
            "azimuthErrorStd": 0.01,
            "elevationErrorMean": 0.0,
            "elevationErrorStd": 0.01,
            "maxReturns": 1,
            "scanRateBaseHz": float(MID360_SPEC["scan_rate_hz"]),
            "reportRateBaseHz": int(MID360_SPEC["points_per_second"]),
            "numberOfEmitters": 1,
            "numberOfChannels": 1,
            "rangeCount": 1,
            "ranges": [{"min": float(MID360_SPEC["near_range_m"]),
                        "max": float(MID360_SPEC["far_range_m"])}],
            "stateResolutionStep": 1,
            "emitterStateCount": 1,
            "emitterStates": [
                {
                    "azimuthDeg": azimuth,
                    "elevationDeg": elevation,
                    "fireTimeNs": fire_ns,
                    "channelId": [1] * n,
                    "rangeId": [0] * n,
                }
            ],
            "intensityMappingType": "LINEAR",
        },
    }
    os.makedirs(os.path.dirname(MID360_JSON), exist_ok=True)
    with open(MID360_JSON, "w") as f:
        json.dump(profile, f, indent=2)
    sz = os.path.getsize(MID360_JSON) / 1024**2
    print(f"[ok]   regenerated {MID360_JSON}", flush=True)
    print(f"         rays={n}/800000 (stride={stride}), file={sz:.2f} MB",
          flush=True)


def author_one(usd_path: str, config_name: str, spec: dict, sensor_name: str) -> None:
    if not os.path.exists(usd_path):
        print(f"[skip] {usd_path} (USD not found)", flush=True)
        return
    stage = Usd.Stage.Open(usd_path)
    default_prim = stage.GetDefaultPrim()
    if not default_prim:
        print(f"[skip] {usd_path} (no defaultPrim)", flush=True)
        return

    sensor_path = default_prim.GetPath().AppendChild(sensor_name)
    sensor = stage.GetPrimAtPath(sensor_path)
    if not sensor.IsValid():
        sensor = UsdGeom.Camera.Define(stage, sensor_path).GetPrim()

    def _set(name: str, value, type_name: Sdf.ValueTypeName) -> None:
        attr = sensor.GetAttribute(name)
        if not attr or not attr.IsValid():
            attr = sensor.CreateAttribute(name, type_name)
        attr.Set(value)

    _set("cameraSensorType", "lidar", Sdf.ValueTypeNames.Token)
    _set("sensorModelPluginName", PLUGIN, Sdf.ValueTypeNames.Token)
    _set("sensorModelConfig", config_name, Sdf.ValueTypeNames.Token)

    _set("clippingRange",
         (spec.get("near_range_m", 0.1), max(spec.get("far_range_m", 100.0), 1000.0)),
         Sdf.ValueTypeNames.Float2)
    _set("focalLength", 50.0, Sdf.ValueTypeNames.Float)
    _set("horizontalAperture", 20.955, Sdf.ValueTypeNames.Float)
    _set("verticalAperture", 15.291, Sdf.ValueTypeNames.Float)
    _set("projection", "perspective", Sdf.ValueTypeNames.Token)

    stage.GetRootLayer().Save()
    print(f"[ok]   {usd_path}", flush=True)
    print(f"         /{default_prim.GetPath().name}/{sensor_name}: "
          f"sensorModelConfig={config_name}", flush=True)


def main():
    targets = []
    if args.target in ("all", "hokuyo"):
        targets.append((HOKUYO_USD, HOKUYO_CFG, HOKUYO_SPEC))
    if args.target in ("all", "mid360"):
        if not args.skip_profile_regen:
            regenerate_mid360_profile(args.mid360_stride)
        targets.append((MID360_USD, MID360_CFG, MID360_SPEC))
    for usd, name, spec in targets:
        author_one(usd, name, spec, args.sensor_prim_name)
    os._exit(0)


if __name__ == "__main__":
    main()
