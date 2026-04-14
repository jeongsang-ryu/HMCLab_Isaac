"""Livox Mid-360 LiDAR sensor visual + mount helper.

The Mid-360 USD is a visual-only mesh (no physics) generated from
the official STP file. Use `attach_mid360_visual()` after spawning
a vehicle to mount the visual at the LiDAR frame.

Assets:
    assets/mid-360-asm.stp   — original CAD
    assets/mid360.usd        — Isaac Sim visual mesh
    assets/mid360.obj/mtl    — colored mesh (for external viewers)
    assets/mid360.stl         — collision mesh (if needed)

Conversion pipeline:
    STP → (convert_stp.py) → OBJ/STL → (obj_to_usd.py) → USD
"""

from __future__ import annotations

import os
from pathlib import Path

ASSETS_DIR = Path(__file__).parent / "assets"
# HOOPS-converted USD (from STEP, with per-part materials)
USD_PATH = str(ASSETS_DIR / "mid360_hoops.usd")
# Fallback: OCP-converted USD (single material)
USD_PATH_FALLBACK = str(ASSETS_DIR / "mid360.usd")
# OBJ (from online converter or convert_stp.py, with .mtl colors)
OBJ_PATH = str(ASSETS_DIR / "mid360.obj")

# Mid-360 dimensions (from STP, in meters)
WIDTH = 0.073    # m
DEPTH = 0.060    # m
HEIGHT = 0.065   # m


def attach_mid360_visual(
    vehicle_prim_path: str,
    mount_link: str = "base_link",
    offset: tuple[float, float, float] = (0.05, 0.0, 0.2),
) -> None:
    """Attach the Mid-360 visual mesh to a vehicle's LiDAR mount point.

    Adds a reference to the Mid-360 USD as a child prim under
    `<vehicle_prim_path>/<mount_link>/mid360_visual`. The visual
    moves with the vehicle automatically via the USD hierarchy.

    Args:
        vehicle_prim_path: e.g. "/World/Ego" or "/World/envs/env_.*/Ego"
        mount_link: link name to attach under (default: base_link)
        offset: (x, y, z) offset from the mount link origin (meters)
    """
    import omni.usd
    from pxr import Gf, Sdf, UsdGeom

    stage = omni.usd.get_context().get_stage()

    # Find matching prims (handles regex patterns for multi-env)
    for prim in stage.Traverse():
        path = prim.GetPath().pathString
        if not path.endswith(f"/{mount_link}"):
            continue
        if vehicle_prim_path.split("/env_")[0] not in path:
            continue

        vis_path = f"{path}/mid360_visual"
        if stage.GetPrimAtPath(vis_path).IsValid():
            continue

        # Add reference to the Mid-360 USD
        vis_prim = stage.DefinePrim(vis_path)
        vis_prim.GetReferences().AddReference(
            assetPath=USD_PATH,
            primPath="/Mid360",
        )

        # Apply offset transform
        xf = UsdGeom.Xformable(vis_prim)
        xf.ClearXformOpOrder()
        xf.AddTranslateOp().Set(Gf.Vec3d(*offset))

    return


__all__ = ["USD_PATH", "WIDTH", "DEPTH", "HEIGHT", "attach_mid360_visual"]
