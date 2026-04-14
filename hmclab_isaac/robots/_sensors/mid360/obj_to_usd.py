"""Convert Mid-360 OBJ to USD for Isaac Sim.

Imports the OBJ mesh into an Isaac Sim stage and saves as USD.
This USD can then be referenced by vehicle articulations as a
visual-only sensor prim (no physics, no collision).

Usage:
    OMNI_KIT_ACCEPT_EULA=YES python hmclab_isaac/robots/_sensors/mid360/obj_to_usd.py

Output:
    hmclab_isaac/robots/_sensors/mid360/assets/mid360.usd
"""

from __future__ import annotations

import os
from pathlib import Path

from isaaclab.app import AppLauncher

launcher = AppLauncher(headless=True)
simulation_app = launcher.app

import omni.usd  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade  # noqa: E402

ASSETS_DIR = Path(__file__).parent / "assets"
OBJ_PATH = ASSETS_DIR / "mid360.obj"
USD_PATH = ASSETS_DIR / "mid360.usd"


def parse_obj(obj_path: Path):
    """Minimal OBJ parser — returns verts, faces, material color."""
    verts = []
    faces = []
    with open(obj_path) as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif line.startswith("f "):
                parts = line.split()
                idx = [int(p.split("/")[0]) - 1 for p in parts[1:]]
                faces.append(tuple(idx))
    return verts, faces


def parse_mtl(mtl_path: Path):
    """Read first Kd color from MTL."""
    color = (0.12, 0.12, 0.14)
    if mtl_path.exists():
        with open(mtl_path) as f:
            for line in f:
                if line.startswith("Kd "):
                    parts = line.split()
                    color = (float(parts[1]), float(parts[2]), float(parts[3]))
                    break
    return color


def build():
    print(f"Parsing {OBJ_PATH} ...")
    verts, faces = parse_obj(OBJ_PATH)
    color = parse_mtl(ASSETS_DIR / "mid360.mtl")
    print(f"  {len(verts)} verts, {len(faces)} faces, color={color}")

    # Create new USD stage
    stage = Usd.Stage.CreateNew(str(USD_PATH))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    root = UsdGeom.Xform.Define(stage, "/Mid360")
    stage.SetDefaultPrim(root.GetPrim())

    # Mesh prim
    mesh = UsdGeom.Mesh.Define(stage, "/Mid360/body")
    from pxr import Vt
    mesh.GetPointsAttr().Set(
        Vt.Vec3fArray([Gf.Vec3f(*v) for v in verts])
    )
    mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray([len(f) for f in faces]))
    indices = []
    for f in faces:
        indices.extend(f)
    mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray(indices))
    mesh.GetDoubleSidedAttr().Set(True)

    # Normals (auto compute by setting subdivision)
    mesh.GetSubdivisionSchemeAttr().Set("none")

    # Material — dark charcoal like real Mid-360
    mat_path = "/Mid360/material"
    mat = UsdShade.Material.Define(stage, mat_path)
    shader = UsdShade.Shader.Define(stage, f"{mat_path}/shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(*color)
    )
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.4)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.1)
    mat.CreateSurfaceOutput().ConnectToSource(
        shader.ConnectableAPI(), "surface"
    )
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)

    stage.Save()
    print(f"  USD saved: {USD_PATH} ({os.path.getsize(USD_PATH)//1024} KB)")

    # Verify
    s = Usd.Stage.Open(str(USD_PATH))
    n = sum(1 for _ in s.Traverse())
    print(f"  Verified: {n} prims")
    print("OBJ_TO_USD_DONE")
    os._exit(0)


if __name__ == "__main__":
    build()
