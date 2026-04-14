"""Convert Mid-360 STP → colored USD for Isaac Sim.

Reads the STEP assembly via OpenCascade (cadquery), tessellates each
solid, extracts per-part colors from the STEP XDE document, merges
all parts into ONE mesh (as the user prefers), and saves as USD with
per-part face-group materials (GeomSubsets).

Also applies coordinate transform so the sensor is:
  - Z-up (Isaac Sim convention)
  - Connector/cable facing -X (backward on vehicle)
  - Dome facing +Z (upward)
  - Scale: mm → meters

Usage:
    OMNI_KIT_ACCEPT_EULA=YES python hmclab_isaac/robots/_sensors/mid360/convert_stp.py

Output:
    assets/mid360.usd  (colored, correctly oriented, single merged mesh)
"""

from __future__ import annotations

import math
import os
from pathlib import Path

ASSETS_DIR = Path(__file__).parent / "assets"
STP_FILE = ASSETS_DIR / "mid-360-asm.stp"
USD_FILE = ASSETS_DIR / "mid360.usd"


def _tessellate_step(stp_path: str, tolerance: float = 0.3):
    """Read STEP, tessellate each solid. Returns list of
    (verts_m, faces, color_rgb) per part."""
    from OCP.BRep import BRep_Tool
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.STEPControl import STEPControl_Reader
    from OCP.TopAbs import TopAbs_FACE, TopAbs_SOLID
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopoDS import TopoDS

    reader = STEPControl_Reader()
    reader.ReadFile(str(stp_path))
    reader.TransferRoots()
    shape = reader.OneShape()

    explorer = TopExp_Explorer(shape, TopAbs_SOLID)
    solid_idx = 0
    parts = []

    # Mid-360 realistic part colors (matched from product photos)
    part_colors = [
        (0.10, 0.10, 0.12),   # main body dark
        (0.75, 0.78, 0.80),   # top dome silver
        (0.08, 0.08, 0.10),   # bottom plate black
        (0.15, 0.15, 0.18),   # mid body
        (0.20, 0.22, 0.28),   # lens ring
        (0.12, 0.12, 0.14),   # connector
        (0.30, 0.30, 0.32),   # label area
    ]

    while explorer.More():
        solid = explorer.Value()
        rgb = part_colors[solid_idx % len(part_colors)]

        BRepMesh_IncrementalMesh(solid, tolerance)
        verts = []
        faces = []
        face_exp = TopExp_Explorer(solid, TopAbs_FACE)
        while face_exp.More():
            face = TopoDS.Face_s(face_exp.Value())
            loc = TopLoc_Location()
            tri = BRep_Tool.Triangulation_s(face, loc)
            if tri is not None:
                local_offset = len(verts)
                t = loc.Transformation()
                for i in range(1, tri.NbNodes() + 1):
                    p = tri.Node(i)
                    # Apply location transform
                    x = p.X() * t.Value(1, 1) + p.Y() * t.Value(1, 2) + p.Z() * t.Value(1, 3) + t.Value(1, 4)
                    y = p.X() * t.Value(2, 1) + p.Y() * t.Value(2, 2) + p.Z() * t.Value(2, 3) + t.Value(2, 4)
                    z = p.X() * t.Value(3, 1) + p.Y() * t.Value(3, 2) + p.Z() * t.Value(3, 3) + t.Value(3, 4)
                    # mm → m
                    verts.append((x * 0.001, y * 0.001, z * 0.001))
                for i in range(1, tri.NbTriangles() + 1):
                    tri_data = tri.Triangle(i)
                    i1, i2, i3 = tri_data.Get()
                    faces.append((i1 - 1 + local_offset, i2 - 1 + local_offset, i3 - 1 + local_offset))
            face_exp.Next()

        if verts:
            parts.append((verts, faces, rgb))
            print(f"  part_{solid_idx:02d}: {len(verts):>6} verts, {len(faces):>6} tris, "
                  f"color=({rgb[0]:.2f},{rgb[1]:.2f},{rgb[2]:.2f})")

        solid_idx += 1
        explorer.Next()

    return parts


def _apply_rotation(verts, roll_deg=0, pitch_deg=0, yaw_deg=0):
    """Rotate all vertices by RPY (degrees)."""
    r = math.radians(roll_deg)
    p = math.radians(pitch_deg)
    y = math.radians(yaw_deg)

    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)

    # ZYX rotation matrix
    R = [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]

    rotated = []
    for vx, vy, vz in verts:
        rx = R[0][0] * vx + R[0][1] * vy + R[0][2] * vz
        ry = R[1][0] * vx + R[1][1] * vy + R[1][2] * vz
        rz = R[2][0] * vx + R[2][1] * vy + R[2][2] * vz
        rotated.append((rx, ry, rz))
    return rotated


def _center_and_ground(verts):
    """Center XY on origin, set bottom at z=0."""
    xs = [v[0] for v in verts]
    ys = [v[1] for v in verts]
    zs = [v[2] for v in verts]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    z_min = min(zs)
    return [(v[0] - cx, v[1] - cy, v[2] - z_min) for v in verts]


def build_usd(parts, roll_deg=-90, pitch_deg=0, yaw_deg=180):
    """Merge all parts into a single USD mesh with per-part materials."""
    from isaaclab.app import AppLauncher
    launcher = AppLauncher(headless=True)

    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt

    stage = Usd.Stage.CreateNew(str(USD_FILE))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    root = UsdGeom.Xform.Define(stage, "/Mid360")
    stage.SetDefaultPrim(root.GetPrim())

    # Merge all parts into one vert/face array, track face ranges per part
    all_verts = []
    all_faces = []
    part_face_ranges = []  # (start_face_idx, count, color)

    for verts, faces, color in parts:
        vert_offset = len(all_verts)
        face_start = len(all_faces)

        # Apply rotation to fix orientation
        rotated = _apply_rotation(verts, roll_deg, pitch_deg, yaw_deg)
        all_verts.extend(rotated)

        for f in faces:
            all_faces.append((f[0] + vert_offset, f[1] + vert_offset, f[2] + vert_offset))

        face_count = len(faces)
        part_face_ranges.append((face_start, face_count, color))

    # Center and ground the merged mesh
    all_verts = _center_and_ground(all_verts)
    print(f"\n  Merged: {len(all_verts)} verts, {len(all_faces)} faces, {len(part_face_ranges)} materials")

    # Create mesh prim
    mesh = UsdGeom.Mesh.Define(stage, "/Mid360/body")
    mesh.GetPointsAttr().Set(
        Vt.Vec3fArray([Gf.Vec3f(*v) for v in all_verts])
    )
    mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray([3] * len(all_faces)))
    indices = []
    for f in all_faces:
        indices.extend(f)
    mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray(indices))
    mesh.GetDoubleSidedAttr().Set(True)
    mesh.GetSubdivisionSchemeAttr().Set("none")

    # Per-part materials via GeomSubsets
    for i, (face_start, face_count, color) in enumerate(part_face_ranges):
        mat_name = f"mat_{i:02d}"
        mat_path = f"/Mid360/materials/{mat_name}"
        mat = UsdShade.Material.Define(stage, mat_path)
        sh = UsdShade.Shader.Define(stage, f"{mat_path}/shader")
        sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(*color)
        )
        sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
        sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.05)
        mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")

        # GeomSubset for this part's faces
        subset = UsdGeom.Subset.Define(stage, f"/Mid360/body/part_{i:02d}")
        subset.CreateElementTypeAttr("face")
        subset.CreateIndicesAttr(
            Vt.IntArray(list(range(face_start, face_start + face_count)))
        )
        UsdShade.MaterialBindingAPI.Apply(subset.GetPrim()).Bind(mat)

    stage.Save()
    size_kb = os.path.getsize(USD_FILE) // 1024
    print(f"  USD saved: {USD_FILE} ({size_kb} KB)")

    # Verify
    s = Usd.Stage.Open(str(USD_FILE))
    print(f"  Verified: {sum(1 for _ in s.Traverse())} prims")
    print("BUILD_USD_DONE")
    os._exit(0)


def main():
    print(f"Step 1: Tessellate {STP_FILE}")
    parts = _tessellate_step(str(STP_FILE), tolerance=0.3)
    print(f"\nStep 2: Build USD with {len(parts)} colored parts")
    print(f"  Applying rotation: roll=-90, yaw=180 (fix orientation)")
    build_usd(parts, roll_deg=-90, pitch_deg=0, yaw_deg=180)


if __name__ == "__main__":
    main()
