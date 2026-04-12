"""Circuit track mesh generator.

Takes a `RacingTrack` (see `_schema.py`) and produces a triangle mesh with:
  - Ground surface between the left/right inner boundaries
  - Left inner wall
  - Right inner wall

Wall top cap, outer walls, and banking visuals are skipped for M3 — the goal
is a collision-accurate, LiDAR-visible track shell. Outer walls and cosmetic
extras can be added later without breaking the interface.

Two entry points:

    build_circuit_mesh(track, wall_height=0.5)
        Pure-numpy mesh construction. Returns a dict with vertices/faces arrays.
        Safe to call without Kit running (no pxr import).

    spawn_circuit(track, prim_path, wall_height=0.5)
        Builds the mesh and registers it as a USD Mesh prim under `prim_path`.
        Adds collision API. Must be called after AppLauncher bootstrapped Kit.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ._schema import RacingTrack

__all__ = ["build_circuit_mesh", "spawn_circuit", "save_obj"]


def build_circuit_mesh(
    track: RacingTrack,
    *,
    wall_height: float = 0.5,
    ground_z_offset: float = 0.0,
    surface_only: bool = False,
) -> dict[str, Any]:
    """Build a triangle mesh for the circuit.

    Args:
        track: Centerline + rpy + widths.
        wall_height: Height of inner walls (ignored if `surface_only=True`).
        ground_z_offset: Global z offset for the ground strip.
        surface_only: If True, emit ONLY the ground quad strip — no walls.
            Useful when you want to combine a smooth 3D road surface with a
            duct mesh on top (or any other wall style).

    The ground surface always follows `track.boundary_points()`, which uses
    each point's local rotation matrix — so 3D tracks (banking + elevation)
    produce a properly tilted surface out of the box.
    """
    left_inner, right_inner = track.boundary_points()  # (N, 3) each, 3D-aware
    if ground_z_offset:
        offset = np.array([0.0, 0.0, ground_z_offset], dtype=np.float64)
        left_inner = left_inner + offset
        right_inner = right_inner + offset

    n = track.num_points
    closed = track.closed

    if surface_only:
        vertices = np.concatenate([left_inner, right_inner], axis=0).astype(np.float32)
        LI_G = 0
        RI_G = n

        faces: list[list[int]] = []
        last = n if closed else n - 1
        for i in range(last):
            j = (i + 1) % n
            # Winding chosen so the triangle normal points UP (+z locally).
            # PhysX static triangle mesh collision only fires from the
            # normal side — if we reverse this the vehicle tunnels through
            # the surface because it approaches from the "back" side.
            faces.append([LI_G + i, RI_G + i, LI_G + j])
            faces.append([RI_G + i, RI_G + j, LI_G + j])
    else:
        up = np.array([0.0, 0.0, wall_height], dtype=np.float64)
        left_top = left_inner + up
        right_top = right_inner + up

        vertices = np.concatenate(
            [left_inner, right_inner, left_top, right_top], axis=0
        ).astype(np.float32)

        LI_G = 0
        RI_G = n
        LI_T = 2 * n
        RI_T = 3 * n

        faces = []
        last = n if closed else n - 1
        for i in range(last):
            j = (i + 1) % n

            # Ground quad — normal up (see surface_only branch above).
            faces.append([LI_G + i, RI_G + i, LI_G + j])
            faces.append([RI_G + i, RI_G + j, LI_G + j])
            # Left inner wall
            faces.append([LI_G + i, LI_G + j, LI_T + j])
            faces.append([LI_G + i, LI_T + j, LI_T + i])
            # Right inner wall (reversed winding)
            faces.append([RI_G + j, RI_G + i, RI_T + i])
            faces.append([RI_G + j, RI_T + i, RI_T + j])

    return {
        "vertices": vertices,
        "faces": np.asarray(faces, dtype=np.int32),
        "n_points": n,
        "length": track.length(),
    }


def save_obj(mesh: dict[str, Any], path: str) -> None:
    """Write mesh dict to a Wavefront .obj file (useful for offline inspection)."""
    verts = mesh["vertices"]
    faces = mesh["faces"]
    with open(path, "w") as f:
        f.write(f"# circuit mesh: {len(verts)} verts, {len(faces)} faces\n")
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")


def spawn_circuit(
    track: RacingTrack,
    prim_path: str,
    *,
    wall_height: float = 0.5,
    ground_z_offset: float = 0.0,
    surface_only: bool = False,
    color: tuple[float, float, float] = (0.4, 0.4, 0.4),
) -> dict[str, Any]:
    """Spawn the circuit mesh as a USD prim and attach collision API.

    Must be called after `AppLauncher`. Pass `surface_only=True` to emit just
    the ground strip — use this alongside `spawn_duct_track` for a 3D road
    under duct pipes.
    """
    # Lazy import — pxr only available after Kit boot.
    import omni.usd
    from pxr import Gf, UsdGeom, UsdPhysics, Vt

    mesh = build_circuit_mesh(
        track,
        wall_height=wall_height,
        ground_z_offset=ground_z_offset,
        surface_only=surface_only,
    )
    verts = mesh["vertices"]
    faces = mesh["faces"]

    stage = omni.usd.get_context().get_stage()
    UsdGeom.Xform.Define(stage, prim_path)
    usd_mesh = UsdGeom.Mesh.Define(stage, f"{prim_path}/mesh")

    usd_mesh.GetPointsAttr().Set(
        Vt.Vec3fArray([Gf.Vec3f(float(v[0]), float(v[1]), float(v[2])) for v in verts])
    )
    usd_mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray([3] * len(faces)))
    usd_mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray(faces.flatten().tolist()))
    usd_mesh.GetDoubleSidedAttr().Set(True)
    usd_mesh.GetDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*color)]))

    UsdPhysics.CollisionAPI.Apply(usd_mesh.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(usd_mesh.GetPrim()).GetApproximationAttr().Set(
        "none"
    )

    return mesh
