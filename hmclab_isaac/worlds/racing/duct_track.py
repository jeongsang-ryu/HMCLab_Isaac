"""Dual-pipe duct track mesh generator.

Two cylindrical pipes run parallel along both sides of the centerline,
sitting on the ground. Each pipe has optional ring-shaped ribs at regular
intervals along the path — gives the track a visible "duct" look that's
easy to see for LiDAR and useful for reward shaping.

```
  [LEFT PIPE with ribs]   centerline   [RIGHT PIPE with ribs]
```

Ported from `f1tenth_rl/envs/duct_track_builder.py`. Reworked to consume a
`RacingTrack` (the 8-col schema in `_schema.py`) instead of a raw f1tenth
centerline CSV, so any track in `_tracks_data/` works.

Entry points:

    build_duct_mesh(track, pipe_radius=0.2, pipe_offset=1.0, ...)
        Pure-numpy. Returns dict with separate duct+rib vertices/faces so
        the spawner can paint them different colors.

    spawn_duct_track(track, prim_path, ..., duct_color=..., rib_color=...)
        Kit-level. Creates two USD Mesh prims (one for pipes, one for ribs),
        applies physics + UsdPreviewSurface materials.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ._schema import RacingTrack, _rpy_to_matrix

__all__ = ["build_duct_mesh", "spawn_duct_track"]


# ----------------------------------------------------------------------
# Frame derivation
# ----------------------------------------------------------------------
def _track_frames(track: RacingTrack) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (positions_3d, left_dir, up_dir, tangent) per point.

    If the track has nonzero rpy, use the local rotation matrix. Otherwise
    fall back to Frenet-style frames derived from the path tangent.
    """
    positions = track.positions.astype(np.float32)
    n = len(positions)

    rpy_nonzero = np.any(np.abs(track.rpy) > 1e-6)
    if rpy_nonzero:
        left = np.zeros((n, 3), dtype=np.float32)
        up = np.zeros((n, 3), dtype=np.float32)
        tangent = np.zeros((n, 3), dtype=np.float32)
        for i in range(n):
            R = _rpy_to_matrix(track.rpy[i])
            tangent[i] = R @ np.array([1.0, 0.0, 0.0])
            left[i] = R @ np.array([0.0, 1.0, 0.0])
            up[i] = R @ np.array([0.0, 0.0, 1.0])
    else:
        # 2D Frenet fallback — tangent from centered finite diff, left from
        # cross product with world up.
        tangent = np.zeros((n, 3), dtype=np.float32)
        for i in range(n):
            nxt = (i + 1) % n if track.closed else min(i + 1, n - 1)
            prv = (i - 1) % n if track.closed else max(i - 1, 0)
            d = positions[nxt] - positions[prv]
            d[2] = 0.0
            norm = np.linalg.norm(d)
            tangent[i] = d / max(norm, 1e-8)
        up = np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float32), (n, 1))
        left = np.cross(up, tangent)
        left /= np.maximum(np.linalg.norm(left, axis=1, keepdims=True), 1e-8)

    return positions, left, up, tangent


# ----------------------------------------------------------------------
# Pipe + rib builders (pure numpy)
# ----------------------------------------------------------------------
def _build_pipe_mesh(
    centers_3d: np.ndarray,
    left_dir: np.ndarray,
    up_dir: np.ndarray,
    pipe_radius: float,
    pipe_resolution: int,
    closed: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a cylindrical pipe mesh along the given centers.

    At each path point, generate `pipe_resolution` vertices around the
    circumference using left_dir / up_dir as the local cross-section basis.
    """
    n_pts = len(centers_3d)
    angles = np.linspace(0.0, 2.0 * np.pi, pipe_resolution, endpoint=False)

    vertices = np.zeros((n_pts * pipe_resolution, 3), dtype=np.float32)
    for i in range(n_pts):
        base = i * pipe_resolution
        for j, angle in enumerate(angles):
            offset = (
                left_dir[i] * (pipe_radius * np.cos(angle))
                + up_dir[i] * (pipe_radius * np.sin(angle))
            )
            vertices[base + j] = centers_3d[i] + offset

    faces = []
    last = n_pts if closed else n_pts - 1
    for i in range(last):
        ni = (i + 1) % n_pts
        for j in range(pipe_resolution):
            nj = (j + 1) % pipe_resolution
            v0 = i * pipe_resolution + j
            v1 = i * pipe_resolution + nj
            v2 = ni * pipe_resolution + j
            v3 = ni * pipe_resolution + nj
            faces.append([v0, v1, v3])
            faces.append([v0, v3, v2])
    return vertices, np.asarray(faces, dtype=np.int32)


def _build_ring_rib(
    pipe_center: np.ndarray,
    left_dir: np.ndarray,
    up_dir: np.ndarray,
    tangent: np.ndarray,
    pipe_radius: float,
    rib_thickness: float,
    rib_height: float,
    rib_resolution: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a torus-like rib ring wrapping around one point on a pipe."""
    inner_r = pipe_radius
    outer_r = pipe_radius + rib_height
    half_t = tangent * (rib_thickness / 2.0)
    angles = np.linspace(0.0, 2.0 * np.pi, rib_resolution, endpoint=False)

    verts = []
    for t_offset in (-half_t, half_t):          # back, front
        for r in (inner_r, outer_r):            # inner, outer
            for angle in angles:
                pt = (
                    pipe_center
                    + t_offset
                    + left_dir * (r * np.cos(angle))
                    + up_dir * (r * np.sin(angle))
                )
                verts.append(pt)
    verts = np.asarray(verts, dtype=np.float32)

    n_seg = rib_resolution
    bi = 0
    bo = n_seg
    fi = 2 * n_seg
    fo = 3 * n_seg

    faces = []
    for k in range(n_seg):
        nk = (k + 1) % n_seg
        # outer wall
        faces.append([fo + k, fo + nk, bo + nk])
        faces.append([fo + k, bo + nk, bo + k])
        # inner wall
        faces.append([bi + k, bi + nk, fi + nk])
        faces.append([bi + k, fi + nk, fi + k])
        # front cap
        faces.append([fi + k, fi + nk, fo + nk])
        faces.append([fi + k, fo + nk, fo + k])
        # back cap
        faces.append([bo + k, bo + nk, bi + nk])
        faces.append([bo + k, bi + nk, bi + k])

    return verts, np.asarray(faces, dtype=np.int32)


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------
def build_duct_mesh(
    track: RacingTrack,
    *,
    pipe_radius: float = 0.2,
    pipe_offset: float = 1.0,
    pipe_resolution: int = 16,
    rib_spacing: float = 0.5,
    rib_thickness: float = 0.02,
    rib_height: float = 0.03,
    rib_resolution: int = 12,
) -> dict[str, Any]:
    """Build dual-pipe duct track mesh from a `RacingTrack`.

    Args:
        track: RacingTrack (positions, rpy, widths).
        pipe_radius: Radius of each pipe in meters.
        pipe_offset: Distance from centerline to each pipe center.
        pipe_resolution: Vertices around each pipe circle.
        rib_spacing: Distance between ribs along the path (0 = no ribs).
        rib_thickness: Thickness of each rib along the path direction.
        rib_height: How far ribs protrude outward from the pipe surface.
        rib_resolution: Segments around each rib ring.

    Returns dict with `duct_vertices/faces`, `rib_vertices/faces`, combined
    `vertices/faces`, and metadata.
    """
    positions, left_dir, up_dir, tangent = _track_frames(track)
    n_pts = len(positions)

    # Pipe centers: lateral offset along the local left axis, lifted along
    # the local up axis so the pipe "sits" on the banked surface at each
    # point. For flat 2D tracks up_dir is world-z, so this degrades to the
    # original f1tenth behaviour of "pipe bottom touches z=0".
    left_centers = positions + left_dir * pipe_offset + up_dir * pipe_radius
    right_centers = positions - left_dir * pipe_offset + up_dir * pipe_radius

    left_verts, left_faces = _build_pipe_mesh(
        left_centers, left_dir, up_dir, pipe_radius, pipe_resolution, track.closed
    )
    right_verts, right_faces = _build_pipe_mesh(
        right_centers, left_dir, up_dir, pipe_radius, pipe_resolution, track.closed
    )
    right_faces = right_faces + len(left_verts)

    duct_verts = np.vstack([left_verts, right_verts])
    duct_faces = np.vstack([left_faces, right_faces])

    # Ribs along the path at `rib_spacing` m intervals.
    rib_verts_list: list[np.ndarray] = []
    rib_faces_list: list[np.ndarray] = []
    n_ribs = 0

    if rib_spacing > 0:
        diffs = np.diff(positions[:, :2], axis=0)
        seg_len = np.linalg.norm(diffs, axis=1)
        cum = np.zeros(n_pts, dtype=np.float32)
        cum[1:] = np.cumsum(seg_len)
        total = float(cum[-1])

        v_offset = len(duct_verts)
        for dist in np.arange(0.0, total, rib_spacing):
            idx = int(np.searchsorted(cum, dist) - 1)
            idx = max(0, min(idx, n_pts - 2))
            t_local = (dist - cum[idx]) / max(seg_len[idx], 1e-8)
            t_local = min(max(t_local, 0.0), 1.0)
            nxt = min(idx + 1, n_pts - 1)

            c = positions[idx] * (1 - t_local) + positions[nxt] * t_local
            ld = left_dir[idx] * (1 - t_local) + left_dir[nxt] * t_local
            ld /= max(np.linalg.norm(ld), 1e-8)
            ud = up_dir[idx] * (1 - t_local) + up_dir[nxt] * t_local
            ud /= max(np.linalg.norm(ud), 1e-8)
            td = tangent[idx] * (1 - t_local) + tangent[nxt] * t_local
            td /= max(np.linalg.norm(td), 1e-8)

            left_c = c + ld * pipe_offset + ud * pipe_radius
            right_c = c - ld * pipe_offset + ud * pipe_radius

            for pipe_c in (left_c, right_c):
                rv, rf = _build_ring_rib(
                    pipe_c, ld, ud, td,
                    pipe_radius, rib_thickness, rib_height, rib_resolution,
                )
                rf = rf + v_offset
                v_offset += len(rv)
                rib_verts_list.append(rv)
                rib_faces_list.append(rf)

            n_ribs += 1

    if rib_verts_list:
        rib_verts = np.vstack(rib_verts_list)
        rib_faces = np.vstack(rib_faces_list)
    else:
        rib_verts = np.zeros((0, 3), dtype=np.float32)
        rib_faces = np.zeros((0, 3), dtype=np.int32)

    if len(rib_verts) > 0:
        all_verts = np.vstack([duct_verts, rib_verts])
        all_faces = np.vstack([duct_faces, rib_faces])
    else:
        all_verts = duct_verts
        all_faces = duct_faces

    return {
        "vertices": all_verts,
        "faces": all_faces,
        "duct_vertices": duct_verts,
        "duct_faces": duct_faces,
        "rib_vertices": rib_verts,
        "rib_faces": rib_faces,
        "n_points": n_pts,
        "track_length": track.length(),
        "pipe_radius": pipe_radius,
        "pipe_offset": pipe_offset,
        "n_ribs": n_ribs,
    }


def spawn_duct_track(
    track: RacingTrack,
    prim_path: str,
    *,
    pipe_radius: float = 0.2,
    pipe_offset: float = 1.0,
    pipe_resolution: int = 16,
    rib_spacing: float = 0.5,
    rib_thickness: float = 0.02,
    rib_height: float = 0.03,
    rib_resolution: int = 12,
    duct_color: tuple[float, float, float] = (1.0, 0.5, 0.0),
    rib_color: tuple[float, float, float] = (0.05, 0.05, 0.05),
) -> dict[str, Any]:
    """Spawn the duct mesh as two USD prims (`<prim>/duct` + `<prim>/ribs`).

    Duct pipes and rib rings are created as separate prims so they can wear
    distinct UsdPreviewSurface colors — duct is typically orange, ribs are
    dark to add contrast for both vision and LiDAR.
    """
    import omni.usd
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade, Vt

    mesh_data = build_duct_mesh(
        track,
        pipe_radius=pipe_radius,
        pipe_offset=pipe_offset,
        pipe_resolution=pipe_resolution,
        rib_spacing=rib_spacing,
        rib_thickness=rib_thickness,
        rib_height=rib_height,
        rib_resolution=rib_resolution,
    )

    stage = omni.usd.get_context().get_stage()
    UsdGeom.Xform.Define(stage, prim_path)

    def _create_mesh(path: str, verts: np.ndarray, faces_arr: np.ndarray, color: tuple[float, float, float]):
        mesh = UsdGeom.Mesh.Define(stage, path)
        mesh.GetPointsAttr().Set(
            Vt.Vec3fArray([Gf.Vec3f(float(v[0]), float(v[1]), float(v[2])) for v in verts])
        )
        mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray([3] * len(faces_arr)))
        mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray(faces_arr.flatten().tolist()))
        mesh.GetDoubleSidedAttr().Set(True)

        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).GetApproximationAttr().Set("none")

        mat_path = path + "_Mat"
        mat = UsdShade.Material.Define(stage, mat_path)
        sh = UsdShade.Shader.Define(stage, mat_path + "/Shader")
        sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))
        )
        sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.6)
        mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)

    _create_mesh(
        f"{prim_path}/duct", mesh_data["duct_vertices"], mesh_data["duct_faces"], duct_color
    )

    if len(mesh_data["rib_vertices"]) > 0:
        # Rib faces were indexed into the combined vertex array during
        # construction; rewind to local indices for the separate mesh prim.
        rib_faces_local = mesh_data["rib_faces"] - len(mesh_data["duct_vertices"])
        _create_mesh(
            f"{prim_path}/ribs", mesh_data["rib_vertices"], rib_faces_local, rib_color
        )

    return mesh_data
