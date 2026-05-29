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
    derive a smooth path frame from the centerline using **forward
    differences** (not centered) and propagating left/up with a parallel
    transport frame so the cross-section never collapses in tight turns.

    The previous centered-difference Frenet fallback failed at near-cusps
    where ``positions[i+1] ≈ positions[i-1]``: the tangent vanished, got
    clamped to a near-arbitrary unit vector, and the resulting cross-
    sections produced twisted/ring-shaped artifacts in the duct mesh.
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
        return positions, left, up, tangent

    # ---- 2D path frame via forward-diff tangent + PT-frame propagation ----
    tangent = np.zeros((n, 3), dtype=np.float32)
    for i in range(n):
        nxt = (i + 1) % n if track.closed else min(i + 1, n - 1)
        d = positions[nxt] - positions[i]
        d[2] = 0.0
        nrm = np.linalg.norm(d)
        if nrm < 1e-8:
            # zero-length segment (degenerate); reuse previous tangent
            tangent[i] = tangent[i - 1] if i > 0 else np.array([1.0, 0.0, 0.0],
                                                                dtype=np.float32)
        else:
            tangent[i] = d / nrm

    up = np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float32), (n, 1))
    left = np.zeros((n, 3), dtype=np.float32)

    # Initial left from world-up × tangent (planar). If that degenerates
    # (tangent is parallel to up, i.e., vertical path), fall back to +Y.
    init_left = np.cross(up[0], tangent[0])
    if np.linalg.norm(init_left) < 1e-6:
        init_left = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    left[0] = init_left / np.linalg.norm(init_left)

    # Parallel transport: at each step rotate the previous left vector
    # by the rotation that takes tangent[i-1] to tangent[i]. This keeps
    # left smoothly varying without zero crossings.
    for i in range(1, n):
        t_prev = tangent[i - 1]
        t_curr = tangent[i]
        axis = np.cross(t_prev, t_curr)
        s = np.linalg.norm(axis)
        c = float(np.dot(t_prev, t_curr))
        if s < 1e-8:
            # Tangents nearly parallel — no rotation needed
            left[i] = left[i - 1]
        else:
            axis = axis / s
            angle = np.arctan2(s, c)
            # Rodrigues' rotation formula on left[i-1]
            v = left[i - 1]
            left[i] = (
                v * np.cos(angle)
                + np.cross(axis, v) * np.sin(angle)
                + axis * np.dot(axis, v) * (1.0 - np.cos(angle))
            )
        # Re-orthogonalize against the new tangent (numerical safety)
        left[i] = left[i] - np.dot(left[i], t_curr) * t_curr
        nrm = np.linalg.norm(left[i])
        if nrm < 1e-6:
            # Catastrophic loss — fall back to up × tangent
            left[i] = np.cross(up[i], t_curr)
            nrm = np.linalg.norm(left[i])
            if nrm < 1e-6:
                left[i] = np.array([0.0, 1.0, 0.0], dtype=np.float32)
                nrm = 1.0
        left[i] = left[i] / nrm

    # Re-derive up to be exactly perpendicular to (left, tangent) at each
    # point — slight drift from parallel transport could make up not
    # exactly world-z, which is fine and even desirable for banked tracks.
    up = np.cross(tangent, left)
    nrm = np.linalg.norm(up, axis=1, keepdims=True)
    up = up / np.maximum(nrm, 1e-8)

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
    make_rigid: bool = False,
) -> dict[str, Any]:
    """Spawn the duct mesh as two USD prims (`<prim>/duct` + `<prim>/ribs`).

    Duct pipes and rib rings are created as separate prims so they can wear
    distinct UsdPreviewSurface colors — duct is typically orange, ribs are
    dark to add contrast for both vision and LiDAR.
    """
    import omni.usd
    from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics, UsdShade, Vt

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
        # Hard-faceted rendering: explicitly disable subdivision so Hydra
        # renders the triangles as-authored. Without this the default
        # subdivisionScheme on UsdGeom.Mesh causes some renderers to apply
        # Catmull-Clark smoothing, which rounds the rib's sharp corners
        # into bulges that look like "twists" along the duct.
        mesh.CreateSubdivisionSchemeAttr().Set("none")
        # Author flat (face-varying) normals so the renderer doesn't
        # average across rib seams and produce dark/inverted shading.
        # Each triangle gets a normal computed from its own vertices.
        v_arr = np.asarray(verts, dtype=np.float32)
        f_arr = np.asarray(faces_arr, dtype=np.int64)
        e1 = v_arr[f_arr[:, 1]] - v_arr[f_arr[:, 0]]
        e2 = v_arr[f_arr[:, 2]] - v_arr[f_arr[:, 0]]
        face_normals = np.cross(e1, e2)
        n_len = np.linalg.norm(face_normals, axis=1, keepdims=True)
        face_normals = face_normals / np.maximum(n_len, 1e-8)
        # faceVarying: 1 normal per vertex per face → replicate per-face normal
        fv_normals = np.repeat(face_normals, 3, axis=0)
        mesh.CreateNormalsAttr().Set(
            Vt.Vec3fArray([Gf.Vec3f(float(n[0]), float(n[1]), float(n[2]))
                           for n in fv_normals])
        )
        mesh.SetNormalsInterpolation("faceVarying")
        mesh.GetDoubleSidedAttr().Set(False)   # with proper normals, one-sided is fine

        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).GetApproximationAttr().Set("none")

        if make_rigid:
            # Promote the static collider to a KINEMATIC rigid body so it
            # becomes a valid ContactSensor.force_matrix filter target
            # (force_matrix only resolves rigid bodies). Kinematic = stays
            # bolted in place like a static wall but is solver-visible, and
            # kinematic bodies are allowed to keep triangle-mesh ("none")
            # colliders. Contact-report API lets the sensor receive the
            # car-vs-wall force directly — no net−Σ subtraction needed.
            rb = UsdPhysics.RigidBodyAPI.Apply(mesh.GetPrim())
            rb.CreateRigidBodyEnabledAttr().Set(True)
            rb.CreateKinematicEnabledAttr().Set(True)
            PhysxSchema.PhysxRigidBodyAPI.Apply(mesh.GetPrim())
            cr = PhysxSchema.PhysxContactReportAPI.Apply(mesh.GetPrim())
            cr.CreateThresholdAttr().Set(0.0)

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

    rigid_mesh_names = ["duct"]
    if len(mesh_data["rib_vertices"]) > 0:
        # Rib faces were indexed into the combined vertex array during
        # construction; rewind to local indices for the separate mesh prim.
        rib_faces_local = mesh_data["rib_faces"] - len(mesh_data["duct_vertices"])
        _create_mesh(
            f"{prim_path}/ribs", mesh_data["rib_vertices"], rib_faces_local, rib_color
        )
        rigid_mesh_names.append("ribs")

    # Leaf names of the mesh prims (relative to ``prim_path``) that were
    # promoted to kinematic rigid bodies — the env uses these to build the
    # ContactSensor wall filter.
    mesh_data["rigid_mesh_names"] = rigid_mesh_names if make_rigid else []
    return mesh_data


def spawn_static_boxes(
    prim_root: str,
    frenet,
    count: int,
    *,
    size: float = 0.5,
    seed: int = 0,
    lat_frac: float = 0.5,
    random_color: bool = True,
    side: str = "both",
) -> list[str]:
    """Spawn ``count`` static cube obstacles at random on-track positions.

    Each box is a KINEMATIC rigid body (valid ContactSensor.force_matrix
    filter target, like the duct walls) with a bright visual so the
    vision policy can see and learn to avoid it. Positions are sampled at
    random arc-lengths with a random lateral offset inside the track.

    Returns the leaf names (relative to ``prim_root``) of the spawned
    boxes so the env can add them to the collision filter.
    """
    if count <= 0:
        return []
    import omni.usd
    from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics, UsdShade

    stage = omni.usd.get_context().get_stage()
    UsdGeom.Xform.Define(stage, prim_root)

    rng = np.random.default_rng(int(seed))
    n_pts = int(frenet.num_points)
    cl_xy = frenet.cl_xy.detach().cpu().numpy()        # (N, 2)
    cl_nrm = frenet.cl_normal.detach().cpu().numpy()   # (N, 2) +left
    cl_w = frenet.cl_widths.detach().cpu().numpy()     # (N, 2)
    # random centerline indices + lateral offsets within the track
    idx = rng.integers(0, n_pts, size=count)
    cl = cl_xy[idx]                                    # (count, 2)
    nrm = cl_nrm[idx]                                  # (count, 2)
    half = np.minimum(cl_w[idx, 0], cl_w[idx, 1])      # (count,)
    # Side-biased lateral offset. normal is +left, so the car's RIGHT is
    # the negative direction. "right" keeps boxes in the policy's actual
    # (right-hugging) line; magnitude 0.15..0.8 of the half-width.
    r = rng.uniform(0.0, 1.0, size=count)
    if side == "right":
        lat = -(0.15 + 0.65 * r) * half
    elif side == "left":
        lat = (0.15 + 0.65 * r) * half
    else:
        lat = (2.0 * r - 1.0) * lat_frac * half
    bx = cl[:, 0] + lat * nrm[:, 0]
    by = cl[:, 1] + lat * nrm[:, 1]
    # Per-box vivid random colour (saturated so it pops on camera). The
    # learner generalises over an "obstacle = solid coloured box" concept
    # rather than a single hue. Deterministic via the same seed.
    if random_color:
        base = rng.uniform(0.55, 1.0, size=(count, 3))
        drop = rng.integers(0, 3, size=count)          # zero one channel
        for j in range(count):
            base[j, drop[j]] *= 0.12
        box_colors = base
    else:
        box_colors = np.tile(np.array([0.9, 0.05, 0.05]), (count, 1))

    names: list[str] = []
    for k in range(count):
        nm = f"box_{k}"
        path = f"{prim_root}/{nm}"
        cube = UsdGeom.Cube.Define(stage, path)
        cube.GetSizeAttr().Set(float(size))
        prim = cube.GetPrim()
        UsdGeom.Xformable(prim).AddTranslateOp().Set(
            Gf.Vec3d(float(bx[k]), float(by[k]), float(size) * 0.5)
        )
        UsdPhysics.CollisionAPI.Apply(prim)
        rb = UsdPhysics.RigidBodyAPI.Apply(prim)
        rb.CreateRigidBodyEnabledAttr().Set(True)
        rb.CreateKinematicEnabledAttr().Set(True)
        PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        cr = PhysxSchema.PhysxContactReportAPI.Apply(prim)
        cr.CreateThresholdAttr().Set(0.0)
        # bright visual so the camera (and learner) can see the obstacle
        mat_path = path + "_Mat"
        mat = UsdShade.Material.Define(stage, mat_path)
        sh = UsdShade.Shader.Define(stage, mat_path + "/Shader")
        sh.CreateIdAttr("UsdPreviewSurface")
        c = box_colors[k]
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(float(c[0]), float(c[1]), float(c[2]))
        )
        sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.7)
        mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat)
        names.append(nm)
    return names
