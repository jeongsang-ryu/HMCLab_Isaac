"""RacingTrack schema — shared contract for all racing worlds.

File format (whitespace- or comma-separated, '#' comments allowed):

    x  y  z  roll  pitch  yaw  d_left  d_right

Semantics at each row:
    (x, y, z)           centerline position in the map frame
    (roll, pitch, yaw)  local frame orientation at that point
                        (x-axis = heading, y+ = left, z+ = up — ROS REP-103)
    d_left              track half-width to the left  (+y in local frame)
    d_right             track half-width to the right (-y in local frame)

2D tracks use z = roll = pitch = 0. 3D tracks (banking, elevation, ramps)
populate all fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class RacingTrack:
    """Parsed centerline representation used by racing worlds and envs."""

    name: str
    positions: np.ndarray  # (N, 3)
    rpy: np.ndarray        # (N, 3)  roll, pitch, yaw
    widths: np.ndarray     # (N, 2)  d_left, d_right
    closed: bool = True
    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path, *, closed: bool = True) -> "RacingTrack":
        """Load a track from an 8-column text file."""
        path = Path(path)
        data = np.loadtxt(path, delimiter=None, comments="#")
        if data.ndim != 2 or data.shape[1] != 8:
            raise ValueError(
                f"{path} must have 8 columns (x y z r p y d_left d_right); got {data.shape}"
            )
        return cls(
            name=path.stem,
            positions=data[:, 0:3].astype(np.float64),
            rpy=data[:, 3:6].astype(np.float64),
            widths=data[:, 6:8].astype(np.float64),
            closed=closed,
        )

    def save(self, path: str | Path) -> None:
        """Write the track to an 8-column text file (schema-conformant)."""
        path = Path(path)
        data = np.concatenate([self.positions, self.rpy, self.widths], axis=1)
        header = "x y z roll pitch yaw d_left d_right"
        np.savetxt(path, data, header=header, fmt="%.6f")

    @classmethod
    def from_2d_centerline(
        cls,
        name: str,
        xy: np.ndarray,
        d_left: np.ndarray,
        d_right: np.ndarray,
        *,
        closed: bool = True,
    ) -> "RacingTrack":
        """Build from legacy 2D data (x, y, w_left, w_right). Fills z=rpy=0
        and computes yaw from the path tangent for vehicle spawning.

        Uses **forward differences** for tangent (xy[i+1]-xy[i]) rather than
        centered differences. Centered differences can vanish at sharp
        cusps (when xy[i+1] ≈ xy[i-1]) and yield arbitrary yaws there;
        forward diff is robust as long as adjacent samples are distinct.
        """
        n = len(xy)
        positions = np.zeros((n, 3), dtype=np.float64)
        positions[:, 0:2] = xy
        rpy = np.zeros((n, 3), dtype=np.float64)
        rpy[:, 2] = _forward_diff_yaw(xy, closed=closed)
        widths = np.stack([d_left, d_right], axis=1)
        return cls(name=name, positions=positions, rpy=rpy, widths=widths, closed=closed)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    @property
    def num_points(self) -> int:
        return self.positions.shape[0]

    def length(self) -> float:
        """Total centerline length in meters."""
        diffs = np.diff(self.positions, axis=0)
        if self.closed:
            diffs = np.vstack([diffs, self.positions[0] - self.positions[-1]])
        return float(np.linalg.norm(diffs, axis=1).sum())

    def spawn_pose(self, progress: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        """Return (position, rpy) for a vehicle at fractional progress [0, 1).

        Vehicles spawned with this pose face the centerline heading automatically
        (yaw from tangent), so the env can feed this directly into
        `ArticulationCfg.InitialStateCfg`.
        """
        if not 0.0 <= progress < 1.0:
            progress = progress % 1.0
        idx = int(progress * self.num_points) % self.num_points
        return self.positions[idx].copy(), self.rpy[idx].copy()

    def densify(self, sub: int) -> "RacingTrack":
        """Return a new RacingTrack with ``sub`` linearly-interpolated
        intermediate points inserted between each pair of adjacent samples.

        Positions and widths are linearly interpolated; yaw is
        **recomputed** from forward-diff of the densified positions to
        avoid the wrap-around bug (linear interp of yaws near ±π flips
        them by 180°, which causes downstream mesh builders to place the
        left/right boundary on the wrong side → visible ring artifacts).
        """
        if sub <= 1:
            return self
        n = self.num_points
        last = n if self.closed else n - 1
        new_pos = []
        new_w = []
        for i in range(last):
            j = (i + 1) % n
            for k in range(sub):
                t = k / sub
                new_pos.append(self.positions[i] * (1 - t) + self.positions[j] * t)
                new_w.append(self.widths[i] * (1 - t) + self.widths[j] * t)
        if not self.closed:
            new_pos.append(self.positions[-1])
            new_w.append(self.widths[-1])
        new_pos = np.array(new_pos)
        new_w = np.array(new_w)
        # Re-derive yaw from forward-diff of new positions
        new_rpy = np.zeros((len(new_pos), 3), dtype=np.float64)
        new_rpy[:, 2] = _forward_diff_yaw(new_pos[:, :2], closed=self.closed)
        return RacingTrack(
            name=self.name,
            positions=new_pos, rpy=new_rpy, widths=new_w, closed=self.closed,
        )

    def tangents(self) -> np.ndarray:
        """Per-point unit tangent vectors in world XY plane, shape (N, 2).

        Computed from forward-difference of positions; for the last point on
        an open track, falls back to the previous segment direction.
        """
        xy = self.positions[:, :2]
        n = len(xy)
        t = np.zeros((n, 2), dtype=np.float64)
        for i in range(n):
            nxt = (i + 1) % n if self.closed else min(i + 1, n - 1)
            dx = xy[nxt, 0] - xy[i, 0]
            dy = xy[nxt, 1] - xy[i, 1]
            mag = float(np.hypot(dx, dy))
            if mag < 1e-12:
                t[i] = t[i - 1] if i > 0 else np.array([1.0, 0.0])
            else:
                t[i, 0] = dx / mag
                t[i, 1] = dy / mag
        return t

    def arclen(self) -> np.ndarray:
        """Cumulative arc-length from start, shape (N,). Always starts at 0."""
        xy = self.positions[:, :2]
        diffs = np.diff(xy, axis=0)
        seg = np.linalg.norm(diffs, axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        return s

    def curvatures(self) -> np.ndarray:
        """Per-point discrete curvature (1/radius), shape (N,).

        Uses the Menger curvature of three consecutive points
        (i-1, i, i+1): kappa = 4 * area / (|a| * |b| * |c|). For a closed
        loop wraps the indices; for an open track endpoints get 0.
        """
        xy = self.positions[:, :2]
        n = len(xy)
        kappa = np.zeros(n, dtype=np.float64)
        for i in range(n):
            if self.closed:
                im = (i - 1) % n
                ip = (i + 1) % n
            else:
                if i == 0 or i == n - 1:
                    continue
                im, ip = i - 1, i + 1
            p0 = xy[im]
            p1 = xy[i]
            p2 = xy[ip]
            a = float(np.linalg.norm(p1 - p0))
            b = float(np.linalg.norm(p2 - p1))
            c = float(np.linalg.norm(p2 - p0))
            denom = a * b * c
            if denom < 1e-12:
                continue
            # 2x triangle area via cross product z-component
            cross = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0])
            kappa[i] = 2.0 * cross / denom   # signed curvature (+ = left turn)
        return kappa

    def boundary_points(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute left/right boundary point arrays in world frame.

        Returns:
            (left (N, 3), right (N, 3)) — suitable for trimesh wall extrusion.
        """
        left = np.zeros_like(self.positions)
        right = np.zeros_like(self.positions)
        for i in range(self.num_points):
            R = _rpy_to_matrix(self.rpy[i])
            left_dir = R @ np.array([0.0, 1.0, 0.0])   # local +y
            right_dir = R @ np.array([0.0, -1.0, 0.0]) # local -y
            left[i] = self.positions[i] + left_dir * self.widths[i, 0]
            right[i] = self.positions[i] + right_dir * self.widths[i, 1]
        return left, right


# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------
def _forward_diff_yaw(xy: np.ndarray, *, closed: bool) -> np.ndarray:
    """Compute per-point yaw from forward-difference tangents of 2D xy data.

    Returned yaws are in [-pi, pi] (np.arctan2 convention). If a forward
    segment has zero length, reuse the previous yaw to avoid undefined
    arctan2(0, 0) artifacts.
    """
    n = len(xy)
    yaws = np.zeros(n, dtype=np.float64)
    prev = 0.0
    for i in range(n):
        nxt = (i + 1) % n if closed else min(i + 1, n - 1)
        dx = float(xy[nxt, 0] - xy[i, 0])
        dy = float(xy[nxt, 1] - xy[i, 1])
        if dx == 0.0 and dy == 0.0:
            yaws[i] = prev
        else:
            yaws[i] = float(np.arctan2(dy, dx))
            prev = yaws[i]
    return yaws


def _rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
    """Roll-pitch-yaw (XYZ intrinsic, ROS convention) to 3x3 rotation matrix."""
    r, p, y = rpy
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx
