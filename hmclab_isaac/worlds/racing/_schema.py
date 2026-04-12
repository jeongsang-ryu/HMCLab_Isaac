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
        and computes yaw from the path tangent for vehicle spawning."""
        n = len(xy)
        positions = np.zeros((n, 3), dtype=np.float64)
        positions[:, 0:2] = xy
        rpy = np.zeros((n, 3), dtype=np.float64)
        tangent = np.zeros_like(xy)
        for i in range(n):
            nxt = (i + 1) % n if closed else min(i + 1, n - 1)
            prv = (i - 1) % n if closed else max(i - 1, 0)
            tangent[i] = xy[nxt] - xy[prv]
        rpy[:, 2] = np.arctan2(tangent[:, 1], tangent[:, 0])
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
