"""Generate a 3D test oval racing track.

Produces an elliptical centerline with:
  - straight sections that climb and descend (sinusoidal z)
  - banked corners (non-zero roll)
  - pitch derived analytically from dz/ds so the vehicle stays normal to
    the surface on the slopes

Output conforms to the 8-column RacingTrack schema.

Usage:
    python scripts/generate_test_oval.py \\
        --dst hmclab_isaac/worlds/racing/_tracks_data/test_oval.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def generate_oval(
    *,
    a: float = 30.0,          # semi-major axis (x direction)
    b: float = 15.0,          # semi-minor axis (y direction)
    n_points: int = 400,
    amp_z: float = 3.0,       # vertical peak-to-trough / 2 in meters
    amp_roll: float = 0.15,   # banking amplitude in radians (~8.6°)
    half_width: float = 2.0,  # track half-width (d_left = d_right)
) -> np.ndarray:
    """Parametric ellipse with hills on straights and banking on corners.

    Shape of an oval (a > b), driven CCW starting at t=0 (= right side,
    heading +y). Straights run north-south (t≈0, t≈π); corners are on
    top/bottom (t≈π/2, t≈3π/2).

    Height profile: `z(t) = amp_z * sin(t)` — peak at top corner, trough
    at bottom corner, so the climbs happen on the right straight and the
    descents on the left straight.

    Banking: `roll(t) = amp_roll * max(0, -cos(2t))` — 0 on straights,
    full-value on corners, smoothly eased in/out.
    """
    t = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)

    x = a * np.cos(t)
    y = b * np.sin(t)
    z = amp_z * np.sin(t)

    # Derivatives for tangent + pitch
    dx_dt = -a * np.sin(t)
    dy_dt = b * np.cos(t)
    dz_dt = amp_z * np.cos(t)
    ds_dt = np.sqrt(dx_dt**2 + dy_dt**2)           # arc length speed (horizontal)
    ds3_dt = np.sqrt(ds_dt**2 + dz_dt**2)          # full 3D arc length speed

    yaw = np.arctan2(dy_dt, dx_dt)
    pitch = np.arctan2(dz_dt, ds_dt)
    roll = amp_roll * np.maximum(0.0, -np.cos(2.0 * t))

    d_left = np.full(n_points, half_width)
    d_right = np.full(n_points, half_width)

    data = np.column_stack([x, y, z, roll, pitch, yaw, d_left, d_right])
    return data.astype(np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dst", type=Path, required=True)
    parser.add_argument("--a", type=float, default=30.0)
    parser.add_argument("--b", type=float, default=15.0)
    parser.add_argument("--n-points", type=int, default=400)
    parser.add_argument("--amp-z", type=float, default=3.0)
    parser.add_argument("--amp-roll-deg", type=float, default=10.0)
    parser.add_argument("--half-width", type=float, default=2.0)
    args = parser.parse_args()

    data = generate_oval(
        a=args.a,
        b=args.b,
        n_points=args.n_points,
        amp_z=args.amp_z,
        amp_roll=np.deg2rad(args.amp_roll_deg),
        half_width=args.half_width,
    )
    args.dst.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        args.dst,
        data,
        header="x y z roll pitch yaw d_left d_right",
        fmt="%.6f",
    )

    n = data.shape[0]
    diff = np.diff(data[:, :3], axis=0, append=data[:1, :3])
    length = float(np.linalg.norm(diff, axis=1).sum())
    print(f"wrote {args.dst}")
    print(f"  N={n}  length={length:.1f} m")
    print(f"  z range: [{data[:, 2].min():.2f}, {data[:, 2].max():.2f}]")
    print(f"  roll range (deg): [{np.rad2deg(data[:, 3]).min():.2f}, {np.rad2deg(data[:, 3]).max():.2f}]")
    print(f"  pitch range (deg): [{np.rad2deg(data[:, 4]).min():.2f}, {np.rad2deg(data[:, 4]).max():.2f}]")


if __name__ == "__main__":
    main()
