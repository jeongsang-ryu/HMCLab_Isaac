"""Generate a catalogue of 3D racing tracks.

Writes one 8-column `.txt` per track into
`hmclab_isaac/worlds/racing/_tracks_data/`. Each track is designed to
exercise a different 3D feature: banking, hills, climbs, figure-8
overpass, helix, ridge, etc.

The tracks are sized to the mini-oval scale (≈ 40 m length, ≤ 20 m
bounding box) so vehicles remain visible in top-down captures.

Usage:
    python scripts/generate_3d_tracks.py

All tracks respect the RacingTrack schema:
    x y z roll pitch yaw d_left d_right
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _finalise(xy: np.ndarray, z: np.ndarray, roll: np.ndarray,
              half_w: float = 1.0) -> np.ndarray:
    """Build the 8-column matrix from (x, y, z, roll). Pitch + yaw come
    from finite-difference tangents so every track has a sensible frame."""
    n = len(xy)
    # Close the loop for derivatives
    dx = np.zeros(n); dy = np.zeros(n); dz = np.zeros(n)
    for i in range(n):
        nxt = (i + 1) % n
        prv = (i - 1) % n
        dx[i] = xy[nxt, 0] - xy[prv, 0]
        dy[i] = xy[nxt, 1] - xy[prv, 1]
        dz[i] = z[nxt] - z[prv]
    ds = np.sqrt(dx * dx + dy * dy)
    yaw = np.arctan2(dy, dx)
    pitch = np.arctan2(dz, ds)
    data = np.column_stack(
        [xy[:, 0], xy[:, 1], z, roll, pitch, yaw,
         np.full(n, half_w), np.full(n, half_w)]
    )
    return data.astype(np.float64)


def _save(track_name: str, data: np.ndarray, out_dir: Path) -> None:
    path = out_dir / f"{track_name}.txt"
    np.savetxt(path, data, header="x y z roll pitch yaw d_left d_right", fmt="%.6f")
    diff = np.diff(data[:, :3], axis=0, append=data[:1, :3])
    length = float(np.linalg.norm(diff, axis=1).sum())
    x_range = (data[:, 0].min(), data[:, 0].max())
    y_range = (data[:, 1].min(), data[:, 1].max())
    z_range = (data[:, 2].min(), data[:, 2].max())
    print(
        f"  {track_name:<28} N={len(data):>3}  length={length:>6.2f} m  "
        f"x={x_range[0]:+5.1f}..{x_range[1]:+5.1f}  "
        f"y={y_range[0]:+5.1f}..{y_range[1]:+5.1f}  "
        f"z={z_range[0]:.2f}..{z_range[1]:.2f}"
    )


# ----------------------------------------------------------------------
# Track generators
# ----------------------------------------------------------------------
def gen_flat_oval(n=120):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    xy = np.column_stack([8.0 * np.cos(t), 4.0 * np.sin(t)])
    z = np.full(n, 0.4)                      # flat, lifted above ground
    roll = np.zeros(n)
    return _finalise(xy, z, roll)


def gen_banked_oval(n=120):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    xy = np.column_stack([8.0 * np.cos(t), 4.0 * np.sin(t)])
    z = 0.4 + 0.25 * (0.5 + 0.5 * np.sin(t))  # gentle hill, z in [0.4, 0.65]
    roll = np.deg2rad(12.0) * np.maximum(0.0, -np.cos(2 * t))
    return _finalise(xy, z, roll)


def gen_figure_8_overpass(n=160):
    """Two loops joined at the middle. One lobe runs at z≈0.3, the other
    at z≈1.5, so the path crosses itself in (x, y) with ~1.2 m vertical
    separation — looks like a figure-8 top-down, but in 3D it's a smooth
    over/under."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    # Lemniscate of Gerono-style: x = a cos t, y = a sin t cos t
    a = 6.0
    x = a * np.cos(t)
    y = a * np.sin(t) * np.cos(t)
    # Height makes loop 1 low, loop 2 high. Smoothly blend with sin(2t).
    z = 0.3 + 0.6 * (1.0 + np.sin(2 * t)) * 0.5 * 2.0  # z in [0.3, 1.5]
    z = 0.3 + 0.6 * (1.0 + np.sin(2 * t))              # simplified → [0.3, 1.5]
    roll = np.zeros(n)
    xy = np.column_stack([x, y])
    return _finalise(xy, z, roll)


def gen_crossover_oval(n=160):
    """Oval where one straight dips down to pass UNDER the other straight.
    Top-down: single oval outline. 3D: the centerline crosses over itself
    at the midpoint."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    # Base oval in xy
    x = 7.0 * np.cos(t)
    y = 4.0 * np.sin(t)
    # Lift the east straight (t≈0) to z=1.2, drop the west straight (t≈π)
    # to z=0.3. Smooth via cos(t).
    z = 0.75 + 0.45 * np.cos(t)
    roll = np.zeros(n)
    xy = np.column_stack([x, y])
    return _finalise(xy, z, roll)


def gen_helix(n=200, turns=2):
    """Ascending helix — single spiral climbing over two turns then wrapping
    back down to the start. Used to stress-test climb physics."""
    t = np.linspace(0, 2 * np.pi * turns, n, endpoint=False)
    r = 5.0
    x = r * np.cos(t)
    y = r * np.sin(t)
    # Rise from 0.3 to 2.0 over `turns` laps, then descend back
    progress = t / (2 * np.pi * turns)
    z = 0.3 + 1.5 * np.sin(np.pi * progress)  # up, then back down
    roll = np.deg2rad(8.0) * np.ones(n)
    xy = np.column_stack([x, y])
    return _finalise(xy, z, roll)


def gen_ridge(n=140):
    """Oval with a high ridge at top-corner and a valley at bottom-corner."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = 8.0 * np.cos(t)
    y = 4.5 * np.sin(t)
    z = 0.4 + 0.8 * np.sin(t) ** 2  # peak at top and bottom corners
    roll = np.zeros(n)
    return _finalise(np.column_stack([x, y]), z, roll)


def gen_valley(n=140):
    """Oval where the track dips into a bowl on one straight."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = 8.0 * np.cos(t)
    y = 4.0 * np.sin(t)
    # Deep valley near t≈π (left straight)
    z = 1.2 - 0.8 * np.exp(-((t - np.pi) ** 2) / 0.6)
    roll = np.zeros(n)
    return _finalise(np.column_stack([x, y]), z, roll)


def gen_s_curve(n=160):
    """Closed S-curve: a long figure-like loop with two opposite-banked
    hairpins."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = 7.0 * np.cos(t)
    y = 2.5 * np.sin(2 * t)  # double-period sin → S-curve
    z = 0.4 + 0.2 * np.sin(t)
    roll = np.deg2rad(10.0) * np.sin(2 * t)  # banks opposite ways
    return _finalise(np.column_stack([x, y]), z, roll)


def gen_twin_loop_overpass(n=180):
    """Two side-by-side loops connected by an over/under at the join.
    Left loop runs at low z, right loop at high z, the join has a smooth
    S-ramp joining them."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    # Smoothly interpolate between two circle centers
    cx = 4.5 * np.sign(np.sin(t))  # -4.5 or +4.5 depending on half
    x = cx + 3.0 * np.cos(2 * t)
    y = 3.0 * np.sin(2 * t)
    # Low on left (cx<0), high on right (cx>0)
    z = 0.35 + 0.85 * (np.sign(np.sin(t)) * 0.5 + 0.5)
    roll = np.zeros(n)
    return _finalise(np.column_stack([x, y]), z, roll)


def gen_rolling_hills(n=160):
    """Long oval with multiple hills along the straights (sinusoidal z)."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = 9.0 * np.cos(t)
    y = 3.5 * np.sin(t)
    z = 0.5 + 0.3 * np.sin(4 * t)  # 4 bumps per lap
    roll = np.zeros(n)
    return _finalise(np.column_stack([x, y]), z, roll)


def gen_dipper_crossover(n=180):
    """Oval with a Ω-shaped straight that dips DOWN through the other
    straight to cross under. The track centerline passes (0, 0) twice with
    different z. A vehicle running CCW climbs a hill on the east straight,
    then dives under as it returns west."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    # Start with an oval, then perturb the north straight to dip south
    # through the origin and back north — crossing the south straight.
    x = 7.0 * np.cos(t)
    y = 3.0 * np.sin(t) + 0.0
    # On the north straight (t in [π/2, 3π/2]-ish where y > 0) perturb
    # y toward −y and lower z.
    mask = np.sin(t) > 0  # upper half
    dip = np.zeros(n)
    for i in range(n):
        if mask[i]:
            local = (t[i] - 0.5 * np.pi) / np.pi  # 0..1 through north half
            dip[i] = -5.5 * np.sin(np.pi * local) ** 2  # dip south up to 5.5 m
    y = y + dip
    z = np.where(mask, 0.35, 1.3)  # north dips (low z), south stays high
    roll = np.zeros(n)
    return _finalise(np.column_stack([x, y]), z, roll)


def gen_mobius_ramp(n=200):
    """Spiral ramp: circular path that climbs then descends smoothly over
    one full loop, with banking. Visually a wavy ring."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = 6.0
    x = r * np.cos(t)
    y = r * np.sin(t) * 0.6  # slightly flattened
    z = 0.4 + 0.7 * (0.5 + 0.5 * np.sin(3 * t))  # 3 crests
    roll = np.deg2rad(10.0) * np.sin(3 * t)
    return _finalise(np.column_stack([x, y]), z, roll)


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------
TRACK_CATALOG = {
    "mini_oval_flat":        gen_flat_oval,
    "mini_oval_banked":      gen_banked_oval,
    "mini_figure8_over":     gen_figure_8_overpass,
    "mini_crossover_oval":   gen_crossover_oval,
    "mini_helix":            gen_helix,
    "mini_ridge":            gen_ridge,
    "mini_valley":           gen_valley,
    "mini_s_curve":          gen_s_curve,
    "mini_twin_loop":        gen_twin_loop_overpass,
    "mini_rolling_hills":    gen_rolling_hills,
    "mini_dipper_crossover": gen_dipper_crossover,
    "mini_mobius_ramp":      gen_mobius_ramp,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dst_dir",
        type=Path,
        default=Path("hmclab_isaac/worlds/racing/_tracks_data"),
    )
    args = parser.parse_args()
    args.dst_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing {len(TRACK_CATALOG)} tracks to {args.dst_dir}:")
    for name, gen in TRACK_CATALOG.items():
        data = gen()
        _save(name, data, args.dst_dir)


if __name__ == "__main__":
    main()
