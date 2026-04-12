"""Convert an f1tenth_racetracks centerline CSV to the RacingTrack 8-column schema.

The f1tenth_racetracks format is `x_m, y_m, w_tr_right_m, w_tr_left_m` (2D, `#`
comment header). We synthesize z=roll=pitch=0 and derive yaw from the path
tangent so vehicles spawn facing forward along the track.

Usage:
    python scripts/convert_f1tenth_track.py \\
        --src /path/to/Austin_centerline.csv \\
        --dst hmclab_isaac/worlds/racing/_tracks_data/austin.txt \\
        --name austin
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from hmclab_isaac.worlds.racing import RacingTrack


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True, help="input CSV path")
    parser.add_argument("--dst", type=Path, required=True, help="output .txt path")
    parser.add_argument("--name", type=str, default=None, help="track name (default: file stem)")
    parser.add_argument(
        "--open",
        action="store_true",
        help="treat track as open (no loop closure). Default: closed.",
    )
    args = parser.parse_args()

    raw = np.loadtxt(args.src, delimiter=",", comments="#")
    if raw.ndim != 2 or raw.shape[1] < 4:
        raise SystemExit(
            f"expected >=4 columns (x, y, w_right, w_left); got shape {raw.shape}"
        )

    xy = raw[:, 0:2]
    d_right = raw[:, 2]
    d_left = raw[:, 3]

    track = RacingTrack.from_2d_centerline(
        name=args.name or args.src.stem,
        xy=xy,
        d_left=d_left,
        d_right=d_right,
        closed=not args.open,
    )

    args.dst.parent.mkdir(parents=True, exist_ok=True)
    track.save(args.dst)

    print(f"wrote {args.dst}")
    print(f"  points: {track.num_points}")
    print(f"  length: {track.length():.1f} m")
    print(f"  closed: {track.closed}")


if __name__ == "__main__":
    main()
