"""Run every M1–M6 smoke test in sequence and report pass/fail.

Each sub-test runs in its own Python subprocess because Isaac Lab can only
host one `DirectRLEnv` per Kit session (see M4 write-up). Expect ~5-10 min
total on an RTX 4080-class GPU.

M7 (ROS2 bridge) is deferred — see docs/deferred_m7_ros2.md.

Usage:
    python scripts/smoke_all.py                 # run all
    python scripts/smoke_all.py --skip demo     # skip by test name
    python scripts/smoke_all.py --only m4,m5    # run a subset
    python scripts/smoke_all.py --fast-fail     # exit on first failure
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = REPO_ROOT / "scripts"


# Each entry: (short_name, milestone, command list).
# The test runner inherits the current environment and activates conda
# via the shell before running the script, so the caller should invoke this
# from an already-activated `hmclab` env.
TESTS: list[tuple[str, str, list[str]]] = [
    ("schema", "M1", [sys.executable, "-c", (
        "from hmclab_isaac.worlds.racing import RacingTrack;"
        "import numpy as np;"
        "xy = np.array([[0,0],[1,0],[2,0.5],[3,1]], dtype=float);"
        "dL = np.full(4, 2.5); dR = np.full(4, 2.5);"
        "t = RacingTrack.from_2d_centerline('smoke', xy, dL, dR, closed=False);"
        "assert t.num_points == 4 and t.length() > 0;"
        "print('M1_SCHEMA_OK')"
    )]),
    ("track_build", "M3", [sys.executable, "-c", (
        "from hmclab_isaac.worlds.racing import RacingTrack, build_circuit_mesh;"
        "import os;"
        "t = RacingTrack.load('hmclab_isaac/worlds/racing/_tracks_data/austin.txt');"
        "m = build_circuit_mesh(t, wall_height=0.5);"
        "assert m['n_points'] == t.num_points;"
        "assert len(m['vertices']) == 4*t.num_points;"
        "print('M3_TRACK_BUILD_OK')"
    )]),
    ("racing_demo", "M4", [
        sys.executable, str(SCRIPTS / "smoke_m4.py"), "--which", "demo", "--steps", "10"
    ]),
    ("racing_solo", "M4", [
        sys.executable, str(SCRIPTS / "smoke_m4.py"), "--which", "solo", "--steps", "10"
    ]),
    ("racing_h2h", "M4", [
        sys.executable, str(SCRIPTS / "smoke_m4.py"), "--which", "h2h", "--steps", "10"
    ]),
    ("rover_spawn", "M5", [sys.executable, str(SCRIPTS / "smoke_m5.py")]),
    ("offroad_demo", "M6", [
        sys.executable, str(SCRIPTS / "smoke_m6.py"), "--which", "demo", "--steps", "10"
    ]),
    ("offroad_nav", "M6", [
        sys.executable, str(SCRIPTS / "smoke_m6.py"), "--which", "nav", "--steps", "10"
    ]),
]

SUCCESS_MARKERS = {
    "schema": "M1_SCHEMA_OK",
    "track_build": "M3_TRACK_BUILD_OK",
    "racing_demo": "M4_SMOKE_DEMO_OK",
    "racing_solo": "M4_SMOKE_SOLO_OK",
    "racing_h2h": "M4_SMOKE_H2H_OK",
    "rover_spawn": "M5_SMOKE_OK",
    "offroad_demo": "M6_SMOKE_DEMO_OK",
    "offroad_nav": "M6_SMOKE_NAV_OK",
}


def run_one(
    name: str, milestone: str, cmd: list[str], log_dir: Path, timeout: float
) -> tuple[bool, float, str]:
    log_file = log_dir / f"{name}.log"
    env = os.environ.copy()
    env.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    t0 = time.time()
    with log_file.open("w") as f:
        try:
            result = subprocess.run(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                cwd=REPO_ROOT,
                env=env,
                timeout=timeout,
            )
            returncode = result.returncode
            timed_out = False
        except subprocess.TimeoutExpired:
            returncode = -1
            timed_out = True
    dt = time.time() - t0
    text = log_file.read_text(errors="replace")
    marker = SUCCESS_MARKERS.get(name)
    # Tests that force-exit after the success marker (os._exit) still count
    # as pass even if returncode != 0, provided the marker is in the log.
    marker_hit = marker is None or marker in text
    ok = (returncode == 0 and marker_hit) or marker_hit
    if timed_out and not marker_hit:
        ok = False
    return ok, dt, str(log_file)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip", default="", help="comma-separated test names to skip")
    parser.add_argument("--only", default="", help="comma-separated test names to run (others skipped)")
    parser.add_argument("--fast-fail", action="store_true", help="exit on first failure")
    parser.add_argument("--log-dir", default="/tmp/hmclab_smoke", help="where per-test logs go")
    args = parser.parse_args()

    skip = {s for s in args.skip.split(",") if s}
    only = {s for s in args.only.split(",") if s}

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"running {len(TESTS)} smoke tests; logs in {log_dir}\n", flush=True)

    results: list[tuple[str, str, bool, float, str]] = []
    total_start = time.time()
    for name, milestone, cmd in TESTS:
        if name in skip or (only and name not in only):
            print(f"  SKIP  [{milestone}] {name}", flush=True)
            continue
        print(f"  RUN   [{milestone}] {name} ... ", end="", flush=True)
        # Per-test timeout: simple imports short, Kit-based tests long.
        # Kit cold boot is ~60s; warm boot ~10s; give enough margin.
        timeout = 30.0 if milestone in ("M1", "M3") else 150.0
        ok, dt, log_path = run_one(name, milestone, cmd, log_dir, timeout)
        status = "OK" if ok else "FAIL"
        print(f"{status} ({dt:.1f}s)", flush=True)
        results.append((name, milestone, ok, dt, log_path))
        if not ok and args.fast_fail:
            break

    total = time.time() - total_start
    print()
    passed = sum(1 for _, _, ok, _, _ in results if ok)
    failed = [(n, m, lp) for n, m, ok, _, lp in results if not ok]
    print(f"summary: {passed}/{len(results)} passed, total {total:.1f}s", flush=True)
    if failed:
        print("\nfailures:")
        for n, m, lp in failed:
            print(f"  [{m}] {n}  → {lp}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
