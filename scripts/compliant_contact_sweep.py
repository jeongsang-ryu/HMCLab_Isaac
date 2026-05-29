"""Compliant-contact case study for UNICORN_1 — DRIVING vibration test.

Each (compliant_stiffness, compliant_damping) case runs in its own Kit
subprocess so the tire material is set BEFORE sim.reset() and PhysX
sees the right value (runtime override is not honoured by PhysX once
the scene has been compiled).

Scenario: drive forward at constant wheel velocity, traverse a small
bump, measure base_link vertical acceleration and pitch rate.

Outer driver:    python scripts/compliant_contact_sweep.py
Inner (per case): python scripts/compliant_contact_sweep.py --case-idx N \\
                         --case-out /tmp/.../case_N.npz

Outputs:
  /tmp/compliant_sweep/timeseries.png   — vertical accel & pitch rate per case
  /tmp/compliant_sweep/summary.png      — RMS values per case
  /tmp/compliant_sweep/case_*.npz       — per-case raw data
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

import numpy as np


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------
CASES = [
    dict(name="rigid",          stiffness=0.0,    damping=0.0),
    dict(name="very_soft_1e3",  stiffness=1.0e3,  damping=1.0e1),
    dict(name="soft_1e4",       stiffness=1.0e4,  damping=1.0e2),
    dict(name="default_1e5",    stiffness=1.0e5,  damping=1.0e3),
    dict(name="stiff_1e6",      stiffness=1.0e6,  damping=1.0e4),
]


# ---------------------------------------------------------------------------
# Inner: actually run sim for one case
# ---------------------------------------------------------------------------
def run_one_case(case_idx: int, out_path: str, args) -> int:
    from isaaclab.app import AppLauncher

    # Build a fresh AppLauncher arg set (no kit-args from outer)
    sub_parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(sub_parser)
    sub_args = sub_parser.parse_args([])
    sub_args.headless = True
    sub_args.enable_cameras = False
    launcher = AppLauncher(sub_args)
    sim_app = launcher.app

    import torch  # noqa: E402
    import isaaclab.sim as sim_utils  # noqa: E402
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402

    from hmclab_isaac.robots.vehicles.UNICORN import UNICORN_1 as v  # noqa: E402

    case = CASES[case_idx]
    print(f"[inner case={case['name']}] k={case['stiffness']:.1e} "
          f"c={case['damping']:.1e}", flush=True)

    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(
            dt=args.dt, render_interval=4, device="cuda:0",
        )
    )
    sim_utils.DomeLightCfg(intensity=2000.0).func(
        "/World/Light", sim_utils.DomeLightCfg(intensity=2000.0))

    gp = sim_utils.GroundPlaneCfg(
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.5, dynamic_friction=1.3, restitution=0.0,
        ),
    )
    gp.func("/World/ground", gp)

    # Series of small bumps every 30 cm in the vehicle's path so each
    # tire encounters one quickly regardless of forward speed.
    bump_positions_x = [0.30, 0.60, 0.90, 1.20, 1.50, 1.80]
    bump_h = 0.015     # 1.5 cm tall — under wheel radius (0.096)
    bump_len = 0.04    # 4 cm long
    for i, bx in enumerate(bump_positions_x):
        bump = sim_utils.MeshCuboidCfg(
            size=(bump_len, 1.5, bump_h),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.7, 0.5, 0.2)),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        )
        bump.func(f"/World/Bump_{i}", bump,
                  translation=(bx, 0.0, bump_h / 2.0))

    init_state = v.CFG.init_state.replace(pos=(0.0, 0.0, float(v.INIT_HEIGHT)))
    robot_cfg = v.CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        init_state=init_state,
    )

    class _SceneCfg(InteractiveSceneCfg):
        robot = robot_cfg

    scene = InteractiveScene(_SceneCfg(num_envs=1, env_spacing=3.0,
                                       replicate_physics=True))

    # Apply tire material BEFORE sim.reset(): PhysX caches materials at
    # scene compile time, so post-reset overrides do not propagate.
    n_mat = v.apply_tire_material(
        "/World/envs/env_.*/Robot",
        compliant_stiffness=case["stiffness"],
        compliant_damping=case["damping"],
        restitution=min(float(v.TIRE_RESTITUTION), 1.0),
    )
    # Verify USD-side: read back the value we just wrote.
    import omni.usd as _omni_usd  # noqa: E402
    from pxr import PhysxSchema as _PhSch  # noqa: E402
    _stage = _omni_usd.get_context().get_stage()
    for _p in _stage.Traverse():
        if "PhysicsMaterials/tire_compliant" in _p.GetPath().pathString:
            _api = _PhSch.PhysxMaterialAPI(_p)
            _ks = _api.GetCompliantContactStiffnessAttr().Get()
            _kd = _api.GetCompliantContactDampingAttr().Get()
            print(f"[inner] {_p.GetPath()}  k={_ks}  c={_kd}", flush=True)
            break
    print(f"[inner] tire material applied to {n_mat} prim(s) before reset",
          flush=True)
    sim.reset()

    art = scene["robot"]
    joint_names = art.joint_names
    import re as _re
    drive_re = _re.compile(r"_wheel_joint$")
    drive_ids = [i for i, n in enumerate(joint_names) if drive_re.search(n)]
    device = sim.device

    # Settling phase: let chassis drop onto wheels (~1 sec)
    settle_steps = int(round(1.0 / args.dt))
    drive_target = torch.zeros((1, len(drive_ids)), device=device)
    art.set_joint_velocity_target(drive_target, joint_ids=drive_ids)
    art.write_data_to_sim()
    for _ in range(settle_steps):
        sim.step()
        scene.update(args.dt)

    # Drive phase: constant wheel ω that gives ~2 m/s linear speed.
    # v = ω × r → ω = v / r
    target_speed = 2.0
    target_omega = target_speed / float(v.WHEEL_RADIUS)
    drive_target.fill_(target_omega)

    # Log over `drive_time` seconds.
    n_steps = int(round(args.drive_time / args.dt))
    z_log = np.empty(n_steps, dtype=np.float32)
    vz_log = np.empty(n_steps, dtype=np.float32)
    az_log = np.empty(n_steps, dtype=np.float32)
    pitch_rate_log = np.empty(n_steps, dtype=np.float32)
    x_log = np.empty(n_steps, dtype=np.float32)

    prev_vz = float(art.data.root_lin_vel_w[0, 2])
    for i in range(n_steps):
        art.set_joint_velocity_target(drive_target, joint_ids=drive_ids)
        art.write_data_to_sim()
        sim.step()
        scene.update(args.dt)

        vz = float(art.data.root_lin_vel_w[0, 2])
        z = float(art.data.root_pos_w[0, 2])
        x = float(art.data.root_pos_w[0, 0])
        # body-frame angular vel: (wx, wy, wz)
        wy = float(art.data.root_ang_vel_b[0, 1])
        # numerical accel in world Z
        az = (vz - prev_vz) / args.dt
        prev_vz = vz

        z_log[i] = z
        vz_log[i] = vz
        az_log[i] = az
        pitch_rate_log[i] = wy
        x_log[i] = x

    np.savez(
        out_path,
        t=np.arange(n_steps) * args.dt,
        z=z_log, vz=vz_log, az=az_log,
        pitch_rate=pitch_rate_log, x=x_log,
        stiffness=case["stiffness"],
        damping=case["damping"],
        name=case["name"],
    )
    print(f"[inner case={case['name']}] saved {out_path}  "
          f"x_final={x_log[-1]:.2f}m  az_rms={np.sqrt((az_log**2).mean()):.3f}m/s²  "
          f"pitch_rms={np.sqrt((pitch_rate_log**2).mean()):.3f}rad/s",
          flush=True)

    # Skip sim_app.close() — it hangs in headless Kit. os._exit() at outer scope
    # tears down the process, and the OS reclaims GPU resources cleanly.
    return 0


# ---------------------------------------------------------------------------
# Outer: launch one subprocess per case, then plot
# ---------------------------------------------------------------------------
def main_outer(args) -> int:
    os.makedirs(args.out_dir, exist_ok=True)
    case_paths = []

    for idx, case in enumerate(CASES):
        out_path = os.path.join(args.out_dir, f"case_{idx:02d}_{case['name']}.npz")
        case_paths.append((case, out_path))
        print(f"\n[outer] launching case {idx} = {case['name']}", flush=True)
        env = os.environ.copy()
        env["OMNI_KIT_ACCEPT_EULA"] = "YES"
        cmd = [
            sys.executable, "-u", __file__,
            "--case-idx", str(idx),
            "--case-out", out_path,
            "--out-dir", args.out_dir,
            "--dt", str(args.dt),
            "--drive-time", str(args.drive_time),
        ]
        ret = subprocess.run(cmd, env=env, check=False)
        if ret.returncode != 0:
            print(f"[outer] case {idx} FAILED with exit {ret.returncode}",
                  flush=True)

    # ------------------------------------------------------------------ plot
    import matplotlib  # noqa: E402
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402

    results = {}
    for case, path in case_paths:
        if os.path.exists(path):
            d = np.load(path, allow_pickle=True)
            results[case["name"]] = {
                "t": d["t"], "az": d["az"],
                "pitch_rate": d["pitch_rate"], "z": d["z"], "x": d["x"],
                "stiffness": float(d["stiffness"]),
                "damping": float(d["damping"]),
            }

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    for name, r in results.items():
        axes[0].plot(r["t"], r["z"], label=name, linewidth=1.2)
        axes[1].plot(r["t"], r["az"], label=name, linewidth=1.0)
        axes[2].plot(r["t"], r["pitch_rate"], label=name, linewidth=1.0)
    axes[0].set_ylabel("base_link z [m]")
    axes[0].set_title("UNICORN_1 driving over bump — chassis dynamics")
    axes[0].grid(alpha=0.3); axes[0].legend(loc="upper right", fontsize=8)
    axes[1].set_ylabel("vert. accel az [m/s²]")
    axes[1].grid(alpha=0.3)
    axes[2].set_ylabel("pitch rate ωy [rad/s]")
    axes[2].set_xlabel("t [s]"); axes[2].grid(alpha=0.3)
    fig.tight_layout()
    ts_path = os.path.join(args.out_dir, "timeseries.png")
    fig.savefig(ts_path, dpi=130)
    plt.close(fig)

    # RMS over the entire drive window
    metrics = {}
    for name, r in results.items():
        metrics[name] = dict(
            az_rms=float(np.sqrt((r["az"] ** 2).mean())),
            pitch_rms=float(np.sqrt((r["pitch_rate"] ** 2).mean())),
            stiffness=r["stiffness"], damping=r["damping"],
        )

    fig2, axs = plt.subplots(1, 2, figsize=(12, 5))
    names = list(metrics.keys())
    x = np.arange(len(names))
    az_vals = [metrics[n]["az_rms"] for n in names]
    pr_vals = [metrics[n]["pitch_rms"] for n in names]

    axs[0].bar(x, az_vals, color="#4C72B0")
    axs[0].set_xticks(x); axs[0].set_xticklabels(names, rotation=15)
    axs[0].set_ylabel("vertical accel RMS [m/s²]")
    axs[0].set_title("base_link vertical vibration")
    axs[0].grid(axis="y", alpha=0.3)

    axs[1].bar(x, pr_vals, color="#DD8452")
    axs[1].set_xticks(x); axs[1].set_xticklabels(names, rotation=15)
    axs[1].set_ylabel("pitch rate RMS [rad/s]")
    axs[1].set_title("chassis pitch oscillation")
    axs[1].grid(axis="y", alpha=0.3)

    fig2.suptitle("Compliant-contact effect on UNICORN_1 driving vibration")
    fig2.tight_layout()
    sm_path = os.path.join(args.out_dir, "summary.png")
    fig2.savefig(sm_path, dpi=130)
    plt.close(fig2)

    if not metrics:
        print("[outer] no case results — nothing to summarise", flush=True)
        return 1

    # Pick optimum (minimise az_rms + pitch_rms after normalising)
    az_norm = max(m["az_rms"] for m in metrics.values()) or 1.0
    pr_norm = max(m["pitch_rms"] for m in metrics.values()) or 1.0
    score = {
        n: m["az_rms"] / az_norm + m["pitch_rms"] / pr_norm
        for n, m in metrics.items()
    }
    best = min(score.items(), key=lambda kv: kv[1])

    print()
    print("[outer] === Summary (RMS over drive window) ===", flush=True)
    for n, m in metrics.items():
        print(f"  {n:<16s}  k={m['stiffness']:.1e}  c={m['damping']:.1e}  "
              f"az_rms={m['az_rms']:.3f} m/s²  "
              f"pitch_rms={m['pitch_rms']:.3f} rad/s  "
              f"score={score[n]:.3f}", flush=True)
    print(f"\n[outer] optimum (lowest combined vibration): {best[0]}",
          flush=True)
    print(f"[outer] saved: {ts_path}", flush=True)
    print(f"[outer] saved: {sm_path}", flush=True)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="/tmp/compliant_sweep")
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--drive-time", type=float, default=8.0,
                        help="Driving phase duration (s).")
    parser.add_argument("--case-idx", type=int, default=-1,
                        help="Inner mode: run this case index. <0 → outer driver.")
    parser.add_argument("--case-out", type=str, default="",
                        help="Inner mode: write npz here.")
    args = parser.parse_args()

    if args.case_idx >= 0:
        rc = run_one_case(args.case_idx, args.case_out, args)
        # Kit shutdown can hang; force exit so the outer driver can advance.
        os._exit(rc)
    else:
        rc = main_outer(args)
        sys.exit(rc)
