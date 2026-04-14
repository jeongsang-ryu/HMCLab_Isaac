"""Suspension visualisation demo for MuSHR Nano v2.

Three phases show how the simplified per-wheel prismatic suspension
reacts to different load inputs:

  --phase 1  body pinned in air, push up on each wheel one-at-a-time
             -> wheel_suspension prismatic visibly compresses per corner
  --phase 2  body free on ground, press chassis DOWN -> all 4 shocks
             compress uniformly
  --phase 3  body free on ground, apply lateral then longitudinal force
             at chassis CoG -> roll then pitch (load transfer)

Outputs PNG/MP4/GIF under Itutorial/outputs/suspension_demo_phase{N}.*

Run:
    OMNI_KIT_ACCEPT_EULA=YES \
      /home/js/anaconda3/envs/hmclab_test/bin/python \
      scripts/suspension_demo.py --phase 1
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--phase", type=int, required=True, choices=[1, 2, 3])
parser.add_argument("--steps", type=int, default=360)
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--gui", action="store_true", default=True)
parser.add_argument("--compliant", action="store_true",
                    help="Apply compliant (soft) contact to wheel colliders")
parser.add_argument("--compliant-stiffness", type=float, default=8e3)
parser.add_argument("--compliant-damping", type=float, default=50.0)
parser.add_argument("--lock-suspension", action="store_true",
                    help="Lock wheel_suspension prismatic so only tire compliance is visible")
parser.add_argument("--cam", type=str, default="side", choices=["side", "wheel"],
                    help="Camera preset: full side view or close-up on rear-left wheel")
parser.add_argument("--label", type=str, default="", help="Tag suffix for output files")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = False
args.enable_cameras = True
launcher = AppLauncher(args)
sim_app = launcher.app

# --- Post-AppLauncher ---
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Itutorial")
)
from _common import get_robot_spec, post_spawn_fix  # noqa: E402
from hmclab_isaac.utils.capture import ChaseCamRecorder  # noqa: E402


def apply_compliant_contact_to_wheels(
    root_prim_path: str,
    stiffness: float,
    damping: float,
) -> int:
    """Bind a compliant-contact PhysicsMaterial to every wheel collider.

    PhysX's compliant contact makes rigid-body contacts behave like a
    spring-damper without deforming geometry — cheap "soft tire" feel.
    """
    import omni.usd
    from pxr import Sdf, UsdPhysics, UsdShade
    from pxr import PhysxSchema

    stage = omni.usd.get_context().get_stage()
    mat_path = f"{root_prim_path}/compliant_tire_mat"
    if not stage.GetPrimAtPath(mat_path).IsValid():
        mat = UsdShade.Material.Define(stage, mat_path)
        UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
        # Set friction on the rigid material
        phys_mat_api = UsdPhysics.MaterialAPI(mat.GetPrim())
        phys_mat_api.CreateStaticFrictionAttr(1.2)
        phys_mat_api.CreateDynamicFrictionAttr(1.0)
        phys_mat_api.CreateRestitutionAttr(0.0)
        # Compliant contact attributes on PhysxMaterialAPI
        px_api = PhysxSchema.PhysxMaterialAPI.Apply(mat.GetPrim())
        px_api.CreateCompliantContactStiffnessAttr(float(stiffness))
        px_api.CreateCompliantContactDampingAttr(float(damping))

    mat_prim = stage.GetPrimAtPath(mat_path)
    mat = UsdShade.Material(mat_prim)

    n = 0
    for prim in stage.Traverse():
        p = prim.GetPath().pathString
        if not p.startswith(root_prim_path):
            continue
        if "_wheel_link" not in p:
            continue
        # Bind material on this wheel link and all collider children
        if prim.HasAPI(UsdPhysics.CollisionAPI) or any(
            c.HasAPI(UsdPhysics.CollisionAPI) for c in prim.GetChildren()
        ):
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                mat,
                bindingStrength=UsdShade.Tokens.strongerThanDescendants,
                materialPurpose="physics",
            )
            n += 1
    print(f"[compliant] bound to {n} wheel link(s) @ k={stiffness} d={damping}", flush=True)
    return n

OUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "Itutorial", "outputs"
)


def side_view_cam(pos_offset=(0.0, -1.5, 0.4), look_at=(0.0, 0.0, 0.1)):
    """Build a fixed side-view camera recorder (reuses ChaseCam for IO)."""
    rec = ChaseCamRecorder(
        out_dir=OUT_DIR,
        tag="_side",
        prim_path="/World/SideCam",
        behind=0.0,
        above=0.0,
        look_up=0.0,
        width=960,
        height=540,
    )
    return rec, pos_offset, look_at


def main() -> int:
    phase = args.phase
    tag = f"suspension_demo_phase{phase}"

    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=0.005, device="cuda:0")
    )
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=2500.0, color=(0.95, 0.95, 0.95)).func(
        "/World/Light",
        sim_utils.DomeLightCfg(intensity=2500.0, color=(0.95, 0.95, 0.95)),
    )

    spec = get_robot_spec("mushr")

    init_z = 0.28 if phase == 1 else spec.init_z
    init_pos = (0.0, 0.0, init_z)
    ego_cfg = spec.cfg.replace(prim_path="/World/Ego")
    ego_cfg = ego_cfg.replace(init_state=ego_cfg.init_state.replace(pos=init_pos))
    ego_cfg.spawn.func(ego_cfg.prim_path, ego_cfg.spawn, translation=init_pos)
    post_spawn_fix(spec, ego_cfg.prim_path)
    if args.compliant:
        apply_compliant_contact_to_wheels(
            ego_cfg.prim_path,
            stiffness=args.compliant_stiffness,
            damping=args.compliant_damping,
        )
    ego = Articulation(ego_cfg)

    side, pos_offset, look_at = side_view_cam()

    sim.reset()
    side.on_reset()

    # Figure out indices we need
    body_names = ego.body_names
    joint_names = ego.joint_names
    print(f"bodies={body_names}", flush=True)
    print(f"joints={joint_names}", flush=True)

    if args.lock_suspension:
        susp_ids = [i for i, n in enumerate(joint_names) if "suspension" in n]
        k_lock = torch.full((1, len(susp_ids)), 5e5, device=sim.device)
        d_lock = torch.full((1, len(susp_ids)), 1e3, device=sim.device)
        ego.write_joint_stiffness_to_sim(k_lock, joint_ids=susp_ids)
        ego.write_joint_damping_to_sim(d_lock, joint_ids=susp_ids)
        print(f"[lock] suspension joints {susp_ids} stiffened to 5e5", flush=True)

    base_ids, _ = ego.find_bodies("base_link")
    rear_drive_ids = [i for i, n in enumerate(joint_names) if "back" in n and "throttle" in n]
    wheel_body_order = [
        "front_left_wheel_link",
        "front_right_wheel_link",
        "back_left_wheel_link",
        "back_right_wheel_link",
    ]
    wheel_body_ids = []
    for wn in wheel_body_order:
        ids, _ = ego.find_bodies(wn)
        if ids:
            wheel_body_ids.append(ids[0])

    device = sim.device
    num_env = 1
    num_bodies = ego.num_bodies

    # Pose lock (phase 1) — hold the chassis in the air
    locked_root_state = ego.data.root_state_w.clone()
    locked_root_state[0, 0:3] = torch.tensor(init_pos, device=device)
    locked_root_state[0, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
    locked_root_state[0, 7:13] = 0.0

    steps = args.steps
    dt = sim.get_physics_dt()

    # Frame list and writer
    frame_step = max(1, int(round(1.0 / (args.fps * dt))))

    # Initial warmup so the camera sees a frame
    for _ in range(5):
        sim.step()
        ego.update(dt)

    def update_side_cam(phase: int):
        root_pos = ego.data.root_pos_w[0].cpu().numpy()
        import torch as _t

        if args.cam == "wheel":
            # Close-up on back-left wheel, low angle so contact patch + tire
            # compression (or ground penetration) is clearly visible.
            bx, by, bz = float(root_pos[0]), float(root_pos[1]), float(root_pos[2])
            # Back-left wheel is roughly at (bx-0.16, by+0.12) with r≈0.05.
            wx, wy, wz = bx - 0.16, by + 0.12, 0.05
            eye = (wx - 0.25, wy + 0.50, wz + 0.10)
            target = (wx, wy, wz)
        else:
            eye = (
                float(root_pos[0]) + 0.0,
                float(root_pos[1]) + 1.6,
                float(root_pos[2]) + 0.25,
            )
            target = (float(root_pos[0]), float(root_pos[1]), float(root_pos[2]) + 0.02)

        eyes_t = _t.tensor([eye], device=side._camera._device, dtype=_t.float32)
        tgts_t = _t.tensor([target], device=side._camera._device, dtype=_t.float32)
        side._camera.set_world_poses_from_view(eyes_t, tgts_t)

    # ------------------------------------------------------------------
    # Phase-specific inputs
    # ------------------------------------------------------------------
    forces = torch.zeros((num_env, num_bodies, 3), device=device)
    torques = torch.zeros((num_env, num_bodies, 3), device=device)
    positions = torch.zeros((num_env, num_bodies, 3), device=device)

    if phase == 1:
        # Rotate through each wheel — apply upward force ramp
        wheel_cycle_steps = steps // max(len(wheel_body_ids), 1)
        peak_force = 6.0  # N, enough to lift the wheel against 2000 N/m spring
    elif phase == 2:
        peak_force = 80.0  # N, compress all four shocks
    elif phase == 3:
        peak_lat = 40.0  # N sustained lateral (≈ 1.3 g for ~3 kg)
        peak_long = 50.0  # N longitudinal

    captured = 0
    max_frames = 240

    import time as _time
    step_times = []

    for i in range(steps):
        phase_t = i / steps  # 0..1

        # reset wrench
        forces.zero_()
        torques.zero_()

        if phase == 1:
            # Pin the chassis each step
            ego.write_root_pose_to_sim(locked_root_state[:, 0:7])
            ego.write_root_velocity_to_sim(locked_root_state[:, 7:13])

            # Cycle which wheel gets force
            if wheel_body_ids:
                per = steps / len(wheel_body_ids)
                w_idx = min(int(i / per), len(wheel_body_ids) - 1)
                local_t = (i - w_idx * per) / per  # 0..1 within this wheel's window
                # Triangle ramp: up then back down
                ramp = 1.0 - abs(2 * local_t - 1.0)
                f = peak_force * ramp
                forces[0, wheel_body_ids[w_idx], 2] = f  # +Z world (local frame ≈ world since wheel unrotated when still)

        elif phase == 2:
            # Downward force on chassis CoG — ramp up, hold, ramp down
            if phase_t < 0.4:
                ramp = phase_t / 0.4
            elif phase_t < 0.7:
                ramp = 1.0
            else:
                ramp = max(0.0, 1.0 - (phase_t - 0.7) / 0.3)
            forces[0, base_ids[0], 2] = -peak_force * ramp

        elif phase == 3:
            # First half: lateral; second half: longitudinal
            if phase_t < 0.5:
                local = phase_t / 0.5
                # Triangle then sinusoidal
                if local < 0.5:
                    ramp = local / 0.5
                else:
                    ramp = 1.0 - (local - 0.5) / 0.5
                forces[0, base_ids[0], 1] = peak_lat * ramp  # +Y lateral
                # apply at higher-than-CoG to produce clear roll
                positions[0, base_ids[0], 2] = 0.08
            else:
                local = (phase_t - 0.5) / 0.5
                if local < 0.5:
                    ramp = local / 0.5
                else:
                    ramp = 1.0 - (local - 0.5) / 0.5
                forces[0, base_ids[0], 0] = peak_long * ramp  # +X longitudinal
                positions[0, base_ids[0], 2] = 0.08

        ego.set_external_force_and_torque(
            forces=forces,
            torques=torques,
            positions=positions,
            is_global=True,
        )

        # Rear-drive small idle damping off; no throttle command needed
        ego.write_data_to_sim()
        _t0 = _time.perf_counter()
        sim.step()
        step_times.append(_time.perf_counter() - _t0)
        ego.update(dt)

        # Update side camera and capture
        if i % frame_step == 0 and captured < max_frames:
            update_side_cam(phase)
            side.capture_frame()
            captured += 1

    # Rename side-cam output to phase-tagged
    suffix = f"_{args.label}" if args.label else ""
    side.tag = tag + suffix
    out = side.finalize()

    # Step-time stats (skip first 30 warmup steps)
    import statistics as _stats
    warm = step_times[30:]
    if warm:
        med = _stats.median(warm) * 1000
        mean = _stats.mean(warm) * 1000
        p95 = sorted(warm)[int(len(warm) * 0.95)] * 1000
        print(f"STEP_STATS label={args.label or 'base'} median={med:.3f}ms mean={mean:.3f}ms p95={p95:.3f}ms n={len(warm)}", flush=True)
    print(f"PHASE_{phase}_DONE outputs={out}", flush=True)

    # Convert MP4 -> GIF with palette for a small, clean file
    mp4 = out.get("mp4")
    if mp4 and os.path.exists(mp4):
        gif = os.path.join(OUT_DIR, f"{tag}{suffix}.gif")
        palette = os.path.join(OUT_DIR, f"{tag}{suffix}_palette.png")
        # Build palette, then render GIF
        os.system(
            f"ffmpeg -y -v error -i {mp4} -vf "
            f"'fps={args.fps},scale=480:-1:flags=lanczos,palettegen' {palette}"
        )
        os.system(
            f"ffmpeg -y -v error -i {mp4} -i {palette} -filter_complex "
            f"'fps={args.fps},scale=480:-1:flags=lanczos[x];[x][1:v]paletteuse' {gif}"
        )
        if os.path.exists(gif):
            print(f"GIF_OK {gif}", flush=True)

    return 0


if __name__ == "__main__":
    import traceback

    rc = 0
    try:
        rc = main()
    except Exception:
        traceback.print_exc()
        rc = 1
    os._exit(rc)
