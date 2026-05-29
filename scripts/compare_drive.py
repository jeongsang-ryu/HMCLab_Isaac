"""Headless scripted drive test for comparing MuSHR vs UNICORN_2.

Spawns one vehicle on flat ground and runs the same 6-second action
sequence:
    0–2 s: forward only (throttle=1, steer=0)
    2–4 s: forward + right turn (throttle=1, steer=-1)
    4–6 s: forward + right turn (continued)

Logs (x, y, yaw, speed) every 0.1 s and writes a CSV to /tmp/. Run
twice (one per vehicle) and the trajectories can be compared directly.

Run:
    OMNI_KIT_ACCEPT_EULA=YES python scripts/compare_drive.py --vehicle MUSHR
    OMNI_KIT_ACCEPT_EULA=YES python scripts/compare_drive.py --vehicle UNICORN_2
"""
from __future__ import annotations

import argparse
import math
import os
import re

from isaaclab.app import AppLauncher

VEHICLE_CHOICES = ["MUSHR", "UNICORN_2"]
SEQUENCE = [
    # (start_s, end_s, throttle, steer)  — back to forward + turn
    (0.0, 2.0,  1.0,  0.0),
    (2.0, 4.0,  1.0, -1.0),
    (4.0, 6.0,  1.0, -1.0),
]
TOTAL_T = 6.0
LOG_DT = 0.1

parser = argparse.ArgumentParser()
parser.add_argument("--vehicle", choices=VEHICLE_CHOICES, default="UNICORN_2")
parser.add_argument("--max-wheel-rps", type=float, default=300.0)
parser.add_argument("--dt", type=float, default=0.005)
parser.add_argument("--out", type=str, default="")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = False
launcher = AppLauncher(args)
sim_app = launcher.app

import torch  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402


def _build_mushr_cfg():
    """Inline MUSHR_SUS_CFG (matches keyboard_teleop_mushr.py)."""
    from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg

    USD = ("/home/js/hmcl_issac_project/reference_repos/WheeledLab/"
           "source/wheeledlab_assets/data/Robots/UWRLL/mushr_nano_v2.usd")
    actuators = {
        "steering_joints": ImplicitActuatorCfg(
            joint_names_expr=["front_left_wheel_steer", "front_right_wheel_steer"],
            velocity_limit_sim=10.0, effort_limit_sim=3.2, stiffness=100.0, damping=10.0, friction=0.0,
        ),
        "throttle_joints": DCMotorCfg(
            joint_names_expr=[".*throttle"],
            saturation_effort=1.05, effort_limit_sim=0.25, velocity_limit_sim=450.0,
            stiffness=0.0, damping=1000.0, friction=0.0,
        ),
        "suspension": ImplicitActuatorCfg(
            joint_names_expr=[".*_suspension"],
            effort_limit_sim=None, velocity_limit_sim=None,
            stiffness=1.0e8, damping=0.0, friction=0.5,
        ),
    }
    return ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=USD,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                max_linear_velocity=1000.0, max_angular_velocity=100000.0,
                max_depenetration_velocity=100.0, max_contact_impulse=0.0,
                enable_gyroscopic_forces=True,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005, stabilization_threshold=0.001,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.05)),
        actuators=actuators,
    ), 0.4   # MAX_STEER for MuSHR


def _quat_to_yaw(q):
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def main() -> int:
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=args.dt, device="cuda:0")
    )
    sim_utils.GroundPlaneCfg(
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=4.0, dynamic_friction=3.5, restitution=0.0,
            friction_combine_mode="multiply",
        ),
    ).func(
        "/World/ground",
        sim_utils.GroundPlaneCfg(
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=4.0, dynamic_friction=3.5, restitution=0.0,
                friction_combine_mode="multiply",
            ),
        ),
    )
    sim_utils.DomeLightCfg(intensity=1500.0).func(
        "/World/Light", sim_utils.DomeLightCfg(intensity=1500.0)
    )

    if args.vehicle == "MUSHR":
        cfg, max_steer = _build_mushr_cfg()
        steer_re = re.compile(r"_wheel_steer$")
        drive_re = re.compile(r"_wheel_throttle$")
    else:  # UNICORN_2
        from hmclab_isaac.robots.vehicles.UNICORN.UNICORN_2 import CFG, MAX_STEER
        cfg, max_steer = CFG, MAX_STEER
        steer_re = re.compile(r"_steering_joint$")
        drive_re = re.compile(r"_wheel_joint$")

    robot_cfg = cfg.replace(prim_path="/World/envs/env_.*/Robot")

    class _SceneCfg(InteractiveSceneCfg):
        robot = robot_cfg

    scene = InteractiveScene(_SceneCfg(num_envs=1, env_spacing=2.5,
                                       replicate_physics=True))
    sim.reset()

    art = scene["robot"]
    joint_names = art.joint_names
    steer_ids = [i for i, n in enumerate(joint_names) if steer_re.search(n)]
    drive_ids = [i for i, n in enumerate(joint_names) if drive_re.search(n)]
    print(f"[bench] {args.vehicle}: joints={len(joint_names)}  "
          f"steer={steer_ids}  drive={drive_ids}", flush=True)

    dev = sim.device
    drive_target = torch.zeros((1, len(drive_ids)), device=dev)
    steer_target = torch.zeros((1, len(steer_ids)), device=dev)

    # Trajectory log: (t, x, y, yaw_rad, speed)
    log = []
    n_steps = int(round(TOTAL_T / args.dt))
    log_every = max(1, int(round(LOG_DT / args.dt)))

    def _command_at(t: float) -> tuple[float, float]:
        for s, e, thr, steer in SEQUENCE:
            if s <= t < e:
                return thr, steer
        return 0.0, 0.0

    init_pos = art.data.root_pos_w[0].clone()
    init_yaw = _quat_to_yaw(art.data.root_quat_w[0])

    for step in range(n_steps):
        t = step * args.dt
        thr, steer = _command_at(t)
        wheel_speed = thr * args.max_wheel_rps
        steer_angle = steer * max_steer
        drive_target.fill_(wheel_speed)
        steer_target.fill_(steer_angle)
        art.set_joint_velocity_target(drive_target, joint_ids=drive_ids)
        art.set_joint_position_target(steer_target, joint_ids=steer_ids)
        art.write_data_to_sim()
        sim.step()
        scene.update(args.dt)
        if step % log_every == 0:
            p = art.data.root_pos_w[0]
            v = art.data.root_lin_vel_w[0]
            yaw = _quat_to_yaw(art.data.root_quat_w[0])
            sp = art.data.joint_pos[0, steer_ids].tolist()
            wv = art.data.joint_vel[0, drive_ids].tolist()  # wheel angular velocity (rad/s)
            wv_avg = sum(wv) / len(wv) if wv else 0.0
            log.append((t,
                        float(p[0] - init_pos[0]),
                        float(p[1] - init_pos[1]),
                        yaw - init_yaw,
                        float(torch.linalg.norm(v[:2])),
                        sp[0] if len(sp) > 0 else 0.0,
                        sp[1] if len(sp) > 1 else 0.0,
                        thr, steer, wv_avg, float(p[2])))

    # Save CSV
    out_path = args.out or f"/tmp/traj_{args.vehicle}.csv"
    with open(out_path, "w") as f:
        f.write("t,x,y,yaw_rad,speed,steer_fl,steer_fr,thr_cmd,steer_cmd,wheel_w_avg,z\n")
        for row in log:
            f.write(",".join(f"{v:.4f}" for v in row) + "\n")
    print(f"[bench] wrote {out_path}  ({len(log)} samples)", flush=True)

    # Summary metrics
    print(f"\n=== {args.vehicle} summary ===", flush=True)
    for ts in (1.0, 2.0, 3.0, 4.0, 5.0, 5.9):
        r = next((r for r in log if abs(r[0] - ts) < LOG_DT / 2), None)
        if r:
            print(f"  t={ts:.1f}s  pos=({r[1]:+.2f},{r[2]:+.2f})  "
                  f"yaw={math.degrees(r[3]):+6.1f}°  v={r[4]:.2f}  "
                  f"wheel_w={r[9]:+6.1f}rad/s  "
                  f"slip={(r[9]*0.0525 - r[4]):+.2f}  z={r[10]:.3f}  "
                  f"steer[fl,fr]=({r[5]:+.3f},{r[6]:+.3f})  "
                  f"cmd(thr,st)=({r[7]:+.1f},{r[8]:+.1f})",
                  flush=True)

    os._exit(0)


if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception:
        traceback.print_exc()
        os._exit(1)
