"""Sensor-less RL throughput benchmark.

Spawns N MuSHR (or F1Tenth) vehicles via InteractiveScene with
`replicate_physics=True`, no LiDAR / no camera, and steps with random
joint targets. Optionally runs a small MLP forward+backward to include
the typical PPO-sized network cost.

Reports median/mean physics step time, GPU memory, and total
environment-steps/sec.

Usage:
    OMNI_KIT_ACCEPT_EULA=YES \
      /home/js/anaconda3/envs/hmclab_test/bin/python \
      scripts/rl_throughput_benchmark.py --num-envs 4096 --steps 600
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num-envs", type=int, default=4096, dest="num_envs")
parser.add_argument("--steps", type=int, default=600)
parser.add_argument("--robot", type=str, default="mushr",
                    choices=["mushr", "f1tenth"])
parser.add_argument("--dt", type=float, default=0.005)
parser.add_argument("--with-net", action="store_true",
                    help="run a 256x256 MLP fwd+bwd each step (rough PPO cost)")
parser.add_argument("--obs-dim", type=int, default=67, dest="obs_dim")
parser.add_argument("--act-dim", type=int, default=2, dest="act_dim")
parser.add_argument("--label", type=str, default="bench")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = False
launcher = AppLauncher(args)
sim_app = launcher.app

import torch  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.assets import ArticulationCfg  # noqa: E402

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Itutorial")
)
from _common import get_robot_spec  # noqa: E402


def main() -> int:
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=args.dt, device=getattr(args, "device", "cuda:0"))
    )

    spec = get_robot_spec(args.robot)
    robot_cfg: ArticulationCfg = spec.cfg.replace(prim_path="/World/envs/env_.*/Robot")
    robot_cfg = robot_cfg.replace(
        init_state=robot_cfg.init_state.replace(pos=(0.0, 0.0, spec.init_z))
    )

    scene_cfg = InteractiveSceneCfg(
        num_envs=args.num_envs,
        env_spacing=2.5,
        replicate_physics=True,
    )

    # Add ground + light
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=2500.0, color=(0.95, 0.95, 0.95)).func(
        "/World/Light",
        sim_utils.DomeLightCfg(intensity=2500.0, color=(0.95, 0.95, 0.95)),
    )

    # Build scene with robot articulation entry
    from isaaclab.scene import InteractiveSceneCfg as _ISCfg

    class _SceneCfg(_ISCfg):
        robot = robot_cfg

    scene_cfg = _SceneCfg(
        num_envs=args.num_envs,
        env_spacing=2.5,
        replicate_physics=True,
    )
    scene = InteractiveScene(scene_cfg)

    sim.reset()

    art = scene["robot"]
    joint_names = art.joint_names
    steer_ids, drive_ids = spec.resolve_joints(joint_names)
    print(f"[bench] joints={len(joint_names)} steer={steer_ids} drive={drive_ids}",
          flush=True)

    dev = sim.device
    N = args.num_envs

    # Action targets
    drive_target = torch.zeros((N, len(drive_ids)), device=dev)
    steer_target = torch.zeros((N, len(steer_ids)), device=dev)

    # Optional NN
    net = None
    optim = None
    if args.with_net:
        net = torch.nn.Sequential(
            torch.nn.Linear(args.obs_dim, 256),
            torch.nn.Tanh(),
            torch.nn.Linear(256, 256),
            torch.nn.Tanh(),
            torch.nn.Linear(256, args.act_dim * 2),  # mean + log_std
        ).to(dev)
        optim = torch.optim.Adam(net.parameters(), lr=3e-4)

    # GPU mem before
    torch.cuda.reset_peak_memory_stats(dev) if "cuda" in str(dev) else None

    step_times = []
    nn_times = []

    # Warm-up
    for _ in range(20):
        steer_target.uniform_(-spec.max_steer, spec.max_steer)
        drive_target.uniform_(-1.0, 1.0).mul_(8.0)
        art.set_joint_position_target(steer_target, joint_ids=steer_ids)
        art.set_joint_velocity_target(drive_target, joint_ids=drive_ids)
        art.write_data_to_sim()
        sim.step()
        scene.update(args.dt)

    t_total_start = time.perf_counter()
    for i in range(args.steps):
        # Random actions
        steer_target.uniform_(-spec.max_steer, spec.max_steer)
        drive_target.uniform_(-1.0, 1.0).mul_(8.0)
        art.set_joint_position_target(steer_target, joint_ids=steer_ids)
        art.set_joint_velocity_target(drive_target, joint_ids=drive_ids)
        art.write_data_to_sim()

        if "cuda" in str(dev):
            torch.cuda.synchronize(dev)
        t0 = time.perf_counter()
        sim.step()
        if "cuda" in str(dev):
            torch.cuda.synchronize(dev)
        step_times.append(time.perf_counter() - t0)

        # Build a fake obs from root state (no sensor)
        if net is not None:
            root_pos = art.data.root_pos_w
            root_vel = art.data.root_lin_vel_w
            joint_pos = art.data.joint_pos
            # Pad/concat to obs_dim
            obs_parts = [root_pos, root_vel, joint_pos]
            obs = torch.cat(obs_parts, dim=-1)
            if obs.shape[-1] < args.obs_dim:
                pad = torch.zeros(N, args.obs_dim - obs.shape[-1], device=dev)
                obs = torch.cat([obs, pad], dim=-1)
            else:
                obs = obs[:, : args.obs_dim]

            t1 = time.perf_counter()
            out = net(obs)
            loss = out.pow(2).mean()
            optim.zero_grad()
            loss.backward()
            optim.step()
            if "cuda" in str(dev):
                torch.cuda.synchronize(dev)
            nn_times.append(time.perf_counter() - t1)

        scene.update(args.dt)

    t_total = time.perf_counter() - t_total_start

    warm = step_times[10:]
    median_ms = statistics.median(warm) * 1000
    mean_ms = statistics.mean(warm) * 1000
    p95_ms = sorted(warm)[int(len(warm) * 0.95)] * 1000
    env_steps = N * args.steps
    throughput = env_steps / t_total

    print(
        f"BENCH label={args.label} num_envs={N} robot={args.robot} "
        f"with_net={args.with_net} "
        f"step_median_ms={median_ms:.3f} step_mean_ms={mean_ms:.3f} "
        f"step_p95_ms={p95_ms:.3f} "
        f"total_time_s={t_total:.2f} env_steps_per_sec={throughput:,.0f}",
        flush=True,
    )
    if nn_times:
        nn_warm = nn_times[10:]
        print(
            f"NN_BENCH label={args.label} "
            f"nn_median_ms={statistics.median(nn_warm)*1000:.3f} "
            f"nn_mean_ms={statistics.mean(nn_warm)*1000:.3f}",
            flush=True,
        )
    if "cuda" in str(dev):
        peak = torch.cuda.max_memory_allocated(dev) / (1024 ** 3)
        print(f"GPU_MEM label={args.label} peak_alloc_GB={peak:.2f}", flush=True)
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
