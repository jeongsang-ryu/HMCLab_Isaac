"""Tutorial 01 — spawn a single vehicle on a flat ground plane.

Supports `--robot f1tenth|mushr`. Captures a top-down screenshot + a
rear-top chase-cam clip.

Run:
    OMNI_KIT_ACCEPT_EULA=YES python Itutorial/01_spawn_on_ground.py --robot f1tenth
    OMNI_KIT_ACCEPT_EULA=YES python Itutorial/01_spawn_on_ground.py --robot mushr
"""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=120)
parser.add_argument("--robot", type=str, default="TOY_01")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = False
args.enable_cameras = True
launcher = AppLauncher(args)
simulation_app = launcher.app

# --- Post-AppLauncher imports ---
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402

from _common import apply_ackermann, get_robot_spec  # noqa: E402
from hmclab_isaac.utils.capture import ChaseCamRecorder, TopDownRecorder  # noqa: E402


OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")


def main() -> int:
    spec = get_robot_spec(args.robot)
    tag_topdown = f"tut01_ground_{spec.name}_topdown"
    tag_chase = f"tut01_ground_{spec.name}_chase"

    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=0.01, device="cuda:0")
    )
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=2500.0, color=(0.95, 0.95, 0.95)).func(
        "/World/Light", sim_utils.DomeLightCfg(intensity=2500.0, color=(0.95, 0.95, 0.95))
    )

    init_pos = (0.0, 0.0, spec.init_z)
    ego_cfg = spec.cfg.replace(prim_path="/World/Ego")
    ego_cfg = ego_cfg.replace(init_state=ego_cfg.init_state.replace(pos=init_pos))
    ego_cfg.spawn.func(ego_cfg.prim_path, ego_cfg.spawn, translation=init_pos)
    ego = Articulation(ego_cfg)

    topdown = TopDownRecorder(
        center=(0.0, 0.0),
        size=4.0,
        prim_path="/World/TopDownCam",
        out_dir=OUT_DIR,
        tag=tag_topdown,
    )
    chase = ChaseCamRecorder(
        out_dir=OUT_DIR,
        tag=tag_chase,
        prim_path="/World/ChaseCam",
        behind=2.0, above=1.2, look_up=0.1,
    )

    sim.reset()
    # topdown.on_reset()
    # chase.on_reset()
    print(
        f">>> spawned {spec.name} — joints={ego.num_joints} bodies={ego.num_bodies}",
        flush=True,
    )

    # for i in range(args.steps):
    while simulation_app.is_running():

        sim.step()
        ego.update(sim.get_physics_dt())
        # if i % 2 == 0:
        if True:
            # topdown.capture_frame()
            pos = ego.data.root_pos_w[0].cpu().numpy()
            q = ego.data.root_quat_w[0].cpu().numpy()
            import math
            w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
            yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
            # chase.update_follow((float(pos[0]), float(pos[1]), float(pos[2])), yaw)
            # chase.capture_frame()

    pos = ego.data.root_pos_w[0]
    print(
        f">>> final pose z={pos[2].item():.3f} speed={ego.data.root_lin_vel_w[0].norm().item():.3f}",
        flush=True,
    )
    print(f"TUT01_OK robot={spec.name}", flush=True)
    print(f"  topdown: {topdown.finalize()}", flush=True)
    print(f"  chase:   {chase.finalize()}", flush=True)
    return 0


if __name__ == "__main__":
    import traceback
    code = 0
    try:
        code = main()
    except Exception:
        traceback.print_exc()
        code = 1
    os._exit(code)
