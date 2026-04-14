"""Straight-line max-speed test on a flat ground plane.

Spawns a single f1tenth in a plain environment (no track meshes), applies
zero steering + full throttle, logs speed vs time. Pure dynamics check.

Usage:
    OMNI_KIT_ACCEPT_EULA=YES python Itutorial/rl/straight_test.py --target 10
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--robot", type=str, default="f1tenth")
parser.add_argument("--target", type=float, default=10.0, help="Target speed (m/s)")
parser.add_argument("--seconds", type=float, default=8.0, help="Run duration in sim seconds")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--high_friction", action="store_true",
                    help="Use spawn_high_friction_ground instead of default plane")
parser.add_argument("--mu_static", type=float, default=2.5)
parser.add_argument("--mu_dynamic", type=float, default=2.2)
parser.add_argument("--capture", action="store_true", help="Save chase + side MP4")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = bool(args.capture)
launcher = AppLauncher(args)
simulation_app = launcher.app

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg  # noqa: E402
from isaaclab.scene import InteractiveSceneCfg  # noqa: E402
from isaaclab.sim import SimulationCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402

from _common import apply_ackermann, get_robot_spec, spawn_high_friction_ground  # noqa: E402


@configclass
class FlatEnvCfg(DirectRLEnvCfg):
    decimation: int = 2
    episode_length_s: float = 30.0
    sim: SimulationCfg = SimulationCfg(dt=0.005, render_interval=2, device="cuda:0")
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1, env_spacing=0.0, replicate_physics=True
    )
    action_space: int = 2
    observation_space: int = 1
    state_space: int = 0
    robot_name: str = "f1tenth"
    drive_effort_limit: float = 500.0
    drive_damping: float = 1000.0
    drive_velocity_limit: float = 400.0
    chassis_mass: float = 4.5
    chassis_com_z: float = -0.06


class FlatEnv(DirectRLEnv):
    cfg: FlatEnvCfg

    def __init__(self, cfg, render_mode=None, **kwargs):
        self._spec = get_robot_spec(cfg.robot_name)
        # 4WD: power both rear and front wheels
        acts = dict(getattr(self._spec.cfg, "actuators", {}) or {})
        acts["drive"] = ImplicitActuatorCfg(
            joint_names_expr=["rear_.*_wheel_joint"],
            effort_limit_sim=cfg.drive_effort_limit,
            velocity_limit_sim=cfg.drive_velocity_limit,
            stiffness=0.0, damping=cfg.drive_damping,
        )
        acts["front_wheels"] = ImplicitActuatorCfg(
            joint_names_expr=["front_.*_wheel_joint"],
            effort_limit_sim=cfg.drive_effort_limit,
            velocity_limit_sim=cfg.drive_velocity_limit,
            stiffness=0.0, damping=cfg.drive_damping,
        )
        self._spec.cfg = self._spec.cfg.replace(actuators=acts)
        super().__init__(cfg, render_mode, **kwargs)
        self._steer, self._drive = self._spec.resolve_joints(self.ego.joint_names)
        # 4WD: also drive front wheels via apply_ackermann's drive_ids
        names = list(self.ego.joint_names)
        front_drive = [i for i, n in enumerate(names) if "front_" in n and "_wheel_joint" in n]
        self._drive = list(self._drive) + front_drive
        # Anti-wheelie: chassis mass + CoM
        try:
            link = "base_link"
            bn = list(self.ego.body_names)
            if link in bn:
                idx = bn.index(link)
                view = self.ego.root_physx_view
                m = view.get_masses().clone(); m[:, idx] = cfg.chassis_mass
                view.set_masses(m, indices=torch.arange(self.num_envs, device="cpu"))
                c = view.get_coms().clone(); c[:, idx, 2] = cfg.chassis_com_z
                view.set_coms(c, indices=torch.arange(self.num_envs, device="cpu"))
                print(f"[setup] chassis mass={cfg.chassis_mass}kg com_z={cfg.chassis_com_z}m", flush=True)
        except Exception as exc:
            print(f"[setup] chassis tune skipped: {exc}", flush=True)

    def _setup_scene(self):
        ego_cfg = self._spec.cfg.replace(prim_path="/World/envs/env_.*/Ego")
        self.ego = Articulation(ego_cfg)
        if getattr(self.cfg, "_high_friction", False):
            mu_s = float(getattr(self.cfg, "_mu_static", 1.5))
            mu_d = float(getattr(self.cfg, "_mu_dynamic", 1.3))
            cfg_g = sim_utils.GroundPlaneCfg(
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=mu_s, dynamic_friction=mu_d, restitution=0.0,
                )
            )
            cfg_g.func("/World/ground", cfg_g)
            print(f"[setup] high-friction ground μ_s={mu_s} μ_d={mu_d}", flush=True)
        else:
            sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())

        if getattr(self.cfg, "_capture", False):
            from isaaclab.sensors import Camera
            from hmclab_isaac.utils.capture import chase_camera_cfg
            self.cam_ch = Camera(chase_camera_cfg("/World/ChaseCam", width=1280, height=720))
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["ego"] = self.ego
        sim_utils.DomeLightCfg(intensity=2000.0).func(
            "/World/Light", sim_utils.DomeLightCfg(intensity=2000.0))

    def _pre_physics_step(self, actions): self._a = actions
    def _apply_action(self):
        apply_ackermann(self.ego, self._spec,
                        steer_rad=self._a[:, 0], speed_mps=self._a[:, 1],
                        steer_ids=self._steer, drive_ids=self._drive,
                        force_steer=True)
    def _get_observations(self):
        return {"policy": torch.zeros(self.num_envs, 1, device=self.device)}
    def _get_rewards(self):
        return torch.zeros(self.num_envs, device=self.device)
    def _get_dones(self):
        trunc = self.episode_length_buf >= self.max_episode_length - 1
        return torch.zeros_like(trunc), trunc
    def _reset_idx(self, env_ids):
        if env_ids is None: env_ids = self.ego._ALL_INDICES
        super()._reset_idx(env_ids)
        st = self.ego.data.default_root_state[env_ids].clone()
        st[:, 0] = self.scene.env_origins[env_ids, 0]
        st[:, 1] = self.scene.env_origins[env_ids, 1]
        st[:, 2] = self._spec.init_z
        st[:, 3] = 1.0; st[:, 4:7] = 0.0; st[:, 7:] = 0.0
        self.ego.write_root_pose_to_sim(st[:, :7], env_ids)
        self.ego.write_root_velocity_to_sim(st[:, 7:], env_ids)
        self.ego.write_joint_state_to_sim(
            self.ego.data.default_joint_pos[env_ids],
            torch.zeros_like(self.ego.data.default_joint_pos[env_ids]),
            None, env_ids,
        )


def main() -> int:
    cfg = FlatEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.robot_name = args.robot
    cfg._high_friction = args.high_friction
    cfg._mu_static = args.mu_static
    cfg._mu_dynamic = args.mu_dynamic
    cfg._capture = args.capture
    env = FlatEnv(cfg)
    env.reset()

    # Dump body masses
    masses = env.ego.data.default_mass[0]
    body_names = env.ego.body_names
    total = 0.0
    for i, n in enumerate(body_names):
        m = float(masses[i].item())
        total += m
        print(f"  body {n:40s} {m:8.4f} kg", flush=True)
    print(f"  TOTAL_MASS = {total:.4f} kg", flush=True)

    N = env.num_envs
    action = torch.zeros(N, 2, device=env.device)
    action[:, 0] = 0.0              # steer straight
    action[:, 1] = args.target      # target speed m/s

    total_steps = int(args.seconds / (cfg.sim.dt * cfg.decimation))
    print(f"[straight] target={args.target} m/s  steps={total_steps}  dt={cfg.sim.dt*cfg.decimation}s",
          flush=True)
    samples = []
    frames = []
    # Step-response sequence: 0→10, hold, →0 (decel), →6, hold
    seg_dt = cfg.sim.dt * cfg.decimation
    schedule = [
        (0.0, args.target),    # 가속 0→target
        (2.5, 0.0),            # 감속 target→0
        (4.0, args.target * 0.7),  # 다시 부분 가속
        (6.0, 0.0),            # 정지
    ]
    for step in range(total_steps):
        t = step * seg_dt
        # find current target
        cur_tgt = schedule[0][1]
        for tt, tv in schedule:
            if t >= tt: cur_tgt = tv
        action[:, 1] = cur_tgt
        env.step(action)
        v = env.ego.data.root_lin_vel_w[0, :2].norm().item()
        samples.append((t, v))
        if step % 25 == 0:
            print(f"  t={t:5.2f}s  tgt={cur_tgt:5.2f}  v={v:6.3f} m/s", flush=True)

        if args.capture and step % 2 == 0:
            # update chase cam to follow ego
            import math as _m
            p = env.ego.data.root_pos_w[0]; q = env.ego.data.root_quat_w[0]
            w_, x_, y_, z_ = float(q[0]), float(q[1]), float(q[2]), float(q[3])
            yaw_ = _m.atan2(2*(w_*z_ + x_*y_), 1 - 2*(y_*y_ + z_*z_))
            px, py, pz = float(p[0]), float(p[1]), float(p[2])
            eye = (px - 2.5*_m.cos(yaw_), py - 2.5*_m.sin(yaw_), pz + 1.4)
            tgt = (px, py, pz + 0.2)
            try:
                env.cam_ch.set_world_poses_from_view(
                    torch.tensor([eye], device=env.cam_ch._device, dtype=torch.float32),
                    torch.tensor([tgt], device=env.cam_ch._device, dtype=torch.float32))
                env.cam_ch.update(dt=0.0, force_recompute=True)
                out = getattr(env.cam_ch.data, "output", None)
                rgb = out.get("rgb") if out else None
                if rgb is not None:
                    import numpy as _np
                    arr = rgb.detach().cpu().numpy() if hasattr(rgb, "detach") else _np.asarray(rgb)
                    if arr.ndim == 4: arr = arr[0]
                    if arr.shape[-1] == 4: arr = arr[..., :3]
                    if arr.dtype != _np.uint8: arr = _np.clip(arr, 0, 255).astype(_np.uint8)
                    frames.append(arr)
            except Exception as exc:
                print(f"[capture] {exc}", flush=True)

    # Stats
    v_arr = [s[1] for s in samples]
    import numpy as np
    v_np = np.array(v_arr)
    # time to reach 90% of max
    peak = v_np.max()
    t_peak_idx = int(np.argmax(v_np))
    t_peak = samples[t_peak_idx][0]
    # time to cross various thresholds
    def t_cross(thresh):
        idx = np.argmax(v_np >= thresh)
        if v_np[idx] < thresh: return None
        return samples[idx][0]
    thresholds = [2.0, 5.0, 8.0, 10.0]
    t_th = {th: t_cross(th) for th in thresholds}

    if args.capture and frames:
        import imageio.v2 as imageio, os as _os
        OUT_DIR = "/home/js/hmcl_issac_project/HMCLab_Isaac/Itutorial/outputs"
        _os.makedirs(OUT_DIR, exist_ok=True)
        tag = f"straight_{args.robot}_target{args.target:.0f}"
        if args.high_friction: tag += "_hf"
        path = _os.path.join(OUT_DIR, f"{tag}.mp4")
        imageio.mimsave(path, frames, fps=30, codec="libx264", macro_block_size=None)
        print(f"saved {path}", flush=True)

    print(f"STRAIGHT_OK peak={peak:.2f} t_peak={t_peak:.2f}s "
          f"final={v_np[-1]:.2f} t_reach(2,5,8,10)="
          f"{t_th[2.0]}, {t_th[5.0]}, {t_th[8.0]}, {t_th[10.0]}", flush=True)

    env.close()
    return 0


if __name__ == "__main__":
    import traceback
    try: main()
    except Exception:
        traceback.print_exc()
    os._exit(0)
