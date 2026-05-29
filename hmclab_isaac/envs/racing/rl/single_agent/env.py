"""Single-agent vision-based racing env on the UNICORN_3 platform.

Lifecycle (DirectRLEnv hooks):
    __init__          load track + FrenetField, resolve joint ids, alloc buffers
    _setup_scene      ground + per-env duct mesh + robot + tiled camera + light
    _pre_physics_step cache the policy action (clipped)
    _apply_action     map action → wheel velocity / steering position targets
    _get_observations dict {"policy": proprio, "images": rgb, "privileged": frenet}
    _get_rewards      progress + speed bonus − lateral/heading/smoothness
    _get_dones        off-track / spinout / reverse / lap-success / timeout
    _reset_idx        random track-progress spawn with lateral & yaw jitter
"""
from __future__ import annotations

import math
import re
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, ContactSensorCfg, TiledCamera

from hmclab_isaac.worlds.racing._schema import RacingTrack
from hmclab_isaac.worlds.racing.frenet import FrenetField
from hmclab_isaac.worlds.racing.duct_track import spawn_duct_track

if TYPE_CHECKING:
    from .cfg import UnicornRacingEnvCfg


_STEER_RE = re.compile(r"^(fl|fr|rl|rr)_steering$")
_DRIVE_RE = re.compile(r"^(fl|fr|rl|rr)_wheel$")


def _yaw_from_quat(q: torch.Tensor) -> torch.Tensor:
    """Extract yaw (Z-axis rotation) from (B, 4) quaternion (w, x, y, z)."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class UnicornRacingEnv(DirectRLEnv):
    """Time-trial racing env: front camera + IMU → drive/steer commands."""

    cfg: "UnicornRacingEnvCfg"

    def __init__(self, cfg: "UnicornRacingEnvCfg", render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Joint id resolution — done lazily once we know the joint_names order
        names = self._robot.joint_names
        self._steer_ids = [i for i, n in enumerate(names) if _STEER_RE.search(n)]
        self._drive_ids = [i for i, n in enumerate(names) if _DRIVE_RE.search(n)]
        if not self._drive_ids or not self._steer_ids:
            raise RuntimeError(
                f"UNICORN_3 joint regex mismatch — names={names}\n"
                f"  drive_ids={self._drive_ids} steer_ids={self._steer_ids}"
            )

        # Track + Frenet field (numpy → GPU tensors)
        track = RacingTrack.load(cfg.track_path()).densify(cfg.track_subdivide)
        self._track = track
        self._frenet = FrenetField(track, device=self.device)

        # Per-env scratch buffers
        E = self.num_envs
        d = self.device
        self._action_scaled = torch.zeros((E, 2), device=d)   # [throttle, steer] in command units
        self._action_raw = torch.zeros((E, 2), device=d)       # last raw policy action ∈ [-1, 1]
        self._global_step = 0                                  # env-steps seen (for speed curriculum)
        self._cur_max_rps = float(self.cfg.max_wheel_rps)      # current curriculum throttle scale
        self._prev_action = torch.zeros((E, 2), device=d)
        self._prev_prev_steer = torch.zeros((E,), device=d)
        self._prev_vx = torch.zeros((E,), device=d)
        self._prev_s = torch.zeros((E,), device=d)
        self._reverse_counter = torch.zeros((E,), dtype=torch.long, device=d)
        self._lap_count = torch.zeros((E,), dtype=torch.long, device=d)
        # Cumulative forward arc-length since the env's last reset
        self._forward_progress = torch.zeros((E,), device=d)
        # Stuck detection: counter increments each step the body XY speed is
        # below cfg.stuck_speed_mps and resets to 0 whenever the vehicle
        # moves. Wall collision → wheels spin, body velocity ≈ 0 → counter
        # climbs → terminate.
        self._stuck_counter = torch.zeros((E,), dtype=torch.long, device=d)
        # External override for spawn progress. If set to a (num_envs,) tensor
        # the next `_reset_idx` call will use these values instead of
        # sampling uniform random. Cleared back to None when consumed (so
        # subsequent env-internal resets revert to random spawn). Lets a
        # caller force two envs (e.g. user + bot) to spawn at the same track
        # position for fair side-by-side racing.
        self._forced_spawn_progress: torch.Tensor | None = None
        # Track each env's "starting s" (for relative progress)
        self._spawn_s = torch.zeros((E,), device=d)
        # Cached Frenet of last step (for reward + dones without recomputing)
        self._frenet_cache: dict[str, torch.Tensor] = {}
        # Terminal-reward bookkeeping: filled by _get_dones, consumed by _get_rewards
        self._terminal_reward = torch.zeros((E,), device=d)
        # Collision-penalty diagnostics (filled each step in _get_rewards;
        # read by the GUI Collision Monitor so the panel shows EXACTLY the
        # values applied to the reward — single source of truth).
        self._dbg_wall_F = torch.zeros((E,), device=d)
        self._dbg_opp_F = torch.zeros((E, 1), device=d)
        self._dbg_my_spd = torch.zeros((E,), device=d)
        self._dbg_rel = torch.zeros((E, 1), device=d)
        self._dbg_wall_pen = torch.zeros((E,), device=d)
        self._dbg_opp_pen = torch.zeros((E,), device=d)
        # Cumulative collision counters (monotonic). A trainer reads these
        # at log time, diffs against its own snapshot, and reports the
        # interval collision rate. Counts (env,step) pairs with a contact.
        self._coll_steps = 0      # total env-steps seen
        self._coll_n_wall = 0     # env-steps with wall contact
        self._coll_n_opp = 0      # env-steps with opponent contact
        self._coll_n_any = 0      # env-steps with any collision

        # ── Phase-2 opponents (N scripted pure-pursuit cars) ──
        # NOTE: self._opponents (list of Articulation) is created in
        # _setup_scene (called from super().__init__ ABOVE). Don't reset it
        # here. Only the shared PP controller + per-opponent scratch buffers.
        self._opp_pp = None
        self._opp_count = int(getattr(cfg, "opponent_count", 1)) \
            if getattr(cfg, "opponent_enabled", False) else 0
        # scaled action buffer per opponent: (count, E, 2)
        self._opp_action_scaled = torch.zeros(
            (max(self._opp_count, 1), E, 2), device=d
        )
        if getattr(cfg, "opponent_enabled", False):
            from hmclab_isaac.algos.pure_pursuit import (
                CenterlinePurePursuit, PP_PRESETS,
            )
            slow_preset = PP_PRESETS.get(
                cfg.opponent_preset, PP_PRESETS["pp_slow"]
            )
            fast_preset = PP_PRESETS.get("pp_fast", slow_preset)
            K = max(self._opp_count, 1)
            n_fast = max(0, min(int(getattr(cfg, "opponent_fast_count", 0)), K))
            spread = float(getattr(cfg, "opponent_lateral_spread", 0.0))
            w_amp = float(getattr(cfg, "opponent_weave_amp", 0.0))
            w_len = float(getattr(cfg, "opponent_weave_wavelength", 18.0))
            jit = float(getattr(cfg, "opponent_speed_jitter", 0.0))
            # One PP controller per opponent, each with a distinct line and
            # pace so the fleet doesn't move as one centerline blob. The
            # last `n_fast` opponents use the pp_fast preset → mixed traffic.
            self._opp_pp = []
            for k in range(K):
                frac = (2.0 * k / (K - 1) - 1.0) if K > 1 else 0.0  # -1..+1
                preset = fast_preset if k >= K - n_fast else slow_preset
                self._opp_pp.append(
                    CenterlinePurePursuit(
                        self._frenet,
                        wheelbase=0.344,
                        max_steer_rad=cfg.max_steer_rad,
                        max_wheel_rps=cfg.max_wheel_rps,
                        wheel_radius=0.0525,
                        lateral_offset=spread * frac,
                        weave_amp=w_amp * (0.6 + 0.4 * ((k % 3) / 2.0)),
                        weave_wavelength=w_len,
                        weave_phase=2.0 * math.pi * k / K,
                        speed_scale=1.0 + jit * frac,
                        **preset,
                    )
                )

    # ------------------------------------------------------------------
    # Scene
    # ------------------------------------------------------------------
    def _setup_scene(self):
        from hmclab_isaac.robots.vehicles.UNICORN import UNICORN_3

        collide = bool(getattr(self.cfg, "collision_penalty_enabled", False))

        # 1. Robot (learner)
        robot_cfg = UNICORN_3.CFG.replace(prim_path="/World/envs/env_.*/Robot")
        if collide:
            # PhysX contact-report API must be baked at spawn for the
            # ContactSensor to receive forces on base_link.
            robot_cfg.spawn.activate_contact_sensors = True
        self._robot = Articulation(robot_cfg)

        # 1.5. Opponent robots (N scripted PP cars). Same USD, separate prims
        # /Opponent_0../Opponent_{N-1} → same env physics scene so they CAN
        # collide with the learner and each other. No cameras (PP is
        # state-based); the learner sees them through its own camera.
        self._opponents: list = []
        if self.cfg.opponent_enabled:
            for k in range(int(self.cfg.opponent_count)):
                ocfg = UNICORN_3.CFG.replace(
                    prim_path=f"/World/envs/env_.*/Opponent_{k}"
                )
                if collide:
                    # Filtered contact (force_matrix_w) needs the opponent
                    # body to emit contact reports too.
                    ocfg.spawn.activate_contact_sensors = True
                self._opponents.append(Articulation(ocfg))

        # 2. Tiled camera (per-env render product, batched on GPU)
        cam_cfg = UNICORN_3.make_tiled_camera(
            robot_prim_path="/World/envs/env_.*/Robot",
            rate_hz=self.cfg.cam_rate_hz,
            width=self.cfg.cam_width,
            height=self.cfg.cam_height,
        )
        self._camera = TiledCamera(cam_cfg)

        # 2.5. Optional high-res recording camera — same mount/intrinsics
        # as the policy camera but at a higher resolution. Used only for
        # MP4 dumps; the actor still consumes self._camera.
        self._record_cam = None
        if getattr(self.cfg, "record_cam_enabled", False):
            from isaaclab.sensors import TiledCameraCfg
            rec_cfg = TiledCameraCfg(
                prim_path="/World/envs/env_.*/Robot/base_link/cam/hd_cam",
                update_period=1.0 / float(self.cfg.cam_rate_hz),
                height=int(self.cfg.record_cam_height),
                width=int(self.cfg.record_cam_width),
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=UNICORN_3.CAMERA_FOCAL_LENGTH,
                    horizontal_aperture=UNICORN_3.CAMERA_HORIZONTAL_APERTURE,
                    clipping_range=UNICORN_3.CAMERA_CLIPPING_RANGE,
                ),
                offset=TiledCameraCfg.OffsetCfg(
                    pos=(0.0, 0.0, 0.0),
                    rot=(1.0, 0.0, 0.0, 0.0),
                    convention="world",
                ),
            )
            self._record_cam = TiledCamera(rec_cfg)

        # 2.7. Contact sensor on the chassis. base_link is held off the
        # ground by the wheels, so its only contacts are duct walls and
        # other cars. filter_prim_paths_expr lists each opponent's base_link
        # → force_matrix_w isolates car-vs-car force; net_forces_w minus that
        # is the wall force. Ground never reaches base_link, so it drops out.
        self._contact = None
        # Each opponent contributes MULTIPLE collision bodies (chassis hull +
        # 4 wheel cylinders). If we filtered only the opponent's base_link,
        # a learner-chassis-vs-opponent-WHEEL contact would land in
        # net_forces_w but NOT in force_matrix_w, and the wall = net − Σ(opp)
        # subtraction would misattribute that car-vs-car force as a wall hit.
        # So filter every opponent rigid body that can realistically touch
        # the learner, and remember how many columns belong to each opponent.
        # EVERY opponent rigid body that carries a collider (verified from
        # the UNICORN_3 USD: chassis hull + 4 wheel cylinders + 4 knuckle
        # convex hulls + 4 upper-arm convex hulls = 13). If any collider is
        # missing from this list, a learner-chassis-vs-that-body contact
        # leaks into net_forces_w but not force_matrix_w and the
        # wall = net − Σ(opp) subtraction misreads it as a phantom wall hit.
        self._opp_filter_bodies = [
            "base_link",
            "fl_wheel", "fr_wheel", "rl_wheel", "rr_wheel",
            "fl_knuckle", "fr_knuckle", "rl_knuckle", "rr_knuckle",
            "fl_upper_arm", "fr_upper_arm", "rl_upper_arm", "rr_upper_arm",
        ]
        self._opp_nbody = len(self._opp_filter_bodies)

        # 3. Per-env duct track. We spawn it under env_0; clone_environments
        # then copies the prim subtree into env_1...env_N. With collision
        # penalty on, the duct meshes are promoted to KINEMATIC rigid
        # bodies so the ContactSensor can filter the wall directly via
        # force_matrix (no net−Σopp subtraction, no XY ground trick).
        track = RacingTrack.load(self.cfg.track_path()).densify(self.cfg.track_subdivide)
        pipe_offset = self.cfg.track_pipe_offset
        if pipe_offset <= 0:
            pipe_offset = float(track.widths.mean())
        track_res = spawn_duct_track(
            track,
            "/World/envs/env_0/track",
            pipe_radius=self.cfg.track_pipe_radius,
            pipe_offset=pipe_offset,
            rib_spacing=self.cfg.track_rib_spacing,
            duct_color=(1.0, 0.5, 0.0),
            rib_color=(0.05, 0.05, 0.05),
            make_rigid=collide,
        )
        self._track_filter_names = list(track_res.get("rigid_mesh_names", []))

        # 3.2. Static obstacle boxes at random on-track positions. Spawned
        # under env_0/boxes and cloned. They share the duct's collision
        # filter group, so hitting a box incurs the wall-type penalty and
        # the vision policy must learn to steer around them.
        self._box_filter_names: list = []
        self._boxes: list = []
        self._box_size = float(getattr(self.cfg, "static_box_size", 0.5))
        n_box = int(getattr(self.cfg, "static_box_count", 0))
        if collide and n_box > 0:
            from hmclab_isaac.worlds.racing.duct_track import spawn_static_boxes
            fr_tmp = FrenetField(track, device=self.device)
            self._box_filter_names = spawn_static_boxes(
                "/World/envs/env_0/boxes",
                fr_tmp,
                n_box,
                size=self._box_size,
                seed=int(getattr(self.cfg, "static_box_seed", 0)),
                random_color=True,
                side=str(getattr(self.cfg, "static_box_side", "both")),
            )
            # Wrap each kinematic box in a RigidObject so its pose can be
            # re-randomised every episode (per env) in _reset_idx. spawn=None
            # → attach to the already-authored prim.
            for nm in self._box_filter_names:
                bcfg = RigidObjectCfg(
                    prim_path=f"/World/envs/env_.*/boxes/{nm}",
                    spawn=None,
                )
                self._boxes.append(RigidObject(bcfg))
            print(f"[racing-env] spawned {len(self._box_filter_names)} "
                  f"static obstacle box(es) (per-episode random pose+color)",
                  flush=True)

        # 3.5. Contact sensor on the CHASSIS (base_link, single body) with a
        # TWO-group filter:
        #   [ opponent rigid bodies ... ][ track + box obstacle meshes ... ]
        # force_matrix_w then gives car-vs-opp and car-vs-wall DIRECTLY,
        # each as its own column block. Ground is not a filter prim → it is
        # excluded automatically.
        # NOTE: the sensor MUST track exactly one body. PhysX's contact
        # force-matrix aligns each filter prim against the sensor bodies;
        # with N>1 sensor bodies the filter resolves to num_envs entries
        # but PhysX expects num_envs*N, so force_matrix breaks. Detecting a
        # wheel-vs-wall graze therefore needs separate per-wheel sensors
        # (deferred); base_link covers the dominant chassis-vs-wall contact
        # and any hard hit still trips the spin/stuck terminations.
        self._n_opp_cols = 0
        self._n_trk_cols = 0
        if collide:
            filt = []
            if self.cfg.opponent_enabled:
                for k in range(int(self.cfg.opponent_count)):
                    for b in self._opp_filter_bodies:
                        filt.append(f"/World/envs/env_.*/Opponent_{k}/{b}")
            self._n_opp_cols = len(filt)
            for nm in self._track_filter_names:
                filt.append(f"/World/envs/env_.*/track/{nm}")
            for nm in self._box_filter_names:
                filt.append(f"/World/envs/env_.*/boxes/{nm}")
            # Boxes share the wall penalty group → n_trk_cols counts both.
            self._n_trk_cols = (
                len(self._track_filter_names) + len(self._box_filter_names)
            )
            cs_cfg = ContactSensorCfg(
                prim_path="/World/envs/env_.*/Robot/base_link",
                update_period=0.0,          # sample every physics step
                history_length=0,
                track_air_time=False,
                filter_prim_paths_expr=filt,
            )
            self._contact = ContactSensor(cs_cfg)

        # 4. Per-env ground plane. The default grid asset is centered at
        # origin and 100 m wide, which leaves most of the env grid on
        # nothing. Spawn one ground per env (cloned with the rest of env_0)
        # sized to env_spacing so every env sits on a textured floor.
        # When env_spacing is 0 (ghost-mode racing — multiple envs sharing
        # the same origin), fall back to a generous 100 m × 100 m plane.
        # Manual colored ground mesh (replaces the textured GroundPlane so the
        # visual color is controllable via UsdPreviewSurface.diffuseColor and the
        # domain-randomizer can recolor it). Static collider + UsdPhysics
        # material for friction (bound at "physics" purpose).
        ground_size = max(float(self.cfg.scene.env_spacing), 100.0)
        self._spawn_colored_ground(
            "/World/envs/env_0/ground",
            size=ground_size,
            static_friction=self.cfg.ground_static_friction,
            dynamic_friction=self.cfg.ground_dynamic_friction,
        )

        # 5. Clone (templated /World/envs/env_.* prims propagate)
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        # 6. Domain randomization: collect material/friction handles + initial sample.
        self._setup_domain_randomization()

        # 6. Register
        self.scene.articulations["robot"] = self._robot
        for k, opp in enumerate(self._opponents):
            self.scene.articulations[f"opponent_{k}"] = opp
        self.scene.sensors["camera"] = self._camera
        if self._record_cam is not None:
            self.scene.sensors["record_cam"] = self._record_cam
        if self._contact is not None:
            self.scene.sensors["contact"] = self._contact
        for k, bx in enumerate(self._boxes):
            self.scene.rigid_objects[f"box_{k}"] = bx

        # 7. Dome light
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.95, 0.95, 0.95))
        light_cfg.func("/World/Light", light_cfg)

        # 8. Deactivate any vehicle-baked Camera prims (e.g. the SG2 prim
        # baked into UNICORN_3.usd). With them active Hydra/Replicator would
        # register *two* render products per env (SG2 + our front_cam),
        # doubling the GPU cost per env step. We keep the SG2 USD intact
        # but flip the prim's active flag off — Isaac Sim removes it from
        # composition until reactivated.
        import omni.usd
        stage = omni.usd.get_context().get_stage()
        disabled = 0
        for prim in stage.Traverse():
            if prim.GetTypeName() != "Camera":
                continue
            path = prim.GetPath().pathString
            if not (path.startswith("/World/envs/")
                    and ("/Robot/" in path or "/Opponent_" in path)):
                continue
            # Spare our front_cam (we authored that one ourselves)
            if path.endswith(f"/Robot/base_link/cam/{UNICORN_3.CAMERA_PRIM_NAME}"):
                continue
            prim.SetActive(False)
            disabled += 1
        if disabled:
            print(f"[racing-env] disabled {disabled} vehicle-baked Camera prim(s)",
                  flush=True)

    # ------------------------------------------------------------------
    # Colored ground (replaces sim_utils.GroundPlaneCfg's grid texture so the
    # visual color can be recolored by the domain randomizer). Static collider
    # + UsdPhysics MaterialAPI for friction bound at "physics" purpose.
    # ------------------------------------------------------------------
    def _spawn_colored_ground(self, prim_path: str, size: float,
                              static_friction: float, dynamic_friction: float) -> None:
        import omni.usd
        from pxr import UsdGeom, UsdShade, UsdPhysics, Gf, Sdf, Vt
        stage = omni.usd.get_context().get_stage()
        half = float(size) * 0.5
        # 2-triangle quad at z=0
        mesh = UsdGeom.Mesh.Define(stage, prim_path)
        mesh.GetPointsAttr().Set(Vt.Vec3fArray([
            Gf.Vec3f(-half, -half, 0.0), Gf.Vec3f(half, -half, 0.0),
            Gf.Vec3f(half,  half,  0.0), Gf.Vec3f(-half, half, 0.0),
        ]))
        mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray([3, 3]))
        mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray([0, 1, 2, 0, 2, 3]))
        mesh.CreateSubdivisionSchemeAttr().Set("none")
        # Static collider (no rigid body → immovable, like the original GroundPlane)
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr().Set("none")
        # Physics material (friction) — bound at "physics" purpose
        pm = UsdShade.Material.Define(stage, prim_path + "_PhysMat")
        pmAPI = UsdPhysics.MaterialAPI.Apply(pm.GetPrim())
        pmAPI.CreateStaticFrictionAttr(float(static_friction))
        pmAPI.CreateDynamicFrictionAttr(float(dynamic_friction))
        pmAPI.CreateRestitutionAttr(0.0)
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(
            pm, UsdShade.Tokens.weakerThanDescendants, "physics"
        )
        # Visual material (UsdPreviewSurface, no texture — DR target)
        vm = UsdShade.Material.Define(stage, prim_path + "_Mat")
        sh = UsdShade.Shader.Define(stage, prim_path + "_Mat/Shader")
        sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.3, 0.3, 0.3))
        sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)
        vm.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(vm)   # default purpose = visual

    # ------------------------------------------------------------------
    # Domain randomization (temporal): resample track/ground color + friction
    # every dr_resample_steps env-steps. Updates the env-0 source materials,
    # which the cloned envs reference, so all envs change together.
    # ------------------------------------------------------------------
    def _setup_domain_randomization(self) -> None:
        self._dr = None
        if not bool(getattr(self.cfg, "dr_enabled", False)):
            return
        import omni.usd
        from pxr import Usd, UsdShade, UsdPhysics, PhysxSchema
        stage = omni.usd.get_context().get_stage()
        E = self.num_envs

        def _shaders(suffix):
            out = []
            for k in range(E):
                pr = stage.GetPrimAtPath(f"/World/envs/env_{k}/track/{suffix}_Mat/Shader")
                if pr.IsValid():
                    out.append(UsdShade.Shader(pr))
            return out

        duct_sh, rib_sh = _shaders("duct"), _shaders("ribs")
        ground_sh, phys_mats = [], []
        for k in range(E):
            sh_pr = stage.GetPrimAtPath(f"/World/envs/env_{k}/ground_Mat/Shader")
            if sh_pr.IsValid():
                ground_sh.append(UsdShade.Shader(sh_pr))
            pm_pr = stage.GetPrimAtPath(f"/World/envs/env_{k}/ground_PhysMat")
            if pm_pr.IsValid():
                phys_mats.append(pm_pr)

        self._dr = {"duct": duct_sh, "rib": rib_sh, "ground": ground_sh, "phys": phys_mats}
        self._dr_last = -10 ** 9
        print(f"[racing-env][DR] handles: duct={len(duct_sh)} rib={len(rib_sh)} "
              f"ground_shaders={len(ground_sh)} phys_mats={len(phys_mats)}", flush=True)
        self._resample_domain(force=True)

    def _resample_domain(self, force: bool = False) -> None:
        if self._dr is None:
            return
        gstep = getattr(self, "_global_step", 0)   # not yet set during _setup_scene
        if not force and (gstep - self._dr_last) < int(self.cfg.dr_resample_steps):
            return
        self._dr_last = gstep
        import numpy as np
        from pxr import Gf, Sdf
        rng = np.random.default_rng()

        def _set_color(shaders, c):
            for sh in shaders:
                inp = sh.GetInput("diffuseColor")
                if not inp:
                    inp = sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)
                inp.Set(Gf.Vec3f(float(c[0]), float(c[1]), float(c[2])))

        if self.cfg.randomize_track_color:
            _set_color(self._dr["duct"], rng.uniform(0.1, 1.0, 3))
            _set_color(self._dr["rib"], rng.uniform(0.0, 0.3, 3))
        if self.cfg.randomize_ground_color:
            _set_color(self._dr["ground"], rng.uniform(0.15, 0.85, 3))
        if self.cfg.randomize_friction:
            f = float(rng.uniform(1.0 - self.cfg.friction_noise_frac, 1.0 + self.cfg.friction_noise_frac))
            s = max(0.2, float(self.cfg.ground_static_friction) * f)
            d = max(0.2, float(self.cfg.ground_dynamic_friction) * f)
            applied = []
            for pr in self._dr["phys"]:
                for attr_name, val in (("physics:staticFriction", s), ("physics:dynamicFriction", d)):
                    a = pr.GetAttribute(attr_name)
                    if a and a.IsValid():
                        a.Set(val)
                # read-back verification (catches silent skips if attr name is wrong)
                rs = pr.GetAttribute("physics:staticFriction")
                rd = pr.GetAttribute("physics:dynamicFriction")
                applied.append((
                    float(rs.Get()) if rs and rs.IsValid() else None,
                    float(rd.Get()) if rd and rd.IsValid() else None,
                ))
            if force or not getattr(self, "_dr_friction_verified", False):
                print(f"[racing-env][DR] friction set: target s={s:.3f} d={d:.3f} | "
                      f"read-back per phys_mat: {applied}", flush=True)
                self._dr_friction_verified = True

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # Per-dim clip: throttle ∈ [0, 1] (no reverse), steer ∈ [-1, 1]. Even
        # though the network may emit values outside, we clip here so the
        # vehicle physically cannot back up.
        # Speed curriculum: ramp the throttle scale from start_rps → max_wheel_rps
        # over the first speed_curriculum_steps env-steps, then hold. Lets the
        # policy learn cornering at low speed first and transfer to full speed.
        self._global_step += self.num_envs
        if bool(getattr(self.cfg, "dr_enabled", False)):
            self._resample_domain()
        if getattr(self.cfg, "speed_curriculum_enabled", False):
            s = float(self.cfg.speed_curriculum_start_rps)
            e = float(self.cfg.max_wheel_rps)
            frac = min(1.0, self._global_step / max(1, int(self.cfg.speed_curriculum_steps)))
            self._cur_max_rps = s + (e - s) * frac
        else:
            self._cur_max_rps = float(self.cfg.max_wheel_rps)

        throttle = actions[:, 0].clamp(0.0, 1.0)
        steer    = actions[:, 1].clamp(-1.0, 1.0)
        self._action_raw = torch.stack([throttle, steer], dim=-1)
        # scaled command units (learner uses the curriculum-ramped throttle scale)
        self._action_scaled[:, 0] = throttle * self._cur_max_rps
        self._action_scaled[:, 1] = steer    * self.cfg.max_steer_rad

        # ── Opponent pure-pursuit actions (N cars, shared PP controller) ──
        for k, opp in enumerate(self._opponents):
            oxy = opp.data.root_pos_w[:, :2] - self.scene.env_origins[:, :2]
            oyaw = _yaw_from_quat(opp.data.root_quat_w)
            ospd = opp.data.root_lin_vel_b[:, 0]
            opp_a = self._opp_pp[k].step(oxy, oyaw, speed=ospd)   # (E, 2)
            self._opp_action_scaled[k, :, 0] = opp_a[:, 0] * self.cfg.max_wheel_rps
            self._opp_action_scaled[k, :, 1] = opp_a[:, 1] * self.cfg.max_steer_rad

    def _apply_action(self) -> None:
        E = self.num_envs
        drive_target = self._action_scaled[:, 0:1].expand(E, len(self._drive_ids)).contiguous()
        steer_target = self._action_scaled[:, 1:2].expand(E, len(self._steer_ids)).contiguous()
        self._robot.set_joint_velocity_target(drive_target, joint_ids=self._drive_ids)
        self._robot.set_joint_position_target(steer_target, joint_ids=self._steer_ids)
        for k, opp in enumerate(self._opponents):
            od = self._opp_action_scaled[k, :, 0:1].expand(E, len(self._drive_ids)).contiguous()
            os_ = self._opp_action_scaled[k, :, 1:2].expand(E, len(self._steer_ids)).contiguous()
            opp.set_joint_velocity_target(od, joint_ids=self._drive_ids)
            opp.set_joint_position_target(os_, joint_ids=self._steer_ids)

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------
    def _compute_frenet(self) -> dict[str, torch.Tensor]:
        """Project root XY onto centerline; cache for reuse within a step."""
        root_pos = self._robot.data.root_pos_w
        xy = root_pos[:, :2] - self.scene.env_origins[:, :2]  # track frame, env-local
        yaw = _yaw_from_quat(self._robot.data.root_quat_w)
        return self._frenet.query(xy, yaw)

    def _get_observations(self) -> dict:
        # Note: _get_dones() and _get_rewards() also need Frenet at the current
        # step. DirectRLEnv.step order is: dones → rewards → reset → observations.
        # We compute Frenet fresh here (post-reset for any reset envs) so the
        # obs sent to the policy is consistent with the env state.
        f = self._compute_frenet()
        self._frenet_cache = f

        # --- image (B, H, W, C) → (B, C, H, W), optionally normalized
        rgb = self._camera.data.output["rgb"]   # uint8 or float32, (B, H, W, 3)
        if rgb.dtype == torch.uint8:
            img = rgb.float()
        else:
            img = rgb
        if self.cfg.cam_image_normalize:
            img = img / 255.0
        # channel-first for rsl_rl CNN
        img = img.permute(0, 3, 1, 2).contiguous()

        # --- proprioception (6 dims) — sim2real-friendly: only IMU-derivable
        # quantities + the controller's own action history. Drops vy_body
        # (sideslip), wheel ω, roll/pitch, contact flag, AND steering-joint
        # encoder reading. The policy gets to infer its own steering state
        # from the last commanded steer value.
        v_b = self._robot.data.root_lin_vel_b           # (E, 3)
        w_b = self._robot.data.root_ang_vel_b           # (E, 3)
        # ax = finite diff of vx
        ax = v_b[:, 0] - self._prev_vx

        proprio = torch.stack(
            [
                v_b[:, 0],                         # vx_body
                ax,                                # ax (Δvx since last step)
                w_b[:, 2],                         # yaw rate
                self._action_raw[:, 0],            # last throttle command
                self._action_raw[:, 1],            # last steer command
                self._prev_prev_steer,             # steer command one step before
            ],
            dim=-1,
        )
        assert proprio.shape == (self.num_envs, 6)

        # --- privileged (11 dims) for asymmetric critic
        # Lateral wall distances assuming d_signed positive = left of centerline
        # → left wall is (d_left - d_signed), right wall is (d_right + d_signed)
        priv = torch.stack(
            [
                f["s_norm"],
                f["d_signed"],
                f["psi_err"],
                f["d_left"] - f["d_signed"],
                f["d_right"] + f["d_signed"],
                f["kappa_lookahead"][:, 0],
                f["kappa_lookahead"][:, 1],
                f["kappa_lookahead"][:, 2],
                f["kappa_lookahead"][:, 3],
                f["kappa_lookahead"][:, 4],
                f["d_left"] + f["d_right"],     # total track width
            ],
            dim=-1,
        )

        return {
            "policy": proprio,
            "images": img,
            "privileged": priv,
        }

    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------
    def _get_rewards(self) -> torch.Tensor:
        # _get_dones populated _frenet_cache for this step; reuse it.
        f = self._frenet_cache
        s = f["s"]
        d_signed = f["d_signed"]
        psi_err = f["psi_err"]

        # Progress: forward arc-length traveled
        ds = self._frenet.lap_delta(s, self._prev_s)
        ds = ds.clamp(-0.5, 0.5)   # cap insane teleports (during reset, etc.)

        # Forward speed component along track tangent
        v_b = self._robot.data.root_lin_vel_b
        v_par = v_b[:, 0] * torch.cos(psi_err) + v_b[:, 1] * torch.sin(psi_err) * 0.0  # body-x ≈ along heading
        # Simpler: project body velocity onto tangent in world frame
        yaw = _yaw_from_quat(self._robot.data.root_quat_w)
        v_world_x = v_b[:, 0] * torch.cos(yaw) - v_b[:, 1] * torch.sin(yaw)
        v_world_y = v_b[:, 0] * torch.sin(yaw) + v_b[:, 1] * torch.cos(yaw)
        tang_angle = f["tangent_angle"]
        v_par = v_world_x * torch.cos(tang_angle) + v_world_y * torch.sin(tang_angle)

        # Smoothness — diff in [-1, 1] action space
        d_steer = (self._action_raw[:, 1] - self._prev_action[:, 1]).abs()
        d_throttle = (self._action_raw[:, 0] - self._prev_action[:, 0]).abs()

        c = self.cfg
        # Simplified reward — v_par (signed) is the core signal. Backward
        # motion ⇒ v_par < 0 ⇒ negative reward, no separate reverse penalty.
        # Other terms exist for backwards-compat with sweep experiments but
        # default cfg sets their weights to 0.
        r = (
            c.w_progress * ds
            + c.w_speed * v_par
            - c.w_lat_err * d_signed * d_signed
            - c.w_heading_err * psi_err * psi_err
            - c.w_steer_smooth * d_steer
            - c.w_throttle_smooth * d_throttle
            + c.r_alive
        )
        # ── Contact-based collision penalty ───────────────────────────────
        # Wall:  −w_wall · |my planar speed|        per step the chassis
        #        touches a duct wall (force_matrix track-mesh block).
        # Opp :  −w_opp  · |relative speed vs opp k| per step the chassis
        #        touches opponent k (force_matrix opponent block).
        # Both are read DIRECTLY from force_matrix_w (duct meshes are
        # kinematic rigid bodies). Ground is not in the filter → excluded.
        if self._contact is not None:
            # Sensor matches several car bodies (chassis + 4 wheels).
            # Sum over that body axis → total force on the whole car.
            net_v = self._contact.data.net_forces_w.sum(dim=1)   # (E, 3)
            thr = self.cfg.contact_force_threshold
            v_w = self._robot.data.root_lin_vel_w[:, :2]          # (E, 2)
            my_speed = v_w.norm(dim=-1)                           # (E,)
            E = self.num_envs
            K = len(self._opponents)
            P = self._opp_nbody

            n_opp = self._n_opp_cols
            n_trk = self._n_trk_cols
            fm = getattr(self._contact.data, "force_matrix_w", None)
            if fm is not None and fm.numel() > 0:
                fm = fm.sum(dim=1)                                # (E, M, 3)
                M = fm.shape[1]
                if M == n_opp + n_trk and n_opp == K * P and n_trk > 0:
                    # ── Option B: DIRECT two-group split ──
                    # Opponent block + track-wall block, each read straight
                    # from force_matrix. No net−Σ subtraction, no XY trick.
                    # Ground isn't in the filter so it drops out for free.
                    opp_block = (
                        fm[:, :n_opp].view(E, K, P, 3).sum(dim=2)  # (E,K,3)
                    )
                    trk_block = fm[:, n_opp:n_opp + n_trk].sum(dim=1)  # (E,3)
                    per_opp = opp_block
                    opp_mag = per_opp.norm(dim=-1)                 # (E, K)
                    wall_v = trk_block                             # (E, 3)
                    wall_F = wall_v.norm(dim=-1)                   # direct
                else:
                    # ── Fallback: legacy net−Σ(opp) with XY ground reject ──
                    if M == K * P:
                        per_opp = fm.view(E, K, P, 3).sum(dim=2)
                    else:
                        per_opp = fm.sum(dim=1, keepdim=True)
                    opp_mag = per_opp.norm(dim=-1)
                    wall_v = net_v - per_opp.sum(dim=1)
                    wall_F = wall_v[:, :2].norm(dim=-1)
            else:
                per_opp = None
                opp_mag = None
                wall_F = net_v[:, :2].norm(dim=-1)

            hit_wall = (wall_F > thr).float()
            wall_pen = self.cfg.w_collide_wall * my_speed * hit_wall
            r = r - wall_pen

            opp_pen_tot = torch.zeros_like(r)
            rel_dbg = torch.zeros((E, max(K, 1)), device=self.device)
            if opp_mag is not None:
                n_grp = opp_mag.shape[1]
                for k, opp in enumerate(self._opponents):
                    if k >= n_grp:
                        break
                    ov = opp.data.root_lin_vel_w[:, :2]
                    rel = (v_w - ov).norm(dim=-1)                 # (E,)
                    rel_dbg[:, k] = rel
                    hit_k = (opp_mag[:, k] > thr).float()
                    opp_pen_tot = opp_pen_tot + (
                        self.cfg.w_collide_opp * rel * hit_k
                    )
                r = r - opp_pen_tot

            # Single source of truth for the live diagnostic panel: store
            # exactly what was applied this step (env-0 readable downstream).
            self._dbg_wall_F = wall_F
            self._dbg_opp_F = (
                opp_mag if opp_mag is not None
                else torch.zeros((E, 1), device=self.device)
            )
            self._dbg_my_spd = my_speed
            self._dbg_rel = rel_dbg
            self._dbg_wall_pen = wall_pen
            self._dbg_opp_pen = opp_pen_tot

            # Cumulative collision tally for periodic rate reporting.
            wall_hit = wall_pen > 0
            opp_hit = opp_pen_tot > 0
            self._coll_steps += int(self.num_envs)
            self._coll_n_wall += int(wall_hit.sum().item())
            self._coll_n_opp += int(opp_hit.sum().item())
            self._coll_n_any += int((wall_hit | opp_hit).sum().item())

        r = r + self._terminal_reward
        # Reset the terminal-reward latch so it only fires once per terminal step
        self._terminal_reward = torch.zeros_like(self._terminal_reward)
        # Cache for next-step deltas. ORDER MATTERS:
        # prev_prev_steer must capture the OLD _prev_action steer BEFORE we
        # overwrite _prev_action with the just-applied action. Previously
        # these two lines were swapped, so prev_prev_steer ended up equal to
        # the current step's steer (always identical to "prev steer").
        self._prev_prev_steer = self._prev_action[:, 1].clone()
        self._prev_action = self._action_raw.clone()
        self._prev_vx = v_b[:, 0].clone()
        self._prev_s = s.clone()
        return r

    # ------------------------------------------------------------------
    # Dones
    # ------------------------------------------------------------------
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # First hook after physics — compute Frenet & cache for _get_rewards.
        f = self._compute_frenet()
        self._frenet_cache = f
        s = f["s"]
        d_signed = f["d_signed"]
        d_left = f["d_left"]
        d_right = f["d_right"]
        psi_err = f["psi_err"]

        # NOTE: off-track termination was removed at user request — leaving
        # only stuck / spin / reverse / lap_success / timeout. A vehicle that
        # pokes through a wall stays in the episode until `stuck` (≥1 s of
        # body speed < 0.2 m/s) catches it.

        w_b = self._robot.data.root_ang_vel_b
        spin = w_b[:, 2].abs() > self.cfg.spin_yawrate_rad
        if not self.cfg.spin_termination_enabled:
            spin = torch.zeros_like(spin)

        v_b = self._robot.data.root_lin_vel_b
        going_backward = v_b[:, 0] < self.cfg.reverse_speed_mps
        self._reverse_counter = torch.where(
            going_backward, self._reverse_counter + 1, torch.zeros_like(self._reverse_counter)
        )
        rev_done = self._reverse_counter >= self.cfg.reverse_duration_steps

        # Stuck detection — body XY speed below threshold for N consecutive
        # steps after a brief grace period. Body velocity stays low whenever
        # the chassis is wedged against a duct wall, regardless of whether
        # the wheels are spinning.
        v_w = self._robot.data.root_lin_vel_w
        body_speed_xy = (v_w[:, 0] * v_w[:, 0] + v_w[:, 1] * v_w[:, 1]).sqrt()
        low_speed = body_speed_xy < self.cfg.stuck_speed_mps
        self._stuck_counter = torch.where(
            low_speed, self._stuck_counter + 1, torch.zeros_like(self._stuck_counter)
        )
        stuck = (
            (self.episode_length_buf > self.cfg.stuck_grace_steps)
            & (self._stuck_counter >= self.cfg.stuck_duration_steps)
        )

        # ds_step_pos (re-)used below for lap counting
        ds_step_pos = self._frenet.lap_delta(s, self._prev_s).clamp_min(0.0)

        # Lap detection: integrate forward arc-length since spawn.
        # Each step's positive ds adds to _forward_progress; when the cumulative
        # crosses (lap_count+1)*total_length we bump lap_count. This is robust
        # against the centered s wrap-around and avoids the spurious lap-success
        # triggers a "near spawn" proxy produces for a stationary car.
        total_len = self._frenet.total_length
        self._forward_progress = self._forward_progress + ds_step_pos
        lap_threshold = (self._lap_count + 1).float() * total_len
        lap_event = self._forward_progress >= lap_threshold
        self._lap_count = torch.where(lap_event, self._lap_count + 1, self._lap_count)
        lap_success = self._lap_count >= self.cfg.target_laps

        terminated = spin | rev_done | stuck | lap_success
        truncated = self.episode_length_buf >= self.max_episode_length - 1

        # Pre-stage the terminal reward for next _get_rewards call within the
        # same step. Off-track/spin/reverse → big negative; lap_success → +
        # bonus. Both are signaled exactly once because _get_dones runs
        # before _get_rewards inside DirectRLEnv.step.
        t = torch.zeros_like(self._terminal_reward)
        bad = spin | rev_done | stuck
        t = torch.where(bad, torch.full_like(t, self.cfg.r_terminal_offtrack), t)
        t = torch.where(lap_success, torch.full_like(t, self.cfg.r_terminal_lap), t)
        self._terminal_reward = t

        return terminated, truncated

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        super()._reset_idx(env_ids)

        n = env_ids.numel()
        if n == 0:
            return

        # Random progress along track — overridable via _forced_spawn_progress
        if self._forced_spawn_progress is not None:
            progress = self._forced_spawn_progress[env_ids].clamp(0.0, 0.99999)
        else:
            progress = torch.rand((n,), device=self.device)
        idx = (progress * self._frenet.num_points).long() % self._frenet.num_points

        spawn_xy = self._frenet.cl_xy[idx]                  # (n, 2)
        spawn_tang_yaw = self._frenet.tang_angle[idx]       # (n,)

        # Lateral jitter (perp to tangent, inside walls)
        # Use a fraction of min(d_left, d_right) at the spawn point.
        d_left = self._frenet.cl_widths[idx, 0]
        d_right = self._frenet.cl_widths[idx, 1]
        max_lat = torch.minimum(d_left, d_right) * self.cfg.init_lateral_jitter
        lat = (torch.rand((n,), device=self.device) * 2.0 - 1.0) * max_lat
        normal = self._frenet.cl_normal[idx]                # (n, 2)
        spawn_xy = spawn_xy + lat.unsqueeze(-1) * normal

        # Yaw jitter
        yaw_jitter = (
            (torch.rand((n,), device=self.device) * 2.0 - 1.0) * self.cfg.init_yaw_jitter
        )
        spawn_yaw = spawn_tang_yaw + yaw_jitter

        # Compose quaternion (w, x, y, z) for yaw-only rotation
        half = spawn_yaw * 0.5
        cw = torch.cos(half)
        sw = torch.sin(half)
        quat = torch.stack(
            [cw, torch.zeros_like(cw), torch.zeros_like(cw), sw],
            dim=-1,
        )

        # Z lift
        z_track = torch.zeros_like(spawn_xy[:, 0])   # 2D test track has z=0
        z_world = z_track + self.cfg.track_z_offset

        env_origins = self.scene.env_origins[env_ids]
        pos_world = torch.stack(
            [
                spawn_xy[:, 0] + env_origins[:, 0],
                spawn_xy[:, 1] + env_origins[:, 1],
                z_world + env_origins[:, 2],
            ],
            dim=-1,
        )

        # Root state: [pos(3), quat(4), lin_vel(3), ang_vel(3)] = 13 dim
        default = self._robot.data.default_root_state[env_ids].clone()
        default[:, 0:3] = pos_world
        default[:, 3:7] = quat
        default[:, 7:] = 0.0
        self._robot.write_root_pose_to_sim(default[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default[:, 7:], env_ids)

        # Reset joints to default state (zero velocities, neutral steer)
        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self._robot.data.default_joint_vel[env_ids].clone()
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # Reset per-env scratch
        self._prev_action[env_ids] = 0.0
        self._prev_prev_steer[env_ids] = 0.0
        self._prev_vx[env_ids] = 0.0
        self._reverse_counter[env_ids] = 0
        self._stuck_counter[env_ids] = 0
        self._lap_count[env_ids] = 0
        self._forward_progress[env_ids] = 0.0
        self._terminal_reward[env_ids] = 0.0
        # cache initial Frenet for prev_s + spawn_s
        out = self._frenet.query(spawn_xy, spawn_yaw)
        self._prev_s[env_ids] = out["s"]
        self._spawn_s[env_ids] = out["s"]

        # ── Opponents reset — opponent k spawned (k+1)*opponent_ahead_m of
        # arc-length in front of the learner, on the centerline, facing the
        # tangent (no jitter — PP tracks the line). The learner must pass
        # them one by one (8 m, 16 m, 24 m, …).
        if self._opponents:
            total = self._frenet.total_length
            for k, opp in enumerate(self._opponents):
                s_opp = (out["s"]
                         + (k + 1) * self.cfg.opponent_ahead_m) % total
                opp_idx = torch.searchsorted(
                    self._frenet.cl_arclen, s_opp
                ).clamp(0, self._frenet.num_points - 1)
                opp_xy = self._frenet.cl_xy[opp_idx]
                opp_yaw = self._frenet.tang_angle[opp_idx]
                ho = opp_yaw * 0.5
                cwo = torch.cos(ho)
                swo = torch.sin(ho)
                opp_quat = torch.stack(
                    [cwo, torch.zeros_like(cwo),
                     torch.zeros_like(cwo), swo], dim=-1
                )
                opp_pos_world = torch.stack(
                    [
                        opp_xy[:, 0] + env_origins[:, 0],
                        opp_xy[:, 1] + env_origins[:, 1],
                        z_world + env_origins[:, 2],
                    ],
                    dim=-1,
                )
                od = opp.data.default_root_state[env_ids].clone()
                od[:, 0:3] = opp_pos_world
                od[:, 3:7] = opp_quat
                od[:, 7:] = 0.0
                opp.write_root_pose_to_sim(od[:, :7], env_ids)
                opp.write_root_velocity_to_sim(od[:, 7:], env_ids)
                ojp = opp.data.default_joint_pos[env_ids].clone()
                ojv = opp.data.default_joint_vel[env_ids].clone()
                opp.write_joint_state_to_sim(ojp, ojv, None, env_ids)
                self._opp_action_scaled[k, env_ids] = 0.0

        # ── Static boxes reset — re-randomise each box's position PER ENV
        # every episode (random arc-length + random lateral within the
        # track). Different layout per env and per reset so the vision
        # policy must generalise obstacle avoidance, not memorise a map.
        if self._boxes:
            n = env_ids.numel()
            npts = self._frenet.num_points
            half_box = 0.5 * self._box_size
            for bx in self._boxes:
                b_idx = torch.randint(
                    0, npts, (n,), device=self.device
                )
                b_xy = self._frenet.cl_xy[b_idx]               # (n, 2)
                b_nrm = self._frenet.cl_normal[b_idx]          # (n, 2)
                b_w = self._frenet.cl_widths[b_idx]            # (n, 2)
                b_half = torch.minimum(b_w[:, 0], b_w[:, 1])
                # Side-biased: normal is +left → car's RIGHT = negative.
                # "right" keeps obstacles in the policy's right-hugging
                # line (magnitude 0.15..0.8 of half-width).
                _bs = str(getattr(self.cfg, "static_box_side", "both"))
                _r = torch.rand(n, device=self.device)
                if _bs == "right":
                    b_lat = -(0.15 + 0.65 * _r) * b_half
                elif _bs == "left":
                    b_lat = (0.15 + 0.65 * _r) * b_half
                else:
                    b_lat = (2.0 * _r - 1.0) * 0.6 * b_half
                bxw = b_xy[:, 0] + b_lat * b_nrm[:, 0] + env_origins[:, 0]
                byw = b_xy[:, 1] + b_lat * b_nrm[:, 1] + env_origins[:, 1]
                bzw = torch.full_like(bxw, half_box) + env_origins[:, 2]
                pose = torch.zeros((n, 7), device=self.device)
                pose[:, 0] = bxw
                pose[:, 1] = byw
                pose[:, 2] = bzw
                pose[:, 3] = 1.0                               # quat w=1
                bx.write_root_pose_to_sim(pose, env_ids)
                zv = torch.zeros((n, 6), device=self.device)
                bx.write_root_velocity_to_sim(zv, env_ids)
