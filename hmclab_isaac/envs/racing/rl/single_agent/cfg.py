"""Configuration for the single-agent vision-based racing env.

UNICORN_3 + duct track + front-mounted SG2 camera (128x128 RGB).
Asymmetric obs design (actor=image+proprio, critic=image+proprio+Frenet privileged)
intended for use with rsl_rl's ``obs_groups`` machinery.
"""
from __future__ import annotations

import os
from dataclasses import field

import numpy as np
from gymnasium import spaces

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass


# Tracks data lives in the worlds/racing package
_TRACKS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "..", "..", "worlds", "racing", "_tracks_data",
)
_TRACKS_DIR = os.path.abspath(_TRACKS_DIR)


@configclass
class UnicornRacingEnvCfg(DirectRLEnvCfg):
    """DirectRLEnv config — visual racing for UNICORN_3 on a duct track."""

    # --- simulation / control rates ------------------------------------------
    decimation: int = 4
    # 60 s timeout — my_track is 148 m so at the agent's expected ~5 m/s
    # cruise speed one lap takes ~30 s and we want margin for warmup + slips.
    # test track is 60 m so 60 s is also fine there (just unused headroom).
    episode_length_s: float = 60.0
    sim: SimulationCfg = SimulationCfg(
        dt=0.005,
        render_interval=4,
        device="cuda:0",
    )

    # --- action/observation spaces -------------------------------------------
    # Action: [throttle ∈ [0, 1], steer ∈ [-1, 1]]. Reverse is disabled —
    # throttle's lower bound is 0, so any negative network output is clipped
    # to "stop". The env still scales these by max_wheel_rps / max_steer_rad.
    action_space = spaces.Box(
        low=np.array([0.0, -1.0], dtype=np.float32),
        high=np.array([1.0, 1.0], dtype=np.float32),
    )

    # Observation: 49158-dim flat vector = 6 proprio + 49152 image (128*128*3).
    # The env actually returns a dict via `_get_observations`; this number is a
    # gym-API placeholder. Trainers consume the dict directly via obs_groups.
    observation_space: int = 49158
    state_space: int = 17   # proprio (6) + privileged Frenet (11)

    # --- scene ---------------------------------------------------------------
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=32,
        # Spacing covers the largest track's bbox plus margin. my_track bbox
        # is ~24 m × 27 m, so 60 m gives ~30 m clearance between envs even
        # accounting for the densified duct mesh's pipe radius.
        env_spacing=60.0,
        replicate_physics=True,
    )

    # --- track ---------------------------------------------------------------
    track_name: str = "test"
    track_subdivide: int = 10
    track_pipe_radius: float = 0.20
    track_pipe_offset: float = 0.0   # 0 = auto from track widths
    track_rib_spacing: float = 0.5
    track_z_offset: float = 0.30
    tracks_dir: str = _TRACKS_DIR

    # --- vehicle action mapping ----------------------------------------------
    max_wheel_rps: float = 150.0     # 150 × 0.0525 ≈ 7.9 m/s — back to the proven PPO-success setup (run 212329 = dense + curriculum 90→150 = 178 laps). Curriculum→210 ramped too fast (less dwell at easy low speeds) → 0 laps.
    max_steer_rad: float = 0.785     # ±45° (was ±28°) — tighter cornering
    init_lateral_jitter: float = 0.3
    init_yaw_jitter: float = 0.2

    # --- speed curriculum ----------------------------------------------------
    # Ramp the throttle scale from `speed_curriculum_start_rps` up to
    # `max_wheel_rps` over the first `speed_curriculum_steps` env-steps, then
    # hold. On-policy (PPO/A2C) couldn't learn cornering at a static high speed
    # (0 laps at 16M); learning corners slow first and transferring to fast is
    # the standard fix. Applied env-side so SAC/PPO/A2C all get it identically.
    speed_curriculum_enabled: bool = True    # ramp 90→max_wheel_rps(150) over speed_curriculum_steps. = the proven PPO-success config (212329). dense ALONE @ static 11 m/s = 0 laps; curriculum→210 (too-fast ramp) = 0 laps; curriculum→150 = 178 laps.
    speed_curriculum_start_rps: float = 90.0    # ≈4.7 m/s start (corners within tire grip)
    speed_curriculum_steps: int = 8_000_000     # ramp over first 8M env-steps

    # --- camera --------------------------------------------------------------
    cam_width: int = 128
    cam_height: int = 128
    cam_rate_hz: float = 50.0
    cam_image_normalize: bool = True   # divide by 255 in _get_observations

    # --- optional recording-only camera (NOT used by the policy) ------------
    # Spawned only when ``record_cam_enabled`` is True. Higher resolution so
    # MP4 dumps look clean, but the actor still sees the 128×128 cam_*.
    record_cam_enabled: bool = False
    record_cam_width: int = 640
    record_cam_height: int = 480

    # --- reward weights ------------------------------------------------------
    # Simplified reward inspired by Jaritz 2018 / A3C-TORCS and the user's
    # request: r = v·cos(α) + alive + terminal. No hand-engineered penalties
    # on lateral / heading / smoothness — the policy is free to choose its
    # own racing line and steering profile. Forward speed along the track
    # tangent is the only continuous signal; backward motion automatically
    # generates a negative reward through the signed v_par (no clamp).
    w_progress: float = 0.0          # ds was redundant with v_par × dt
    w_speed: float = 1.0             # v · cos(α) — the only continuous reward
    w_lat_err: float = 0.5           # dense steering signal (−0.5·d²) — THIS cracked on-policy: minimal+curriculum=0 laps@10.7M, dense+curriculum=178 laps@781k. #1 tuning knob.
    w_heading_err: float = 0.3       # dense steering signal (−0.3·ψ_err²): align with track tangent
    w_steer_smooth: float = 0.0      # large steering allowed
    w_throttle_smooth: float = 0.0   # large throttle changes allowed
    r_alive: float = 0.0             # REMOVED: free alive bonus → crawl local-optimum (PPO-convergence fix). Reward = progress + wall only.
    r_terminal_offtrack: float = -20.0   # bad termination penalty
    r_terminal_lap: float = 50.0     # lap completion bonus

    # --- termination thresholds ---------------------------------------------
    offtrack_margin: float = 0.05    # |d| > (d_side − margin) → off-track
    # Spin-out termination: |yaw rate| > spin_yawrate_rad ends the episode
    # instantly (−r_terminal_offtrack). A high-speed wall impact spins the
    # car past this threshold in one step, so this is effectively the
    # "hard wall hit → immediate end" path. Set spin_termination_enabled
    # False to disable that instant cut (e.g. to observe sustained wall
    # contact, or to let the policy learn to recover from a slide).
    spin_termination_enabled: bool = False   # line-study: let car scrape walls & recover (soft wall penalty only)
    spin_yawrate_rad: float = 8.0
    reverse_speed_mps: float = -0.5
    reverse_duration_steps: int = 25     # 25 × 0.02 s = 0.5 s
    # Stuck: body XY speed below `stuck_speed_mps` for
    # `stuck_duration_steps` consecutive steps. With a low 0.2 m/s threshold
    # the policy could converge to "crawl forward at 0.3 m/s" — never stuck,
    # never terminated, but accruing +0.1 alive/step for the full 60 s.
    # Raised threshold to 1.0 m/s so anything slower than a slow walk is
    # treated as stuck, forcing the policy to actually drive.
    stuck_speed_mps: float = 0.4         # was 1.0 — allow hard braking into corners (esp. low-speed curriculum) without "stuck" termination
    stuck_duration_steps: int = 50       # was 25 — back to 1.0 s
    stuck_grace_steps: int = 50          # was 25 — 1.0 s spawn-warmup

    # --- ground / friction ---------------------------------------------------
    # Tweaked up a touch from 1.2/1.05 — a little more grip for stronger
    # brakes and corner traction without going back to carpet-sticky.
    ground_static_friction: float = 1.35
    ground_dynamic_friction: float = 1.2

    # --- domain randomization (temporal: resampled every dr_resample_steps) ---
    # Periodically resample track color / ground color / friction so the vision
    # policy generalises over appearance and the dynamics policy over grip,
    # instead of overfitting to "yellow walls on a fixed floor". Updates the
    # shared (env_0-source) materials so all cloned envs change together.
    dr_enabled: bool = True
    randomize_track_color: bool = True
    randomize_ground_color: bool = True
    randomize_friction: bool = True
    dr_resample_steps: int = 3000        # resample every N env-steps (≈ a few episodes)
    friction_noise_frac: float = 0.4     # ground friction ×(1 ± this), clamped ≥0.2

    # --- opponent (Phase 2 multi-agent) -------------------------------------
    # When enabled, a scripted pure-pursuit car is spawned in EACH env at
    # /World/envs/env_.*/Opponent. It shares the track with the learner and
    # can collide with it (same env → same physics scene). The learner only
    # sees the opponent through its own camera (no extra obs); SAC is
    # unchanged. The opponent spawns `opponent_ahead_m` of track arc-length
    # in front of the learner so the policy practises catching / passing.
    opponent_enabled: bool = False
    opponent_type: str = "pp"            # only "pp" for now
    opponent_preset: str = "pp_slow"     # PP_PRESETS key
    opponent_ahead_m: float = 8.0        # arc-length gap between cars
    opponent_count: int = 5              # # of PP cars; spawned at
                                         # (k+1)*ahead_m ahead of the learner
                                         # so it must pass them one by one
    # --- opponent behavioural diversity -------------------------------------
    # If every bot drives the exact centerline the learner just memorises
    # "avoid the centerline". So each bot pursues a line offset off-center,
    # with a slow lateral weave and a per-bot pace. Offsets are auto-clamped
    # to stay inside the duct. Set any to 0 to disable that dimension.
    opponent_lateral_spread: float = 0.6   # bots fanned across ±this (m)
    opponent_weave_amp: float = 0.35       # sinusoidal lateral weave (m)
    opponent_weave_wavelength: float = 18.0  # arc length per weave cycle (m)
    opponent_speed_jitter: float = 0.25    # per-bot speed ×(1 ± this)
    # Of the `opponent_count` bots, the last `opponent_fast_count` use the
    # pp_fast preset (~2× cruise speed) instead of `opponent_preset` — a
    # mix of slow and fast traffic to overtake.
    opponent_fast_count: int = 2

    # --- static obstacle boxes ----------------------------------------------
    # `static_box_count` cube obstacles at random on-track positions the
    # learner must see (bright red) and steer around. Each is a kinematic
    # rigid body wired into the same collision filter as the duct wall, so
    # hitting one incurs the wall-type collision penalty. Positions are
    # sampled once at scene build (seeded) and cloned to every env.
    static_box_count: int = 0
    static_box_size: float = 0.5
    static_box_seed: int = 0
    # Which side of the centerline to place boxes on. The policy laps
    # counter-clockwise hugging its RIGHT, so "right" puts the obstacles
    # actually in its driving line (boxes on the unused left would never
    # be encountered). "both" = symmetric, "left" = mirror. Right = the
    # −normal side (FrenetField normal is +left).
    static_box_side: str = "right"

    # --- collision penalty (ContactSensor on Robot/base_link) ---------------
    # base_link carries the chassis convex hull + device boxes but is held
    # ABOVE the ground by the wheels, so it never contacts the floor — every
    # contact force on it is a wall or an opponent. force_matrix_w (filtered
    # to each opponent's base_link) isolates car-vs-car contact; the remainder
    # of the net force is the duct-wall contact. Ground is excluded for free.
    #   wall penalty   ∝ |my world planar speed|        (per step in contact)
    #   opp  penalty   ∝ |relative speed vs that opponent| (per step, per opp)
    collision_penalty_enabled: bool = True
    w_collide_wall: float = 0.7      # lowered 1.5→0.7 (< w_speed 1.0): fast near-wall cornering stays net-positive so PPO can push through corners (was over-punishing the racing line)
    w_collide_opp: float = 0.5
    contact_force_threshold: float = 1.0   # N — ignore numerical contact noise

    # --- curriculum ----------------------------------------------------------
    target_laps: int = 1   # bumped externally during training (1→2→4)

    # --- viewer --------------------------------------------------------------
    viewer: ViewerCfg = ViewerCfg(eye=(5.0, 5.0, 5.0), lookat=(0.0, 0.0, 0.3))

    # internal: keys returned in the obs dict — let trainer configs reference
    actor_obs_keys: list[str] = field(default_factory=lambda: ["policy", "images"])
    critic_obs_keys: list[str] = field(
        default_factory=lambda: ["policy", "images", "privileged"]
    )

    def track_path(self) -> str:
        return os.path.join(self.tracks_dir, f"{self.track_name}.txt")
