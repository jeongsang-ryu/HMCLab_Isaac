"""ArticulationCfg for the 1/10 scale Ackermann vehicle with Livox Mid-360 LiDAR.

Ported from the original f1tenth_rl project. The USD file bundles:
  - Chassis + 4 wheels (Ackermann kinematics, wheelbase 0.2255 m)
  - Front steering joints (±0.6 rad)
  - Passive front wheels, driven rear wheels
  - Mounted frames: `lidar_link`, `camera_link` (for sensor attachment)

The Cfg exposed here is a *template* — leave `prim_path` as default and
`.replace(prim_path=...)` on it from an env cfg to spawn multiple instances
(ego / opponent / etc.).
"""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

_ASSET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
# Use the `physics` variant directly: the top-level `f1tenth_mid360.usd` wraps
# the physics USD via a payload, and Isaac Lab's `UsdFileCfg` spawns with
# references (not payloads), so the payload never loads and all visuals go
# missing in the GUI. Loading the physics USD directly composes both
# articulation and the `/visuals` + `/colliders` scopes the links reference.
USD_PATH = os.path.join(_ASSET_DIR, "configuration", "f1tenth_mid360_physics.usd")

F1TENTH_MID360_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=20.0,
            max_angular_velocity=2000.0,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=8,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.05),  # wheel radius = 0.0365 m, small air gap
        joint_pos={
            "front_left_steering_joint": 0.0,
            "front_right_steering_joint": 0.0,
            "front_left_wheel_joint": 0.0,
            "front_right_wheel_joint": 0.0,
            "rear_left_wheel_joint": 0.0,
            "rear_right_wheel_joint": 0.0,
        },
    ),
    actuators={
        "steering": ImplicitActuatorCfg(
            joint_names_expr=["front_.*_steering_joint"],
            effort_limit_sim=50.0,
            stiffness=200.0,
            damping=20.0,
        ),
        "drive": ImplicitActuatorCfg(
            joint_names_expr=["rear_.*_wheel_joint"],
            effort_limit_sim=500.0,
            stiffness=0.0,
            damping=200.0,
        ),
        "front_wheels": ImplicitActuatorCfg(
            joint_names_expr=["front_.*_wheel_joint"],
            effort_limit_sim=0.0,
            stiffness=0.0,
            damping=0.5,
        ),
    },
)

# Vehicle specs (kept as module constants so envs can import without a running sim)
WHEELBASE = 0.2255
TRACK_WIDTH = 0.200
WHEEL_RADIUS = 0.0365
MAX_STEER = 0.6

# Sensor mount offsets (matching the USD prim positions)
LIDAR_MOUNT_OFFSET = (0.05, 0.0, 0.2)
FRONT_CAMERA_MOUNT_OFFSET = (0.122, 0.0, 0.257)
