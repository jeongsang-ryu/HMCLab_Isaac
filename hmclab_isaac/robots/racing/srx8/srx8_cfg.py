"""ArticulationCfg for the 1/8-scale Serpent SRX8-inspired buggy.

USD is generated programmatically by `scripts/build_srx8_usd.py` from
primitives (box chassis + cylinder wheels) with an articulation that
mirrors the MuSHR 10-joint layout:

    back/front × left/right × wheel_suspension   (prismatic, ±1 cm)
    back/front × left/right × wheel_throttle     (revolute, continuous, 4WD)
    front × left/right × wheel_steer             (revolute, ±30°)

Scale (1/8 off-road buggy):
    chassis   0.50 × 0.30 × 0.12 m
    wheelbase 0.325 m
    track     0.28 m
    wheel Ø   0.11 m
"""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

_ASSET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
USD_PATH = os.path.join(_ASSET_DIR, "srx8.usd")

WHEELBASE = 0.325
TRACK_WIDTH = 0.28
WHEEL_RADIUS = 0.055
MAX_STEER = 0.52   # ~30°

SRX8_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=30.0,
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
        # base_link sits at wheel-hub height. Spawn slightly above the
        # wheel radius so the wheels settle cleanly on the ground.
        pos=(0.0, 0.0, 0.08),
        joint_pos={
            ".*_wheel_suspension": 0.0,
            ".*_wheel_throttle": 0.0,
            "front_.*_wheel_steer": 0.0,
        },
    ),
    actuators={
        # 4WD — all four throttle joints drive; 1/8 buggy is bigger / faster
        # than MuSHR so its torque budget is higher.
        "throttle": ImplicitActuatorCfg(
            joint_names_expr=[".*_wheel_throttle"],
            effort_limit_sim=8.0,
            stiffness=0.0,
            damping=25.0,
        ),
        "steering": ImplicitActuatorCfg(
            joint_names_expr=["front_.*_wheel_steer"],
            effort_limit_sim=80.0,
            stiffness=500.0,
            damping=30.0,
        ),
        "suspension": ImplicitActuatorCfg(
            joint_names_expr=[".*_wheel_suspension"],
            effort_limit_sim=0.0,
            stiffness=3000.0,
            damping=80.0,
        ),
    },
)
