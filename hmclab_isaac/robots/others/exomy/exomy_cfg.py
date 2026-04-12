"""ArticulationCfg for the ExoMy 6-wheel Mars rover (RLRoverLab asset).

ExoMy is an open-source ESA ExoMars-inspired small-scale rover:
  - 6 independently driven wheels (`.*Drive_Joint`)
  - 6 independently steered wheels (`.*Steer_Joint`) — every wheel turns
  - 3 passive bogie pivots (`.*Bogie_Joint`)
  - Supports skid-steer, crab-walk, spot-turn

The USD + textures were copied from `reference_repos/RLRoverLab`. The cfg
here drops the RLRoverLab-specific `RoverArticulation` subclass so we can
use the stock `isaaclab.assets.Articulation`.
"""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

_ASSET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
USD_PATH = os.path.join(_ASSET_DIR, "exomy.usd")

EXOMY_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=USD_PATH,
        activate_contact_sensors=True,
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=0.04, rest_offset=0.01
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            max_linear_velocity=1.5,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
            disable_gravity=False,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.2),
        joint_pos={".*Steer_Joint": 0.0},
        joint_vel={".*Steer_Joint": 0.0, ".*Drive_Joint": 0.0},
    ),
    actuators={
        "steering": ImplicitActuatorCfg(
            joint_names_expr=[".*Steer_Joint"],
            velocity_limit=6.0,
            effort_limit=12.0,
            stiffness=8000.0,
            damping=1000.0,
        ),
        "drive": ImplicitActuatorCfg(
            joint_names_expr=[".*Drive_Joint"],
            velocity_limit=6.0,
            effort_limit=12.0,
            stiffness=100.0,
            damping=4000.0,
        ),
        "bogies": ImplicitActuatorCfg(
            joint_names_expr=[".*Bogie_Joint"],
            velocity_limit=15.0,
            effort_limit=0.0,
            stiffness=0.0,
            damping=0.0,
        ),
    },
)

# Vehicle specs (approximate; confirm against URDF)
WHEELBASE = 0.30
TRACK_WIDTH = 0.25
WHEEL_RADIUS = 0.08
MAX_STEER = 1.05  # ~60 degrees
