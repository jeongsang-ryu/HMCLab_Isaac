"""Robot registry — loads robot specs from YAML files by exact codename.

Usage:
    from robots import load_robot
    spec = load_robot("HMC_F1TENTH_01")

Each YAML file in this directory defines one robot. The filename must match
the codename (e.g., `HMC_F1TENTH_01.yaml`). Only exact codename matches
are accepted — no aliases, no fuzzy matching.
"""

from __future__ import annotations

import importlib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

_DIR = Path(__file__).parent
_CACHE: dict[str, "RobotSpec"] = {}


@dataclass
class RobotSpec:
    codename: str
    description: str
    cfg: object                 # ArticulationCfg
    wheelbase: float
    wheel_radius: float
    max_steer: float
    steer_regex: str
    drive_regex: str
    init_z: float
    force_steer: bool = True
    post_spawn_hooks: list = None

    def resolve_joints(self, joint_names: list[str]) -> tuple[list[int], list[int]]:
        steer_re = re.compile(self.steer_regex)
        drive_re = re.compile(self.drive_regex)
        steer_ids = [i for i, n in enumerate(joint_names) if steer_re.search(n)]
        drive_ids = [i for i, n in enumerate(joint_names) if drive_re.search(n)]
        return steer_ids, drive_ids


def available_robots() -> list[str]:
    """List all registered codenames (YAML files in this directory)."""
    return sorted(
        p.stem for p in _DIR.glob("*.yaml")
    )


def load_robot(codename: str) -> RobotSpec:
    """Load a robot spec by exact codename.

    Raises KeyError if the codename doesn't match any YAML file.
    """
    if codename in _CACHE:
        return _CACHE[codename]

    yaml_path = _DIR / f"{codename}.yaml"
    if not yaml_path.exists():
        avail = available_robots()
        raise KeyError(
            f"Unknown robot codename '{codename}'. "
            f"Available: {avail}"
        )

    with open(yaml_path) as f:
        cfg_data = yaml.safe_load(f)

    assert cfg_data["codename"] == codename, (
        f"YAML codename '{cfg_data['codename']}' != filename '{codename}'"
    )

    # Import the ArticulationCfg from the specified module
    veh = cfg_data["vehicle"]
    mod = importlib.import_module(veh["module"])
    art_cfg = getattr(mod, veh["cfg"])

    # Apply actuator overrides if specified
    overrides = cfg_data.get("actuator_overrides")
    if overrides:
        from isaaclab.actuators import ImplicitActuatorCfg
        new_actuators = {}
        for name, params in overrides.items():
            new_actuators[name] = ImplicitActuatorCfg(
                joint_names_expr=params["joint_names_expr"],
                effort_limit_sim=float(params.get("effort_limit_sim", 0)),
                stiffness=float(params.get("stiffness", 0)),
                damping=float(params.get("damping", 0)),
            )
        art_cfg = art_cfg.replace(actuators=new_actuators)

    ctrl = cfg_data.get("control", {})

    # Post-spawn hooks
    hooks = []
    ps = cfg_data.get("post_spawn", {})
    if ps.get("add_chassis_collider"):
        hooks.append("add_chassis_collider")

    spec = RobotSpec(
        codename=codename,
        description=cfg_data.get("description", ""),
        cfg=art_cfg,
        wheelbase=float(veh["wheelbase"]),
        wheel_radius=float(veh["wheel_radius"]),
        max_steer=float(veh["max_steer"]),
        steer_regex=ctrl.get("steer_regex", ""),
        drive_regex=ctrl.get("drive_regex", ""),
        init_z=float(veh.get("init_z", 0.05)),
        force_steer=bool(ctrl.get("force_steer", True)),
        post_spawn_hooks=hooks,
    )
    _CACHE[codename] = spec
    return spec


def run_post_spawn(spec: RobotSpec, prim_path: str) -> None:
    """Execute post-spawn hooks defined in the YAML."""
    if not spec.post_spawn_hooks:
        return
    for hook in spec.post_spawn_hooks:
        if hook == "add_chassis_collider":
            from hmclab_isaac.robots.racing.mushr_nano_v2.mushr_nano_v2_cfg import (
                add_chassis_collider,
            )
            add_chassis_collider(prim_path)
