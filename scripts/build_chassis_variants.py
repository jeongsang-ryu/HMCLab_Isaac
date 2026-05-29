"""Rename Xforms and joints in chassis USDs to a unified convention.

Takes the source USD as-is (geometry, references, RB hierarchy, joints, etc.
all unchanged), and only renames prim names + updates joint body0/body1
relationships to match. Output file lives next to the source.

Usage:
    python build_chassis_variants.py fiesta
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from isaacsim import SimulationApp
_app = SimulationApp({"headless": True})

from pxr import Sdf, Usd, UsdPhysics  # noqa: E402

ASSETS_ROOT = Path("/home/js/hmcl_issac_project/HMCLab_Isaac/assets/robots/chassis")

# ---------------------------------------------------------------------------
# Per-chassis mapping: original prim/joint names -> unified names
# ---------------------------------------------------------------------------

FIESTA_RB_RENAME = {
    "link_chassis_06":         "chassis",
    "link_wheel_fl_02":        "fl_wheel",
    "link_wheel_fr_02":        "fr_wheel",
    "link_wheel_rl_02":        "rl_wheel",
    "link_wheel_rr_02":        "rr_wheel",
    "link_under_arm_fl":       "fl_lower_arm",
    "link_under_arm_fr":       "fr_lower_arm",
    "link_under_arm_rl":       "rl_lower_arm",
    "link_under_arm_rr":       "rr_lower_arm",
    "link_upper_arm_fl":       "fl_upper_arm",
    "link_upper_arm_fr":       "fr_upper_arm",
    "link_upper_arm_rl":       "rl_upper_arm",
    "link_upper_arm_rr":       "rr_upper_arm",
    "link_nuckle_1_fl":        "fl_knuckle1",   # steering input
    "link_nuckle_1_fr":        "fr_knuckle1",
    "link_nuckle_2_fl":        "fl_knuckle2",   # wheel hub
    "link_nuckle_2_fr":        "fr_knuckle2",
    "link_nuckle_rl":          "rl_knuckle",
    "link_nuckle_rr":          "rr_knuckle",
    "link_front_sus_spring_1": "fl_shock_upper",
    "link_front_sus_spring_2": "fr_shock_upper",
    "link_rear_sus_spring_1":  "rl_shock_upper",
    "link_rear_sus_spring_2":  "rr_shock_upper",
    "link_front_sus_rod_1":    "fl_shock_lower",
    "link_front_sus_rod_2":    "fr_shock_lower",
    "link_rear_sus_rod_1":     "rl_shock_lower",
    "link_rear_sus_rod_2":     "rr_shock_lower",
}

FIESTA_JOINT_RENAME = {
    "dof_under_fl": "fl_lower_pivot", "dof_under_fr": "fr_lower_pivot",
    "dof_under_rl": "rl_lower_pivot", "dof_under_rr": "rr_lower_pivot",
    "upper_fl":     "fl_upper_pivot", "upper_fr":     "fr_upper_pivot",
    "upper_rl":     "rl_upper_pivot", "upper_rr":     "rr_upper_pivot",
    # steering (front only — nuckle_1 is the steering input)
    "dof_nuckle1_fl": "fl_steer", "dof_nuckle1_fr": "fr_steer",
    # knuckle closures (front: nuckle2; rear: nuckle)
    "dof_nuckle2_fl": "fl_lower_knuckle", "dof_nuckle2_fr": "fr_lower_knuckle",
    "dof_nuckle_rl":  "rl_lower_knuckle", "dof_nuckle_rr":  "rr_lower_knuckle",
    # upper-arm to knuckle (closed-loop break candidates — kept as-is)
    "dof_upper2_fl": "fl_upper_knuckle", "dof_upper2_fr": "fr_upper_knuckle",
    "dof_upper2_rl": "rl_upper_knuckle", "dof_upper2_rr": "rr_upper_knuckle",
    # spring pivots
    "dof_spring_fl": "fl_spring_pivot_upper", "dof_spring_fr": "fr_spring_pivot_upper",
    "dof_spring_rl": "rl_spring_pivot_upper", "dof_spring_rr": "rr_spring_pivot_upper",
    "sus_fl": "fl_spring_pivot_lower", "sus_fr": "fr_spring_pivot_lower",
    "sus_rl": "rl_spring_pivot_lower", "sus_rr": "rr_spring_pivot_lower",
    # spring slide (prismatic)
    "dof_spring_rod_fl": "fl_spring_slide", "dof_spring_rod_fr": "fr_spring_slide",
    "dof_spring_rod_rl": "rl_spring_slide", "dof_spring_rod_rr": "rr_spring_slide",
    # wheel spin (drive) — `_joint` suffix avoids name collision with the
    # `fl_wheel`/etc RB Xform at the same articulation root.
    "dof_wheel_fl": "fl_wheel_joint", "dof_wheel_fr": "fr_wheel_joint",
    "dof_wheel_rl": "rl_wheel_joint", "dof_wheel_rr": "rr_wheel_joint",
}

CHASSIS_REGISTRY = {
    "fiesta": {
        "src":               "traxxas_final.usd",
        "dst":               "fiesta_renamed.usd",
        "articulation_root": "/Root/assembly_v2/assembly_v2",
        "rb_rename":         FIESTA_RB_RENAME,
        "joint_rename":      FIESTA_JOINT_RENAME,
    },
}


# ---------------------------------------------------------------------------
# Rename helpers
# ---------------------------------------------------------------------------

def _rename_via_namespace_editor(stage: Usd.Stage, art_root: str,
                                   rename_map: dict[str, str]) -> int:
    """Rename prims using Usd.NamespaceEditor — apply each rename
    immediately. Joint body0/body1 relationship targets follow automatically."""
    n = 0
    for old_name, new_name in rename_map.items():
        if old_name == new_name:
            continue
        old_path = f"{art_root}/{old_name}"
        prim = stage.GetPrimAtPath(old_path)
        if not prim:
            print(f"  [skip] not found: {old_path}")
            continue
        editor = Usd.NamespaceEditor(stage)
        ok = editor.RenamePrim(prim, new_name)
        if not ok:
            print(f"  [fail rename] {old_path} -> {new_name}")
            continue
        ok = editor.ApplyEdits()
        if not ok:
            print(f"  [fail apply] {old_path} -> {new_name}")
            continue
        n += 1
    return n


# ---------------------------------------------------------------------------
# Main rename pipeline
# ---------------------------------------------------------------------------

def rename_chassis(name: str) -> Path:
    cfg = CHASSIS_REGISTRY[name]
    src_path = ASSETS_ROOT / name / cfg["src"]
    out_path = ASSETS_ROOT / name / cfg["dst"]
    art_root = cfg["articulation_root"]

    # 1. Copy source to output verbatim.
    shutil.copy(src_path, out_path)
    print(f"[rename] copied {src_path.name} -> {out_path.name}")

    # 2. Open as stage and rename via NamespaceEditor (handles joint
    #    body0/body1 relationship target updates automatically).
    stage = Usd.Stage.Open(str(out_path))

    n_rb = _rename_via_namespace_editor(stage, art_root, cfg["rb_rename"])
    print(f"[rename] {n_rb} RB prims renamed")

    n_j = _rename_via_namespace_editor(stage, art_root, cfg["joint_rename"])
    print(f"[rename] {n_j} joints renamed")

    stage.GetRootLayer().Save()
    print(f"[rename] saved {out_path}")
    return out_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("chassis", choices=list(CHASSIS_REGISTRY.keys()) + ["all"])
    args = p.parse_args()
    targets = list(CHASSIS_REGISTRY.keys()) if args.chassis == "all" else [args.chassis]
    for name in targets:
        print(f"\n=== {name} ===")
        rename_chassis(name)
    _app.close()


if __name__ == "__main__":
    main()
