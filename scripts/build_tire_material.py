"""Author a shared compliant-contact tire material and bind it to the
``/Flattened_Prototype_13/tire_0`` prototype mesh of the SRC chassis USDs.

Layout produced:

    hmclab_isaac/robots/materials/tire_compliant.usd
      /tire_compliant   (Material — default prim)
        UsdPhysics.MaterialAPI       (friction + restitution)
        PhysxSchema.PhysxMaterialAPI (compliantContact{Stiffness,Damping})

    SRC_dw.usd / SRC_simple.usd
      /Root/PhysicsMaterials/tire_compliant   (Material, references the above)
      /Flattened_Prototype_13/tire_0   ← MaterialBindingAPI bound w/ purpose=physics

Because the 4 tire instances share the prototype, binding once propagates
to all 4 tires. The wheel rim (`/Flattened_Prototype_20/wheel_0`) is
left untouched (rigid contact for the rim).

Run:
    OMNI_KIT_ACCEPT_EULA=YES \\
      /home/js/anaconda3/envs/hmclab_test/bin/python \\
      scripts/build_tire_material.py
"""
from __future__ import annotations

import argparse
import os
import shutil

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--static-friction", type=float, default=1.5)
parser.add_argument("--dynamic-friction", type=float, default=1.3)
parser.add_argument("--restitution", type=float, default=0.0)
parser.add_argument("--compliant-stiffness", type=float, default=1.0e5,
                    help="Compliant contact spring stiffness [N/m]")
parser.add_argument("--compliant-damping", type=float, default=1.0e3,
                    help="Compliant contact damping [N·s/m]")
parser.add_argument("--material-name", type=str, default="tire_compliant")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
launcher = AppLauncher(args)
sim_app = launcher.app

from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade  # noqa: E402
from pxr import PhysxSchema  # noqa: E402

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROBOTS = os.path.join(_REPO, "hmclab_isaac", "robots")
MATERIALS_DIR = os.path.join(ROBOTS, "materials")
MATERIAL_USD = os.path.join(MATERIALS_DIR, f"{args.material_name}.usd")

CHASSIS_USDS = [
    os.path.join(ROBOTS, "chassis", "SRX", "SRC_dw.usd"),
    os.path.join(ROBOTS, "chassis", "SRX", "SRC_simple.usd"),
]

TIRE_PROTOTYPE_MESH = "/Flattened_Prototype_13/tire_0"


def _ensure_material_usd() -> None:
    """Create or refresh the standalone material USD."""
    os.makedirs(MATERIALS_DIR, exist_ok=True)
    if os.path.exists(MATERIAL_USD):
        os.remove(MATERIAL_USD)

    stage = Usd.Stage.CreateNew(MATERIAL_USD)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    mat = UsdShade.Material.Define(stage, Sdf.Path(f"/{args.material_name}"))
    stage.SetDefaultPrim(mat.GetPrim())

    phys = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    phys.CreateStaticFrictionAttr().Set(float(args.static_friction))
    phys.CreateDynamicFrictionAttr().Set(float(args.dynamic_friction))
    phys.CreateRestitutionAttr().Set(float(args.restitution))

    physx = PhysxSchema.PhysxMaterialAPI.Apply(mat.GetPrim())
    physx.CreateCompliantContactStiffnessAttr().Set(float(args.compliant_stiffness))
    physx.CreateCompliantContactDampingAttr().Set(float(args.compliant_damping))

    stage.GetRootLayer().Save()
    print(f"[ok] wrote material USD: {MATERIAL_USD}", flush=True)
    print(f"     friction static={args.static_friction} dynamic={args.dynamic_friction} "
          f"restitution={args.restitution}", flush=True)
    print(f"     compliant stiffness={args.compliant_stiffness} "
          f"damping={args.compliant_damping}", flush=True)


def _bind_in_chassis(usd_path: str) -> None:
    print(f"\n=== {os.path.basename(usd_path)} ===", flush=True)
    if not os.path.exists(usd_path):
        print(f"  (not found)", flush=True)
        return

    backup = usd_path.replace(".usd", ".bak.usd")
    if not os.path.exists(backup):
        shutil.copy(usd_path, backup)
        print(f"  backup → {backup}", flush=True)

    stage = Usd.Stage.Open(usd_path)

    # 1. Add a Material prim under /Root/PhysicsMaterials that references
    #    the standalone material USD. Created idempotently.
    holder_path = f"/Root/PhysicsMaterials/{args.material_name}"
    holder = stage.GetPrimAtPath(holder_path)
    if not holder.IsValid():
        # Define parent scope first
        if not stage.GetPrimAtPath("/Root/PhysicsMaterials").IsValid():
            stage.DefinePrim("/Root/PhysicsMaterials", "Scope")
        holder = stage.DefinePrim(holder_path, "Material")
    # Compute relative reference path so it stays portable
    rel = os.path.relpath(MATERIAL_USD, os.path.dirname(usd_path))
    refs = holder.GetReferences()
    refs.ClearReferences()
    refs.AddReference(rel)
    print(f"  added Material reference: {holder_path} → {rel}", flush=True)

    # 2. Apply MaterialBindingAPI on the tire prototype mesh and bind
    #    with purpose='physics'.
    tire = stage.GetPrimAtPath(TIRE_PROTOTYPE_MESH)
    if not tire.IsValid():
        print(f"  [warn] tire prototype mesh not found at {TIRE_PROTOTYPE_MESH}",
              flush=True)
        return
    binding = UsdShade.MaterialBindingAPI.Apply(tire)
    mat_obj = UsdShade.Material(holder)
    binding.Bind(mat_obj, materialPurpose="physics")
    print(f"  bound material on {TIRE_PROTOTYPE_MESH}  (purpose=physics)",
          flush=True)

    stage.GetRootLayer().Save()
    print(f"  saved → {usd_path}", flush=True)


def main():
    _ensure_material_usd()
    for p in CHASSIS_USDS:
        _bind_in_chassis(p)
    print("\nDONE", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
