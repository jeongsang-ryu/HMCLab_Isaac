"""Racing worlds.

Contract: every module in this package must either produce or consume a
`RacingTrack` (see `_schema.py`). This gives `envs/racing/*` a common interface
for spawning vehicles, generating meshes, and computing progress rewards.
"""

from ._schema import RacingTrack

# `circuit_track` + `duct_track` import pxr/omni lazily inside their `spawn_*`
# functions, so re-exporting the builder + spawner at package level is safe.
# Only actually *calling* spawn_* requires Kit.
from .circuit_track import build_circuit_mesh, save_obj, spawn_circuit
from .duct_track import build_duct_mesh, spawn_duct_track

__all__ = [
    "RacingTrack",
    "build_circuit_mesh",
    "save_obj",
    "spawn_circuit",
    "build_duct_mesh",
    "spawn_duct_track",
]
