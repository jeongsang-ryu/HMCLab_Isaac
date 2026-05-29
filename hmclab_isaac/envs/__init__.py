"""Environments: composed (robot + world + task). Split by category and usage.

Importing this package triggers gym registration of all envs underneath, so
``import hmclab_isaac.envs`` (or any submodule import) is enough to make
``gym.make("HMCLab-Racing-Single-Visual-v0")`` work.
"""
from . import racing  # noqa: F401  (triggers gym.register side effects)
