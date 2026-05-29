"""HAMA — DBXL chassis family. Variants are 1:1 with their USD/.py file."""
from . import HAMA_1, HAMA_2  # noqa: F401

# Convenience alises so callers can ``from ...HAMA import HAMA_1_CFG``.
HAMA_1_CFG = HAMA_1.CFG
HAMA_2_CFG = HAMA_2.CFG

__all__ = ["HAMA_1", "HAMA_2", "HAMA_1_CFG", "HAMA_2_CFG"]
