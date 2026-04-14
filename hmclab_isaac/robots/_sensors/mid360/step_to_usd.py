"""Convert Mid-360 STEP/STP → USD using Omniverse's built-in CAD converter.

This uses `omni.kit.converter.hoops_core` which handles STEP natively
with colors, materials, and proper geometry — much better than manual
OCP tessellation.

Usage:
    OMNI_KIT_ACCEPT_EULA=YES python hmclab_isaac/robots/_sensors/mid360/step_to_usd.py

Output:
    assets/mid360.usd
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

launcher = AppLauncher(headless=True)
simulation_app = launcher.app

ASSETS_DIR = Path(__file__).parent / "assets"
# Accept both .stp and .step
INPUT = str(ASSETS_DIR / "mid-360-asm.step")
if not os.path.exists(INPUT):
    INPUT = str(ASSETS_DIR / "mid-360-asm.stp")
OUTPUT = str(ASSETS_DIR / "mid360.usd")


async def convert():
    # Enable the HOOPS converter extension
    import omni.kit.app
    mgr = omni.kit.app.get_app().get_extension_manager()
    mgr.set_extension_enabled_immediate("omni.kit.converter.hoops_core", True)

    # Wait for extension to load
    for _ in range(30):
        await asyncio.sleep(0.1)
        simulation_app.update()

    from omni.kit.converter.hoops_core import get_instance

    converter = get_instance()
    if converter is None:
        sys.stdout.write("ERROR: HOOPS converter not available\n")
        sys.stdout.flush()
        return

    sys.stdout.write(f"Converting {INPUT} -> {OUTPUT}\n")
    sys.stdout.flush()

    convert_options = {
        "bOptimize": "true",
        "instancing": "true",
    }

    await converter.create_converter_task(INPUT, OUTPUT, convert_options)

    # Wait for completion
    for _ in range(100):
        await asyncio.sleep(0.2)
        simulation_app.update()

    if os.path.exists(OUTPUT):
        size = os.path.getsize(OUTPUT) // 1024
        sys.stdout.write(f"SUCCESS: {OUTPUT} ({size} KB)\n")
    else:
        sys.stdout.write(f"FAILED: output not created\n")
    sys.stdout.flush()


def main():
    import asyncio
    task = asyncio.ensure_future(convert())

    # Pump the Kit event loop until the conversion is done
    while not task.done():
        simulation_app.update()

    sys.stdout.write("STEP_TO_USD_DONE\n")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
