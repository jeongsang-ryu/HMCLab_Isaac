"""Car Tutorial 00 — List all registered robots.

No Isaac Sim needed — just reads the YAML files.

Run:
    python Itutorial/car/00_list_robots.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from robots import available_robots
from pathlib import Path
import yaml

robots_dir = Path(__file__).parent.parent / "robots"

print(f"\n{'='*60}")
print(f"  Registered Robots ({len(available_robots())})")
print(f"{'='*60}\n")

for codename in available_robots():
    with open(robots_dir / f"{codename}.yaml") as f:
        data = yaml.safe_load(f)
    veh = data.get("vehicle", {})
    ctrl = data.get("control", {})
    print(f"  {codename}")
    print(f"    {data.get('description', '')}")
    print(f"    wheelbase={veh.get('wheelbase')}m  wheel_r={veh.get('wheel_radius')}m  "
          f"max_steer={veh.get('max_steer')}rad")
    print(f"    drive: {ctrl.get('drive_regex', '?')}")
    print()
