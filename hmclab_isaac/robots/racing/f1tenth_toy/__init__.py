"""F1Tenth Toy — chassis/sensor 분리 구조.

디렉토리:
    chassis/TOY_01_chassis.usd  — 샤시+바퀴+조인트 (Articulation)
    sensors/mid360.obj          — LiDAR 비주얼 (→ GUI에서 USD로 저장)
    sensors/nuc14pro.obj        — NUC 비주얼 (→ GUI에서 USD로 저장)
"""

import os

_DIR = os.path.dirname(os.path.abspath(__file__))
CHASSIS_USD = os.path.join(_DIR, "chassis", "TOY_01_chassis.usd")
SENSOR_DIR = os.path.join(_DIR, "sensors")
