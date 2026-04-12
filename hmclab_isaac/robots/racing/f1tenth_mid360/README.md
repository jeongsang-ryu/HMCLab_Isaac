# f1tenth_mid360

1/10-scale Ackermann 레이싱 차량 + Livox Mid-360 LiDAR + 전방 카메라.

## 구성

```
f1tenth_mid360/
├── assets/
│   ├── f1tenth_mid360.usd         # 최상위 USD (아래 4개를 참조)
│   └── configuration/
│       ├── f1tenth_mid360_base.usd
│       ├── f1tenth_mid360_physics.usd
│       ├── f1tenth_mid360_robot.usd
│       └── f1tenth_mid360_sensor.usd
├── patterns/                      # Livox Mid-360 스캔 패턴
│   ├── mid360_pattern_cfg.py      # configclass
│   ├── mid360_pattern.py          # torch 기반 pattern 함수
│   └── mid360-real-centr.csv      # 실제 센서 로그 (800K 포인트, 40프레임 사이클)
├── f1tenth_mid360_cfg.py          # ArticulationCfg + 상수 (휠베이스 등)
├── sensors.py                     # 이 차량용 센서 factory 래퍼
└── ros2_graph.py                  # ROS2 bridge graph (M7 스텁)
```

## 제원

| 항목 | 값 |
|---|---|
| 휠베이스 | 0.2255 m |
| 트랙 폭 | 0.200 m |
| 바퀴 반경 | 0.0365 m |
| 최대 조향 | ±0.6 rad |
| LiDAR 마운트 | `(0.05, 0, 0.2)` @ base_link |
| 카메라 마운트 | `(0.122, 0, 0.257)` @ base_link |

## ⚠️ Import 순서

이 모듈은 `isaaclab.sim`에 의존해요. **반드시 `AppLauncher` 부팅 후**에 import해야 `pxr` 에러가 안 나요.

## 사용 예 (env cfg 안)

```python
from hmclab_isaac.robots.racing.f1tenth_mid360 import (
    F1TENTH_MID360_CFG, make_mid360, make_front_camera,
)

# 차량 2대 스폰 (head-to-head)
ego_cfg = F1TENTH_MID360_CFG.replace(prim_path="/World/envs/env_.*/Ego")
opp_cfg = F1TENTH_MID360_CFG.replace(prim_path="/World/envs/env_.*/Opponent")

# ego에 Mid-360 장착, opponent를 동적 target으로
from isaaclab.sensors.ray_caster import MultiMeshRayCasterCfg
ego_lidar = make_mid360(
    robot_prim_path="/World/envs/env_.*/Ego",
    mesh_targets=[
        "/World/ground/geom",
        "/World/Track/walls",
        MultiMeshRayCasterCfg.RaycastTargetCfg(
            prim_expr="/World/envs/env_.*/Opponent/base_link",
            track_mesh_transforms=True,
        ),
    ],
)

# 전방 카메라 (ego만)
ego_camera = make_front_camera("/World/envs/env_.*")
```

## 원본

[f1tenth_rl](../../../../../../../isaacworkspace/f1tenth_rl) 프로젝트에서 이식.
URDF 소스: `saye_mid360` Gazebo 모델.
