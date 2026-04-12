# robots/

로봇 재료 (USD + Cfg). `racing/`과 `offroad/` 두 카테고리로 나뉘고, 각 로봇은 자기 서브패키지를 가져요.

## 서브 폴더

- `_sensors/` — 카테고리 공용 센서 factory. 여기에 있는 함수는 어떤 로봇이든 쓸 수 있게 `prim_path`를 인자로 받음.
- `racing/` — F1Tenth 같은 레이싱 차량. 각 폴더는 `<코드네임>/` 형태.
- `offroad/` — 로버, skid-steer 등. 동일 패턴.

## 새 로봇 추가 방법

```
robots/racing/<codename>/
├── __init__.py
├── assets/
│   ├── <codename>.urdf
│   └── <codename>.usd
├── <codename>_cfg.py        # ArticulationCfg
└── ros2_graph.py            # (옵션) 표준 ROS2 토픽 그래프
```

`<codename>_cfg.py` 안에서는 `_sensors.lidar.make_*` / `_sensors.camera.make_*` 팩토리를 호출해서 기본 센서 구성을 제공합니다.
