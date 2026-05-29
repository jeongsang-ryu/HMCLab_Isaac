# _sensors/

카테고리 공용 센서 factory. 모든 로봇이 동일한 센서 cfg를 쉽게 쓰도록 함수로 감쌌어요.

## 원칙

- **prim_path 주입**: 센서는 로봇의 특정 프레임에 붙기 때문에 factory가 `robot_prim_path`를 인자로 받음. 같은 센서를 ego/opponent 둘 다에 달 때 문자열만 바꾸면 됨.
- **Lazy import**: Isaac Lab 센서 모듈은 `pxr` 의존이라 `AppLauncher` 전에 import하면 깨져요. factory 함수 **본문 안에서** import함.
- **pattern_cfg 외부 주입**: Mid360/Velodyne/Bpearl 등 각자 다른 ray pattern 클래스를 인자로 받음. factory는 센서 타입에 불가지론적.

## 파일

- `lidar.py` — `make_raycaster_cfg`, `make_multimesh_raycaster_cfg`
- `camera.py` — `make_tiled_camera_cfg`
- `imu.py` — IMU 후처리 (스텁, M2에서 채움)
