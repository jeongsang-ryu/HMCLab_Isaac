# scripts/

차량/센서 관련 USD 빌드 + 텔레옵 + 진단 스크립트.

## 텔레옵 (운전)

키보드 + Linux 조이스틱 (`/dev/input/jsX`) 둘 다 지원 — 두 입력 모두 활성, 조이스틱이 활성이면 그게 우선.

| 스크립트 | 대상 | 비고 |
|---|---|---|
| `keyboard_teleop_chassis.py` | `chassis/SRX/SRC_simple.usd` (UNICORN_2 actuator cfg 재사용) | LiDAR 없음, 가벼움 |
| `keyboard_teleop_lidar.py`   | HAMA_1/HAMA_2/UNICORN_1/UNICORN_2 (LiDAR 디버그 viz 포함) | `--vehicle <NAME>` |
| `keyboard_teleop_mushr.py`   | MuSHR (WheeledLab USD) — UNICORN 비교 reference | |

조이스틱 매핑 디폴트 (Xbox-style): 왼쪽 스틱 Y = 스로틀, 오른쪽 스틱 X = 조향, A 버튼 = 브레이크. 본인 패드 매핑은 `joystick_probe.py` 로 확인 후 `--js-axis-steer N` / `--js-axis-throttle M` 등으로 오버라이드.

```bash
# 평지:
OMNI_KIT_ACCEPT_EULA=YES python scripts/keyboard_teleop_chassis.py
# 트랙:
OMNI_KIT_ACCEPT_EULA=YES python scripts/keyboard_teleop_chassis.py --track mini_oval_banked
# 조이스틱 끄기:
OMNI_KIT_ACCEPT_EULA=YES python scripts/keyboard_teleop_chassis.py --joystick ""
```

공통 입력 코드는 `hmclab_isaac/utils/teleop_input.py` (LinuxJoystick + CarKeyboard + CarController).

## 진단

| 스크립트 | 용도 |
|---|---|
| `joystick_probe.py`         | `/dev/input/jsX` 이벤트 실시간 출력 (axis/button 번호 확인용) |
| `inspect_vehicle_mass.py`   | 차량 mass 분포 + CG + 휠리 임계 |
| `compare_drive.py`          | MuSHR vs UNICORN_2 6초 시퀀스 헤드리스 비교 (CSV 로그) |
| `verify_install.py`         | 설치 sanity (Python/Sim/Lab/torch/hmclab_isaac import + CUDA) |

```bash
OMNI_KIT_ACCEPT_EULA=YES python scripts/verify_install.py
```

## USD 빌드 파이프라인

| 스크립트 | 용도 |
|---|---|
| `build_chassis_variants.py` | SRC chassis 변형 (SRC_simple, SRC_dw 등) 빌드 |
| `build_vehicles.py`         | HAMA_*, UNICORN_*, ROBORACER 어셈블리 (chassis + device 결합) |
| `build_tire_material.py`    | tire 마찰/compliant material 재빌드 + chassis USD 에 binding |
| `author_lidar_schemas.py`   | Mid-360 / Hokuyo USD 에 RTX LiDAR Camera 스키마 author |

## 트랙

| 스크립트 | 용도 |
|---|---|
| `generate_3d_tracks.py`     | `worlds/racing/_tracks_data/*.txt` 트랙 생성 |

## 실행 환경

모든 Isaac Sim 의존 스크립트는 다음 환경 변수가 필요:

```bash
export OMNI_KIT_ACCEPT_EULA=YES
```

권장 Python: `/home/js/anaconda3/envs/hmclab_test/bin/python`.
