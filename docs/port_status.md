# 포팅 상태: full port vs preview only

각 로봇 폴더(`hmclab_isaac/robots/*/<codename>/`)는 다음 두 상태 중 하나입니다. README 상단의 명시가 기준.

## preview only

**정의**: 차량의 존재와 외형만 카탈로그에 등록된 상태. HMCLab_Isaac 내에서 아직 사용 불가.

**포함된 것:**
- `README.md` — 외형, 조인트 구조, 센서 구성, 원본 출처, 비교표
- `preview.jpg` — Isaac Sim GUI 스크린샷

**포함 안 된 것:**
- USD 파일 사본 (원본 경로만 README에 명시)
- `ArticulationCfg` Python 객체 (`*_cfg.py` 없음)
- `__init__.py` export
- 센서 factory (`sensors.py`)
- ROS2 graph (`ros2_graph.py`)
- env 통합 (`envs/*/` 에서 이 로봇을 import하지 않음)
- 스모크 테스트 커버리지

**쓰려면 무엇이 필요한가:**
preview only 상태의 로봇을 사용하려면 "full port" 단계로 올려야 합니다. 작업량은 로봇 1대당 1~3시간 (단순한 경우) ~ 반나절 (센서 복잡하거나 조인트 이름 재매핑 필요한 경우).

**언제 preview only로 남겨두는가:**
- 지금 당장 쓸 계획이 없지만 후보로 저장해 두고 싶을 때
- 랩원들이 "어떤 차량이 있는지" 한눈에 비교하고 선택할 때
- 원본 repo에서 바로 로드할 수 있고, 복사본 유지 비용이 부담될 때 (대용량 USD)

## full port

**정의**: HMCLab_Isaac의 일급 시민으로 편입된 상태. env에서 바로 사용 가능하고 CI smoke 테스트가 커버함.

**포함된 것 (전부 필수):**

1. **`assets/` 폴더에 USD 사본** — 원본 repo가 사라져도 동작. 대용량이면 LFS 고려.
2. **`<codename>_cfg.py`** — `ArticulationCfg` 인스턴스. spawn(USD 경로, physics props, articulation props), init_state, actuators(joint 그룹별 control 설정). 모듈 상수(휠베이스, 휠 반경, 최대 조향각 등)도 여기.
3. **`__init__.py`** — 패키지 최상위에서 `EXOMY_CFG` 같은 이름으로 cfg와 상수를 re-export.
4. **`README.md`** — preview only의 내용 + 제원 표 + 사용 예 코드 + 포팅 상태 명시.
5. **`preview.jpg`** — 갤러리 이미지.
6. **env 통합** — `envs/*/` 중 최소 1개에서 이 로봇을 import하고 사용. e.g. `envs/offroad/_base.py`가 `from hmclab_isaac.robots.others.exomy import EXOMY_CFG`.
7. **스모크 테스트** — `scripts/smoke_all.py`의 테스트 리스트에 해당 env 또는 전용 spawn 테스트가 포함돼 있고 통과.

**선택 사항 (필요 시 추가):**

- **`sensors.py`** — 해당 차량 전용 센서 factory (마운트 오프셋 고정). `_sensors/lidar.py`나 `camera.py`의 공용 factory를 wrapping.
- **`ros2_graph.py`** — 해당 차량의 표준 ROS2 토픽 세트를 붙이는 `attach()` 함수. M7 재오픈 시.
- **`patterns/`** — 차량 고유 센서 스캔 패턴 (예: Mid-360 non-repetitive pattern CSV).

## 현재 상태 (2026-04-11 기준)

| 로봇 | 상태 | env에서 사용 | smoke 커버 |
|---|---|---|---|
| `robots/racing/f1tenth_mid360` | **full port** | `envs/racing/*` (demo/solo/h2h) | racing_demo/solo/h2h |
| `robots/racing/f1tenth_wl` | preview only | ❌ | ❌ |
| `robots/racing/mushr_nano` | preview only | ❌ | ❌ |
| `robots/racing/mushr_nano_v2` | preview only | ❌ | ❌ |
| `robots/others/exomy` | **full port** | `envs/offroad/*` (demo/nav) | rover_spawn, offroad_demo, offroad_nav |
| `robots/others/aau_rover` | preview only | ❌ | ❌ |
| `robots/others/aau_rover_simple` | preview only | ❌ | ❌ |

즉 총 7대 중 **2대(f1tenth_mid360, exomy)만 full port**, 나머지 5대는 카탈로그 preview only.

## preview → full port 절차

1. 원본 USD를 `assets/`에 복사 (필요 시 sub-USD/텍스처까지)
2. `<codename>_cfg.py` 작성
   - 원본 repo의 cfg 파일(`wheeledlab_assets/f1tenth.py` 등)을 참고
   - `class_type` 같은 non-standard 필드는 제거 (`Articulation` 기본 사용)
   - joint 이름 regex는 실제 USD 조인트 이름과 일치하게
3. `__init__.py`에서 cfg export
4. 스폰 smoke 작성 — `scripts/smoke_<codename>.py` 식으로, 10 step만 돌고 exit
5. `scripts/smoke_all.py` TESTS 리스트에 추가
6. 필요하다면 env 통합 (`envs/*/rl/*`에 새 cfg 작성)
7. README 상단의 상태를 `preview only` → `full port`로 수정

이 절차를 랩원 누구든 따라 할 수 있게 각 로봇 README 하단에 "포팅 상태 + 전환 TODO" 섹션을 유지하세요.
