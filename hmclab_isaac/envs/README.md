# envs/

환경 = `robots/` + `worlds/` + task 로직(리워드, termination, obs/action) 의 조합.

## 카테고리 × 용도 2축 분리

| | `racing/` | `offroad/` |
|---|---|---|
| `sim/` | 1 env, 디버그/데모 (GUI, 수동 조작) | 동일 |
| `rl/single_agent/` | N envs × 1 robot (병렬 학습) | 동일 |
| `rl/multi_agent/` | N envs × M robots (head-to-head) | 동일 |

## cfg vs env 분리 규칙

각 환경은 **항상 2개 파일**로 구성:

```
envs/racing/rl/multi_agent/f1tenth_h2h/
├── cfg.py        # @configclass F1TenthH2HEnvCfg(DirectRLEnvCfg)
└── env.py        # class F1TenthH2HEnv(DirectRLEnv)
```

- `cfg.py`: 순수 데이터 — robot cfg import + world cfg import + 하이퍼파라미터. Hydra/argparse로 오버라이드 가능.
- `env.py`: DirectRLEnv 상속 — `_setup_scene`, `_get_observations`, `_get_rewards`, `_reset_idx` 등 로직만.

## ⚠️ Import 순서 함정

`robots/racing/*`, `robots/offroad/*`, `worlds/racing/*` 같은 모듈은 내부적으로 `isaaclab.sim`을 import하며, 이게 `pxr`(USD)에 의존하기 때문에 **반드시 `AppLauncher` 부팅 이후에 import**해야 합니다.

```python
# 올바른 순서
from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

# 여기서부터 robot/world/env 모듈 import 가능
from hmclab_isaac.robots.racing.f1tenth_mid360 import F1TENTH_MID360_CFG
```

AppLauncher 이전에 import하면 `ModuleNotFoundError: No module named 'pxr'`가 납니다. `RacingTrack` (`worlds/racing/_schema.py`)만 예외 — 순수 numpy라 언제든 import 가능.

## 조립 패턴

```python
# cfg.py
from hmclab_isaac.robots.racing.f1tenth_mid360 import F1TENTH_CFG
from hmclab_isaac.robots.racing.f1tenth_mid360.sensors import make_mid360, make_front_camera
from hmclab_isaac.worlds.racing import RacingTrack

@configclass
class MyEnvCfg(DirectRLEnvCfg):
    track_path = "austin.txt"
    ego_cfg = F1TENTH_CFG.replace(prim_path="/World/envs/env_.*/Ego")
    lidar_cfg = make_mid360("/World/envs/env_.*/Ego", mesh_targets=[...])
```
