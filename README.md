# HMCLab_Isaac

연구실 공유 Isaac Sim / Isaac Lab 코드베이스. 차량 로봇을 **racing**과 **offroad** 2가지 카테고리로 구분하여, 로봇·월드·환경(sim/RL, 단일/멀티 에이전트)을 일관성 있게 공유하기 위한 프로젝트.

## 설치

[INSTALL.md](./INSTALL.md) 참고. 고정 버전: Isaac Sim 5.1.0 / Isaac Lab commit `4df6560e187` / Python 3.11.

## 구조

```
HMCLab_Isaac/
├── INSTALL.md                       # 설치 매뉴얼
├── hmclab_isaac/
│   ├── robots/                      # 1. 로봇 (USD + Cfg)
│   │   ├── _sensors/                #    카테고리 공용 센서 factory
│   │   │   ├── lidar.py
│   │   │   ├── camera.py
│   │   │   └── imu.py
│   │   ├── racing/                  #    예: f1tenth_mid360
│   │   └── offroad/                 #    예: rover
│   │
│   ├── worlds/                      # 2. 월드 (트랙, 지형)
│   │   ├── racing/
│   │   │   ├── _schema.py           #    RacingTrack (x y z r p y dL dR)
│   │   │   └── _tracks_data/        #    .txt 센터라인 데이터
│   │   └── offroad/
│   │
│   ├── envs/                        # 3. 환경 (조합된 cfg + env 클래스)
│   │   ├── racing/
│   │   │   ├── sim/                 #    1 env 데모/디버그
│   │   │   └── rl/
│   │   │       ├── single_agent/    #    N envs × 1 robot
│   │   │       └── multi_agent/     #    N envs × M robots (head-to-head)
│   │   └── offroad/
│   │       ├── sim/
│   │       └── rl/
│   │           ├── single_agent/
│   │           └── multi_agent/
│   │
│   └── utils/
│       └── ros2.py                  # Python-driven OmniGraph ROS2 bridge
│
└── scripts/
    └── verify_install.py
```

## 설계 원칙

- **로봇(재료) × 월드(재료) × 환경(요리)** 3분할. env가 robot+world cfg를 import만 해서 조합.
- **cfg와 env 분리**: `..._cfg.py`(순수 데이터, `@configclass`)와 `..._env.py`(`DirectRLEnv` 상속).
- **센서는 factory 함수**: `make_raycaster_cfg(robot_prim_path=...)` 형태로 prim_path 주입받아 재사용.
- **racing 트랙은 스키마 계약**: `RacingTrack` (`x y z r p y dL dR`, 8컬럼) 한 가지로 mesh 생성·스폰·리워드가 모두 구동.
- **Isaac Lab 수정 금지**: 필요 시 `patches/` 폴더에 diff로 관리.

## 버전 정책

- Isaac Sim: 5.1.0 (pip install)
- Isaac Lab: commit `4df6560e187` (v2.3.2 + 15 커밋, flatdict 버그 수정 포함)
- Python: 3.11
