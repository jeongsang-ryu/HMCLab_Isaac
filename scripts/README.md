# scripts/

공용 검증/변환 스크립트 모음.

## verify_install.py

`INSTALL.md` 5단계 마지막에서 호출. Python/Sim/Lab/torch/hmclab_isaac 임포트와 CUDA 체크.

```bash
OMNI_KIT_ACCEPT_EULA=YES python scripts/verify_install.py
```

## smoke_all.py

M1–M6 전체 smoke test를 한 번에 돌림. 각 테스트는 자기만의 Python 프로세스에서 돌고, 실패하면 해당 테스트 로그를 가리킴.

```bash
OMNI_KIT_ACCEPT_EULA=YES python scripts/smoke_all.py
```

**옵션:**
- `--only <names>` — 일부만 실행 (쉼표 구분): `--only racing_demo,rover_spawn`
- `--skip <names>` — 일부만 제외
- `--fast-fail` — 첫 실패 즉시 종료
- `--log-dir PATH` — 로그 디렉토리 변경 (기본 `/tmp/hmclab_smoke`)

**예상 실행시간:** Kit 캐시가 따뜻하면 **~40초**, 콜드 부팅이면 ~2분.

**커버리지:**
| 이름 | 마일스톤 | 검증 대상 |
|---|---|---|
| schema | M1 | RacingTrack 로더 + 쿼리 |
| track_build | M3 | `build_circuit_mesh` (Kit 없이) |
| racing_demo | M4 | F1Tenth 1 env + Austin 트랙 + 10 step |
| racing_solo | M4 | F1Tenth 16 env solo |
| racing_h2h | M4 | Ego + Opponent + 동적 LiDAR 타겟 |
| rover_spawn | M5 | ExoMy 스폰 + 조인트 검증 + 10 step |
| offroad_demo | M6 | ExoMy 1 env + rough terrain |
| offroad_nav | M6 | ExoMy 8 env goal 네비게이션 |

**제외:** M7 ROS2 (deferred — `docs/deferred_m7_ros2.md` 참고)

## smoke_m4.py / smoke_m5.py / smoke_m6.py / smoke_m7.py

개별 마일스톤 단독 실행. `smoke_all.py`가 내부적으로 호출해요. 디버깅할 때 직접 실행 가능:

```bash
python scripts/smoke_m4.py --which demo --steps 20
python scripts/smoke_m5.py
python scripts/smoke_m6.py --which nav --steps 20
```

## convert_f1tenth_track.py

f1tenth_racetracks `*_centerline.csv` (2D, 4컬럼) → HMCLab_Isaac `RacingTrack` 스키마 (3D, 8컬럼) 변환.

```bash
python scripts/convert_f1tenth_track.py \
  --src /path/to/reference_repos/f1tenth_racetracks/Monza/Monza_centerline.csv \
  --dst hmclab_isaac/worlds/racing/_tracks_data/monza.txt \
  --name monza
```
