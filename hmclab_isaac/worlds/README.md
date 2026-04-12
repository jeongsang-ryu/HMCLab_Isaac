# worlds/

월드 재료 (트랙, 지형, 장애물). `racing/`과 `offroad/` 두 카테고리로 나뉘어요.

## racing/

**계약**: 모든 racing 월드는 `RacingTrack` (`_schema.py`)을 생산하거나 소비해야 합니다. 이게 있어서 `envs/racing/*` 어디서든 "트랙이 있다"는 가정하에 동작 가능.

## offroad/

지형, 경사, 바위 등. 공용 스키마는 아직 없음 — offroad는 racing만큼 구조화가 쉽지 않아서 필요 시 추가.
