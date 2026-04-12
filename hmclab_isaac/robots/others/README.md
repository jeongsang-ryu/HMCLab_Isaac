# robots/others/

차량 중에서 표준 4륜 구성(racing/offroad 버기)에 들어가지 않는 것들을 모은 카테고리. 주로 **6륜 Mars 로버 계열**.

## 현재 포함

| 코드네임 | 바디 수 | 조인트 | Drive | Steer | 서스펜션 | 포팅 상태 |
|---|---|---|---|---|---|---|
| [`exomy`](./exomy/) | 19 | 15 | 6 | 6 (전륜 독립) | 3 bogie | **full port** |
| [`aau_rover`](./aau_rover/) | 16 | 15 | 6 | 4 (코너) | 4 rocker + 1 diff | preview only |
| [`aau_rover_simple`](./aau_rover_simple/) | 14 | 13 | 6 | 4 (코너) | 3 bogie | preview only |

## 왜 "others" 인가

- `racing/` = F1Tenth 스타일 1/10 RC + Ackermann (4 휠, 2 steer)
- `offroad/` = 4WD 오프로드 버기 (현재 비어 있음; 나중에 Traxxas Hound 같은 거 포팅)
- `others/` = 그 외 — 현재는 6륜 rocker-bogie 로버만. 드론, 매니퓰레이터, legged 같은 차량 외 로봇이 생기면 여기에 **넣지 말고** 별도 최상위 카테고리 (`drones/`, `manipulators/`) 를 만드세요.

## 사용 시

env 카테고리(`envs/racing/`, `envs/offroad/`)와 **로봇 카테고리는 독립**이에요. 6륜 rover는 `others/` 에 있지만 `envs/offroad/exomy_demo` 같은 offroad env에서 정상적으로 import해서 써도 됩니다. 하나의 로봇이 여러 env 카테고리에 걸쳐서 쓰일 수 있음.
