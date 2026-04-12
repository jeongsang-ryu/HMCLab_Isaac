# f1tenth_solo

16 parallel envs × 1 F1Tenth. 15초 에피소드. rsl_rl/skrl 등 표준 PPO 학습 용도.

관측 = LiDAR(64 bins) + 정규화 속도 (총 65차원).
액션 = [steer, speed] ∈ [-1, 1]^2.

```python
from hmclab_isaac.envs.racing.rl.single_agent.f1tenth_solo import F1TenthSoloEnvCfg
from hmclab_isaac.envs.racing._base import F1TenthRacingBaseEnv

env = F1TenthRacingBaseEnv(F1TenthSoloEnvCfg())
```
