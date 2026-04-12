# exomy_nav

ExoMy × 8 parallel envs, 20초, goal 네비게이션. 베이스 환경 그대로 재사용.

obs = [goal_dx_local, goal_dy_local, sin(yaw), cos(yaw), speed_norm]  (5-dim)
action = [forward_speed_norm, yaw_rate_norm]  (2-dim)

```python
from hmclab_isaac.envs.offroad.rl.single_agent.exomy_nav import ExomyNavEnvCfg
from hmclab_isaac.envs.offroad._base import ExomyOffroadBaseEnv

env = ExomyOffroadBaseEnv(ExomyNavEnvCfg())
```
