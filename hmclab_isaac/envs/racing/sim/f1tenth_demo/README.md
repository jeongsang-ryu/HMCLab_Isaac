# f1tenth_demo

1 env, 60초 에피소드. GUI 확인, 수동 조종 테스트, USD 검사용. `F1TenthRacingBaseEnv` 재사용 (로직 동일, cfg만 다름).

```python
from hmclab_isaac.envs.racing.sim.f1tenth_demo import F1TenthDemoEnvCfg
from hmclab_isaac.envs.racing._base import F1TenthRacingBaseEnv

env = F1TenthRacingBaseEnv(F1TenthDemoEnvCfg())
obs, _ = env.reset()
```
