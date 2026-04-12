# f1tenth_h2h

Head-to-head: learnable **ego** + rule-based **opponent** (constant 3 m/s, 센터라인 추종).

## 확장 포인트

`F1TenthH2HEnv`는 `F1TenthRacingBaseEnv`를 상속해서:
- `_setup_scene`: opponent articulation 추가 스폰
- `_apply_action`: ego + 간단한 pure-pursuit opponent 제어
- `_get_observations`: base obs + opponent 상대 (dx, dy)
- `_get_rewards`: base reward + overtake bonus + opponent collision penalty
- `_reset_idx`: ego reset 후 opponent를 몇 점 앞에 스폰

## 관측/액션

```
obs  = [lidar_64, speed_norm, opp_dx_norm, opp_dy_norm]   # 67-dim
act  = [steer, speed] ∈ [-1, 1]^2
```

## 사용

```python
from hmclab_isaac.envs.racing.rl.multi_agent.f1tenth_h2h import (
    F1TenthH2HEnvCfg, F1TenthH2HEnv,
)
env = F1TenthH2HEnv(F1TenthH2HEnvCfg())
```
