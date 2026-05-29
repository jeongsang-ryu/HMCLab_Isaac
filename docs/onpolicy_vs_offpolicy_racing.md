# On-policy vs Off-policy on Vision-based Racing

**Task:** single-car time-trial on `my_track`, vehicle **UNICORN_3**, learning to drive
from a front camera. Goal: compare the *behavior* of **off-policy (SAC)** vs
**on-policy (PPO, A2C)** under an as-identical-as-possible setup.

_Last updated: 2026-05-26._

---

## 1. Shared setup (identical across all three algorithms)

| Component | Value |
|---|---|
| Env id | `HMCLab-Racing-Single-Visual-v0` (`UnicornRacingEnv`) |
| Vehicle / track | UNICORN_3 / `my_track`, single car (no opponents, no static boxes) |
| Observation (actor) | 128×128 RGB front camera + 6-dim proprio (vx, ax, yaw-rate, last throttle, last steer, prev steer) |
| Observation (critic, **asymmetric**) | + 11-dim privileged Frenet (s_norm, d_signed, ψ_err, L/R wall dist, κ-lookahead ×5, track width) |
| Action (continuous) | `[throttle ∈ [0,1], steer ∈ [-1,1]]` → wheel velocity + steering-position targets |
| **Network (identical)** | CNN (Champion 32/64/64/64, k 8/4/3/3, s 4/2/2/1) → proj-512 → **GRU-256** actor → MLP(256,256); asymmetric value/critic with privileged input |
| Recurrent handling | per-env hidden carry, reset-on-done; SAC = stored-state(replay) 1-step, PPO/A2C = sequence truncated-BPTT over the rollout |

The CNN+GRU backbone and the asymmetric (privileged) critic are byte-identical
between PPO and A2C (shared classes) and architecturally matched to SAC.

---

## 2. Algorithms

| | SAC (off-policy) | Recurrent PPO (on-policy) | Recurrent A2C (on-policy) |
|---|---|---|---|
| Critic | twin **Q(s,a)** + Polyak target | **V(s)** + GAE | **V(s)** + GAE |
| Update | replay 200k, 1 grad/env-step | clipped surrogate, 5 epochs × 4 minibatch | single vanilla-PG step |
| Exploration | **auto-α** entropy (target −dim) | fixed entropy coef | fixed entropy coef |
| Actor head | tanh-Gaussian, state-dep std | plain Gaussian, state-indep std (`log_std_max=0` cap) | same as PPO |
| Script | `scripts/train_racing_sac.py` | `scripts/train_racing_rppo.py --plain-gaussian` | `scripts/train_racing_ra2c.py --plain-gaussian` |

γ = 0.99 (all). lr: SAC 2.5e-5 fixed; PPO/A2C 3e-4 (PPO KL-adaptive, A2C fixed).

---

## 3. What succeeded — the key settings

### ✅ SAC — succeeded immediately, on the SPARSE reward
- **Reward (sparse/minimal):** `r = v_par − w_collide_wall·|speed|·(wall contact) + lap(+50) / bad-term(−20)`. **No lateral/heading shaping.**
- **Speed:** 11 m/s (`max_wheel_rps≈210`), **static (no curriculum).**
- **Result:** **2324 laps**, return ~6900, episode_len 914, wall-collision 0.2% @ 4M env-steps.
- Run: `/tmp/hmclab_runs/report/racing_sac_asym/20260522_182849`.

### ✅ Recurrent PPO — succeeded only with DENSE reward + speed curriculum
- **Reward (dense):** `r = 1.0·v_par − 0.5·d_signed² − 0.3·ψ_err² − 0.7·|speed|·(wall) + lap/term`. The **`−0.5·d²` (lateral) and `−0.3·ψ²` (heading)** terms are the dense steering signal.
- **Speed curriculum:** ramp `max_wheel_rps` 90→150 (4.7→7.9 m/s) over the first 8M env-steps, then hold (`cfg.speed_curriculum_*`).
- **Stability:** plain-Gaussian head + `log_std_max=0.0` cap (prevents the entropy-driven std→bang-bang blow-up), `entropy_coef=0.008`, KL-adaptive lr, `stuck_speed_mps=0.4`.
- **Result:** **178 laps @ 781k env-steps** (still in the low-speed ~5 m/s curriculum phase), return ~7040, episode_len 1519, wall-collision 0%.
- Run: `/tmp/hmclab_runs/report/racing_rppo/20260526_212329`.

### ❗ Decisive A/B test (same curriculum, reward differs)
| Run | Reward | Result |
|---|---|---|
| `racing_rppo/20260526_140806` | **minimal** (no lat/heading) + curriculum | **0 laps @ 10.7M** |
| `racing_rppo/20260526_212329` | **dense** (lat/heading) + curriculum | **178 laps @ 781k** |

→ **The dense steering reward is the key enabler. The speed curriculum alone was NOT enough for on-policy.**

### ⏳ Recurrent A2C — not yet lapping
Weaker than PPO (no clip / single-epoch update); 0 laps so far even with dense reward + curriculum. Needs further tuning (longer/lower curriculum, larger batch) — open.

---

## 4. Experiment journey (on-policy fixes, in order)

| # | Change | Result |
|---|---|---|
| 1 | PPO with SAC-identical head (tanh + state-dep std) | entropy → std clamp ceiling → bang-bang **collapse**, 0 laps |
| 2 | plain-Gaussian + `log_std_max=0` cap | collapse fixed (stable), but **plateau** 0 laps |
| 3 | 16M env-steps @ static 11 m/s | still 0 laps (more steps alone insufficient) |
| 4 | speed ↓150 + curriculum 90→150 | still 0 laps (**speed not the bottleneck**) |
| 5 | **+ dense steering reward (lateral + heading)** | ✅ **PPO laps (178 @ 781k)** |

---

## 5. Key findings

- **Off-policy (SAC)** learns to lap at **full speed from a SPARSE reward** — replay reuse + auto-α exploration find track-following on their own.
- **On-policy (PPO)** **cannot** learn from the same sparse reward (0 laps at 16M, even with a speed curriculum); it needs a **DENSE steering reward** + speed curriculum to lap.
- ⇒ On-policy is far more sensitive to **reward density and task difficulty (curriculum)** than off-policy. This is the central on/off-policy behavioral difference here (sample-efficiency / exploration gap).
- Notable failure mode (fixed): a tanh + state-dependent-std actor under an entropy bonus drives `log_std` to its clamp → saturated bang-bang actions; **plain-Gaussian + `log_std_max` cap** fixes it.
- On-policy runs are **high variance** (Isaac is non-deterministic across runs even at the same seed) — run multiple seeds.

---

## 6. Reproduce

```bash
conda activate hmclab_test && cd /home/js/hmcl_issac_project/HMCLab_Isaac

# SAC (off-policy) — succeeds on minimal reward
OMNI_KIT_ACCEPT_EULA=YES python scripts/train_racing_sac.py \
    --track my_track --num_envs 16 --seed 0 --headless --log-dir <DIR>

# PPO (on-policy) — needs the dense-reward cfg (current cfg.py: w_lat_err=0.5, w_heading_err=0.3) + curriculum
OMNI_KIT_ACCEPT_EULA=YES python scripts/train_racing_rppo.py \
    --track my_track --num_envs 32 --max_iterations 16000 --seed 0 --headless \
    --plain-gaussian --log-dir <DIR>

# A2C (on-policy)
OMNI_KIT_ACCEPT_EULA=YES python scripts/train_racing_ra2c.py \
    --track my_track --num_envs 32 --max_iterations 16000 --seed 0 --headless \
    --plain-gaussian --log-dir <DIR>
```

Key cfg knobs (`hmclab_isaac/envs/racing/rl/single_agent/cfg.py`):
`max_wheel_rps`, `speed_curriculum_*`, `w_lat_err`, `w_heading_err`, `w_collide_wall`, `stuck_speed_mps`.
TensorBoard: `tensorboard --logdir <DIR>` — watch `perf/lap_count`, `perf/return_mean`, `env/max_wheel_rps`.

---

## 7. Next step — lap-time & racing-line comparison (plan)

PPO's success used BOTH a dense reward AND a speed reduction (curriculum→7.9 m/s),
so the cause is confounded. Plan disentangles it, then does the matched comparison —
all at SAC's speed (**11 m/s, `max_wheel_rps=210`**):

**Known so far:** SAC sparse @11 m/s = 2324 laps · PPO sparse @11 m/s = 0 laps · PPO dense + curriculum (low ~5 m/s) = 178 laps.

- **Step 1 — isolate dense reward (DONE):** PPO at static 11 m/s (curriculum OFF) + dense → **0 laps @ 2M** (return ~200 plateau, wall 33-46%). **⇒ dense reward ALONE @ 11 m/s is NOT enough — the speed reduction was also necessary.** So PPO needs **dense reward AND speed reduction/curriculum** (SAC needs neither). run `racing_rppo/20260526_221036`.
- **Step 1b — find PPO's max sustainable speed (NEXT):** dense + curriculum **90→210**; watch `perf/lap_count` vs `env/max_wheel_rps` — the speed at which laps break = PPO's ceiling. cfg now: `max_wheel_rps=210`, `speed_curriculum_enabled=True`, dense.
- **Step 2 — matched comparison:** at PPO's max sustainable speed, run SAC + **same dense reward** there → compare **lap time** (TB `perf/lap_time_*`) and **racing line** (top-down capture; add TopDownRecorder to `play_racing_policy.py`).

**Run (Step 1):**
```bash
OMNI_KIT_ACCEPT_EULA=YES python scripts/train_racing_rppo.py \
    --track my_track --num_envs 32 --max_iterations 16000 --seed 0 --headless \
    --plain-gaussian --log-dir /tmp/hmclab_runs/cmp_highspeed
```
Watch TB `perf/lap_count` (>0 = PPO laps at 11 m/s with dense) and `env/max_wheel_rps` (flat 210 = curriculum off).
