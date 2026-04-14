# 자율주행 출력 → 조인트 입력 수식 연결

자율주행 알고리즘의 출력이 각 바퀴/조향 조인트에 어떻게 전달되는지
수식적으로 정리한 문서.

---

## 기본 원리: 모든 조인트는 토크를 받는다

```
USD 조인트는 매 physics step마다 하나의 토크 값을 받아 회전합니다.
그 토크는 3개 입력의 합산으로 결정됩니다:

  τ_applied = Kp × (θ_target - θ_current)        ← position term
            + Kd × (ω_target - ω_current)          ← velocity term
            + τ_direct                              ← effort term

  τ_joint = clamp(τ_applied, -τ_limit, +τ_limit)

  Kp = stiffness (Nm/rad)
  Kd = damping (Nm·s/rad)
  τ_limit = effort_limit_sim (Nm)
```

---

## 차량 조인트 구성 (TOY_01 기준)

```
차량 = 6개 조인트

조향 조인트 (2개):
  front_left_steering_joint   ← revolute, 각도 제어
  front_right_steering_joint  ← revolute, 각도 제어

구동 조인트 (2개, RWD):
  rear_left_wheel_joint       ← revolute, 속도/토크 제어
  rear_right_wheel_joint      ← revolute, 속도/토크 제어

프리 휠 (2개):
  front_left_wheel_joint      ← revolute, 자유 회전
  front_right_wheel_joint     ← revolute, 자유 회전
```

---

## 조향 조인트 수식

### 입력: 자율주행 → 조향각 (δ)

모든 자율주행 알고리즘은 **가상 중앙 조향각 δ (rad)** 를 출력합니다.

#### Case 1: 단순 조향 (좌우 동일)

```
δ_left  = δ
δ_right = δ

적용: write_joint_position_to_sim(δ, joint_ids=[steer_left, steer_right])
```

#### Case 2: Ackermann 기하학 (좌우 다름)

```
실제 차량은 내륜이 외륜보다 더 많이 꺾여야 합니다.

선회 반경:
  R = L / tan(|δ|)

좌/우 바퀴 각도:
  δ_inner = atan(L / (R - d/2))     ← 안쪽 (더 많이 꺾임)
  δ_outer = atan(L / (R + d/2))     ← 바깥쪽 (덜 꺾임)

  L = wheelbase (축간 거리, m)
  d = track_width (트랙 폭, m)

좌회전 (δ > 0):
  δ_left  = +δ_inner = +atan(L / (R - d/2))
  δ_right = +δ_outer = +atan(L / (R + d/2))

우회전 (δ < 0):
  δ_left  = -δ_outer = -atan(L / (R + d/2))
  δ_right = -δ_inner = -atan(L / (R - d/2))

적용:
  write_joint_position_to_sim(δ_left,  joint_ids=[steer_left])
  write_joint_position_to_sim(δ_right, joint_ids=[steer_right])
```

#### Case 3: Extension AckermannController

```
Extension 내부에서 동일 계산을 수행:

입력: steeringAngle, wheelBase, trackWidth, speed
출력: wheelAngles[left, right], wheelRotationVelocity[4]

→ 우리가 직접 atan 계산할 필요 없음
```

#### 조향 조인트의 PhysX 토크 계산

```
조향 조인트 설정: Kp=8000, Kd=200, τ_limit=500

θ_target = δ (자율주행 출력)
θ_current = 현재 바퀴 각도

τ_steer = 8000 × (δ - θ_current) + 200 × (0 - ω_current)
τ_steer = clamp(τ_steer, -500, +500)

→ 바퀴가 δ로 이동 (PD 제어)

또는 force_steer (우리 코드):
  write_joint_position_to_sim(δ)  → PD 무시, 즉시 각도 설정
```

---

## 구동 조인트 수식

### 자율주행 출력 → 바퀴 각속도/토크 변환

```
자율주행 출력 유형:

  Type A: desired_velocity (m/s)
  Type B: acceleration (m/s²) + brake
  Type C: throttle_percent (-1 ~ +1)
```

---

### 제어 방법 ① Velocity Target

```
자율주행: desired_velocity = v (m/s)
    ↓
단위 변환: ω_target = v / r         (rad/s)
    ↓
PhysX: τ = Kd × (ω_target - ω_current)
    ↓
바퀴에 적용

  r = wheel_radius (m)
  Kd = damping (Nm·s/rad)

예: v=3 m/s, r=0.0365 m
  ω_target = 3 / 0.0365 = 82.2 rad/s
  현재 ω=0 → τ = 30 × 82.2 = 2466 Nm → clamp(15) → 15 Nm

적용:
  set_joint_velocity_target(ω_target)
  → PhysX PD가 토크를 자동 계산
```

### 제어 방법 ② Effort (토크 직접)

```
자율주행: acceleration = a (m/s²)
    ↓
뉴턴 법칙: F = m × a           (N)
    ↓
토크 변환: τ = F × r / n       (Nm)
    ↓
바퀴에 적용

  m = vehicle_mass (kg)
  r = wheel_radius (m)
  n = 구동 바퀴 수 (RWD=2, 4WD=4)

예: a=2 m/s², m=2.5 kg, r=0.0365 m, n=2
  F = 2.5 × 2 = 5.0 N
  τ = 5.0 × 0.0365 / 2 = 0.091 Nm (per wheel)

적용:
  set_joint_effort_target(τ)
  → PhysX에 직접 토크 전달 (Kp=0, Kd=0)

⚠️ 속도 제한이 없어서 무한 가속 가능 → 코드에서 속도 리미터 필요:
  if current_speed > max_speed:
      τ = 0
```

### 제어 방법 ②-B: Effort + 브레이크

```
자율주행: (acceleration, brake_percent)
    ↓
if brake > 0:
    F_brake = -brake × m × g × μ_brake    (N)
    τ = F_brake × r / n
else:
    F_drive = m × acceleration              (N)
    τ = F_drive × r / n

  g = 9.81 m/s²
  μ_brake = 브레이크 마찰계수 (~0.8)

적용:
  set_joint_effort_target(τ)
```

### 제어 방법 ②-C: Effort + 모터 모델 (VESC)

```
자율주행: throttle_percent = t (-1 ~ +1)
    ↓
모터 토크: τ_motor = t × Kt × I_max           (Nm)
    ↓
기어비: τ_wheel = τ_motor × G                   (Nm)
    ↓
바퀴에 적용

  Kt = 60 / (2π × KV)     (Nm/A, 토크 상수)
  I_max = 최대 전류 (A)
  G = gear_ratio

DC 모터 특성 (속도 올라가면 토크 감소):
  τ_motor = Kt × (V_battery - Kt × ω_motor × G) / R_motor
  = 최대 토크에서 선형 감소

적용:
  set_joint_effort_target(τ_wheel)
```

### 제어 방법 ③ Force Write Velocity

```
자율주행: desired_velocity = v (m/s)
    ↓
단위 변환: ω = v / r           (rad/s)
    ↓
PhysX에 직접 속도 설정 (PD 무시, 무한 토크 가정)

적용:
  write_joint_velocity_to_sim(ω)

⚠️ 비현실적 — 실차에서는 즉시 속도 변경 불가
✓ RL/프로토타입에 적합 — 간편하고 즉시 반응
```

### 제어 방법 ④ Force Write Position

```
자율주행: desired_velocity = v (m/s)
    ↓
각도 누적: θ(t) = θ(t-1) + (v/r) × dt     (rad)
    ↓
PhysX에 직접 각도 설정

적용:
  write_joint_position_to_sim(θ)

⚠️ 각도를 매 step 누적해야 함
⚠️ 실차와 안 맞음 (위치 제어 서보 모터가 아닌 이상)
```

### 제어 방법 ⑤ Ramp Velocity

```
자율주행: desired_velocity = v (m/s)
    ↓
Ramp: v_cmd(t) = min(v, v_cmd(t-1) + a_max × dt)     (m/s)
    ↓
단위 변환: ω = v_cmd / r
    ↓
Force write

  a_max = 최대 허용 가속도 (m/s²)

적용:
  write_joint_velocity_to_sim(ω)

✓ ③보다 현실적 — 부드러운 출발, slip 감소
```

---

## Extension 컨트롤러 수식

### DifferentialController

```
입력: (V, ω) = (선속도 m/s, 각속도 rad/s)
출력: (ω_left, ω_right) = 좌/우 바퀴 각속도 (rad/s)

Unicycle 모델:
  ω_left  = (2V - ω×b) / (2r)
  ω_right = (2V + ω×b) / (2r)

  V = linear velocity (m/s)
  ω = angular velocity (rad/s), 좌회전 양수
  b = track width (⚠️ cm 단위!)
  r = wheel radius (⚠️ cm 단위!)

자율주행 연결:
  알고리즘 출력: (steering_angle δ, velocity v)
      ↓
  Bicycle → Unicycle 변환:
    V = v
    ω = v × tan(δ) / L        (L = wheelbase)
      ↓
  DifferentialController.forward([V, ω])
      ↓
  [ω_left, ω_right]
      ↓
  write_joint_velocity_to_sim()
```

### WheelBasePoseController

```
입력: (current_pos, current_yaw, goal_pos, goal_yaw)
출력: (ω_left, ω_right)

내부 로직:
  1. heading_error = atan2(Δy, Δx) - current_yaw
  2. distance = √(Δx² + Δy²)

  if distance < tol:
      V=0, ω=0                    → 정지
  elif |heading_error| > heading_tol:
      V=0, ω=yaw_velocity         → 제자리 회전
  else:
      V=lateral_velocity, ω=0     → 직진

  → DifferentialController.forward([V, ω])
  → [ω_left, ω_right]
```

### Stanley Control

```
입력: (현재 상태, 경로 cx/cy/cyaw)
출력: δ (조향각 rad)

수식:
  1. nearest = argmin ||path[i] - pos||       (최근접 경로점)
  2. heading_error = cyaw[nearest] - vehicle_yaw
  3. crosstrack_error = 횡방향 거리
  4. δ = heading_error + atan(k × cte / (v + ε))

  k = Stanley gain (~0.5)
  ε = 속도 0 방지 상수

출력 δ → 조향 조인트에 적용
```

### PID Control

```
입력: (target_speed, current_speed)
출력: speed_command

수식:
  cmd = Kp × (target - current)

출력 → ω = cmd / r → 구동 조인트에 적용
```

### Quintic Polynomial

```
입력: (시작 x,v,a) → (목표 x,v,a), 시간 T
출력: 경로점 배열 (x, y, yaw, v, a)

수식:
  x(t) = a₀ + a₁t + a₂t² + a₃t³ + a₄t⁴ + a₅t⁵

  경계조건 6개:
    x(0) = x_start       x(T) = x_goal
    x'(0) = v_start      x'(T) = v_goal
    x''(0) = a_start     x''(T) = a_goal

  → 6×6 연립방정식으로 계수 a₀~a₅ 결정
  → jerk 연속성 보장 (3차 미분까지 존재)

출력 → Stanley + PID의 입력으로 사용
```

---

## 전체 파이프라인 요약

```
자율주행 스택
    │
    ▼
┌─────────────────────────────┐
│ 경로 계획                    │
│  QuinticPolynomial           │
│  (start, goal) → 경로 (x,y) │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│ 경로 추종                    │
│  Stanley: 경로 → δ (조향각)  │
│  PID: 목표속도 → v (속도)    │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│ 저수준 변환                  │
│  Ackermann: δ → (δ_L, δ_R)  │
│  Diff/직접: v → (ω_L, ω_R)  │
│  또는: v → a → F → τ        │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│ 조인트 입력                  │
│  조향: position_target (rad) │
│  구동: velocity_target (rad/s│
│        또는 effort (Nm)      │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│ PhysX                        │
│  τ = Kp(θt-θ) + Kd(ωt-ω) + τd│
│  τ_joint = clamp(τ, ±limit)  │
│  → 바퀴 회전 → 차량 이동     │
└─────────────────────────────┘
```

---

## 수치 예제 (TOY_01)

```
파라미터:
  wheelbase L = 0.2255 m
  track_width d = 0.20 m
  wheel_radius r = 0.0365 m
  mass m = 2.5 kg
  max_steer = 0.6 rad (34.4°)
  steer actuator: Kp=8000, Kd=200, τ_limit=500
  drive actuator: Kp=0, Kd=30, τ_limit=15
```

---

### 예제 1: 직진 중 좌회전 (Velocity Control)

**현재 상태:**
```
차량 위치: (1.0, 0.0)   방향: 0° (동쪽)
chassis 속도: 2.0 m/s
후좌 바퀴 ω: 54.8 rad/s
후우 바퀴 ω: 54.8 rad/s
조향 각도: 0 rad
```

**자율주행 출력:**
```
steering_angle = +0.3 rad (좌회전 17.2°)
desired_velocity = 2.0 m/s (속도 유지)
```

**Step 1: 조향 변환**
```
단순: δ_left = δ_right = 0.3 rad

Ackermann:
  R = L / tan(|δ|) = 0.2255 / tan(0.3) = 0.731 m
  δ_left  = atan(L / (R - d/2)) = atan(0.2255 / 0.631) = 0.343 rad (19.7°)
  δ_right = atan(L / (R + d/2)) = atan(0.2255 / 0.831) = 0.265 rad (15.2°)
```

**Step 2: 조향 조인트에 적용**
```
write_joint_position_to_sim:
  front_left_steering_joint  ← 0.343 rad (즉시 설정)
  front_right_steering_joint ← 0.265 rad

또는 set_joint_position_target:
  PhysX 계산: τ = 8000 × (0.343 - 0.0) + 200 × (0 - 0) = 2744 Nm
  clamp: min(2744, 500) = 500 Nm → 바퀴가 0.343 rad으로 이동 시작
  (수 step 후 도달)
```

**Step 3: 구동 변환**
```
ω_target = v / r = 2.0 / 0.0365 = 54.8 rad/s
현재 ω = 54.8 rad/s (이미 목표 속도)
```

**Step 4: 구동 조인트에 적용**
```
set_joint_velocity_target(54.8):
  PhysX 계산: τ = 30 × (54.8 - 54.8) = 0 Nm
  → 현재 속도 유지 (추가 토크 불필요)

write_joint_velocity_to_sim(54.8):
  → PD 무시, 바퀴를 54.8 rad/s로 강제 유지
```

**결과 (1 step 후):**
```
차량이 좌측으로 커브 시작
선회 반경 R = 0.731 m
```

---

### 예제 2: 정지 상태에서 출발 (Effort Control)

**현재 상태:**
```
차량 위치: (0.0, 0.0)   방향: 90° (북쪽)
chassis 속도: 0 m/s
후좌 바퀴 ω: 0 rad/s
후우 바퀴 ω: 0 rad/s
조향 각도: 0 rad
```

**자율주행 출력:**
```
steering_angle = 0 rad (직진)
acceleration = 2.0 m/s² (출발)
```

**Step 1: 조향**
```
δ = 0 → 조향 조인트 변화 없음
```

**Step 2: 가속도 → 토크 변환**
```
F = m × a = 2.5 × 2.0 = 5.0 N          (필요한 힘)
τ_total = F × r = 5.0 × 0.0365 = 0.183 Nm  (총 필요 토크)
τ_per_wheel = 0.183 / 2 = 0.091 Nm      (바퀴당, RWD 2개)
```

**Step 3: 구동 조인트에 적용**
```
set_joint_effort_target(0.091):
  rear_left_wheel_joint  ← τ = 0.091 Nm
  rear_right_wheel_joint ← τ = 0.091 Nm

PhysX (Kp=0, Kd=0):
  τ_applied = 0 + 0 + 0.091 = 0.091 Nm
  clamp: 0.091 < 15 → 그대로 적용
```

**결과 (매 step):**
```
step 0: ω=0.0,  v=0.00 m/s, τ=0.091 Nm
step 1: ω=2.5,  v=0.09 m/s, τ=0.091 Nm  (계속 가속)
step 2: ω=5.0,  v=0.18 m/s, τ=0.091 Nm
...
step 100: ω=250, v=9.13 m/s, τ=0.091 Nm  (⚠️ 무한 가속!)

→ 속도 리미터 필요:
  if v > desired_max_speed:
      τ = 0
```

---

### 예제 3: 고속 주행 중 감속 (Effort + Brake)

**현재 상태:**
```
차량 위치: (5.0, 3.0)   방향: 45°
chassis 속도: 4.0 m/s
후좌 바퀴 ω: 109.6 rad/s
후우 바퀴 ω: 109.6 rad/s
조향 각도: 0.1 rad
```

**자율주행 출력:**
```
steering_angle = 0.1 rad (유지)
acceleration = 0 m/s²
brake = 0.5 (50% 브레이크)
```

**Step 1: 조향**
```
δ = 0.1 rad → 현재 0.1 → 변화 없음
```

**Step 2: 브레이크 → 토크 변환**
```
F_brake = -brake × m × g × μ
        = -0.5 × 2.5 × 9.81 × 0.8
        = -9.81 N

τ_per_wheel = F_brake × r / n
            = -9.81 × 0.0365 / 2
            = -0.179 Nm (음수 = 감속 방향)
```

**Step 3: 구동 조인트에 적용**
```
set_joint_effort_target(-0.179):
  rear_left_wheel_joint  ← τ = -0.179 Nm
  rear_right_wheel_joint ← τ = -0.179 Nm

PhysX:
  τ_applied = 0 + 0 + (-0.179) = -0.179 Nm
  바퀴 회전 감속
```

**결과:**
```
step 0:   ω=109.6, v=4.00 m/s
step 50:  ω=85.2,  v=3.11 m/s
step 100: ω=60.8,  v=2.22 m/s
...점차 감속
```

---

### 예제 4: DifferentialController로 제자리 회전

**현재 상태:**
```
차량 위치: (0.0, 0.0)   방향: 0°
chassis 속도: 0 m/s
```

**자율주행 출력:**
```
linear_velocity = 0 m/s (전진 없음)
angular_velocity = 2.0 rad/s (좌회전)
```

**Step 1: DifferentialController 계산**
```
V = 0 m/s
ω = 2.0 rad/s
b = 20 cm (track width)
r = 3.65 cm (wheel radius)

ω_left  = (2×0 - 2.0×20) / (2×3.65) = -40 / 7.3 = -5.48 rad/s
ω_right = (2×0 + 2.0×20) / (2×3.65) = +40 / 7.3 = +5.48 rad/s
```

**Step 2: 구동 조인트에 적용**
```
write_joint_velocity_to_sim:
  rear_left_wheel_joint  ← -5.48 rad/s (후진 방향)
  rear_right_wheel_joint ← +5.48 rad/s (전진 방향)
  → 좌바퀴 뒤로, 우바퀴 앞으로 → 제자리 좌회전
```

**결과:**
```
차량이 제자리에서 반시계 방향으로 회전
회전 속도 = 2.0 rad/s ≈ 114.6 °/s
1초 후 약 115° 회전
```

---

### 예제 5: WheelBasePoseController로 목표점 이동

**현재 상태:**
```
차량 위치: (0.0, 0.0)   방향: 0° (동쪽)
```

**목표:**
```
goal_position = (2.0, 2.0)
goal_orientation = 45° (북동쪽)
```

**Step 1: 컨트롤러 내부 계산**
```
heading_to_goal = atan2(2.0 - 0.0, 2.0 - 0.0) = 45° = 0.785 rad
heading_error = 0.785 - 0.0 = 0.785 rad

|heading_error| = 0.785 > heading_tol (0.1)
→ 판단: "먼저 방향을 맞춰야 함"
→ V = 0, ω = yaw_velocity = 1.0 rad/s
```

**Step 2: DifferentialController**
```
ω_left  = (0 - 1.0×20) / (2×3.65) = -2.74 rad/s
ω_right = (0 + 1.0×20) / (2×3.65) = +2.74 rad/s
→ 제자리 우회전 (45° 방향으로)
```

**Step 3: 방향 맞춘 후**
```
heading_error < 0.1 rad
distance = √(4+4) = 2.83 m > position_tol (0.1)
→ 판단: "직진"
→ V = lateral_velocity = 0.5 m/s, ω = 0

ω_left = ω_right = 0.5 / 0.0365 = 13.7 rad/s
→ 직진으로 목표점까지 이동
```

**Step 4: 도착**
```
distance < 0.1 m
→ V = 0, ω = 0
→ 정지
```

---

### 예제 6: Stanley 경로 추종 중 횡방향 오차 보정

**현재 상태:**
```
차량 위치: (1.5, 0.3)   방향: 5° (약간 북쪽으로 틀어짐)
chassis 속도: 1.5 m/s
경로: y=0 직선 (동쪽으로)
```

**Stanley 계산:**
```
nearest path point: (1.5, 0.0)
crosstrack_error (cte) = 0.3 m (경로 북쪽으로 벗어남)
heading_error = 0° - 5° = -5° = -0.087 rad
speed = 1.5 m/s
k = 0.5 (Stanley gain)

δ = heading_error + atan(k × cte / (speed + ε))
  = -0.087 + atan(0.5 × 0.3 / (1.5 + 0.001))
  = -0.087 + atan(0.1)
  = -0.087 + 0.0997
  = +0.013 rad (약간 우회전 → 경로로 복귀)
```

**조인트 적용:**
```
조향: write_joint_position_to_sim(0.013)
  → 거의 직진이지만 약간 우측으로 → 경로 복귀

구동: PID speed control
  target=1.5, current=1.5 → cmd = Kp × 0 = 0
  → 현재 속도 유지

  ω = 1.5 / 0.0365 = 41.1 rad/s
  write_joint_velocity_to_sim(41.1)
```

**결과:**
```
수 step 후: 차량이 y=0 경로로 서서히 복귀
cte 감소: 0.3 → 0.25 → 0.18 → ... → 0.0
```

---

## effort_limit_sim = 내부 토크 리밋

```
PhysX는 매 step 토크를 계산한 뒤 반드시 clamp합니다:

  τ_계산 = Kp×(θt-θ) + Kd×(ωt-ω) + τ_direct
  τ_실제 = clamp(τ_계산, -effort_limit, +effort_limit)
                         ↑
                     최대 토크 제한

이 값은 USD/YAML에서 설정:
  actuator_overrides:
    drive:
      effort_limit_sim: 15    ← Nm
```

### 실차 대응

```
effort_limit = 모터 최대 토크 × 기어비

실차 계산:
  Kt = 60 / (2π × KV)                            Nm/A
  motor_max_torque = Kt × I_max                    Nm
  effort_limit_sim = motor_max_torque × gear_ratio  Nm

예: KV=3200, I_max=40A, gear_ratio=5
  Kt = 0.00298 Nm/A
  motor = 0.00298 × 40 = 0.119 Nm
  wheel = 0.119 × 5 = 0.597 Nm
  → effort_limit_sim = 0.6 Nm (현실적)
```

### effort_limit이 동역학에 미치는 영향

```
effort_limit=15 Nm (현재 TOY_01):
  F = 15/0.0365 = 411 N/wheel
  a = 822/2.5 = 329 m/s² ← 비현실적

effort_limit=0.6 Nm (실차 기반):
  F = 0.6/0.0365 = 16.4 N/wheel
  a = 32.8/2.5 = 13.1 m/s² ← 현실적

→ 실차 모터 스펙에 맞추면 전복 없이 현실적 가속
```

---

## 조향 vs 구동 정리

```
          설정 시점 (USD/YAML)              매 step (Python)
          ────────────────                 ────────────────
조향:     Kp=8000, Kd=200, τ_limit=500    position_target = δ (rad)
구동:     Kp=0,    Kd=30,  τ_limit=15     velocity_target = ω (rad/s)
                                          또는 effort_target = τ (Nm)

velocity 제어 = "이상적 모터" (목표 속도 즉시 추종)
effort 제어  = "실제 모터" (토크 → 가속 → 서서히 도달)

지금 단계: velocity 제어 (알고리즘 개발)
나중 Sim2Real: effort 제어 (실차 동역학 매칭)
```
