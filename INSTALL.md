# 설치 매뉴얼

HMCLab 공용 Isaac Sim + Isaac Lab 환경 설치 가이드. 아래 버전으로 **고정**합니다.

| 항목 | 버전 |
|---|---|
| OS | Ubuntu 22.04 이상 |
| GPU | NVIDIA RTX (Ampere 이상 권장) |
| NVIDIA 드라이버 | 535+ |
| CUDA | 12.x (드라이버에 포함) |
| Python | 3.11 |
| Isaac Sim | 5.1.0 |
| Isaac Lab | main 브랜치, 커밋 `4df6560e187` (v2.3.2-15) |

> **Python 3.11 전용**: Isaac Sim 5.1.0 pip wheel은 Python 3.11에서만 설치됩니다. 3.10/3.12에서는 "No matching distribution" 에러가 납니다.
>
> **IsaacLab 커밋 핀 고정 이유**: v2.3.2 태그에는 `flatdict==4.0.1` 고정 버그가 있어 최신 setuptools에서 core `isaaclab` 패키지 빌드가 실패합니다. NVIDIA가 이후 커밋 `8cf5f191cce`에서 수정했고, 본 매뉴얼은 해당 수정이 포함된 `4df6560e187` 커밋으로 고정합니다.

## 작업 디렉토리 구조

```
hmcl_issac_project/
├── HMCLab_Isaac/      ← 랩 공유 코드 (이 레포)
└── IsaacLab/          ← git clone (버전 고정)
```

## 사전 준비

- Anaconda 또는 Miniconda 설치
- `nvidia-smi` 동작 확인 (GPU, 드라이버 535+)
- 디스크 여유: **50GB 이상** (Isaac Sim 관련 의존성 ~25GB, IsaacLab ~100MB, CUDA libs 추가)

## 설치 절차

작업 디렉토리 이동:

```bash
cd ~/hmcl_issac_project   # HMCLab_Isaac가 여기 있다고 가정
```

### 1. Conda 환경 생성

```bash
conda create -n hmclab python=3.11 -y
conda activate hmclab
pip install --upgrade pip
```

### 2. Isaac Sim 5.1.0 설치 (pip)

```bash
pip install 'isaacsim[all,extscache]==5.1.0' \
  --extra-index-url https://pypi.nvidia.com
```

약 5~10분 소요 (네트워크 속도 의존). 경고로 `packaging 23.0 vs wheel requires 24.0+` 메시지가 뜰 수 있으나 무시 가능.

검증 (첫 실행 시 Kit 부팅에 ~50초):

```bash
OMNI_KIT_ACCEPT_EULA=YES python -c "
from isaacsim import SimulationApp
app = SimulationApp({'headless': True})
print('Isaac Sim OK')
app.close()
"
```

> **EULA**: 첫 실행 시 NVIDIA Omniverse EULA 동의가 필요합니다. `OMNI_KIT_ACCEPT_EULA=YES` 환경변수로 자동 동의. `~/.bashrc` 또는 `~/.zshrc`에 넣어두면 편합니다.

### 3. Isaac Lab 설치 (고정 커밋)

```bash
cd ~/hmcl_issac_project
git clone --depth 30 --branch main https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
git checkout 4df6560e187
./isaaclab.sh --install
```

약 5~10분 소요. 의존성에 torch 2.7.0+cu128 재설치가 포함되어 다운로드 양이 큼. 스크립트 마지막의 VSCode settings 생성 단계에서 EULA 프롬프트 때문에 exit 1로 끝나 보일 수 있는데, **실제 pip 패키지 설치는 완료된 상태**이므로 다음 검증 단계로 넘어가면 됩니다.

검증:

```bash
pip list | grep -E "^isaaclab"
# isaaclab, isaaclab_assets, isaaclab_contrib, isaaclab_mimic, isaaclab_rl, isaaclab_tasks 가 보여야 함
```

### 4. HMCLab_Isaac 설치

```bash
cd ~/hmcl_issac_project/HMCLab_Isaac
pip install -e .
```

### 5. 전체 검증

```bash
OMNI_KIT_ACCEPT_EULA=YES python scripts/verify_install.py
```

정상 출력 예:
```
[OK] Python: 3.11.x
[OK] Isaac Sim import: importable
[OK] Isaac Lab: 0.54.3
[OK] Torch + CUDA: torch 2.7.0+cu128, device: NVIDIA GeForce RTX ...
[OK] hmclab_isaac: 0.0.1
SUCCESS
```

> **주의**: `isaaclab.sim`, `isaaclab.scene` 등의 서브모듈은 `AppLauncher`로 Kit을 부팅한 **후에만** 임포트 가능합니다. 부팅 전에 임포트하면 `ModuleNotFoundError: No module named 'pxr'`가 납니다. 모든 Isaac Lab 스크립트는 반드시 다음 패턴으로 시작해야 합니다:
>
> ```python
> from isaaclab.app import AppLauncher
> app_launcher = AppLauncher(headless=True)
> simulation_app = app_launcher.app
> # 이후에만 import isaaclab.sim, isaaclab.scene, ...
> ```

## 업데이트 정책

- Sim/Lab 버전 변경은 랩 차원에서 결정 후 이 매뉴얼 수정.
- 개별 연구원이 임의 업그레이드 금지 (프로젝트 간 호환성 파괴).
- Isaac Lab 소스 자체는 수정하지 않음. 패치가 필요하면 HMCLab_Isaac의 `patches/` 폴더에 diff로 관리.

## 자주 겪는 문제

- **`isaacsim` 설치 중 "No matching distribution"** → Python이 3.11인지 확인 (`python --version`). pypi.nvidia.com 인덱스가 추가됐는지 확인.
- **`import isaaclab.sim` 시 `ModuleNotFoundError: No module named 'pxr'`** → `AppLauncher`로 Kit 부팅 전에 임포트했음. 위 주의사항 참조.
- **`./isaaclab.sh --install` 중 `flatdict` 빌드 실패** → v2.3.2 태그는 이 버그가 있음. 본 매뉴얼의 `4df6560e187` 커밋으로 고정되어 있는지 `git log -1` 확인.
- **PhysX `GPU Create Stream fail` / `CUDA error: out of memory`** → 다른 프로세스가 GPU 점유 중. `nvidia-smi`로 확인하고 필요 시 종료.
- **EULA 프롬프트로 멈춤** → `export OMNI_KIT_ACCEPT_EULA=YES`.
- **GUI 실행 시 Vulkan 오류** → NVIDIA 드라이버 535+ 확인, `vulkaninfo` 동작 확인.
