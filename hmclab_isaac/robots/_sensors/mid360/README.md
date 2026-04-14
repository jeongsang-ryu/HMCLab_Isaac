# Livox Mid-360 LiDAR Sensor Asset

실제 Livox Mid-360 STP 파일에서 변환한 비주얼 메쉬 + Isaac Sim USD.

## 파일 구조

```
assets/
├── mid-360-asm.stp       # 원본 CAD (Livox 공식)
├── mid-360-asm.step      # 동일 파일 (.step 확장자)
├── mid360.usd            # Isaac Sim 비주얼 메쉬 (현재 사용)
├── mid360.obj + .mtl     # OBJ (색상 포함, 외부 뷰어용)
├── mid360.stl            # 충돌 메쉬 (색 없음)
├── mid360.glb            # GLB (범용 3D 뷰어)
└── mid360_preview.png    # 프리뷰 이미지
```

## 변환 파이프라인

### 방법 1: OCP 기반 (현재 사용)

```bash
OMNI_KIT_ACCEPT_EULA=YES python hmclab_isaac/robots/_sensors/mid360/convert_stp.py
```

`STP → OCP tessellate → 파트별 메쉬 → USD (GeomSubset 머티리얼)`

### 방법 2: Omniverse HOOPS CAD Converter (권장, 색상 보존)

Isaac Sim에 내장된 `omni.kit.converter.hoops_core` 확장을 사용:

```python
import omni.kit.converter.hoops_core
converter = omni.kit.converter.hoops_core.get_instance()
await converter.create_converter_task(
    "mid-360-asm.step",  # input
    "mid360.usd",         # output
    {"bOptimize": "true", "instancing": "true"}
)
```

또는 Isaac Sim GUI에서: `File > Import > CAD File` 선택.

참고: https://docs.omniverse.nvidia.com/extensions/latest/ext_cad-converter/manual.html

### 방법 3: 온라인 변환 도구 (가장 간편)

STP/STEP 파일을 다른 포맷으로 변환:

| 변환 | 링크 |
|---|---|
| STP/STEP → OBJ (색상 보존) | https://imagetostl.com/convert/file/stp/to/obj#convert |
| STP/STEP → STL (색 없음) | https://imagetostl.com/convert/file/stp/to/stl#convert |

> 참고: `.stp` = `.step` 동일 포맷 (ISO 10303). 확장자만 다르므로 `cp a.stp a.step`으로 충분.

> **⚠️ pip install Isaac Sim 환경에서는 STEP 파일 직접 import가 안 됩니다.**
> `File > Import`로 STEP를 열면 내부적으로 `kit/kit` 바이너리를 호출하는데,
> pip 설치에는 이 바이너리가 포함되어 있지 않습니다 (Omniverse Launcher 설치에만 있음).
> **OBJ로 변환 후 import하면 pip 환경에서도 정상 작동합니다.**

**추천 워크플로우:**

```
STP (원본 CAD, 보관용)
 ↓  온라인 변환 (imagetostl.com)
OBJ + MTL (색상 보존)
 ↓  Isaac Sim GUI에서 Import
USD (최종 시뮬레이션용)
```

> **STEP vs OBJ 차이:**
> - STEP = 수학적 곡면 (NURBS). 무한 확대해도 매끄러움. CAD 편집 가능.
> - OBJ = 삼각형 메쉬 근사. 확대하면 각진 면. 편집 제한.
> - 변환 시 곡면 → 삼각형으로 tessellate되어 정밀도가 약간 떨어지지만
>   시뮬레이션 비주얼로는 충분.
> - **STEP은 원본으로 보관, OBJ는 시뮬 import용.**

## 차량에 장착하기

```python
from hmclab_isaac.robots._sensors.mid360 import attach_mid360_visual

# 스폰 후 한 줄:
attach_mid360_visual("/World/Ego", mount_link="base_link", offset=(0.05, 0.0, 0.12))
```

`offset`으로 마운트 위치 조절. 물리에 영향 없이 비주얼만 추가.

## 치수

| 항목 | 값 |
|---|---|
| 폭 (X) | 73 mm |
| 깊이 (Y) | 60 mm |
| 높이 (Z) | 65 mm |
| 무게 | 265 g (실물) |
| FOV | 360° × 59° |
