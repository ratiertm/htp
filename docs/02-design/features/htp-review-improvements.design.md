---
template: design
version: 1.3
feature: htp-review-improvements
date: 2026-05-16
author: Mindbuild
project: HTP (Hub Topology Programming)
status: Draft
---

# htp-review-improvements Design Document

> **Summary**: HTPConfig god-object와 htp_runtime.py 967줄을 Bottom-up Incremental 마이그레이션으로 해소. 7개 작은 커밋, 각 단계마다 회귀 57/57 통과를 강제. 공개 API 100% 유지.
>
> **Project**: HTP
> **Version**: post-`6be8746`
> **Author**: Mindbuild
> **Date**: 2026-05-16
> **Status**: Draft
> **Planning Doc**: [htp-review-improvements.plan.md](../../01-plan/features/htp-review-improvements.plan.md)

---

## Context Anchor

| Key | Value |
|-----|-------|
| **WHY** | god-object/god-file이 향후 Phase 확장과 단위 테스트성을 막는 구조적 부채. 지금 안 갚으면 임베딩 라우팅·Predictive Coding 도입 시 폭증. |
| **WHO** | HTP 개발자 본인 + 향후 컨트리뷰터. 사용자(`from htp import …`)는 무영향. |
| **RISK** | 회귀 57/57 깨짐 → 즉시 롤백. `server.py`/`static/index.html` 외부 의존 누설 금지. **순환 import** 발생 가능성이 최대 기술적 위험. |
| **SUCCESS** | (1) 회귀 57/57 + 신규 unit ~20개 통과 (2) `htp_runtime.py` ≤250줄 (3) HTPConfig 직접 참조 50%↓ (4) 모든 `from htp import …` 무변경. |
| **SCOPE** | Phase A = HIGH 1·2만 (HTPConfig DI + htp_runtime.py 파일 분리). MED/LOW 6건은 백로그. |

---

## 1. Overview

### 1.1 Design Goals

1. 공개 API(`htp/__init__.py` 93줄)를 단 한 줄도 깨지 않으면서 내부 결합도를 낮춤
2. 각 마이그레이션 단계가 **독립적으로 revert 가능**해야 함 — 7개 커밋, 모두 green
3. `htp/core/` ↔ `htp/runtime/` 의존 방향을 **단방향 DAG**로 강제 (순환 import 차단)
4. 신규 unit test ~20개로 *공개 API 호환 + 엔진 독립 생성 + 순환 의존 부재*를 영구 검증

### 1.2 Design Principles

- **Single Responsibility**: 한 파일 = 한 엔진 + 그 엔진이 직접 쓰는 dataclass
- **Dependency Inversion**: 엔진은 자기 sub-config만 받음 (전체 HTPConfig를 모름)
- **Backward Compatibility First**: re-export 우선, 새 import 경로는 *추가만* (제거 없음)
- **Fail Loud, Fail Early**: 매 단계 `pytest tests/regression/`을 통과해야만 다음 단계 진입
- **No Speculation**: Phase A 범위 밖 모든 발견은 백로그 등록만, 작업 금지

### 1.3 Selected Strategy

**Option A — Bottom-up Incremental (최대 안전)**

7개 커밋으로 분할, 각 커밋이 독립적으로 의미를 가지며 회귀 57/57을 유지함. 큰 함수 한 번에 옮기지 않고 점진적으로 위임 경로를 추가/교체.

---

## 2. Architecture Overview

### 2.1 Target Module Structure

```
htp/
├── __init__.py                       (변경 없음 — 공개 API 표면)
├── core/
│   ├── __init__.py                   (export 확장)
│   ├── config.py                     ★ NEW (Step 1) — sub-config 4종
│   ├── weight_matrix.py              ★ NEW (Step 3) — WeightMatrix
│   ├── hub_formation.py              ★ NEW (Step 4) — HubFormationEngine
│   ├── pruning.py                    ★ NEW (Step 5) — PruningEngine + PruneStrategy
│   ├── activation.py                 ★ NEW (Step 6) — ActivationEngine + Node + tag/terminal
│   └── node_generation_engine.py    (위치 유지)
├── runtime/
│   ├── htp_runtime.py                ≤250줄 — HTPRuntime + HTPConfig facade + demo + re-exports
│   ├── region_runtime.py             (변경 없음)
│   ├── brain_runtime.py              (변경 없음)
│   ├── cortical_connections.py       (변경 없음)
│   └── async_brain_runtime.py        (변경 없음)
├── thalamus/                         (변경 없음)
├── memory/                           (변경 없음)
└── llm/                              (변경 없음)

tests/
├── regression/                       57개 (변경 없음, 항상 통과)
└── unit/                             ★ NEW (Steps 4-7에 분산)
    ├── __init__.py
    ├── test_config_isolation.py      sub-config 독립성 + HTPConfig facade 호환 (~5개)
    ├── test_engine_di.py             각 엔진을 sub-config만으로 생성 가능 (~5개)
    ├── test_import_paths.py          공개 API + 기존 경로 모두 작동 (~5개)
    └── test_no_circular_deps.py      htp/core/ → htp/runtime/ import 부재 (~3개)
```

### 2.2 Dependency Direction (DAG)

```
                      htp/__init__.py
                            │
            ┌───────────────┼────────────────┐
            │               │                │
            ▼               ▼                ▼
        htp/runtime/    htp/thalamus/    htp/memory/    htp/llm/
            │               │                │             │
            └────────────┬──┴────────┬───────┴─────────────┘
                         ▼           ▼
                    htp/core/    (외부 라이브러리 only — torch 등)

규칙:
  - htp/core/*.py  ∈ {torch, dataclasses, typing, enum}만 import
  - htp/runtime/*.py 가 htp/core/* import 가능 (반대 금지)
  - htp/__init__.py 는 모든 곳에서 import
```

### 2.3 HTPConfig Facade Pattern

```python
# htp/core/config.py (NEW — Step 1)
@dataclass
class HubConfig:
    hebbian_lr: float = 0.05
    hub_pr_threshold: float = 2.5
    threshold: float = 0.5
    # ... (HubFormationEngine이 쓰는 필드만)

@dataclass
class PruneConfig:
    decay_rate: float = 0.001
    prune_threshold: float = 0.01
    # ...

@dataclass
class NGEConfig:
    maturity_calls: int = 5
    global_cooldown: int = 10
    max_gen_per_run: int = 1
    # ...

@dataclass
class ActivationConfig:
    max_depth: int = 20
    # ...


# htp/runtime/htp_runtime.py (변경됨 — Step 1 + Step 7)
@dataclass
class HTPConfig:
    """Facade. 기본값 그대로, 또는 sub-config 직접 주입."""
    hub:        HubConfig        = field(default_factory=HubConfig)
    prune:      PruneConfig      = field(default_factory=PruneConfig)
    nge:        NGEConfig        = field(default_factory=NGEConfig)
    activation: ActivationConfig = field(default_factory=ActivationConfig)

    # 호환 레이어: 기존 HTPConfig(hub_pr_threshold=3.0) 같은 사용 코드 지원
    def __init__(self, **kwargs):
        # 1. sub-config 직접 전달 케이스
        self.hub        = kwargs.pop("hub", None)        or HubConfig()
        self.prune      = kwargs.pop("prune", None)      or PruneConfig()
        self.nge        = kwargs.pop("nge", None)        or NGEConfig()
        self.activation = kwargs.pop("activation", None) or ActivationConfig()
        # 2. flat 키워드 호환 (구버전 호출 패턴)
        for k, v in kwargs.items():
            for sub in (self.hub, self.prune, self.nge, self.activation):
                if hasattr(sub, k):
                    setattr(sub, k, v)
                    break
            else:
                raise TypeError(f"Unknown HTPConfig field: {k}")

    # 호환 레이어: cfg.hub_pr_threshold 같은 옛 속성 접근 지원
    def __getattr__(self, name: str):
        for sub in (self.hub, self.prune, self.nge, self.activation):
            if hasattr(sub, name):
                return getattr(sub, name)
        raise AttributeError(name)
```

---

## 3. Migration Plan (7 Steps)

각 Step은 독립 커밋. 각 커밋 직전·직후 반드시 `pytest tests/regression/ -v` 통과.

### Step 1 — sub-config 신설 (HTPConfig는 facade 변환만)

**변경**:
- `htp/core/config.py` 생성 — 4개 sub-config dataclass (Hub/Prune/NGE/Activation)
- `htp/runtime/htp_runtime.py`의 `HTPConfig`를 facade로 변환 (위 §2.3 코드)
- 엔진들은 **여전히 HTPConfig 받음** — `cfg.hub_pr_threshold` 같은 접근은 `__getattr__` facade로 통과

**검증**: `pytest tests/regression/`  → 57/57 통과 (행동 무변경)

**롤백 단위**: 단일 커밋 revert

### Step 2 — `HubFormationEngine`만 sub-config 직접 받기

**변경**:
- `HubFormationEngine.__init__(wm, hub_cfg: HubConfig)` 시그니처
- `HTPRuntime`이 `HubFormationEngine(self.wm, self.cfg.hub)`로 주입
- 엔진 내부의 `self.cfg.hub_pr_threshold` 같은 접근을 `self.cfg.hub_pr_threshold` (HubConfig)로 변경 (실제로는 동일)

**검증**: `pytest tests/regression/test_phase1_hub_formation.py` + 전체

**롤백 단위**: 단일 커밋

### Step 3 — `WeightMatrix` 파일 분리

**변경**:
- `htp/core/weight_matrix.py` 생성 — `WeightMatrix` 클래스 이동
- `htp/runtime/htp_runtime.py`: `from htp.core.weight_matrix import WeightMatrix` + 기존 위치 삭제
- `htp/core/__init__.py`: `WeightMatrix` export 추가
- **공개 API 보존**: `htp/runtime/htp_runtime.py`에서 `WeightMatrix` re-export 유지 (사용자 코드 무변경)

**검증**: 전체 회귀 + `python -c "from htp import WeightMatrix; from htp.runtime.htp_runtime import WeightMatrix"`

**롤백 단위**: 단일 커밋

### Step 4 — `HubFormationEngine` 파일 분리 + unit test 시작

**변경**:
- `htp/core/hub_formation.py` 생성 — `HubFormationEngine` 이동
- `htp/runtime/htp_runtime.py`: re-export 유지
- `htp/core/__init__.py`: export 추가
- `tests/unit/__init__.py` + `tests/unit/test_engine_di.py` 일부 신설 — `test_hub_formation_engine_with_only_hub_config()` 등

**검증**: 회귀 + `pytest tests/unit/test_engine_di.py -v`

**롤백 단위**: 단일 커밋

### Step 5 — `PruningEngine` + `PruneStrategy` 파일 분리

**변경**:
- `htp/core/pruning.py` 생성 — `PruningEngine`, `PruneStrategy` 이동
- `PruningEngine.__init__(wm, prune_cfg: PruneConfig)` 시그니처 변경
- `HTPRuntime`이 `PruningEngine(self.wm, self.cfg.prune)`로 주입
- re-export 유지, `htp/core/__init__.py` export 추가
- `tests/unit/test_engine_di.py` 확장 (pruning 케이스)

**검증**: 회귀 + unit

**롤백 단위**: 단일 커밋

### Step 6 — `ActivationEngine` + 데코레이터 + `Node` 파일 분리

**변경**:
- `htp/core/activation.py` 생성 — `ActivationEngine`, `Node`, `tag`, `terminal`, `FIRE_FLOOR` 이동
- `ActivationEngine.__init__(wm, activation_cfg: ActivationConfig)` 시그니처 변경
- `HTPRuntime`이 `ActivationEngine(self.wm, self.cfg.activation)`로 주입
- re-export 유지 (`tag`, `terminal`, `Node`, `FIRE_FLOOR`도 포함)
- `htp/core/__init__.py` export 추가
- `tests/unit/test_engine_di.py` 확장 (activation 케이스)

**검증**: 회귀 12/12 라우팅 테스트가 가장 핵심 — `from htp import tag, terminal` 동작 확인

**롤백 단위**: 단일 커밋

### Step 7 — 마무리: 교차 검증 unit test + 문서 갱신

**변경**:
- `tests/unit/test_config_isolation.py` — HTPConfig facade 호환 (~5개)
- `tests/unit/test_import_paths.py` — 공개 API + 기존 경로 모두 작동 (~5개)
- `tests/unit/test_no_circular_deps.py` — `import ast` 기반 순환 의존 검사 (~3개)
- `CLAUDE.md` 갱신: 파일 구조 트리 + Phase 1 엔진 위치
- `docs/03-review/htp-project-review.md`: §4-A 트리 갱신 (선택)
- `htp_runtime.py` 최종 줄 수 확인 (목표 ≤250)

**검증**: 회귀 57/57 + unit ~20개 + `wc -l htp/runtime/htp_runtime.py` ≤ 250

**롤백 단위**: 단일 커밋

---

## 4. API Surface (변경 없음)

### 4.1 Public Imports (`htp/__init__.py`)

모든 심볼 그대로:

```python
# Phase 1
from .runtime.htp_runtime import (
    HTPConfig, WeightMatrix, HubFormationEngine, PruningEngine,
    ActivationEngine, HTPRuntime, Node, RunResult,
    tag, terminal, FIRE_FLOOR,
)
# ... (변경 없음)
```

내부 `htp/runtime/htp_runtime.py`는 다음과 같이 re-export:

```python
# Re-exports for backward compatibility (Steps 3-6)
from htp.core.weight_matrix  import WeightMatrix
from htp.core.hub_formation  import HubFormationEngine
from htp.core.pruning        import PruningEngine, PruneStrategy
from htp.core.activation     import ActivationEngine, Node, tag, terminal, FIRE_FLOOR
from htp.core.config         import HubConfig, PruneConfig, NGEConfig, ActivationConfig
```

### 4.2 New Symbols (선택적 import용)

```python
# 새 권장 import 경로 (옵셔널 — 강제하지 않음)
from htp.core.config import HubConfig, PruneConfig, NGEConfig, ActivationConfig
from htp.core.hub_formation import HubFormationEngine

# 하위 호환 (그대로 사용 가능)
from htp import HubFormationEngine, HTPConfig  # 변경 없음
```

---

## 5. Risks and Mitigation

| Risk | Step | Mitigation |
|------|------|------------|
| 순환 import (`htp/core/activation.py`가 `HTPRuntime` 참조 필요?) | Step 6 | `ActivationEngine`은 `HTPRuntime` 모르며 `WeightMatrix`만 안다. `Node` 콜백 시그니처는 `Callable[[Any], Any]`로 추상 |
| `static/index.html`/`server.py`가 내부 경로 의존 | Step 1 직전 | `grep -r "htp\.runtime\|htp\.core" server.py static/` 사전 확인 (server.py는 `from htp import HTPRuntime`만 쓸 가능성 높음) |
| `HTPConfig(hub_pr_threshold=3.0)` 같은 옛 호출 깨짐 | Step 1 | `__init__` flat 키워드 호환 레이어 + `__getattr__` 위임 (§2.3) |
| 한 엔진의 sub-config가 다른 엔진 필드를 참조해야 함 | Step 2 | 발견 즉시 해당 필드를 적절한 sub-config로 이동 — 사전에 `htp_runtime.py`의 모든 `cfg.*` 접근을 grep으로 분류 (아래 §6) |
| Step 7에서 `htp_runtime.py` > 250줄 | Step 7 | demo 함수를 `htp/runtime/_demo.py`로 분리하면 ~50줄 추가 절감 가능 (옵션) |
| unit test가 잘못된 가정 인코딩 | Steps 4-7 | 각 테스트는 "행동" 검증이 아니라 "구조" 검증 (생성 가능성, 순환 부재, import 경로 존재) — 행동 검증은 기존 회귀 57개로 충분 |

---

## 6. Pre-flight Inventory (Step 1 시작 전 수행)

```bash
# 6.1 HTPConfig 필드 사용 위치 매핑 (어떤 필드가 어느 엔진에 속하는지 결정 근거)
grep -nE "cfg\.|self\.cfg\." htp/runtime/htp_runtime.py | sort -t: -k3

# 6.2 외부 의존 확인 (server.py, static/, tests/가 내부 경로를 쓰는가?)
grep -rE "htp\.runtime\.htp_runtime\.|htp\.core\." server.py static/ tests/

# 6.3 현 회귀 베이스라인
pytest tests/regression/ -v --tb=no | tail -5

# 6.4 현 줄 수 베이스라인
wc -l htp/runtime/htp_runtime.py htp/core/node_generation_engine.py
```

이 4개 커맨드의 결과를 `docs/02-design/features/htp-review-improvements.preflight.md`(선택)에 기록하면 Step 1 시작 시 정확한 필드 분배가 가능.

---

## 7. Testing Strategy

### 7.1 Regression (변경 없음)

| 테스트 슈트 | 통과 기준 | 검증 시점 |
|-----------|----------|---------|
| `tests/regression/` (57개) | 100% | **매 Step 직후** |

### 7.2 New Unit Tests (~20개, Steps 4-7에 분산)

| 파일 | 테스트 개수 | 검증 대상 | 추가 Step |
|------|----------|---------|---------|
| `test_config_isolation.py` | ~5 | sub-config 독립 생성 / HTPConfig facade 호환 (flat 키워드 + `__getattr__`) / 기본값 일치 | Step 7 |
| `test_engine_di.py` | ~5-7 | 각 엔진을 sub-config만으로 생성 / HTPConfig 없이 동작 / 잘못된 config 타입 거부 | Steps 4-6 |
| `test_import_paths.py` | ~5 | `from htp import …` 모든 심볼 / `from htp.runtime.htp_runtime import …` 옛 경로 / `from htp.core.* import …` 새 경로 | Step 7 |
| `test_no_circular_deps.py` | ~3 | `ast.parse`로 `htp/core/*.py`가 `htp.runtime` 미참조 / `htp/core/*` 내부 의존 DAG / 전체 `import htp` 무에러 | Step 7 |

### 7.3 Match Rate 목표

Static + Functional + Contract 3축에서 ≥ 90%. Runtime은 라이브러리 코드라 N/A (HTTP 서버 없음).

---

## 8. Test Plan (gap-detector L1-L3 적용 안 됨)

이 feature는 라이브러리 리팩토링이라 L1(API)/L2(UI)/L3(E2E) 적용 불가. **Check 단계 평가는 다음으로 대체**:

| Level | 적용 | 대체 검증 |
|-------|----|---------|
| L1 (API) | N/A | `python -c "from htp import *"` + `python -m htp.runtime.htp_runtime` (demo) |
| L2 (UI) | N/A | — |
| L3 (E2E) | N/A | `pytest tests/regression/` (행동 동일성) |
| L4 (Perf) | 부분 | `time python -c "import htp"` 전후 비교 (loose budget: ±20%) |
| L5 (Security) | N/A | — |

---

## 9. Decision Record

| Decision | Options | Selected | Rationale |
|----------|---------|----------|-----------|
| DI 방식 | Manual / DI framework / Service locator | **Manual Constructor Injection** | Plan 결정 — 외부 라이브러리 없이 dataclass + 명시 주입으로 충분 |
| Config 구조 | 단일 / Sub-config / Pydantic | **Sub-config + Facade** | Plan 결정 — 호환성과 결합도 둘 다 해소 |
| 파일 위치 | `htp/engines/` / `htp/core/` 확장 / `htp/runtime/` 분할 | **`htp/core/` 확장** | Plan 결정 — 기존 `node_generation_engine.py`와 의미 통일 |
| API 호환 | Breaking / 100% | **100% 유지** | Plan 결정 |
| 마이그레이션 전략 | Bottom-up(7) / Big-Bang(1) / Two-Phase(2) | **Bottom-up Incremental (7 steps)** | Design Checkpoint 3 사용자 선택 — 최대 안전성 |
| 테스트 추가량 | 0 / ~5 / ~20 | **~20 (전면 unit)** | Plan 결정 — 향후 Phase 확장 안전망 |
| 의존 방향 | 양방향 허용 / 단방향 DAG | **단방향 DAG (`core` ← `runtime`)** | 순환 import 영구 차단 |
| HTPConfig 호환 | 제거 / facade 유지 | **Facade + `__getattr__` 위임** | 옛 호출 패턴(`HTPConfig(hub_pr_threshold=3.0)`) 보존 |

---

## 10. Architecture Considerations

### 10.1 Project Level

Research / Library (Python package). bkit 표준 3-level 외 — Enterprise에 가깝지만 microservices/DI framework 미사용.

### 10.2 Key Tools

| Category | Used |
|----------|------|
| Language | Python 3.10+ |
| ML Framework | PyTorch |
| Test | pytest |
| Lint | (있다면) — 없으면 본 사이클 범위 밖 |
| Type Check | (옵션) — Step 4 이후 mypy 추가 검토 가능 |

---

## 11. Implementation Guide

### 11.1 Recommended Order

위 §3 Migration Plan 그대로. Step 1 → Step 7 순서.

### 11.2 Key Files Reference

| Step | Files Created | Files Modified |
|------|---------------|----------------|
| 1 | `htp/core/config.py` | `htp/runtime/htp_runtime.py` (HTPConfig facade) |
| 2 | — | `htp/runtime/htp_runtime.py` (HubFormationEngine ctor) |
| 3 | `htp/core/weight_matrix.py` | `htp/runtime/htp_runtime.py`, `htp/core/__init__.py` |
| 4 | `htp/core/hub_formation.py`, `tests/unit/test_engine_di.py` (시작) | 위 동일 |
| 5 | `htp/core/pruning.py` | 위 동일 + `test_engine_di.py` 확장 |
| 6 | `htp/core/activation.py` | 위 동일 + `test_engine_di.py` 확장 |
| 7 | `tests/unit/test_config_isolation.py`, `test_import_paths.py`, `test_no_circular_deps.py` | `CLAUDE.md` |

### 11.3 Session Guide (Module Map)

| Module Key | Description | Estimated Time | Independent? |
|-----------|------------|---------------|--------------|
| `step-1` | sub-config 신설 + HTPConfig facade | ~30분 | Yes |
| `step-2` | HubFormationEngine DI 전환 | ~15분 | No (Step 1 의존) |
| `step-3` | WeightMatrix 파일 분리 | ~15분 | No (Step 1 의존) |
| `step-4` | HubFormationEngine 파일 분리 + unit test 시작 | ~30분 | No (Step 2, 3 의존) |
| `step-5` | PruningEngine 파일 분리 + DI | ~30분 | No (Step 1, 3 의존) |
| `step-6` | ActivationEngine 파일 분리 + DI | ~45분 | No (Step 1, 3 의존) |
| `step-7` | 통합 unit test + 문서 갱신 | ~45분 | No (Steps 1-6 의존) |

**권장 세션 분할**:
- Session 1: Steps 1-3 (인프라 + 첫 파일 이동) — `/pdca do htp-review-improvements --scope step-1,step-2,step-3`
- Session 2: Steps 4-6 (나머지 엔진들) — `/pdca do htp-review-improvements --scope step-4,step-5,step-6`
- Session 3: Step 7 (마무리 + 모든 unit test) — `/pdca do htp-review-improvements --scope step-7`

---

## 12. Next Steps

1. [ ] Pre-flight inventory 실행 (§6의 4개 grep/pytest)
2. [ ] `/pdca do htp-review-improvements --scope step-1,step-2,step-3` (Session 1)
3. [ ] 회귀 통과 확인 후 Session 2 진행
4. [ ] Session 3 후 `/pdca analyze htp-review-improvements`
5. [ ] Match Rate ≥ 90% 이면 `/pdca report`

---

## Version History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 0.1 | 2026-05-16 | Initial — Bottom-up Incremental (Option A) 선택 | Mindbuild |
