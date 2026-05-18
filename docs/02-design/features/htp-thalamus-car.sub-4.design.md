---
template: design
feature: htp-thalamus-car (sub-4)
date: 2026-05-18
author: Mindbuild
project: HTP (Hub Topology Programming)
predecessor: Bridge Integration (시스템 A↔B 검증 PASS), sub-3 (CoherenceGate)
plan_ref: docs/01-plan/features/htp-thalamus-car.plan.md §5 Stage 4-5 + §3 보완
status: Draft (사용자 확인 대기)
---

# HTP sub-4 Design — ExternalRegion + LLMRegion + PipelinedBrainRuntime

**범위**: Plan §5 Stage 4 (ExternalRegion + LLMRegion + CostRouter 4-Level) + Stage 5
(PipelinedBrainRuntime). Plan §3 보완 작업 C-1~C-4 모두 흡수.

**선행 검증**: Bridge Integration §9 Q1/Q2/Q3 3/3 PASS — 시스템 A 의 가치 검증 완료.
LLMRegion 진입 명분 확보. 227/227 baseline.

---

## Context Anchor

| 키 | 값 |
|----|----|
| **WHY** | LLMRegionRuntime 이 RegionRuntime 을 상속해 PageRank/Hebbian/NGE 를 불필요하게 끌어옴 → graphify 상 LLM 관련 isolated 노드 다수. ExternalRegion 추상으로 분리 필요 (G3). |
| **WHO** | HTP 사용자 — LLM 호출을 비용 인식 가능하게 Region 처럼 다루고 싶음. |
| **RISK** | (1) 회귀 깨짐 — `LLMRegionRuntime` 사용 코드 (`async_brain_runtime.py` 38줄) 깨질 위험. (2) CostRouter 4-method 호출 경로 변경 시 외부 통합 깨짐. (3) PipelinedBrainRuntime 의 비동기 동시성 버그. |
| **SUCCESS** | (1) Stage 4 후 graphify isolated 노드 50% 이상 감소. (2) `select_level` 4-level 라우팅 결정 동작. (3) Stage 5 후 throughput ≥ 1.5× 순차. (4) 회귀 227/227 + 신규 본선 테스트 통과. |
| **SCOPE** | ExternalRegion / LLMRegion / CostRouter.select_level / LLMRegionRuntime archive 이동 / PipelinedBrainRuntime. Embedding bridge (sub-5 이미 완료) 또는 vector default 전환 (sub-6) 은 OUT. |

---

## 1. 현재 상태 진단

### 1-1. 기존 LLM 코드 구조

```
htp/llm/  (609 줄)
  __init__.py           — 공개 export
  llm_node.py           (145줄)  Anthropic API 동기/비동기 래퍼 + 비용 추적 + MockLLMNode
  llm_region_runtime.py (178줄)  LLMRegionRuntime(RegionRuntime) — 상속 ⚠
  cost_router.py         (86줄)  CostRouter — 7-method (update/pressure/status/
                                  suggest_model/routing_score/should_block/report)
```

### 1-2. 진단 결과 — G3 재확인

`LLMRegionRuntime(RegionRuntime)` 이 상속함으로써:

| RegionRuntime 메서드 | LLM 에서 사용? | graphify 영향 |
|----------------------|:--------------:|---------------|
| `_ensure_built` | ✓ (LLMNode 빌드용) | 부분 사용 |
| `run` / `arun` | ✓ | LLM 만 호출 |
| `collect_signal` (PageRank/fire_rate) | ✗ | **dead code → isolated** |
| `apply_suppression` | ✗ | **dead code → isolated** |
| `_entropy_concentration` | ✗ | **dead code → isolated** |

→ LLM Region 에는 의미 없는 RegionRuntime 메서드 다수가 dead code. graphify 분석 시
LLM 관련 노드가 hub 그래프에서 isolated 로 나타남 (Plan G3).

### 1-3. 사용 코드

```
htp/__init__.py:49     export LLMRegionRuntime
htp/llm/__init__.py:5  export LLMRegionRuntime
htp/runtime/async_brain_runtime.py:38   docstring 예시
```

→ 실제 import 사용처는 export 만. 회귀 영향 최소.

---

## 2. Architecture Options

### Option A — Minimal (변경 최소)

`ExternalRegion` 추상만 신설하고, `LLMRegion(ExternalRegion)` 도 사실상 기존
`LLMRegionRuntime` 의 얇은 wrapper. `LLMRegionRuntime` 유지 (deprecate 안 함).

| 장점 | 단점 |
|------|------|
| 회귀 위험 최소 | G3 해결 안 됨 (LLMRegionRuntime 잔존) |
| 30-40분 작업 | graphify isolated 감소 미달 |
| | C-1~C-4 보완 일부만 흡수 |

### Option B — Clean (Plan §5 전면 흡수) — **권장**

`ExternalRegion` 추상 (RegionRuntime 비상속) + `LLMRegion(ExternalRegion)` 신규.
`LLMRegionRuntime` 은 `archive/deprecated_phase4/` 이동. `CostRouter.select_level()`
4-level 추가, 기존 7-method 보존. `PipelinedBrainRuntime` 신규.
C-1~C-4 모두 흡수.

| 장점 | 단점 |
|------|------|
| Plan §5 + §3 완전 충족 | 작업량 ~3h |
| G3 (graphify isolated) 본질 해결 | LLMRegionRuntime 사용처 (export) 도 정리 필요 |
| PipelinedBrainRuntime 즉시 진입 | |
| Stage 4 + 5 한 cycle 마무리 | |

### Option C — Pragmatic (Stage 4 만, Stage 5 분리)

Option B 와 동일하나 PipelinedBrainRuntime 은 별도 sub-4-b cycle 로 분리.
이번 cycle 은 Stage 4 + C-1~C-4 만.

| 장점 | 단점 |
|------|------|
| 작은 스코프로 빠른 검증 (~1.5h) | Plan §5 미완 (Stage 5 후속) |
| Stage 4 결과 확인 후 Stage 5 계획 정밀화 가능 | sub-cycle 추가 (관리 비용) |

### 비교 매트릭스

| 항목 | A | **B** | C |
|------|:--:|:----:|:--:|
| Plan §5 흡수 | 부분 | ✓ 완전 | Stage 4 만 |
| G3 (isolated) 해결 | ✗ | ✓ | ✓ |
| 회귀 위험 | 최저 | 중 | 중 |
| 작업량 | 0.5h | 3h | 1.5h |
| 후속 cycle 필요 | 많음 | 적음 | sub-4-b |
| **권장** | | **★** | |

---

## 3. 선택안 (Option B) 상세 설계

### 3-1. ExternalRegion 추상

**위치**: `htp/runtime/external_region.py` (Plan FR-16)

```python
"""
ExternalRegion — RegionRuntime 비상속 추상. PageRank/Hebbian/NGE 없이
Region 의 핵심 interface (collect_signal, run, apply_suppression) 만 만족.

graphify: LLM/외부 API 노드를 brain-like 구조 그래프에서 분리.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from htp.thalamus.region_signal import RegionSignal


class ExternalRegion(ABC):
    """Region interface 만 만족하는 외부 호출 추상.

    RegionRuntime 의 PageRank/Hebbian/NGE 를 끌어오지 않음.
    하위 클래스: LLMRegion, FutureToolRegion (예: search/RAG), ...
    """

    region_name: str
    specialty:   str
    step:        int

    @abstractmethod
    def run(self, data: Any) -> Any:
        """동기 외부 호출. 반환: 도메인별 결과 dict."""
        ...

    async def arun(self, data: Any) -> Any:
        """비동기 외부 호출. default = sync wrap.
        하위 클래스가 진짜 async 면 override."""
        return self.run(data)

    @abstractmethod
    def collect_signal(self) -> RegionSignal:
        """Thalamus 가 보는 외부 region 의 신호 — 가짜 hub_strength/fire_rate.
        precision 은 외부 호출 신뢰도 (예: CostRouter.pressure 역수)."""
        ...

    def apply_suppression(self, strength: float) -> None:
        """default = no-op. 외부 호출은 suppression 영향 없음 (또는 호출 빈도로 변환)."""
        pass
```

### 3-2. LLMRegion(ExternalRegion)

**위치**: `htp/llm/llm_region.py` (Plan FR-17)

```python
"""LLMRegion — ExternalRegion 의 LLM 구현. CostRouter 통합.

기존 LLMRegionRuntime 의 LLM 호출 로직을 받되 RegionRuntime 비상속.
"""
from __future__ import annotations

import time
import numpy as np

from htp.runtime.external_region import ExternalRegion
from htp.thalamus.region_signal  import RegionSignal
from .llm_node                   import LLMNode, MockLLMNode
from .cost_router                import CostRouter


# 기존 llm_region_runtime.py 의 SPECIALTY_PROMPTS 그대로 재사용
SPECIALTY_PROMPTS: dict[str, str] = { ... }   # 5개


class LLMRegion(ExternalRegion):
    """LLM 호출을 ExternalRegion 으로 추상화.

    LLMNode 는 내부 멤버 (C-2 옵션 A — LLMNode 내부 멤버로 유지).
    """

    def __init__(
        self,
        region_name: str,
        specialty: str,
        model: str = "claude-sonnet-4-6",
        system: str | None = None,
        budget: float = 0.01,
        use_mock: bool = False,
    ):
        self.region_name = region_name
        self.specialty   = specialty
        self.step        = 0

        self.model    = model
        self.system   = system or SPECIALTY_PROMPTS.get(
            specialty, f"You are a {specialty} specialist. Return JSON.",
        )
        self.router   = CostRouter(budget_per_step=budget)
        self.use_mock = use_mock

        NodeClass = MockLLMNode if use_mock else LLMNode
        self._llm_node = NodeClass(
            name=f"{region_name}_llm",
            model=self.router.suggest_model(self.model),
            system=self.system,
            tags=set(specialty.replace("_", " ").split()),
        )
        self._last_result: dict | None = None

    def run(self, data):
        if self.router.should_block():
            return {"text": "cost_blocked", "label": "blocked"}
        result = self._llm_node.run(data)
        if self._llm_node._token_log:
            last = self._llm_node._token_log[-1]
            self.router.update(last["cost"], last["ms"])
        self.step += 1
        self._last_result = result
        return result

    async def arun(self, data):
        if self.router.should_block():
            return self._last_result or {"text": "cost_blocked", "label": "blocked"}
        t0 = time.perf_counter()
        result = await self._llm_node.arun(data)
        elapsed = (time.perf_counter() - t0) * 1000
        if self._llm_node._token_log:
            last = self._llm_node._token_log[-1]
            self.router.update(last["cost"], elapsed)
        self.step += 1
        self._last_result = result
        return result

    def collect_signal(self) -> RegionSignal:
        """CostRouter pressure 를 precision 의 역수로 환산."""
        import torch
        pressure = self.router.pressure()   # 0~1
        precision = 1.0 / max(0.1, 1.0 - pressure)  # pressure 높을수록 precision 낮음
        return RegionSignal(
            region_id=self.region_name,
            hub_strength=0.0,
            fire_rate=float(min(1.0, self.step / 100)),
            top_hubs=[],
            overload=self.router.should_block(),
            output_vec=torch.zeros(1),    # 외부 Region 은 의미 vec 없음 (or future bridge)
            precision=precision,
        )

    def cost_report(self) -> str:
        lines = [f"  [{self.region_name}] {self.router.report()}"]
        lines.append(self._llm_node.cost_report())
        return "\n".join(lines)
```

### 3-3. CostRouter.select_level (C-3 보존 + 4-Level)

**위치**: `htp/llm/cost_router.py` (Plan FR-18)

기존 7-method (update/pressure/status/suggest_model/routing_score/should_block/report) **보존**.
추가:

```python
def select_level(self, query_complexity: float = 0.5) -> int:
    """4-Level 의사결정 트리 — 비용 압박과 쿼리 복잡도 기반.

    Level 1 (Local — TF-IDF/keyword) : pressure < 0.3 AND complexity < 0.3
    Level 2 (sLLM — local 384-dim)   : pressure < 0.5 AND complexity < 0.5  (EmbeddingBridge)
    Level 3 (API 소형 — Haiku)       : pressure < 0.8 AND complexity < 0.8
    Level 4 (API 대형 — Sonnet/Opus) : 그 외 (high complexity OR low pressure)

    Note: pressure 가 높을수록 비싼 모델 회피 → 압박 시 level 하향.
    Plan §5 Stage 4 FR-18.
    """
    p = self.pressure()
    if p > 0.8:
        return 1   # 비용 극압박 — 로컬 폴백
    if p > 0.5 and query_complexity < 0.5:
        return 2   # 가벼운 쿼리 — sLLM
    if query_complexity > 0.8:
        return 4   # 복잡 쿼리 — 대형 모델
    if query_complexity > 0.5:
        return 3   # 중간 쿼리 — API 소형
    return 2       # default — sLLM
```

### 3-4. LLMRegionRuntime archive 이동 (Plan FR-20)

```bash
mkdir -p archive/deprecated_phase4/
git mv htp/llm/llm_region_runtime.py archive/deprecated_phase4/
```

`htp/llm/__init__.py` 와 `htp/__init__.py` 의 export 제거. backward-compat 위해 deprecation warning 옵션:

```python
# htp/llm/__init__.py 변경
from .llm_node     import LLMNode, MockLLMNode
from .cost_router  import CostRouter
from .llm_region   import LLMRegion             # 신규
# LLMRegionRuntime  → archive/deprecated_phase4/ 이동 (sub-4)

__all__ = ["LLMNode", "MockLLMNode", "CostRouter", "LLMRegion"]
```

### 3-5. PipelinedBrainRuntime (Plan FR-22)

**위치**: `htp/runtime/pipelined_brain.py`

```python
"""PipelinedBrainRuntime — L3 파이프라인 병렬성.

기존 AsyncBrainRuntime 보존 (대체 아닌 추가). 차이점:
  AsyncBrainRuntime  : asyncio.gather 로 동일 step 의 region 병렬
  PipelinedBrainRuntime: step N+1 region 호출이 step N 의 PFC binding 과 겹침
                          (3-stage pipeline: collect → bind → top-down)

Throughput 목표: ≥ 1.5× AsyncBrainRuntime (Plan SUCCESS §3).
"""
from __future__ import annotations

import asyncio
from collections import deque
from typing import Any

from .brain_runtime import BrainRuntime


class PipelinedBrainRuntime(BrainRuntime):
    """3-stage pipeline 병렬 실행."""

    def __init__(self, *args, buffer_size: int = 3, **kwargs):
        super().__init__(*args, **kwargs)
        self.buffer_size = buffer_size
        self._stage_collect = deque(maxlen=buffer_size)
        self._stage_bind    = deque(maxlen=buffer_size)
        self._stage_topdown = deque(maxlen=buffer_size)

    async def pipelined_step(self, inputs: list[Any]) -> list[Any]:
        """N 개 입력을 pipeline 으로 실행. 출력 순서 보존."""
        ...   # 상세 알고리즘은 Do 단계에서 결정
```

### 3-6. Module Map (Session Guide)

| Module | 위치 | 변경 | 의존 |
|--------|------|------|------|
| M1 ExternalRegion | `htp/runtime/external_region.py` 신규 | +60줄 | abc, RegionSignal |
| M2 LLMRegion | `htp/llm/llm_region.py` 신규 | +130줄 | ExternalRegion, LLMNode, CostRouter |
| M3 CostRouter +select_level | `htp/llm/cost_router.py` 확장 | +25줄 | (기존 7-method 유지) |
| M4 LLMRegionRuntime archive | `archive/deprecated_phase4/llm_region_runtime.py` | git mv | — |
| M5 __init__ 갱신 | `htp/llm/__init__.py`, `htp/__init__.py` | -2 +1 export | — |
| M6 PipelinedBrainRuntime | `htp/runtime/pipelined_brain.py` 신규 | +120줄 | BrainRuntime, asyncio |
| M7 LLM Region 데모 (C-1) | `examples/llm_region_demo.py` 신규 | +50줄 | LLMRegion (MockLLMNode) |
| M8 테스트 | `tests/regression/test_sub4_*.py` 신규 | +200줄 | — |

### 3-7. Recommended Session Plan

| Session | Modules | 소요 | 검증 |
|---------|---------|------|------|
| **A** Stage 4 코어 | M1 + M2 + M3 + M5 | ~1.5h | 신규 LLMRegion + CostRouter 4-level 단위 테스트 |
| **B** archive + 데모 | M4 + M7 | ~30분 | graphify 재실행 isolated 감소 확인 (C-4) |
| **C** Pipeline (Stage 5) | M6 + M8 | ~1h | throughput ≥ 1.5× 측정 |

총 ~3h. 회귀 1-pass 끝까지 보존.

---

## 4. C-1 ~ C-4 보완 결정

### C-1: LLMRegion 사용 데모 추가 (Session B)

`examples/llm_region_demo.py` — MockLLMNode 로 actual API 키 없이 실행 가능.
3 Region (language / code / memory) ingest → BrainRuntime 통합 데모.

### C-2: LLMNode 클래스 처리 정책 — **옵션 A 채택**

LLMNode 는 LLMRegion 의 **내부 멤버 (`self._llm_node`)** 로 유지.
이유:
- LLMNode 자체는 HTP 노드가 아니라 Anthropic API 래퍼 + 비용 추적
- LLMRegion 의 한 구현 디테일 (다른 ExternalRegion 은 LLMNode 불필요)
- 외부 import 없음 (`htp/__init__.py` 에서만 export)

### C-3: CostRouter 기존 인터페이스 보존 (§3-3)

7-method (`update`/`pressure`/`status`/`suggest_model`/`routing_score`/`should_block`/`report`)
모두 유지. `select_level` 만 추가.

### C-4: graphify isolated 감소 정량 기준

Stage 4 Go/No-Go:
1. `/graphify` 또는 동등 분석 도구로 LLM 관련 노드 (`llm_*`, `cost_router_*`) 의
   isolated 노드 수 측정
2. archive 이동 후 재측정 → **50% 이상 감소** 확인
3. 측정 결과 docs/03-analysis/htp-sub-4.analysis.md 에 기록

---

## 5. 테스트 계획

### 5-1. 자동 테스트 (신규 ~10건)

```python
# tests/regression/test_sub4_external_region.py
test_external_region_abstract_cannot_instantiate
test_external_region_subclass_must_implement_run
test_external_region_subclass_must_implement_collect_signal
test_external_region_default_arun_wraps_sync

# tests/regression/test_sub4_llm_region.py
test_llm_region_inherits_external_region
test_llm_region_mock_run_returns_dict
test_llm_region_collect_signal_precision_reflects_pressure
test_llm_region_cost_blocked_returns_cached
test_llm_region_async_run_works

# tests/regression/test_sub4_cost_router.py
test_cost_router_existing_7_methods_preserved
test_cost_router_select_level_default
test_cost_router_select_level_high_pressure_to_level_1
test_cost_router_select_level_high_complexity_to_level_4
test_cost_router_select_level_mid_returns_2_or_3

# tests/regression/test_sub4_pipeline.py  (Stage 5)
test_pipelined_brain_runtime_inherits
test_pipelined_step_preserves_order
test_pipelined_throughput_at_least_1_5x_async   # micro-benchmark
```

### 5-2. 수동 검증

- `/graphify htp/llm` 으로 isolated 노드 변화 측정 (C-4)
- `examples/llm_region_demo.py` 실행 확인 (C-1)

### 5-3. Go/No-Go (Plan §SUCCESS)

- 신규 자동 테스트 모두 통과
- 기존 227 PASS 유지 → 237/237 신규 baseline
- graphify isolated 50% 이상 감소
- PipelinedBrainRuntime throughput ≥ 1.5× AsyncBrainRuntime (단순 mock 시나리오)

---

## 6. DAG

### 6-1. 변경 후 의존 방향

```
htp/runtime/external_region.py
  ↑ (한정 import)
htp/llm/llm_region.py
  → htp/runtime/external_region    (단방향)
  → htp/llm/llm_node
  → htp/llm/cost_router
  → htp/thalamus/region_signal     (RegionSignal — 기존 사용)

htp/runtime/pipelined_brain.py
  → htp/runtime/brain_runtime      (상속)
  → asyncio
```

### 6-2. DAG 룰 (test_no_circular_deps.py 영향)

기존 룰 영향 없음:
- `htp/core/*` 는 변경 없음 (변경 없음 = 그대로 통과)
- `htp/knowledge/*` 는 변경 없음
- `htp/thalamus/*` 는 변경 없음

새 DAG 룰 추가 검토:
- `htp/llm/llm_region.py` 가 `htp/runtime` 을 import — 정상 (llm 는 runtime 의 외부 layer)
- `htp/runtime/external_region.py` 가 `htp/thalamus.region_signal` 만 import — 정상 (thalamus 는 모든 layer 의 base)

→ 신규 DAG 테스트 불필요. 기존 룰만으로 안전.

---

## 7. Risk + Mitigation

| Risk | 가능성 | 영향 | 완화 |
|------|:------:|:----:|------|
| `LLMRegionRuntime` 외부 사용자 코드 깨짐 | 낮음 | 중 | export 제거 + deprecation note. 외부 사용자 발견 시 hot-fix |
| `CostRouter.select_level` 의 4-level 기준 부적합 | 중 | 낮음 | Plan §SUCCESS Level 1-2 비율 70% 측정 후 조정. 초기 default 만 sane. |
| `PipelinedBrainRuntime` 의 concurrency 버그 | 중 | 중 | buffer_size=3 작게 시작. 단위 테스트로 ordering 보존 검증. |
| graphify isolated 50% 미달 | 중 | 낮음 | 측정 결과 분석 → 추가 archive 후보 식별 (e.g., 더 많은 dead code) |

---

## 8. 구현 순서

```
1. Session A — Stage 4 코어 (ExternalRegion + LLMRegion + CostRouter.select_level)
   ↓ 회귀 통과 확인
2. Session B — archive + 데모 + graphify 측정 (C-1, C-4)
   ↓ 회귀 통과 확인
3. Session C — Stage 5 PipelinedBrainRuntime
   ↓ throughput 측정
4. Check phase — Match Rate, Go/No-Go
5. Report — docs/04-report/htp-thalamus-car.sub-4.report.md
```

---

## 9. 사용자 확인 요청

이 design 에 대해 아래 3 항목 확인 후 Do 진입:

1. **Architecture 옵션**: B (Clean, Plan §5 전면 흡수) 동의?
2. **C-2 LLMNode 정책**: 옵션 A (LLMRegion 내부 멤버 유지) 동의?
3. **Session 분할**: A → B → C 순차 진행 또는 한 번에?
