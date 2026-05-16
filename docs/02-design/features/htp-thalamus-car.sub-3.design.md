---
template: design
feature: htp-thalamus-car
sub_cycle: sub-3 (Stage 3)
date: 2026-05-17
author: Mindbuild
project: HTP (Hub Topology Programming)
status: Draft
selected_option: B — Modular Strategy
---

# htp-thalamus-car sub-3 Design — CoherenceGate + Memory novelty 연동

> **Summary**: Region 응답의 temporal binding 도입. **Option B (Modular Strategy)** 채택 — `CoherenceStrategy` Protocol + `PairwiseCoherenceGate` 1차 구현. 향후 N≥16 시 LSH 구현체가 동일 Protocol 의 추가 구현체로 끼움. `MemorySystem.swr_priority` 확장으로 conflict 가 novelty 증폭 신호로 작동.
>
> **Selected Architecture**: Option B — Modular Strategy (router/ 패턴 일관성)
> **Predecessor**: sub-2 (commit `36760fc` — Stage 1 + 2 완료, Match Rate 99%)
> **Test Target**: 140 → **146** (Stage 3 +6)

---

## Context Anchor (Plan 에서 전파)

| Key | Value |
|-----|-------|
| **WHY** | (G2) 다중 Region 응답의 temporal binding 부재 — A가 "사과"로 B가 "토마토"로 해석해도 불일치 미감지. CoherenceGate 가 이를 정량화. |
| **WHO** | HTP 개발자 + BrainRuntime 사용자 (coherence 옵션 off 기본 → 회귀 보호). |
| **RISK** | (R2 from Plan) CoherenceGate O(N²) 스케일링 — N≥16 LSH 전환 임계. (R8 신규) BrainRuntime 흐름 변경으로 Stage 5 통합 테스트 회귀 위험. |
| **SUCCESS** | 누적 140 → **146**. 의도적 conflict 10건 중 ≥9건 감지, 정합 10건 중 false positive ≤1건. SWR priority 가 conflict_magnitude 로 증폭. |
| **SCOPE** | Stage 3 만. LSH 구현체는 sub-3 OUT (N≥16 시 별도 사이클). |

---

## 1. Overview

### 1.1 Selected Architecture: Option B — Modular Strategy

```
htp/thalamus/
├── coherence/                       [신규 패키지 — sub-2 router/ 패턴 일관]
│   ├── __init__.py                  공개 export
│   ├── base.py                      CoherenceStrategy Protocol + 공통 헬퍼
│   └── pairwise.py                  PairwiseCoherenceGate — O(N²) sub-3 기본
│                                    (lsh.py 는 N≥16 시 sub-별도 사이클)
├── types.py                         [신규] BoundResponse dataclass + 공통 타입
├── core_cells.py                    (sub-2 무변경)
└── ...

htp/memory/
└── memory_system.py                 수정 — swr_priority(novelty, reward, conflict)

htp/runtime/
└── brain_runtime.py                 수정 — coherence DI + Region 응답 binding
```

### 1.2 Design Goals

| ID | Goal | 측정 방법 |
|----|------|---------|
| G1 | 회귀 140 + 신규 6 = **146/146** 통과 | `pytest -q` |
| G2 | CoherenceStrategy 다형성 (Pairwise + 향후 LSH) | `isinstance(gate, CoherenceStrategy)` |
| G3 | BrainRuntime 기본 동작 무변경 (coherence=None 시 skip) | 기존 통합 테스트 7건 무영향 |
| G4 | DAG — `coherence/*` 는 `htp.runtime` 미참조 | `test_no_circular_deps.py` parametrize 확장 |
| G5 | conflict 감지 precision/recall — 10/9 ≥, false positive ≤1/10 | 의도적 conflict fixture |
| G6 | SWR priority 가 conflict_magnitude 로 단조 증가 | `test_swr_priority_conflict_amplification` |

### 1.3 Design Principles

1. **OCP 일관성** — sub-2 router/ 패턴 그대로 적용. CoherenceStrategy Protocol.
2. **회귀 보호 우선** — BrainRuntime `coherence: CoherenceStrategy | None = None` 기본. None 시 기존 동작.
3. **단일 책임** — types.py 가 BoundResponse 소유, coherence/* 가 알고리즘만, BrainRuntime 이 oracle 역할.
4. **DAG 단방향** — `coherence/* → types.py → numpy` 단방향. `coherence/* → memory/runtime` 금지.

---

## 2. Architecture Detail

### 2.1 CoherenceStrategy Protocol (htp/thalamus/coherence/base.py)

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class CoherenceStrategy(Protocol):
    """Region 응답들의 temporal binding 인터페이스.

    sub-3: PairwiseCoherenceGate (O(N²))
    future: LSHCoherenceGate (N≥16 시 별도 사이클)
    """

    @property
    def mode(self) -> str: ...    # "pairwise" | "lsh" | ...

    def bind(self,
             responses: "list[RegionResponse]") -> "BoundResponse":
        """다중 Region 응답 → 단일 BoundResponse.

        - coherence: 응답 간 평균 일치도 [0, 1]
        - conflict:  최대 불일치 magnitude [0, 1]
        - fused_vec: precision-weighted 평균 출력
        - escalate_to_pfc: conflict 가 threshold 초과 시 True
        """
        ...
```

### 2.2 BoundResponse + 공통 타입 (htp/thalamus/types.py)

```python
@dataclass
class RegionResponse:
    """Region 의 응답 단위 — CoherenceGate input."""
    region_id: str
    output_vec: np.ndarray           # Region 의 의미 출력
    precision: float = 1.0           # Friston precision

@dataclass
class BoundResponse:
    """Plan FR-13 — 다중 응답 binding 결과."""
    responses: list[RegionResponse]   # 원본 보존
    coherence: float                  # 평균 pairwise 일치도
    conflict:  float                  # 최대 pairwise 불일치
    fused_vec: np.ndarray             # precision-weighted 평균
    escalate_to_pfc: bool             # conflict > threshold 시 True
```

### 2.3 PairwiseCoherenceGate (htp/thalamus/coherence/pairwise.py)

```python
class PairwiseCoherenceGate:
    """O(N²) pairwise cosine similarity 기반 binding.

    Plan §R2: N≥16 시 LSH 전환. sub-3 은 N<16 가정.
    """

    def __init__(self,
                 conflict_threshold: float = 0.3,
                 escalation_threshold: float = 0.7):
        """
        conflict_threshold:    1 - cosine > threshold 면 pair conflict.
        escalation_threshold:  최대 conflict 가 threshold 초과 시 PFC 에스컬레이션.
        """
        self.conflict_threshold   = conflict_threshold
        self.escalation_threshold = escalation_threshold

    @property
    def mode(self) -> str: return "pairwise"

    def bind(self, responses: list[RegionResponse]) -> BoundResponse:
        if len(responses) < 2:
            # 단일 응답: 자기 자신 fused, coherence=1, conflict=0
            vec = (responses[0].output_vec if responses
                   else np.zeros(64, dtype=np.float64))
            return BoundResponse(
                responses=list(responses),
                coherence=1.0 if responses else 0.0,
                conflict=0.0,
                fused_vec=vec,
                escalate_to_pfc=False,
            )

        # 1. Pairwise cosine — O(N²)
        N = len(responses)
        sims: list[float] = []
        for i in range(N):
            for j in range(i + 1, N):
                sims.append(_cosine(responses[i].output_vec,
                                    responses[j].output_vec))
        coherence = float(np.mean(sims))
        max_disagreement = max(1.0 - s for s in sims)
        conflict = float(max(0.0, max_disagreement))

        # 2. Precision-weighted 평균 fusion
        weights = np.array([r.precision for r in responses], dtype=np.float64)
        wsum = float(weights.sum()) or 1.0
        stacked = np.stack([r.output_vec for r in responses])
        fused = (weights[:, None] * stacked).sum(axis=0) / wsum

        return BoundResponse(
            responses=list(responses),
            coherence=coherence,
            conflict=conflict,
            fused_vec=fused,
            escalate_to_pfc=(conflict > self.escalation_threshold),
        )


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))
```

### 2.4 BrainRuntime CoherenceGate 삽입 지점

```python
class BrainRuntime:
    def __init__(self, ..., coherence: "CoherenceStrategy | None" = None):
        ...
        # 기본 None — 회귀 보호 (Plan FR-14 의 옵션 활성)
        self.coherence = coherence

    def step(self, ...):
        # ... 기존 Region 응답 수집 후 ...
        if self.coherence is not None and len(region_responses) >= 2:
            bound = self.coherence.bind(region_responses)
            # bound.fused_vec → PFC 입력
            # bound.conflict → MemorySystem.swr_priority 증폭
            # bound.escalate_to_pfc → PFC top-down 트리거
            ...
        # ... 기존 PFC 호출 ...
```

### 2.5 MemorySystem.swr_priority 확장

```python
class MemorySystem:
    def consolidate(self, ..., conflict_magnitude: float = 0.0):
        """Plan FR-15: priority = novelty × reward × (1 + conflict_magnitude).

        기존 동작 (conflict_magnitude=0) 시 priority = novelty × reward
        → 기존 calling convention 무변경 (회귀 보호).
        """
        novelty = self._compute_novelty(...)
        reward  = self._compute_reward(...)
        priority = novelty * reward * (1.0 + conflict_magnitude)
        if priority >= self.swr_threshold:
            self._tag_for_consolidation(...)
```

---

## 3. DAG 의존 방향

```
htp/thalamus/coherence/*  ──→  htp/thalamus/types.py
                          ──→  numpy

htp/thalamus/types.py     ──→  numpy

htp/runtime/brain_runtime.py  ──→  htp/thalamus/coherence/*  (DI)
                              ──→  htp/thalamus/types.py

htp/memory/memory_system.py   ──→  (no new dependency — conflict_magnitude 인자만)

금지: coherence/* → htp.runtime / htp.memory / htp.knowledge / htp.thalamus.router
```

### M8-cont: `tests/unit/test_no_circular_deps.py` parametrize 확장
- `_COHERENCE_DIR = htp/thalamus/coherence` 추가
- 금지: `htp.runtime`, `htp.memory`, `htp.knowledge`

---

## 4. Stage 별 구현 순서 (Session Guide)

### Module Map

| Module | 파일 | 의존 | 테스트 |
|--------|------|-----|------|
| **M1** types | `htp/thalamus/types.py` | numpy | +1 (BoundResponse default) |
| **M2** coherence.base | `htp/thalamus/coherence/base.py` | typing | +1 (Protocol 준수) |
| **M3** coherence.pairwise | `htp/thalamus/coherence/pairwise.py` | M1, M2 | +2 (정합 응답 high coherence, conflict 감지 정확도) |
| **M4** brain_runtime DI | `htp/runtime/brain_runtime.py` | coherence | +1 (coherence=None 기본 회귀 동등) |
| **M5** memory_system | `htp/memory/memory_system.py` | (인자만) | +1 (conflict_magnitude 단조 증폭) |
| **M6** DAG enforcement | `tests/unit/test_no_circular_deps.py` | AST | +0 (parametrize 자동 확장) |

### Recommended Session Plan

| Session | Scope | 누적 테스트 | 소요 |
|---------|-------|----------|------|
| **stage-3-coherence-core** | M1 + M2 + M3 (types + Protocol + Pairwise) | 140 → 144 | ~2.5h |
| **stage-3-integration** | M4 + M5 + M6 (BrainRuntime + Memory + DAG) | 144 → **146** | ~2h |

`/pdca do htp-thalamus-car --scope stage-3-coherence-core` 식으로 분할 실행 가능.

---

## 5. Test Plan (sub-3: +6 신규)

### 5.1 회귀 보호 (140/140)
- BrainRuntime `coherence=None` 기본 → 기존 통합 테스트 7건 무영향
- MemorySystem `conflict_magnitude=0` 기본 → 기존 memory 17 tests 무영향

### 5.2 신규 본선 테스트 (140 → 146)

**Stage 3 (+5, `tests/regression/test_stage3_coherence.py`)**
- `test_bound_response_defaults` — BoundResponse 기본 + dataclass 직렬화
- `test_coherence_strategy_protocol_compliance` — isinstance(PairwiseCoherenceGate, CoherenceStrategy)
- `test_pairwise_coherence_high_agreement` — 유사 응답 3개 → coherence ≥ 0.7
- `test_pairwise_conflict_detection_accuracy` — 의도적 conflict 10건 중 ≥9건 감지, 정합 10건 중 false positive ≤1건
- `test_swr_priority_conflict_amplification` — `priority` 가 conflict_magnitude 에 단조 증가

**BrainRuntime (+1, `tests/regression/test_stage3_brain_runtime_coherence.py`)**
- `test_brain_runtime_coherence_optional` — coherence=None 시 기존 동작 + coherence 주입 시 PFC 입력 변화

### 5.3 Plan SC §의 conflict 감지 기준 (G5)

10건 의도적 conflict fixture + 10건 정합 fixture:
- Conflict 감지 recall ≥ 90% (10건 중 9건)
- Conflict false positive ≤ 10% (10건 중 1건)
- 이 기준은 `test_pairwise_conflict_detection_accuracy` 가 검증

---

## 6. Risks + Mitigations

| ID | Risk | Mitigation |
|----|------|----------|
| R1 | BrainRuntime 통합 변경으로 Stage 5 통합 테스트 (7건) 회귀 | coherence=None 기본 — 기존 호출자 무변경. 회귀 0건 보장 |
| R2 | PairwiseCoherenceGate O(N²) — N≥16 시 성능 저하 | sub-3 OUT-OF-SCOPE. Plan §R2 의 `lsh_transition_n=16` 임계 도달 시 별도 사이클 (`htp-thalamus-coherence-lsh`) |
| R3 | Conflict threshold 0.3 / escalation 0.7 default 가 부정확 | test fixture (의도적 conflict 10건) 로 보정. 필요 시 RoutingConfig 와 유사한 CoherenceConfig 활용 |
| R4 | precision-weighted fusion 의 weights 합 0 (모든 precision=0) | wsum=0 시 1.0 으로 fallback (zero division 방지) |
| R5 | swr_priority 식 `(1 + conflict)` 가 over-amplification | conflict ∈ [0, 1] → priority 최대 2배 증폭. test 가 단조성만 보장. 정량 보정은 sub-4 이후 |

---

## 7. Decision Record

| Decision | Choice | Rationale |
|----------|--------|----------|
| Architecture | **Option B — Modular Strategy** | sub-2 router/ 패턴 일관성. LSH 전환 시 무변경 끼움 |
| CoherenceStrategy Protocol | runtime_checkable | sub-2 RouterStrategy 와 동일 패턴 |
| BoundResponse 위치 | `htp/thalamus/types.py` | Plan §5 file map 일치. coherence/* 가 types 참조 (단방향) |
| BrainRuntime coherence 기본값 | `None` | 회귀 보호 (Stage 5 통합 7건 무영향) |
| swr_priority 식 | `novelty × reward × (1 + conflict)` | Plan FR-15 명세 그대로. conflict_magnitude=0 시 기존 동작 동등 |
| Conflict threshold 기본 | 0.3 (pair) / 0.7 (escalation) | 신경과학 직관 — 0.3 미만 일치는 noise. 0.7 초과 conflict 는 명백 |
| LSH 구현체 | sub-3 OUT-OF-SCOPE | Plan §R2 — N≥16 시 별도 사이클 진입 trigger |

---

## 8. Out-of-Scope (sub-3)

- LSH coherence 구현체 (`coherence/lsh.py`) — Plan §R2 trigger N≥16
- `CoherenceConfig` sub-config 분리 — sub-1 의 sub-config 7개 외 신규. 필요시 sub-4 에서 추가
- Multi-head coherence (응답 차원별 분리 binding) — Plan §8 Multi-head attention 과 함께 별도
- Memory consolidation 후속 ripple 시뮬레이션 — Plan LOW-7 (별도 사이클)

---

## 9. Checkpoint Summary

- **Architecture 선택**: ✅ Option B — Modular Strategy
- **Session 분할**: ✅ 2 sessions (coherence-core / integration)
- **테스트 목표**: 140 → **146**
- **회귀 보호**: BrainRuntime / MemorySystem 모두 기본값으로 기존 동작 유지
- **LSH trigger**: N≥16 시 별도 PDCA 사이클 (`htp-thalamus-coherence-lsh`)
- **다음 액션**: `/pdca do htp-thalamus-car --scope stage-3-coherence-core`
