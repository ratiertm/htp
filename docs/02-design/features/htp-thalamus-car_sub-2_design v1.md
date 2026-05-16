---
template: design
feature: htp-thalamus-car
sub_cycle: sub-2 (Stage 1 + 2)
date: 2026-05-17
revision: "v1.1 — β sweep metrics + async pipeline note 보강"
author: Mindbuild
project: HTP (Hub Topology Programming)
status: Draft → Reviewed
reviewed_by: Claude
review_notes: "#1 정규화 클램프, #2 HybridRouter 인자 전달, #3 냉시작 보호 in-scope, #4 테스트 +1, #5 Session M5 이동, **#6 β sweep entropy/hit-rate 메트릭**, **#7 HybridRouter async pipeline 향후 검토**"
selected_option: B — Clean Strategy
---

# htp-thalamus-car sub-2 Design — Vector Routing + Hybrid

> **Summary**: Thalamus 라우팅을 문자열 태그 매칭에서 벡터 유사도 기반 content-addressable routing으로 전환. **Option B (Clean Strategy)** 채택: `RouterStrategy` Protocol 도입으로 Tag/Vector/Hybrid 3종을 다형성으로 분리. Stage 6 EmbeddingBridge 도 동일 Protocol 의 추가 구현체가 되도록 설계.
>
> **v1.1 보강 (2026-05-17)**: (1) β sweep 시 단순 통과 여부가 아닌 entropy + hit-rate 메트릭 의무 로깅. 이는 "지식 저장소" 사용 시 "얼마나 다양한 노드가 활성화되는가"의 튜닝 핵심 지표. (2) HybridRouter 의 향후 비동기 파이프라인 가능성 design note 명시 — sub-2 OUT-OF-SCOPE 이나 sub-5 (Stage 6 EmbeddingBridge) 진입 시 성능 병목 검토 필수.
>
> **Selected Architecture**: Option B — Clean Strategy
> **Predecessor**: sub-1 (commit `5815e27` — Stage 0 + 0.5 완료, Critical Gap #3 RESOLVED)
> **Test Target**: 118 → **133** (Stage 1 +12, Stage 2 +3) — Review #6 메트릭 sweep test +1

---

## Context Anchor (Plan 에서 전파)

| Key | Value |
|-----|-------|
| **WHY** | 태그 매칭이 HTP 4대 원칙 ("구조는 데이터가 만든다") 와 모순. sub-1 의 knowledge_log 가 vector routing 의 입력 자료로 직결. |
| **WHO** | HTP 개발자 + 향후 Region 추가 컨트리뷰터. `routing_mode="tag"` 기본값 유지로 사용자 무영향. |
| **RISK** | (1) 회귀 118/118 깨짐 → 즉시 롤백. (3) RegionSignature 냉시작 (centroid 영벡터). (11 신규) Strategy 패턴 도입으로 호출 경로 전면 재배선 → 회귀 가능성. |
| **SUCCESS** | 누적 테스트 118 → **132**. vector mode 가 tag mode 와 동등 또는 우위 (empty route 0건). α=0.1~0.9 변화 시 cosine of selected > 0.5. |
| **SCOPE** | Stage 1 + 2 만. Stage 3 CoherenceGate, Stage 4-7 모두 OUT-OF-SCOPE. |

---

## 1. Overview

### 1.1 Selected Architecture: Option B — Clean Strategy

```
htp/thalamus/
├── router/                          [신규 패키지]
│   ├── __init__.py                  공개 export
│   ├── base.py                      RouterStrategy Protocol + RoutingScore dataclass
│   ├── tag_router.py                TagRouter — 기존 keyword 매칭 로직 이관
│   ├── vector_router.py             VectorRouter — RegionSignature.similarity 기반
│   └── hybrid_router.py             HybridRouter — α × vector + (1-α) × tag
├── signature.py                     [신규] RegionSignature(centroid, count, update, similarity)
├── core_cells.py                    수정 — gate() 가 router DI 받아 위임
└── region_signal.py                 수정 — region_signature: RegionSignature | None 필드 추가
```

### 1.2 Design Goals

| ID | Goal | 측정 방법 |
|----|------|---------|
| G1 | 회귀 118 + 신규 15 = **133/133** 통과 | `pytest -q` |
| G2 | Tag/Vector/Hybrid 동일 Protocol 다형성 | `isinstance(router, RouterStrategy)` |
| G3 | Stage 6 EmbeddingBridge 가 RouterStrategy 추가 구현체로 끼워질 수 있어야 함 | 인터페이스 안정성 (CHANGELOG 가 호환 표기) |
| G4 | DAG 확장 — `htp/thalamus/router/*` 는 `runtime/` 미참조 | `test_no_circular_deps.py` parametrize 확장 |
| G5 | **(Review #6)** β sweep 시 `(entropy, active_count, top1_score)` 메트릭 노출 + 단조성 만족 | `test_vector_router_beta_sweep_metrics` |
| G6 | **(Review #7)** HybridRouter 향후 async 도입 trigger 조건 + 옵션 문서화 | §2.5 Design Note + §8 Out-of-Scope |

### 1.3 Design Principles

1. **OCP (Open-Closed)** — 새 라우팅 정책 추가가 기존 코드 변경 없이 가능 (Stage 6 EmbeddingBridge 가 첫 실증)
2. **DI (Dependency Injection)** — `CoreCells(router=...)` 생성자 인자. 기본값은 `TagRouter()` 로 회귀 보호
3. **단일 책임** — `RegionSignature` 는 centroid 관리 + 유사도만. `Router` 는 score 계산만. `CoreCells` 는 gating 만.
4. **DAG 단방향** — `router/*` → `signature.py` 가능, 역방향 금지. `router/*` 는 `runtime/` 미참조.

---

## 2. Architecture Detail

### 2.1 RouterStrategy Protocol (htp/thalamus/router/base.py)

```python
from typing import Protocol, runtime_checkable
from dataclasses import dataclass

@dataclass
class RoutingScore:
    """라우팅 결과 — Region 별 점수."""
    region_id: str
    score: float          # [0.0, 1.0] 정규화
    breakdown: dict       # {"tag": 0.6, "vector": 0.4} 등 진단용

@runtime_checkable
class RouterStrategy(Protocol):
    """Thalamus 라우팅 정책 인터페이스.

    Stage 1: TagRouter (기본), VectorRouter
    Stage 2: HybridRouter
    Stage 6: EmbeddingBridgeRouter (실험 브랜치, 동일 Protocol 추가 구현)
    """
    @property
    def mode(self) -> str: ...   # "tag" | "vector" | "hybrid" | "embedding"

    def score(self,
              signal_text: str | None,
              signal_vec: "np.ndarray | None",
              regions: "list[RegionSignal]") -> list[RoutingScore]:
        """signal → Region 별 score 리스트.

        - tag mode: signal_text 만 사용
        - vector mode: signal_vec + regions[i].region_signature.similarity 사용
        - hybrid mode: 둘 다 사용
        """
        ...
```

### 2.2 RegionSignature (htp/thalamus/signature.py)

```python
@dataclass
class RegionSignature:
    """Region 의 의미 중심점 — Online EMA centroid + cosine similarity.

    Plan FR-06.
    """
    centroid: np.ndarray   # (64,) — 0-vector 초기화 (냉시작)
    count: int = 0
    dim: int = 64

    def update(self, input_vec: np.ndarray) -> None:
        """EMA centroid 업데이트 — lr = 1 / (count + 1).

        cold start (count=0): centroid ← input_vec
        steady (count >> 0): centroid ← centroid + (input_vec - centroid) / (count+1)
        """
        lr = 1.0 / (self.count + 1)
        self.centroid = (1 - lr) * self.centroid + lr * input_vec
        self.count += 1

    def similarity(self, query_vec: np.ndarray) -> float:
        """cosine similarity. centroid 가 영벡터면 0.0 반환 (냉시작 보호)."""
        nc = np.linalg.norm(self.centroid)
        nq = np.linalg.norm(query_vec)
        if nc < 1e-8 or nq < 1e-8:
            return 0.0
        return float(np.dot(self.centroid, query_vec) / (nc * nq))
```

### 2.3 TagRouter (htp/thalamus/router/tag_router.py)

기존 `core_cells.gate()` 의 keyword 매칭 로직을 그대로 이관. 회귀 보호 핵심.

```python
class TagRouter:
    @property
    def mode(self) -> str: return "tag"

    def score(self, signal_text, signal_vec, regions) -> list[RoutingScore]:
        # 기존 htp_runtime._extract 패턴 재사용
        # signal_text 의 keyword 와 region.region_id / region tags 매칭
        # NB: 기존 core_cells.gate() 의 score 계산 알고리즘 1:1 이관
        ...
```

### 2.4 VectorRouter (Stage 1)

```python
class VectorRouter:
    """Content-addressable routing.

    Plan FR-08: dynamic threshold (μ + β×σ).
    Review #1: thr 상한 클램프 (cosine 분포가 [0.9,1.0] 집중 시 부호 반전 방지).
    Review #3: 냉시작 보호 (count=0 → 균등 score).
    Review #6: β sweep 시 entropy + hit-rate 메트릭 의무 기록 — last_metrics 노출.
    """
    def __init__(self, beta: float = 0.5):
        self.beta = beta
        # Review #6: 매 score() 호출 후 진단 가능하도록 마지막 호출의 메트릭 보존.
        # 단순 dict 로 노출 (관측 가능성 우선, 추적 의존성 없음).
        self.last_metrics: dict = {}

    @property
    def mode(self) -> str: return "vector"

    def score(self, signal_text, signal_vec, regions) -> list[RoutingScore]:
        if signal_vec is None:
            self.last_metrics = {"cold_start": False, "empty_vec": True,
                                 "active_count": 0, "entropy": 0.0,
                                 "top1_score": 0.0, "thr": None}
            return [RoutingScore(r.region_id, 0.0, {"vector": 0.0})
                    for r in regions]

        # Review #3: 냉시작 보호 — 모든 Region의 signature가 미초기화면
        # 균등 score 반환. CoreCells의 기존 gating이 최종 결정.
        if all(r.region_signature is None or r.region_signature.count == 0
               for r in regions):
            uniform = 1.0 / max(len(regions), 1)
            # Review #6: cold start 도 메트릭 기록 — 사용자가 "지금 학습 안 됨" 즉시 인지 가능
            self.last_metrics = {"cold_start": True, "empty_vec": False,
                                 "active_count": len(regions),
                                 "entropy": _shannon_entropy([uniform] * len(regions)),
                                 "top1_score": uniform, "thr": None}
            return [RoutingScore(r.region_id, uniform,
                                 {"vector": 0.0, "cold_start": True})
                    for r in regions]

        sims = []
        for r in regions:
            sig = r.region_signature  # RegionSignature | None
            s = sig.similarity(signal_vec) if sig is not None else 0.0
            sims.append((r.region_id, s))
        # Dynamic threshold (μ + β×σ) — 정규화
        scores = np.array([s for _, s in sims])
        mu, sigma = float(scores.mean()), float(scores.std())
        # Review #1: 상한 클램프 — thr > 0.99 시 부호 반전 방지
        thr = min(mu + self.beta * sigma, 0.99)
        normalized = [max(0.0, (s - thr) / (1.0 - thr + 1e-8))
                      for s in scores]

        # Review #6: β sweep 메트릭 — "얼마나 다양한 노드가 활성화되는가" 측정.
        # active_count: normalized > 0 인 Region 수 (small β → 많이 활성 / large β → 집중)
        # entropy: Shannon entropy of normalized distribution. 높을수록 분산 라우팅.
        # top1_score: 최고 점수 Region 의 normalized score. 1에 가까울수록 confident.
        # thr: 적용된 dynamic threshold (디버깅 + 후속 튜닝용).
        active_count = sum(1 for n in normalized if n > 1e-8)
        self.last_metrics = {
            "cold_start":   False,
            "empty_vec":    False,
            "beta":         self.beta,
            "mu":           mu,
            "sigma":        sigma,
            "thr":          thr,
            "active_count": active_count,
            "entropy":      _shannon_entropy(normalized),
            "top1_score":   float(max(normalized)) if normalized else 0.0,
        }
        return [RoutingScore(rid, float(ns), {"vector": float(s)})
                for (rid, s), ns in zip(sims, normalized)]


def _shannon_entropy(scores: list[float]) -> float:
    """Shannon entropy (nats) — 분포가 균일할수록 ln(N), 한쪽 집중일수록 0.

    정규화된 score 분포의 다양성 지표. β sweep 시 entropy 변화로
    "라우팅 다양성 vs 정확도" trade-off 를 정량 관찰.
    """
    arr = np.array([s for s in scores if s > 1e-8], dtype=np.float64)
    if arr.size == 0:
        return 0.0
    p = arr / arr.sum()
    return float(-(p * np.log(p + 1e-12)).sum())
```

**§2.4 Design Note — β Sweep 메트릭 사용 가이드**

| β | 예상 동작 | 활용 시나리오 |
|---|---------|----------|
| 0.0 | thr = μ. 평균 이상 모두 활성 → entropy 높음, active_count 큼 | "지식 저장소" 사용 — 다양한 관련 노드 동시 활성화 (recall 우선) |
| 0.5 | thr = μ + 0.5σ. 균형 | 기본 라우팅 (precision/recall 균형) |
| 1.0 | thr = μ + σ. 1σ 초과만 활성 → entropy 낮음, top1 confident | "실시간 라우팅" — 명확한 정답 하나로 (precision 우선) |

M4 테스트에서 동일 input 으로 β=0.0/0.5/1.0 sweep 시 `vec_router.last_metrics` 의 `(entropy, active_count, top1_score)` 변화를 검증 + 로그 기록. 이는 향후 `RoutingConfig.beta` 를 application context (지식 저장소 vs 실시간 라우팅) 별로 다르게 튜닝하는 근거 자료.

### 2.5 HybridRouter (Stage 2)

```python
class HybridRouter:
    """α × VectorRouter + (1-α) × TagRouter.

    Plan FR-10, FR-11.
    Review #2: Protocol 계약상 signal_text/signal_vec 양쪽 모두 전달.
               각 Router 내부에서 불필요한 인자를 무시하는 것은 Router의 책임.
    Review #7: sub-2 는 동기 호출. 향후 async pipeline 검토 (§2.5 Design Note 참조).
    """
    def __init__(self, tag: TagRouter, vec: VectorRouter, alpha: float = 0.5):
        self.tag = tag
        self.vec = vec
        self.alpha = alpha

    @property
    def mode(self) -> str: return "hybrid"

    def score(self, signal_text, signal_vec, regions) -> list[RoutingScore]:
        # Review #2: 양쪽 인자를 모두 전달 — Protocol 계약 준수
        # Review #7 (sub-2): 동기 순차 호출. VectorRouter 가 무거워지면 (sub-5 EmbeddingBridge
        # 에서 sLLM forward pass 등) TagRouter 결과가 먼저 준비됐어도 vec 완료 대기 →
        # 동기 병목. 향후 async 도입 시 두 호출을 asyncio.gather 로 묶음.
        tag_scores = {s.region_id: s.score
                      for s in self.tag.score(signal_text, signal_vec, regions)}
        vec_scores = {s.region_id: s.score
                      for s in self.vec.score(signal_text, signal_vec, regions)}
        out = []
        for r in regions:
            t = tag_scores.get(r.region_id, 0.0)
            v = vec_scores.get(r.region_id, 0.0)
            mixed = self.alpha * v + (1 - self.alpha) * t
            out.append(RoutingScore(
                r.region_id, mixed,
                breakdown={"tag": t, "vector": v, "alpha": self.alpha}
            ))
        return out
```

**§2.5 Design Note — 비동기 처리 파이프라인 (향후 성능 고려)**

현재 `HybridRouter.score()` 는 `tag.score()` 와 `vec.score()` 를 **동기 순차** 호출. sub-2 범위에선 두 Router 모두 가벼운 numpy 연산이라 병목 없음. 그러나:

| 시점 | 변화 | 병목 가능성 |
|------|------|----------|
| **sub-5 (Stage 6 EmbeddingBridge)** | VectorRouter 가 sLLM forward pass 수행 → 100ms+ 단위 | **HIGH** — TagRouter 결과가 1ms 안에 준비돼도 100ms 동기 대기 |
| **batch ingest 시나리오** | 동일 signal 에 N개 Region 평가, 다수 query 동시 | **MED** — 다중 query asyncio.gather 로 throughput ×N 가능 |
| **CoherenceGate (Stage 3)** | Region 응답 후 pairwise binding O(N²) | **MED** — 별도 사이클, 본 Router 와 독립 |

**향후 async 도입 시 설계 옵션**:

1. **Protocol 확장** — `async def score()` 를 별도 메서드로 (`ascore`). 동기 인터페이스 보존, 비동기는 opt-in.
2. **HybridRouter.ascore()** — `asyncio.gather(self.tag.ascore(...), self.vec.ascore(...))` 로 병렬화. fast path (TagRouter 결과 선반환) 도 검토 가능.
3. **Router 결과 캐싱** — 같은 signal 에 대한 vec.score() 결과를 LRU 캐시. embedding 비용이 크면 hit-rate 측정 후 적용.

**sub-2 결정**: 동기 유지. async 도입은 sub-5 EmbeddingBridge 진입 시 "실제 측정된 병목" 을 근거로 별도 PDCA 사이클 (예: `htp-thalamus-async-pipeline`) 에서 다룸. 미리 도입 시 over-engineering 위험.

**검증 시점 트리거**: sub-5 에서 EmbeddingBridge 의 `score()` 1회 호출 시간 ≥ 50ms 측정되면 즉시 async 사이클 열기.

### 2.6 CoreCells DI 변경

```python
class CoreCells:
    def __init__(self,
                 ...,
                 router: RouterStrategy | None = None):
        ...
        self.router = router or TagRouter()   # 기본값 = 회귀 보호

    def gate(self, signal_text, signal_vec, regions, ...) -> GatingMask:
        # AS-IS: 내장 keyword 매칭
        # TO-BE: self.router.score(...) 위임 후 sigmoid 변환
        routing_scores = self.router.score(signal_text, signal_vec, regions)
        # 기존 sigmoid + theta_bias + precision 계산 그대로 적용
        ...
```

### 2.7 RegionSignal 확장

```python
@dataclass
class RegionSignal:
    region_id: str
    ...
    precision: float = 1.0
    region_signature: "RegionSignature | None" = None   # 신규 (Plan FR-07)
```

---

## 3. DAG 의존 방향 (확장)

```
htp/thalamus/router/*  ──→  htp/thalamus/signature.py
                       ──→  htp/thalamus/region_signal.py
                       ──→  numpy (외부)

htp/thalamus/core_cells.py  ──→  htp/thalamus/router/*  (DI)

금지: router/* → htp.runtime / htp.memory / htp.knowledge
```

`test_no_circular_deps.py` 에 `htp/thalamus/router/` 디렉토리 검사 추가.

---

## 4. Stage 별 구현 순서 (Session Guide)

### Module Map

| Module | 파일 | 의존 | 테스트 |
|--------|------|-----|------|
| **M1** signature | `htp/thalamus/signature.py` | numpy | +3 (init, update EMA, similarity 냉시작) |
| **M2** router.base | `htp/thalamus/router/base.py` | typing | +1 (Protocol 준수) |
| **M3** router.tag | `htp/thalamus/router/tag_router.py` | core_cells 이관 | +2 (회귀 동등성) |
| **M4** router.vector | `htp/thalamus/router/vector_router.py` | signature, M2 | +6 (dynamic threshold, 빈 vec, normalized 합, **냉시작 균등**, **고균일 유사도 클램프**, **β sweep 메트릭 단조성 Review #6**) |
| **M5** region_signal | `htp/thalamus/region_signal.py` | signature | +1 (region_signature 필드) |
| **M6** core_cells DI | `htp/thalamus/core_cells.py` | router/* | +3 (회귀 동등 + vector mode + **런타임 router 교체**) |
| **M7** router.hybrid | `htp/thalamus/router/hybrid_router.py` | M3, M4 | +3 (α=0.1, 0.5, 0.9 연속성) |
| **M8** DAG enforcement | `tests/unit/test_no_circular_deps.py` | AST | +0 (router/ 추가 검사, 기존 parametrize 확장) |

### Recommended Session Plan

| Session | Scope | 누적 테스트 | 소요 |
|---------|-------|----------|------|
| **stage-1-foundation** | M1 + M2 + M3 + M5 (signature + Protocol + Tag 이관 + **RegionSignal 확장**) | 118 → 125 | ~3.5h |
| **stage-1-vector** | M4 + M6 (VectorRouter + core_cells DI) — **+ β sweep 메트릭 (Review #6)** | 125 → 130 | ~2.5h |
| **stage-2-hybrid** | M7 + M8 (HybridRouter + DAG) — **+ async pipeline note 검수 (Review #7)** | 130 → **133** | ~1.5h |

> **Review #5**: M5 (RegionSignal 확장)를 foundation 으로 당김. foundation 세션에서 데이터 구조(RegionSignature + RegionSignal)가 완전히 확정되어, stage-1-vector 가 순수 로직(VectorRouter + CoreCells DI)에만 집중.

`/pdca do htp-thalamus-car --scope stage-1-foundation` 식으로 분할 실행 가능.

---

## 5. Test Plan (sub-2: +13 신규)

### 5.1 회귀 보호 (118/118)
- `routing_mode="tag"` 기본값 유지 → 기존 12/12 routing 테스트 무영향
- CoreCells 기본 router = TagRouter → 회귀 동등성 검증

### 5.2 신규 본선 테스트 (118 → 132)

**Stage 1 (+12, `tests/regression/test_stage1_vector_routing.py`)**
- `test_signature_init_zero_centroid` — 0-vec 초기화
- `test_signature_update_ema` — lr=1/(count+1) 점진 업데이트
- `test_signature_similarity_cold_start` — count=0 시 0.0
- `test_router_strategy_protocol_compliance` — Tag/Vector 모두 isinstance(RouterStrategy)
- `test_tag_router_regression_equivalence` — 기존 12/12 패턴 동등
- `test_vector_router_empty_vec` — signal_vec=None 시 안전
- `test_vector_router_dynamic_threshold` — μ+β×σ 정규화
- `test_vector_router_normalized_sum` — 합 ≤ N
- `test_vector_router_cold_start_uniform` — **(Review #3)** 모든 Region count=0 시 균등 score 반환, empty route 0건
- `test_vector_router_high_uniform_similarity` — **(Review #1)** 모든 Region similarity ≥ 0.95 시 thr 클램프 동작, 부호 반전 없음
- `test_vector_router_beta_sweep_metrics` — **(Review #6)** β∈{0.0, 0.5, 1.0} sweep 시 `last_metrics` 에 `entropy`, `active_count`, `top1_score`, `thr` 모두 기록. 단조성 검증: β↑ → active_count ↓, entropy ↓, top1_score ↑ (precision/recall trade-off). 로그 형식이 후속 RoutingConfig 튜닝의 근거 자료가 됨을 보장
- `test_core_cells_router_di_default_tag` — 기본값 = TagRouter
- `test_core_cells_vector_mode` — knowledge_log 7-entry 입력 시 라우팅 성공
- `test_core_cells_router_swap_at_runtime` — **(Review #4)** Runtime 중 router 교체 → 다음 gate() 정상 동작 (DI 핵심 가치 검증)

**Stage 2 (+3, `tests/regression/test_stage2_hybrid_routing.py`)**
- `test_hybrid_alpha_continuity` — α∈{0.1, 0.5, 0.9} 변화 시 cosine(selected) > 0.5
- `test_hybrid_breakdown_records` — RoutingScore.breakdown 에 tag/vector/alpha 모두 기록
- `test_hybrid_extremes_match_pure` — α=0 → Tag 결과, α=1 → Vector 결과 일치

**Unit (+0, 기존 DAG 확장)**
- `test_no_circular_deps.py` 의 parametrize 에 `htp/thalamus/router/` 추가 — 신규 카운트 없이 자동 확장

### 5.3 Stage 0.5 knowledge_log 활용

`.htp/knowledge_log.jsonl` 의 7 entries (영문 3 + 한국어 3 + bilingual 1) 를 vector routing 테스트 fixture 로 재사용. 기대:
- brain↔ai 0.55 → VectorRouter score ≥ 0.5
- brain↔infra 0.24 → VectorRouter score < 0.3
- 냉시작 (count=0) → 균등 score 반환, empty route 0건 **(Review #3)**

---

## 6. Risks + Mitigations

| ID | Risk | Mitigation |
|----|------|----------|
| R1 | Strategy 패턴 도입으로 호출 경로 재배선 → 회귀 깨짐 | 매 M 직후 `pytest -q` 통과 의무. M3 TagRouter 이관 시 기존 로직 1:1 카피 |
| R2 | RegionSignature 냉시작 (centroid=0) → 모든 similarity=0.0 → vector mode empty route | **(Review #3, RESOLVED in sub-2)** VectorRouter 에 냉시작 감지 로직 추가: 모든 Region count=0 시 균등 score 반환. `test_vector_router_cold_start_uniform` 으로 검증 |
| R3 | VectorRouter dynamic threshold β=0.5 가 너무 strict | M4 test 에서 β=0.0 / 0.5 / 1.0 sweep, knowledge_log 7-entry 분포 기반 검증 |
| R3b | **(Review #1)** thr > 0.99 시 정규화 부호 반전 | `thr = min(mu + beta * sigma, 0.99)` 상한 클램프. `test_vector_router_high_uniform_similarity` 로 검증 |
| R4 | HybridRouter α extreme (0 or 1) 시 다른 router 결과와 미세 차이 | float epsilon 1e-8 허용 (`pytest.approx`) |
| R5 | Option B 가 sub-2 한 사이클로 끝나지 않을 위험 | Session 3분할 (stage-1-foundation / stage-1-vector / stage-2-hybrid) 각각 ~1.5-3.5h |
| R6 | **(Review #6)** β 기본 0.5 가 실제 사용 패턴 (지식 저장소 vs 실시간 라우팅) 과 어긋남 | `VectorRouter.last_metrics` 의 `(entropy, active_count, top1_score)` 를 향후 RoutingConfig 튜닝의 근거. β sweep test 가 단조성 검증 + 로그 형식 보장. sub-3 이후 application context 별 권장 β 표 작성 |
| R7 | **(Review #7)** sub-5 (EmbeddingBridge) 진입 시 HybridRouter 동기 호출이 성능 병목 | sub-2 OUT-OF-SCOPE. sub-5 에서 `EmbeddingBridge.score()` 시간 ≥ 50ms 측정 시 별도 PDCA 사이클 (`htp-thalamus-async-pipeline`) 열기. §2.5 Design Note 의 옵션 1-3 (`ascore` 메서드 / `asyncio.gather` / LRU 캐시) 검토 |

---

## 7. Decision Record

| Decision | Choice | Rationale |
|----------|--------|----------|
| Architecture | **Option B — Clean Strategy** | OCP, Stage 6 EmbeddingBridge 와 동일 Protocol 다형성. sub-2 한 번의 비용으로 sub-5/sub-6 무변경. |
| RouterStrategy 형태 | Protocol (PEP 544) | runtime_checkable. duck typing 으로 isinstance 검증. sub-1 TextEncoder 와 동일 패턴 |
| RegionSignature 위치 | `htp/thalamus/signature.py` (router 패키지 외부) | router 가 signature 를 참조 (역방향 안됨). 의존 깊이 1 절약 |
| Dynamic threshold β | 기본 0.5 | M4 test 에서 0.0/0.5/1.0 sweep 으로 보정 가능. 향후 RoutingConfig.beta 로 노출 가능 |
| **(Review #1)** thr 클램프 | `min(mu + beta * sigma, 0.99)` | 고균일 유사도 분포에서 부호 반전 방지. 0.99 상한은 실용적으로 "모든 Region이 거의 동일"을 의미 |
| **(Review #2)** HybridRouter 인자 | 양쪽 signal 모두 전달 | Protocol 계약 준수. 각 Router 내부에서 불필요 인자 무시는 Router 책임 |
| **(Review #3)** 냉시작 보호 | VectorRouter 내부 균등 score | sub-2 IN-SCOPE. α 동적 조정과 분리된 자기방어 로직 |
| CoreCells router 기본값 | `TagRouter()` | 회귀 보호 (Plan FR-09 `routing_mode="tag"`) |
| TagRouter 구현 | 기존 logic 1:1 이관 (no refactor) | 회귀 0건 보장 우선. 정리는 sub-3 이후 |
| **(Review #5)** M5 세션 이동 | foundation 으로 당김 | 데이터 구조 완전 확정 후 로직 구현 |
| **(Review #6)** β sweep 메트릭 | `VectorRouter.last_metrics` 노출 + Shannon entropy 계산 | "지식 저장소 vs 실시간 라우팅" 의 β 튜닝 근거를 코드에 내장. test 가 단조성 검증으로 회귀 보호 |
| **(Review #7)** HybridRouter async | sub-2 동기 유지, §2.5 Design Note 만 추가 | over-engineering 회피. sub-5 EmbeddingBridge 시 실측 병목 (≥50ms) 발생 후 별도 사이클 진입 |

---

## 8. Out-of-Scope (sub-2)

- CoherenceGate (Stage 3, sub-3)
- ExternalRegion / LLMRegion (Stage 4, sub-4)
- PipelinedBrainRuntime (Stage 5, sub-4)
- EmbeddingBridge 실험 브랜치 (Stage 6, sub-5)
- vector default 전환 (Stage 7, sub-6)
- RegionSignature 의 persistence (centroid 영속화 — Critical Gap #3 와 유사. sub-3 이후 필요시 도입)
- ~~α 동적 조정 (R2 의 cold start fallback 등)~~ → **(Review #3)** VectorRouter 냉시작 균등 score 는 sub-2 에서 해소. α 자동 조정은 여전히 sub-3 이후
- **(Review #7)** HybridRouter 비동기 처리 파이프라인 (asyncio.gather / ascore / LRU 캐시) — sub-5 EmbeddingBridge 진입 시 실측 병목 ≥ 50ms 발생 후 별도 PDCA 사이클로 진행 (`htp-thalamus-async-pipeline`). §2.5 Design Note 가 trigger 조건 + 옵션 명세.
- **(Review #6 부분)** β 의 application context 별 권장값 표 (지식 저장소: β=0.0, 실시간: β=1.0 등) — sub-2 에서는 sweep 메트릭 로깅까지만. 실제 권장값 확정은 sub-3 이후 사용 데이터 축적 후.

---

## 9. Checkpoint Summary

- **Architecture 선택**: ✅ Option B — Clean Strategy
- **Review 반영**: ✅ 7건 (#1 정규화 클램프, #2 인자 전달, #3 냉시작 보호, #4 테스트 +1, #5 Session M5 이동, **#6 β sweep 메트릭 entropy/hit-rate**, **#7 HybridRouter async pipeline 향후 검토 명시**)
- **Session 분할**: ✅ 3 sessions (foundation / vector / hybrid) — M5 를 foundation 으로 이동
- **테스트 목표**: 118 → **133** (Review #6 메트릭 sweep test +1 추가)
- **다음 액션**: `/pdca do htp-thalamus-car --scope stage-1-foundation`
