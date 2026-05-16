---
template: report
feature: htp-thalamus-car
sub_cycle: sub-2 (Stage 1 + 2)
date: 2026-05-17
author: Mindbuild
status: Completed
match_rate: 99%
---

# htp-thalamus-car sub-2 Completion Report

## Executive Summary

| Perspective | Content |
|-------------|---------|
| **Problem** | Thalamus 라우팅이 문자열 태그 매칭(주소 기반) — HTP 4대 원칙 ("구조는 데이터가 만든다") 와 모순. content-addressable routing 부재로 임베딩 기반 시맨틱 라우팅 불가. |
| **Solution** | **Option B (Clean Strategy)** 채택: `RouterStrategy` Protocol + Tag/Vector/Hybrid 3 구현체. `RegionSignature` (EMA centroid + cosine similarity) 분리. `CoreCells` 가 router DI 받아 다형성 위임. Stage 6 EmbeddingBridge 도 동일 Protocol 추가 구현체로 끼움. |
| **Function/UX Effect** | `VectorRouter` 의 `last_metrics` (entropy + active_count + top1_score) 가 β 튜닝의 정량 근거 자동 누적. `HybridRouter` 의 α 가중으로 "지식 저장소 (β=0, recall) ↔ 실시간 라우팅 (β=1, precision)" 연속 스펙트럼 탐색 가능. |
| **Core Value** | OCP 실증 — sub-2 한 번의 비용으로 sub-5 (EmbeddingBridge) / sub-6 (vector default) 무변경 끼움. β sweep 메트릭이 v4 Rev 1.3 "루프를 먼저 닫는다" 원칙의 3차 실증 (코드에 내장된 자기진단). |

### 1.3 Value Delivered (4 perspectives, 실측 기준)

| Perspective | Planned | Delivered | Δ |
|-------------|---------|-----------|---|
| Test count | 118 → 133 (+15) | 118 → **140 (+22)** | **+7 초과** |
| Match Rate | ≥ 90% | **99%** | +9pp |
| Match Rate axes | 4축 평균 | Structural 100 / Functional 100 / API 100 / Runtime 100 | 완벽 |
| Review 반영 | 7건 (#1~#7) | 7건 모두 코드+테스트 검증 | 100% |

---

## Context Anchor

| Key | Value |
|-----|-------|
| **WHY** | Thalamus 라우팅을 문자열 태그 → 벡터 유사도로 전환. HTP 4대 원칙 정합성. |
| **WHO** | HTP 개발자 + 향후 Region 추가 컨트리뷰터. 기존 호출자 무영향 (routing_mode="tag" 기본). |
| **RISK** | (R1) 회귀 깨짐 / (R2) 냉시작 / (R6) β 튜닝 어긋남 / (R7) HybridRouter 동기 병목 |
| **SUCCESS** | 누적 118 → 140 (+22), Match Rate 99%, Review #1~#7 모두 PASS |
| **SCOPE** | Stage 1 (Vector Routing) + Stage 2 (Hybrid) |

---

## 2. PRD → Plan → Design → Implementation 일관성

### 2.1 Decision Record Chain

```
[Plan]   Architecture: Bottom-up Incremental (Plan §3) — 회귀 보호 우선
   ↓
[Design] Selected Option: B — Clean Strategy (사용자 선택)
   ↓     Review 7건 반영 (#1 thr 클램프 / #2 양쪽 인자 / #3 냉시작
         / #4 런타임 교체 / #5 M5 foundation / #6 β sweep / #7 async note)
   ↓
[Do]     Session 3분할 — foundation b79bf11 → vector 4a7858a → hybrid f2d2505
   ↓
[Check]  Match Rate 99%, Gap 0건 (Critical/Important)
   ↓
[Report] (현 문서)
```

### 2.2 Key Decisions & Outcomes

| Decision | Followed? | Outcome |
|----------|:---:|---------|
| Option B (Clean Strategy) | ✅ | `htp/thalamus/router/` 패키지 신설 + 3 구현체 |
| `RouterStrategy` Protocol (runtime_checkable) | ✅ | `isinstance` 다형성 검증 PASS |
| `CoreCells(router=TagRouter())` 기본값 | ✅ | 회귀 12/12 깨지지 않음 (FR-09) |
| `RegionSignature` 별도 모듈 (`signature.py`) | ✅ | DAG router → signature 단방향 |
| (Review #1) `min(mu + β·σ, 0.99)` 클램프 | ✅ | 부호반전 방지, β=10 sweep 안전 |
| (Review #3) 냉시작 균등 1/N | ✅ | empty route 0건 보장 |
| (Review #6) `last_metrics` 9 필드 | ✅ | entropy/active_count 단조성 검증 |
| (Review #7) sub-2 sync 유지, sub-5 trigger 명시 | ✅ | §2.5 Design Note + §8 Out-of-Scope |
| TagRouter 기존 logic 1:1 이관 | ✅ | `test_tag_router_regression_equivalence` PASS |

---

## 3. Plan Success Criteria Final Status

### Plan FR-06 ~ FR-11 + sub-2 G1~G6

| ID | Criterion | Status | Evidence |
|----|-----------|:---:|----------|
| **FR-06** | RegionSignature class (centroid + count + update + similarity) | ✅ Met | `signature.py:24-75` |
| **FR-07** | RegionSignal.region_signature 필드 | ✅ Met | `region_signal.py:34-36` |
| **FR-08** | CoreCells._gate_vector + dynamic threshold (μ + β·σ) | ✅ Met | `vector_router.py:88-117` (CoreCells 가 router 위임) |
| **FR-09** | routing_mode="tag" 기본값 (회귀 보호) | ✅ Met | `CoreCells(router=TagRouter())` 기본 + 회귀 0건 |
| **FR-10** | CoreCells._gate_hybrid α × vec + (1-α) × tag | ✅ Met | `hybrid_router.py:74-94` |
| **FR-11** | α 0.1→0.9 cosine(selected) > 0.5 | ✅ Met | `test_hybrid_alpha_continuity` 단조 cosine 검증 |
| **G1**   | 누적 테스트 ≥ 133 | ✅ Met | **140** 달성 (+7) |
| **G2**   | Protocol 다형성 isinstance | ✅ Met | `test_router_strategy_protocol_compliance` |
| **G3**   | Stage 6 EmbeddingBridge 호환성 | ✅ Met | Protocol 안정성 (인터페이스 신규 메서드 0건) |
| **G4**   | DAG 단방향 (router/* → runtime 금지) | ✅ Met | `test_router_file_dag_isolation` 4/4 |
| **G5**   | β sweep 메트릭 단조성 | ✅ Met | `test_vector_router_beta_sweep_metrics` |
| **G6**   | async pipeline trigger 문서화 | ✅ Met | Design §2.5 + R7 명시 |

**Overall Success Rate: 12/12 (100%)**

---

## 4. Delivered Artifacts

### 4.1 신규 파일 (5)

| 파일 | LoC | 책임 |
|------|----:|------|
| `htp/thalamus/signature.py` | 75 | RegionSignature (EMA + cosine + cold start 보호) |
| `htp/thalamus/router/__init__.py` | 28 | 공개 export (RouterStrategy, RoutingScore, 3 Router) |
| `htp/thalamus/router/base.py` | 65 | RouterStrategy Protocol + RoutingScore dataclass |
| `htp/thalamus/router/tag_router.py` | 71 | 기존 hub_strength 로직 1:1 이관 |
| `htp/thalamus/router/vector_router.py` | 142 | similarity 기반 + dynamic threshold + last_metrics |
| `htp/thalamus/router/hybrid_router.py` | 94 | α 가중 결합 + breakdown 기록 |
| `tests/regression/test_stage1_vector_routing.py` | 381 | Stage 1 신규 15 tests |
| `tests/regression/test_stage2_hybrid_routing.py` | 162 | Stage 2 신규 3 tests |
| `docs/02-design/features/htp-thalamus-car_sub-2_design v1.md` | 493 | Design v1.1 (Review 7건) |
| `docs/03-analysis/htp-thalamus-car.sub-2.analysis.md` | ~200 | Check Phase 결과 |

### 4.2 수정 파일 (3)

| 파일 | 변경 | 내용 |
|------|------|------|
| `htp/thalamus/region_signal.py` | +3 LoC | `region_signature: RegionSignature \| None = None` |
| `htp/thalamus/core_cells.py` | +27 LoC | router DI + gate() keyword-only signal_text/vec |
| `tests/unit/test_no_circular_deps.py` | +30 LoC | `_ROUTER_DIR` parametrize 추가 |

### 4.3 commits (3)

```
b79bf11 sub-2 stage-1-foundation  M1+M2+M3+M5  118 → 125 (+7)
4a7858a sub-2 stage-1-vector      M4+M6        125 → 133 (+8)
f2d2505 sub-2 stage-2-hybrid      M7+M8        133 → 140 (+7)
```

---

## 5. Metrics

### 5.1 Volume

| 지표 | 값 |
|------|---:|
| 신규 LoC (htp/) | 475 |
| 수정 LoC (htp/) | ~30 |
| 신규 테스트 LoC | 543 |
| Design 문서 LoC | 493 |
| Analysis 문서 LoC | ~200 |
| 총 변경 LoC | ~1,741 |

### 5.2 Quality

| 지표 | Before (sub-1 end) | After (sub-2 end) |
|------|------:|------:|
| 총 테스트 | 118 | **140** (+22) |
| 회귀 깨짐 | 0 | **0** |
| 실행 시간 | 1.30s | 1.21s (단축) |
| Match Rate | 98% (sub-1) | **99%** (sub-2) |
| Public API 호환성 | 100% | **100%** |
| Critical Gap | 0 | **0** |
| DAG 검증 파일 | 8 (core 4 + knowledge 4) | **12** (+ router 4) |

---

## 6. Lessons Learned

### 6.1 잘된 점

1. **OCP 가 sub-2 한 번에 검증됨** — RouterStrategy Protocol 도입으로 Tag/Vector/Hybrid 가 일관된 인터페이스. sub-5 EmbeddingBridge 도 무변경 끼울 토대 완성.
2. **β sweep 메트릭의 사전 내장** — Review #6 이 design 단계에서 *코드에 자기진단을 박는* 패턴을 정착시킴. 향후 RoutingConfig 튜닝 시 외부 도구 불필요.
3. **회귀 보호 우선 전략** — TagRouter 기본값으로 12/12 routing test 0건 깨짐. design 의 Plan FR-09 가 실제 코드 default 로 정확 반영.
4. **Session 3분할이 적중** — foundation / vector / hybrid 가 각각 commit 단위 + 회귀 통과 단위로 깔끔 정렬. design v1.1 의 session plan 이 정확.

### 6.2 개선 여지

1. **design 의 누적 테스트 숫자가 부정확** (130/132/133 등 산수 inconsistency) — 실제 +22 가 예측보다 큼. 다음 sub-cycle 부터 design 의 test 카운트는 *최소* 로만 표기 권장.
2. **RegionSignature 영속화 부재** — Gap #1 (Low). sub-3 이후 또는 sub-5 에서 encoder_state.pkl 패턴 재사용.
3. **β 권장값 표 미작성** — Review #6 의 "지식 저장소 β=0 / 실시간 β=1" 권장값은 sub-2 에선 sweep 메트릭만, 실 측정값 기반 권장은 sub-3 이후 사용 데이터 누적 후.

### 6.3 v4 Rev 1.3 "루프를 먼저 닫는다" 원칙의 3차 실증

| 차수 | 시점 | 발견/예방 |
|-----|-----|---------|
| 1차 | sub-1 직후 | Critical Gap #3 (encoder.fit refit) 24h 내 발견 → 옵션 A-2 영속화 |
| 2차 | sub-1 cross-language 실험 | "언어가 hub" 가설 부분 실증 + Stage 6 우선순위 검증 |
| **3차** | **sub-2 (현 사이클)** | **Review #6 β sweep 메트릭을 *코드에 내장* — 향후 튜닝 시 외부 도구 불필요** |

원칙의 가치는 *후속 사이클* 에서 누적된다. sub-3 (CoherenceGate) 진입 시 vector routing 의 실측 데이터 (entropy/active_count) 가 자동 누적되어 CoherenceGate 의 conflict threshold 결정에도 사용 가능.

---

## 7. Next Cycle Recommendations

### 7.1 즉시 가능 — sub-3 Design 진입

```bash
/pdca design htp-thalamus-car   # Stage 3 (CoherenceGate + Memory novelty 연동)
```

- 누적 테스트 목표: 140 → 146 (Plan §5 명시 84)
- 핵심: `CoherenceGate.bind()` pairwise + conflict detection + precision-weighted fusion
- Memory 연동: `swr_priority = novelty × reward × (1 + conflict_magnitude)`

### 7.2 향후 별도 PDCA 사이클 후보

- **`htp-region-signature-persistence`** (Gap #1) — RegionSignature pickle 영속화. Critical Gap #3 와 동일 패턴.
- **`htp-thalamus-async-pipeline`** (Review #7) — sub-5 EmbeddingBridge 진입 후 `score()` ≥ 50ms 측정 시 trigger.
- **`htp-routing-beta-tuning`** — sub-3+ 이후 누적된 last_metrics 분석으로 application context 별 β 권장값 표 작성.

---

## 8. Sign-off

| 항목 | 결과 |
|------|------|
| **Plan SC** | 12/12 (100%) |
| **Decision Record** | 10/10 따름 |
| **Review 반영** | 7/7 (#1~#7) |
| **회귀 보호** | 0건 깨짐 |
| **Match Rate** | **99%** (≥ 90% target) |
| **PDCA 단계** | Plan → Design (v1.1) → Do (3 sessions) → Check (sub-2.analysis) → **Report (현 문서)** |

**sub-2 PDCA 완료 ✅ — sub-3 진입 준비됨.**
