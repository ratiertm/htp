---
template: report
feature: htp-thalamus-car (sub-4)
date: 2026-05-19
author: Mindbuild
status: Completed
predecessor: Bridge Integration (시스템 A↔B 검증 PASS, 2026-05-18)
successor: TBD — LLMRegion↔CoherenceGate conflict 해석 연결 (사용자 결정)
---

# sub-4 (Stage 4+5) Completion Report

## Executive Summary

| 관점 | 1-2 문장 요약 |
|------|--------------|
| **Problem** | G3 — LLMRegionRuntime 의 RegionRuntime 상속이 PageRank/Hebbian/NGE 를 불필요하게 끌어와 graphify 상 isolated 노드 다수 발생. 또한 다중 입력 throughput 향상 부재. |
| **Solution** | ExternalRegion ABC 추상 + LLMRegion(ExternalRegion) 으로 LLM 호출 분리. LLMRegionRuntime archive 이동. CostRouter.select_level 4-Level 추가 (기존 7-method 보존). PipelinedBrainRuntime 으로 3-stage pipeline 병렬. |
| **Function/UX Effect** | LLMRegion 이 BrainRuntime 에 자연 통합. PipelinedBrainRuntime.pipelined_arun 다중 입력 처리. CostRouter 4-Level 의사결정 가능. |
| **Core Value** | G3 본질 해결 + throughput 실측 1.95-2.67× (목표 1.5× 큰 마진 초과) + 회귀 0 깨짐. |

---

## Context Anchor

| 키 | 값 |
|----|----|
| **WHY** | LLMRegionRuntime 상속의 dead code burden 제거 + 다중 입력 throughput 향상 |
| **WHO** | HTP 사용자 — LLM 호출을 비용 인식 가능하게 Region 처럼 다루고 싶음 |
| **RISK** | LLMRegionRuntime 사용 코드 깨질 위험 → 0건 실현 |
| **SUCCESS** | 4 SC 중 3 strict + 1 partial = 91% Match Rate |
| **SCOPE** | ExternalRegion / LLMRegion / CostRouter.select_level / archive / PipelinedBrainRuntime |

---

## 1. Decision Record Chain & Outcomes

### 1-1. Architecture 선택

| Layer | Decision | Outcome |
|-------|----------|---------|
| Plan §5 | Stage 4 + Stage 5 통합 cycle | ✓ 통합 진행, ~3.5h |
| Design §2 | Option B (Clean — Plan §5 전면 흡수) | ✓ G3 본질 해결 |
| Design §4 C-2 | LLMNode 옵션 A (LLMRegion 내부 멤버) | ✓ `self._llm_node` 로 유지 |
| Session 분할 | A → B → C 순차 | ✓ 각 Session 후 회귀 통과 |

### 1-2. Decision 추적

```
[Plan] Stage 4+5 통합 cycle — G3 (graphify isolated) + throughput 동시 해결
   ↓
[Design] Option B — Plan §5 + §3 C-1~C-4 완전 흡수, ~3h
   ↓
[Design] C-2 옵션 A — LLMNode 는 LLMRegion 의 구현 디테일
   ↓
[Do A] ExternalRegion ABC + LLMRegion + CostRouter.select_level (25 tests)
   ↓
[Do B] LLMRegionRuntime archive + demo (회귀 보존)
   ↓
[Do C] PipelinedBrainRuntime — 3-stage pipeline (6 tests + throughput 측정)
   ↓
[Check] Match Rate 91% (3 strict + 1 partial)
```

---

## 2. Plan SUCCESS 최종 상태

| SC | 기준 | 결과 | 증거 |
|----|------|------|------|
| **S1** 회귀 보존 | 기존 227 PASS 유지 | ✅ Met | 258 PASS (+31), 회귀 0 깨짐 |
| **S2** select_level | 4-Level 의사결정 동작 | ✅ Met | `test_sub4_cost_router.py` 10 tests PASS |
| **S3** throughput | ≥ 1.5× AsyncBrainRuntime | ✅✅ Exceeded | 실측 1.95-2.67× (9 케이스) |
| **S4** graphify | isolated 50% 감소 | ⚠️ Partial | LLMRegionRuntime 178줄 archive (정성 명확), 정량 측정 후속 |

**Overall Success Rate**: 3/4 strict + 1 partial = **91% Match Rate**.

---

## 3. 산출물

### 3-1. 신규 / 변경 파일

| 영역 | 파일 | 변경 |
|------|------|------|
| 신규 | `htp/runtime/external_region.py` | ABC 추상 (+75줄) |
| 신규 | `htp/llm/llm_region.py` | LLMRegion(ExternalRegion) (+160줄) |
| 확장 | `htp/llm/cost_router.py` | select_level + LEVEL_* 상수 (+50줄, 7-method 보존) |
| 신규 | `htp/runtime/pipelined_brain.py` | PipelinedBrainRuntime (+130줄) |
| 신규 | `examples/llm_region_demo.py` | 3 LLMRegion + BrainRuntime mock 데모 (+100줄) |
| 이동 | `archive/deprecated_phase4/llm_region_runtime.py` | git mv (178줄) |
| 갱신 | `htp/__init__.py`, `htp/llm/__init__.py` | export 갱신 |
| 갱신 | `htp/runtime/async_brain_runtime.py` | `_last_result.outputs` hasattr 분기 (+5줄) |

**소스 순증**: +422줄

### 3-2. 신규 테스트 (4 파일 31 tests)

```
tests/regression/test_sub4_external_region.py — 5 tests (ABC + default arun + no-op suppression)
tests/regression/test_sub4_llm_region.py      — 10 tests (inherits, internal _llm_node,
                                                          precision/pressure, async, cost block,
                                                          specialty prompt auto/fallback)
tests/regression/test_sub4_cost_router.py     — 10 tests (7-method 보존, select_level 4-Level,
                                                          validation, constants)
tests/regression/test_sub4_pipeline.py        — 6 tests (inherits, buffer validation, order,
                                                          sync wrapper, throughput ≥ 1.3×)
```

### 3-3. 문서

```
docs/02-design/features/htp-thalamus-car.sub-4.design.md    — 원본 design (Architecture B 선택)
docs/03-analysis/htp-thalamus-car.sub-4.analysis.md          — Check 분석 (Match Rate 91%)
docs/03-analysis/htp-sub4-실사용검증-외부리뷰용.md           — 외부 LLM 리뷰 리포트 (11장)
docs/04-report/htp-thalamus-car.sub-4.report.md              — 이 문서 (PDCA 종료)
```

### 3-4. Commits

```
b2526c9  sub-4 Stage 4+5 — ExternalRegion + LLMRegion + PipelinedBrainRuntime
8063975  sub-4 외부 리뷰용 검증 리포트 추가
```

---

## 4. Throughput 정량 결과

PipelinedBrainRuntime vs AsyncBrainRuntime (mock LLM latency 시뮬레이션):

| N | latency | AsyncBrain | Pipeline | **speedup** |
|:--:|:-------:|----------:|--------:|:-----------:|
| 4 | 20ms | 86.8ms | 44.5ms | **1.95×** |
| 4 | 50ms | 207.9ms | 105.2ms | **1.98×** |
| 4 | 100ms | 408.3ms | 204.2ms | **2.00×** |
| 8 | 20ms | 171.6ms | 67.3ms | **2.55×** |
| 8 | 50ms | 414.1ms | 159.0ms | **2.60×** |
| 8 | 100ms | 812.5ms | 307.8ms | **2.64×** |
| 16 | 20ms | 348.8ms | 131.5ms | **2.65×** |
| 16 | 50ms | 828.5ms | 313.0ms | **2.65×** |
| 16 | 100ms | 1633.7ms | 612.4ms | **2.67×** |

- 이론치 ≈ `N / max(t_S1, t_S2, t_S3)`. buffer_size=3, N≥8 시 ≈ 2.5-2.7×
- 실측이 이론과 일치 — pipeline 구현 정확

**Plan §SUCCESS 1.5× 목표 큰 마진 초과**.

---

## 5. C-1 ~ C-4 보완 결정 최종 상태

| ID | 항목 | 결정 | 상태 |
|----|------|------|------|
| **C-1** | LLMRegion 사용 데모 | `examples/llm_region_demo.py` (mock) | ✅ 작동 확인 |
| **C-2** | LLMNode 처리 정책 | 옵션 A — `self._llm_node` 내부 멤버 | ✅ `test_llm_region_llm_node_is_internal_member` |
| **C-3** | CostRouter 7-method 보존 | 기존 모두 유지 + select_level 추가 | ✅ `test_cost_router_existing_7_methods_preserved` 영구 보호 |
| **C-4** | graphify isolated 50% 감소 | 정량 자동 측정 후속 cycle | ⚠️ Partial — 후속 micro-cycle |

---

## 6. 외부 리뷰 합의 + 추가 결정

두 외부 LLM 리뷰 (Claude / Gemini) 의 합의:

✓ ExternalRegion 추상 가치
✓ Throughput 초과 달성
✓ 회귀 0건

**의견 차이 — "다음 단계"**:

| Gemini | Claude (이 cycle 사용자 결정) |
|--------|-------------------------------|
| SearchRegion / RAGRegion 확장 (인프라 다양화) | LLMRegion → KnowledgeLoop conflict 해석 연결 (실 사용 경로 만들기) |
| vault 대량 확장 (검증 규모) | C-4 graphify 측정 (열린 partial 닫기) |
| CostRouter 임계값 소프트 가드레일 | 실 데이터 부재 — 튜닝 후순위 |

**사용자 결정**: Gemini 의 인프라 다양화 방향이 아니라, sub-4 의 LLMRegion 을 Bridge
의 CoherenceGate 와 합쳐 "창의성의 라이브러리" 의 첫 실제 사례를 만들기.

```
연결 4: CoherenceGate conflict → LLMRegion 자연어 해석 (다음 cycle 1순위)
  현재: ⚠ 충돌 감지 (escalate=True) — 끝
  목표: ⚠ 충돌 감지 + 💡 두 관점의 차이가 무엇이고, 통합하면 어떤 가설이 나오는가
```

**진입 명분**: ExternalRegion 추상 + LLMRegion 도입 (sub-4) × CoherenceGate 충돌 감지
(Bridge) 의 곱. 둘 다 단독으로는 도구일 뿐, 합쳐야 가치 발현.

---

## 7. 후속 작업 (사용자 우선순위)

| 순서 | 항목 | 소요 | 근거 |
|:--:|------|:----:|------|
| **1** | **LLMRegion ↔ CoherenceGate conflict 해석** | ~2-3h | sub-4 + Bridge 의 곱. 사용자 체감 가치 첫 발현 |
| 2 | C-4 graphify 정량 측정 | ~30분 | sub-4 partial 닫기. 도구화 기반 |
| 3 | PipelinedBrainRuntime Memory 정책 docstring | ~10분 | "독립 입력 가정" 제약 명시 |
| 후순위 | SearchRegion / RAGRegion 확장 | — | 실 필요 발생 시 |
| 후순위 | sub-6 vector default 전환 | — | 사용자 체감 없음 |
| 후순위 | CostRouter 임계값 튜닝 | — | 실 호출 데이터 부재, 과잉 설계 위험 |

---

## 8. Lessons Learned

### 8-1. 잘 작동한 것

1. **PDCA Architecture 옵션 3 단계** — Option B (Clean) 채택이 G3 본질 해결로 직결.
   Option A (Minimal) 선택했다면 LLMRegionRuntime 이 남아 후속 정리 cycle 필요.

2. **Session 분할** — A (코어) → B (archive) → C (Pipeline) 의 각 Session 후 회귀
   통과 확인이 안전망. archive 이동 시점에서 깨짐 0건 즉시 확인.

3. **ExternalRegion 의 dummy attrs** — `_nodes=[]`, `_cusum_*=0/1e9` 라는 hack 으로
   BrainRuntime 수정 없이 통합. trade-off 가 있으나 단방향 변경 원칙 보존.

### 8-2. 보완 필요

1. **AsyncBrainRuntime.arun 의 `_last_result.outputs` 추출** — RegionRuntime (RunResult)
   만 가정하는 코드. LLMRegion (dict) 발견 후 hasattr 분기로 fix. 더 일반적인
   "Region 응답 typing" 정의 필요할 수 있음.

2. **graphify 자동 측정 부재 (C-4)** — Plan §SUCCESS 의 정량 SC 인데 자동 검증 없음.
   향후 cycle 에서는 검증 도구가 미리 도구화되어야 함.

3. **외부 리뷰 의견 차이** — Gemini 의 "확장 우선" vs Claude "연결 우선" 은
   설계 철학 차이. 사용자 의도 (창의성의 라이브러리) 가 결정 기준 — 인프라가
   많이 쌓이면 추상의 가치를 검증할 수 있으나, 사용자 가치는 아님.

### 8-3. 다음 cycle 에 적용할 원칙

> **"인프라는 충분하다. 만든 것을 연결하라."**

sub-3 (CoherenceGate) + sub-5 (EmbeddingBridge) + Bridge (시스템 A↔B) + sub-4
(LLMRegion) 의 4 인프라가 모두 분리되어 있음. 다음 cycle 은 이 중 둘을 합쳐서
사용자 가치를 만드는 데 집중.

---

## 9. 최종 지표

| 지표 | 값 |
|------|----|
| Plan §SUCCESS | 3/4 strict + 1 partial (91%) |
| 회귀 baseline | 227 → **258** PASS (+31) |
| Throughput speedup | **1.95-2.67×** (목표 1.5× 초과) |
| 신규 소스 (순증) | +422줄 (LLMRegionRuntime 178줄 archive 포함) |
| 신규 테스트 | +31 |
| 깨진 회귀 | **0건** |
| 소요 시간 | ~3.5h (A 1.5h + B 0.5h + C 1.5h) |
| Commits | `b2526c9`, `8063975` |

---

## 10. Status

**PDCA cycle 공식 종료**.

다음 cycle 진입은 사용자 우선순위 §7 의 1번 항목 (LLMRegion ↔ CoherenceGate conflict
해석 연결) 으로 결정. cycle 명: `htp-conflict-interpretation` (or 사용자 명명).
