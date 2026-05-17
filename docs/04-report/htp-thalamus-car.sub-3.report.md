---
template: report
feature: htp-thalamus-car
sub_cycle: sub-3 (Stage 3)
date: 2026-05-17
author: Mindbuild
status: Completed
match_rate: 99%
---

# htp-thalamus-car sub-3 Completion Report

## Executive Summary

| Perspective | Content |
|-------------|---------|
| **Problem** | (G2) 다중 Region 응답의 temporal binding 부재 — A가 "사과", B가 "토마토"로 해석해도 불일치 미감지. 학습 신호로의 conflict 활용 부재. |
| **Solution** | **Option B (Modular Strategy)** 채택 (sub-2 router/ 패턴 일관): `CoherenceStrategy` Protocol + `PairwiseCoherenceGate` (O(N²)). BrainRuntime 의 옵션 hook 으로 추가 (coherence=None 기본 → 회귀 보호). `MemorySystem.swr_priority` 가 `novelty × reward × (1 + conflict_magnitude)` 로 확장 — conflict 가 학습 신호 증폭. |
| **Function/UX Effect** | Region 간 응답 불일치가 자동 감지되어 `BoundResponse.escalate_to_pfc` 로 PFC top-down 트리거. 동일 conflict 가 episode 의 swr_priority 증폭 → 수면 consolidation 시 학습 강화. Plan G5 의 conflict 감지 정확도 (recall 10/10, FP 0/10) 초과 달성. |
| **Core Value** | sub-2 와 동일한 Strategy 패턴 적용으로 *아키텍처 일관성* 확보. 향후 LSH (N≥16) / EmbeddingBridge 모두 동일 Protocol 추가 구현체. v4 Rev 1.3 "루프를 먼저 닫는다" 원칙의 4차 실증 — *학습 신호* 가 코드에 내장된 자기진단 (conflict_magnitude). |

### 1.3 Value Delivered (4 perspectives, 실측 기준)

| Perspective | Planned | Delivered | Δ |
|-------------|---------|-----------|---|
| Test count | 140 → 146 (+6) | 140 → **148 (+8)** | **+2 초과** |
| Match Rate | ≥ 90% | **99%** | +9pp |
| Plan G5 (conflict 감지) | recall ≥ 9/10, FP ≤ 1/10 | **10/10, 0/10** | recall +1, FP -1 |
| 회귀 영향 | Stage 5 통합 7건 무영향 | 0건 깨짐 | 완벽 |

---

## Context Anchor

| Key | Value |
|-----|-------|
| **WHY** | Region 응답 binding + conflict 정량화 + 학습 신호 증폭 |
| **WHO** | HTP 개발자 + BrainRuntime 사용자 (coherence 옵션 off 기본 → 무영향) |
| **RISK** | (R1) Stage 5 회귀 / (R2) O(N²) / (R3) threshold |
| **SUCCESS** | 148 통과, conflict 감지 10/10, swr_priority 단조 증가 |
| **SCOPE** | Stage 3 만. LSH 구현체는 sub-3 OUT (N≥16 trigger) |

---

## 2. Plan → Design → Implementation 일관성

### 2.1 Decision Record Chain

```
[Plan]   Stage 3 = CoherenceGate (FR-12~15) + Memory novelty 연동
   ↓
[Design] Selected Option: B — Modular Strategy (사용자 선택)
   ↓     sub-2 router/ 패턴 일관 (CoherenceStrategy Protocol)
   ↓
[Do]     Session 2분할
   ↓       stage-3-coherence-core  2cc7cfa  140 → 144
   ↓       stage-3-integration     c6961b0  144 → 148
   ↓
[Check]  Match Rate 99%, Gap 0건 (Critical/Important)
   ↓
[Report] (현 문서)
```

### 2.2 Key Decisions & Outcomes

| Decision | Followed? | Outcome |
|----------|:---:|---------|
| Option B (Modular Strategy) | ✅ | `coherence/` 패키지 + Protocol — sub-2 router/ 패턴 일관 |
| `CoherenceStrategy` runtime_checkable Protocol | ✅ | `isinstance(gate, CoherenceStrategy)` PASS |
| `BoundResponse` 위치 `types.py` | ✅ | Plan §5 file map 정확 일치 |
| BrainRuntime coherence=None 기본 | ✅ | Stage 5 통합 7건 회귀 0건 |
| swr_priority = novelty × reward × (1 + conflict) | ✅ | conflict=0 시 기존 식 동등 보장 |
| SQLite schema 변경 회피 | ✅ | in-memory `_conflict_by_episode` dict — migration risk 0 |
| LSH 구현체 sub-3 OUT | ✅ | 별도 trigger (N≥16) 시 `htp-thalamus-coherence-lsh` |

---

## 3. Plan Success Criteria Final Status

### Plan FR-12 ~ FR-15 + sub-3 G1~G6

| ID | Criterion | Status | Evidence |
|----|-----------|:---:|----------|
| **FR-12** | CoherenceGate.bind() pairwise + conflict + precision-weighted fusion | ✅ Met | `pairwise.py:60-105` |
| **FR-13** | BoundResponse dataclass (responses/coherence/conflict/fused_vec/escalate) | ✅ Met | `types.py:29-43` |
| **FR-14** | BrainRuntime CoherenceGate 삽입 (Region 응답 후, PFC 전) | ✅ Met | `brain_runtime.py:312-321` |
| **FR-15** | swr_priority = novelty × reward × (1 + conflict_magnitude) | ✅ Met | `memory_system.py:78-80` + `episode_store.py:104-105` |
| **G1**   | 누적 테스트 ≥ 146 | ✅ Met | **148** 달성 (+2) |
| **G2**   | Protocol 다형성 isinstance | ✅ Met | `test_coherence_strategy_protocol_compliance` |
| **G3**   | BrainRuntime 기본 동작 무변경 | ✅ Met | Stage 5 통합 7건 PASS, memory 17 tests PASS |
| **G4**   | DAG coherence/* → runtime 미참조 | ✅ Met | `test_coherence_file_dag_isolation` 2/2 |
| **G5**   | Conflict 감지 recall ≥ 9/10, FP ≤ 1/10 | ✅ **10/10, 0/10** | `test_pairwise_conflict_detection_accuracy` |
| **G6**   | SWR priority conflict 단조 증가 | ✅ Met | `test_swr_priority_conflict_amplification` |

**Overall Success Rate: 10/10 (100%)**

---

## 4. Delivered Artifacts

### 4.1 신규 파일 (5)

| 파일 | LoC | 책임 |
|------|----:|------|
| `htp/thalamus/types.py` | 39 | RegionResponse + BoundResponse dataclass |
| `htp/thalamus/coherence/__init__.py` | 29 | 공개 export |
| `htp/thalamus/coherence/base.py` | 47 | CoherenceStrategy Protocol |
| `htp/thalamus/coherence/pairwise.py` | 100 | PairwiseCoherenceGate (O(N²)) |
| `tests/regression/test_stage3_coherence.py` | 158 | M1+M2+M3 신규 4 tests |
| `tests/regression/test_stage3_integration.py` | 90 | M4+M5 신규 2 tests |
| `docs/02-design/features/htp-thalamus-car.sub-3.design.md` | 349 | Design (Option B) |
| `docs/03-analysis/htp-thalamus-car.sub-3.analysis.md` | ~200 | Check Phase 결과 |

### 4.2 수정 파일 (4)

| 파일 | 변경 | 내용 |
|------|------|------|
| `htp/runtime/brain_runtime.py` | +52 LoC | coherence DI + `_bind_region_responses` hook |
| `htp/memory/memory_system.py` | +25 LoC | `save(conflict_magnitude=0)` + `swr_priority()` 헬퍼 + in-memory dict |
| `htp/memory/episode_store.py` | +10 LoC | `tag_swr(conflict_map=None)` 식 확장 |
| `tests/unit/test_no_circular_deps.py` | +27 LoC | `_COHERENCE_DIR` parametrize 추가 |

### 4.3 commits (3)

```
8c06eed sub-3 Design — CoherenceGate (Option B Modular Strategy)
2cc7cfa sub-3 stage-3-coherence-core  M1+M2+M3   140 → 144 (+4)
c6961b0 sub-3 stage-3-integration     M4+M5+M6   144 → 148 (+4)
```

---

## 5. Metrics

### 5.1 Volume

| 지표 | 값 |
|------|---:|
| 신규 LoC (htp/) | 215 |
| 수정 LoC (htp/) | ~87 |
| 신규 테스트 LoC | 248 |
| Design 문서 LoC | 349 |
| Analysis 문서 LoC | ~200 |
| 총 변경 LoC | ~1,099 |

### 5.2 Quality

| 지표 | Before (sub-2 end) | After (sub-3 end) |
|------|------:|------:|
| 총 테스트 | 140 | **148** (+8) |
| 회귀 깨짐 | 0 | **0** |
| 실행 시간 | 1.21s | 1.46s (+0.25s, fixture random seed 고정 영향) |
| Match Rate | 99% (sub-2) | **99%** (sub-3) |
| Public API 호환성 | 100% | **100%** |
| Critical Gap | 0 | **0** |
| DAG 검증 파일 | 12 | **14** (+ coherence 2) |

---

## 6. Lessons Learned

### 6.1 잘된 점

1. **sub-2 패턴의 재사용** — RouterStrategy 패턴이 CoherenceStrategy 에 그대로 적용. OCP 가 *프로젝트 레벨* 의 일관성으로 확장됨. 다음 sub-cycle 도 같은 패턴 적용 가능 (Strategy Protocol + 다형성 구현체).
2. **SQLite migration 회피 결정** — Decision Record "swr_priority 식" 에서 in-memory dict 선택. 회귀 risk 0 + 같은 세션 내 정확 동작. trade-off 가 explicit (Gap #1 으로 문서화).
3. **Plan G5 초과 달성** — Conflict 감지 정확도 측정값이 기준 초과 (10/10, 0/10). seed=42 고정으로 deterministic 검증.
4. **2 session 분할 적중** — coherence-core (Pairwise 자체) → integration (BrainRuntime/Memory 연결) 의 자연스러운 분리. 각 commit 이 독립적으로 회귀 보호 검증.

### 6.2 개선 여지

1. **Region output_vec 차원 통일 임시 처리** — `_bind_region_responses` 에서 padding/truncate fallback. Region 마다 다른 차원이 *정상* 인지 *버그* 인지 불명확. sub-4 에서 `ExternalRegion` 도입 시 표준 차원 정의 필요.
2. **conflict_magnitude 영속화 부재** — Gap #1 (Low). 같은 세션 내에서만 priority 증폭 작동. 세션 간 영속화는 sub-4+ 에서 SQLite 마이그레이션 시 함께.
3. **β/α/conflict threshold 의 application context 별 튜닝** — sub-2 의 Review #6 권장값 표 미작성 (sub-2 OUT-OF-SCOPE 였음). sub-3 의 conflict threshold (0.3) + escalation (0.7) 도 동일. sub-4 이후 누적 데이터 기반 튜닝 사이클 필요.

### 6.3 v4 Rev 1.3 "루프를 먼저 닫는다" 원칙의 4차 실증

| 차수 | 시점 | 발견/예방 |
|-----|-----|---------|
| 1차 | sub-1 직후 | Critical Gap #3 (encoder.fit refit) 24h 내 발견 → 옵션 A-2 영속화 |
| 2차 | sub-1 cross-language 실험 | "언어가 hub" 가설 부분 실증 |
| 3차 | sub-2 (β sweep 메트릭) | 코드에 자기진단 내장 — 외부 튜닝 도구 불필요 |
| **4차** | **sub-3 (현 사이클)** | **conflict_magnitude 가 학습 신호로 자동 누적 — 외부 평가 도구 불필요** |

원칙의 진화: 매 sub-cycle 마다 *코드 안에 내장된 자기 측정 메커니즘* 이 누적되어, 다음 sub-cycle 의 튜닝 근거가 자동으로 확보됨. sub-4 (LLMRegion) 진입 시 sub-2 의 β sweep + sub-3 의 conflict 통계가 이미 사용 가능한 학습 자료가 됨.

---

## 7. Next Cycle Recommendations

### 7.1 즉시 가능 — sub-4 또는 L2 sidequest

**선택 A: sub-4 Design 진입 (Stage 4 + 5)**
```bash
/pdca design htp-thalamus-car   # ExternalRegion + LLMRegion + Pipeline
```
- 누적 테스트 목표: 148 → 154 (Plan §5 명시 89)
- 핵심: `ExternalRegion` 추상 + `LLMRegion(ExternalRegion)` + `vec_to_prompt/prompt_to_vec`

**선택 B: L2 sidequest (`htp-knowledge-cli-polish`, TODO.md 백로그)**
```bash
/pdca plan htp-knowledge-cli-polish
```
- 5개 CLI 기능 추가 (batch / stdin / filter / edit / export)
- 1-2일 소요, 0.5 가 "매일 쓰는 prototype" 단계 (L2) 도달

### 7.2 향후 별도 PDCA 사이클 후보

- **`htp-region-output-vec-standard`** — sub-3 Gap #1 + sub-3 Lessons #1 통합. Region 차원 표준화
- **`htp-thalamus-coherence-lsh`** — Plan §R2 trigger N≥16 도달 시
- **`htp-thalamus-async-pipeline`** — sub-5 EmbeddingBridge 진입 후 측정 병목 시
- **`htp-routing-tuning`** — β/α/conflict threshold application context 별 권장값

---

## 8. Sign-off

| 항목 | 결과 |
|------|------|
| **Plan SC (FR-12~15)** | 4/4 (100%) |
| **Plan G (G1~G6)** | 6/6 (100%) |
| **Decision Record** | 7/7 따름 |
| **회귀 보호** | 0건 깨짐 |
| **Match Rate** | **99%** (≥ 90% target) |
| **Plan G5 초과** | recall 10/10 (기준 9), FP 0/10 (기준 ≤1) |
| **PDCA 단계** | Plan → Design (Option B) → Do (2 sessions) → Check (sub-3.analysis) → **Report (현 문서)** |

**sub-3 PDCA 완료 ✅ — sub-4 또는 L2 sidequest 진입 준비됨.**
