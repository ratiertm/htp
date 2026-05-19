---
template: report
feature: htp-conflict-memory
date: 2026-05-19
status: Completed
predecessor: htp-conflict-interpretation + 양적 검증 50건
---

# Report — htp-conflict-memory

## Executive Summary

| 관점 | 1-2 문장 |
|------|---------|
| **Problem** | htp-conflict-interpretation 의 LLM 해석이 일회용 — 같은 충돌 반복 시 매번 ~14s LLM 새 호출. 통찰이 자산화되지 않음. |
| **Solution** | interpretation 을 Episode 로 저장 (state_vec = trigger 의 text 임베딩). 다음 비슷한 충돌 시 CA3 pattern completion 으로 이전 해석 recall. quality_hint heuristic 으로 best-match 정렬. |
| **Function/UX Effect** | CLI ingest 가 escalate=True 시 `📚 이전 유사 충돌` + `💡 새 해석` 동시 노출. 두 번째 비슷한 충돌부터 통찰 *축적*. |
| **Core Value** | **"창의성의 라이브러리" 정의 완성** — 충돌 발견 (sub-3) → 해석 (sub-conflict-interp) → **기억** (이번) → 재활용. 단순 도구가 아닌 *학습하는 시스템*. |

---

## 1. Plan SUCCESS 6/6 strict PASS

| SC | 결과 |
|----|------|
| SC1 회귀 보존 | ✅ 283 → **303** PASS (+20) |
| SC2 Episode 저장 | ✅ winner='conflict_interpreter' 시연 확인 |
| SC3 recall_hint 채워짐 | ✅ v4 시연 mismatch=0.530 < 0.6 → HIT |
| SC4 quality_hint heuristic | ✅ 19 키워드 + 5 단위 테스트 |
| SC5 schema 마이그레이션 | ✅ idempotent ALTER + legacy DB 로드 |
| SC6 실 시연 2회 충돌 | ✅ v4: warm-up → 1회차 → 2회차 recall HIT |

**Match Rate**: **100%**.

---

## 2. 실 시연 결과 (v4 — warm-up + Redis + Kubernetes)

```
warm-up (cold-start 흡수)        12.6s
1회차 Redis (LRU eviction)       14.0s
  💡 "conflict is categorical, not factual — Redis LRU + nginx vs Hebbian/CA3 ..."
  → Memory Episodes: 1

2회차 Kubernetes (pod eviction)  16.7s
  📚 RECALL HIT ✓ mismatch=0.530, quality=0.33
     trigger:    "Redis LRU 캐시 eviction 전략 ↔ 해마 CA3 패턴 완성..."
     prev_interp: "The conflict is categorical, not factual ..."
  💡 새 해석: "두 진술 모두 '인프라' 도메인의 'eviction' 메커니즘이나
              추상 계층이 다름 — Redis LRU 데이터 계층 vs Kubernetes
              pod 리소스 계층"
```

**Insight 누적 시연 의의**:
- 1회차: LLM 새 호출, 새 해석 생성, Episode 저장
- 2회차: **이전 해석 즉시 보임** + **새 LLM 해석이 더 정밀한 비교 추가** (LRU 데이터 vs pod 리소스)
- 누적 → recall: 통찰 자산화 완성

---

## 3. 발견 + Fix (2 bugs)

### Bug 1: recall key 불일치

**원인**: `save_conflict` 가 `interpretation` 임베딩을 state_vec 으로 저장했으나,
`_try_recall_conflict` 는 *text* 임베딩으로 검색. 다른 벡터 공간 비교 → 매칭 실패.

**Fix**: `save_conflict` 의 첫 인자 → `trigger_vec` (= text 임베딩).
"비슷한 input → 이전 해석 recall" 데이터 흐름이 더 자연스러움.

### Bug 2: threshold 너무 엄격

**원인**: `CONFLICT_RECALL_MISMATCH_THRESHOLD = 0.3` 은 64-dim sparse vec 기준.
384-dim e5 dense vec 에서 일반 cosine ~0.85 → L2 ~0.55 → 0.3 미통과.

**Fix**: 0.3 → **0.6** (cosine ~0.82 ↔ "비슷한 도메인" 수준). v4 실측 0.530 으로 HIT.

---

## 4. 산출물

### 신규 (10)

- `htp/memory/quality_hint.py` — 19 키워드 heuristic
- `htp/knowledge/conflict_prompt.py` (이전 cycle, 보강 X)
- `tests/knowledge/test_conflict_memory.py` — 15 tests
- `scripts/conflict_quant_eval.py` — 50건 양적 검증 스크립트
- `docs/01-plan/features/htp-conflict-memory.plan.md`
- `docs/02-design/features/htp-conflict-memory.design.md`
- `docs/03-analysis/htp-conflict-memory.analysis.md`
- `docs/03-analysis/conflict_quant_summary.md` — 양적 분석
- `docs/03-analysis/conflict_quant_raw.jsonl` — raw 50건
- `docs/04-report/htp-conflict-memory.report.md` — 이 문서

### 수정 (6)

- `htp/memory/types.py` — Episode.interpretation_text
- `htp/memory/episode_store.py` — SCHEMA + ALTER + save + _row_to_episode
- `htp/memory/memory_system.py` — save_conflict / recall_conflict / threshold 0.6
- `htp/knowledge/loop.py` — memory DI + recall + save + SKIP_MARKERS
- `htp/knowledge/cli/ingest.py` — 📚 출력
- `tests/unit/test_no_circular_deps.py` — DAG 룰 갱신

**소스 순증 ~256줄. 테스트 +250. 깨진 회귀 0건**.

---

## 5. Decision Record

| Decision | 결과 |
|----------|------|
| 양적 검증 먼저 (50건) | conflict 값과 quality 무관 발견 → 사전 confidence 필터 불가 |
| Episode 확장 (interpretation_text) | 단일 store 일관성 |
| quality_hint heuristic | recall best-match 정렬, 사용자 노출 quality 보장 |
| Architecture B (Auto-create) | sub-conflict-interp 와 동일 패턴 — onboarding 마찰 0 |
| recall 우선 → 그 후 새 LLM | UX 공명 좋음 — "이전엔 이랬다" + "이번엔 더 정밀하다" |
| winner fixed | search_similar 의 winner_filter 단순 |

---

## 6. 핵심 성취 — "창의성의 라이브러리" 정의 완성

이번 cycle 전:
- sub-3 (CoherenceGate): 충돌 *발견*
- sub-4 (LLMRegion): 외부 호출 추상
- Bridge: KnowledgeLoop 통합
- sub-conflict-interpretation: 충돌 *해석*

이번 cycle 후 추가:
- **기억** (interpretation → Episode → state_vec=trigger_vec → recall)
- **재활용** (다음 비슷한 충돌 시 즉시 노출)

이 5가지가 결합해 HTP 는 "**충돌을 발견하고, 해석하고, 기억하고, 재활용하는**"
완전 루프를 갖춤. *학습하는 지식 시스템*.

---

## 7. Lessons Learned

### 잘 작동한 것

1. **양적 검증 먼저** — Plan 결정 (저장 정책 / quality_hint) 의 근거를 데이터로
   확보. "100% LLM 응답률 + 1/3 고품질" 통계가 *모든 interpretation 저장 + recall
   필터* 결정을 정당화.

2. **사용자 의도 명시화** — "interpretation 도 벡터화" 라는 한 줄이 핵심 설계 포인트.
   다만 실제 흐름에서 *recall key* 가 trigger 인지 interpretation 인지는 시연으로
   드러남 (Bug 1). 사용자 의도의 *deeper interpretation* 이 진짜 가치.

3. **시연이 단위 테스트를 보완** — Bug 1, 2 모두 단위 테스트로는 못 잡힘. 실 LLM +
   실 인코더 + 실 데이터 조합에서만 발현.

### 보완 필요

1. **claude CLI cold-start** — Production 영향 미확인. 사용자 설계 cycle 별도 필요.
2. **threshold hard-coded** — 384-dim 전용 0.6. encoder 변경 시 재조정.
3. **trigger context 50자 cap** — 표시 정보 부족. KnowledgeEntry id 저장 후 전체 join 검토.

### 다음 cycle 원칙

> **"한 번에 한 연결만"** (이전 cycle 원칙) 을 유지하되,
> **"양적 검증 → 설계 → 시연"** 순서 정착.

---

## 8. 최종 지표

| 지표 | 값 |
|------|----|
| Plan SUCCESS | 6/6 strict (Match Rate **100%**) |
| 회귀 baseline | 283 → **303** PASS (+20) |
| 신규 소스 | +256 줄 |
| 신규 테스트 | +15 |
| 깨진 회귀 | **0건** |
| 발견 + fix | **2 bugs** (recall key, threshold) |
| 소요 시간 | ~3.5h (양적 검증 0.5h + Plan/Design 1h + Do 1.5h + 시연 0.5h) |

---

## 9. Status

**PDCA cycle 공식 종료**. 다음 cycle 사용자 결정.
