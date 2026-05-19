---
template: plan
feature: htp-conflict-recall-guardrail
date: 2026-05-20
author: Mindbuild
predecessor: htp-conflict-memory + 외부 리뷰 + 컨테이너 실측 3회
status: Do 진행 (Phase 1·2 만 — Phase 3 별도 cycle)
directive: docs/01-plan/features/claude_code_지시서_conflict_recall_phase1-2.md
---

# Plan — htp-conflict-recall-guardrail

**한 줄**: htp-conflict-memory 의 외부 리뷰 "Full MERGE GO" 판정이 컨테이너 실측
3회로 **NO-GO** 로 정정됨. 결함을 *테스트로 고정* + trigger-key 미검증을 *xfail 로
명문화* + 측정 스크립트를 *재현 가능하게 커밋* 하는 것이 이 cycle 의 전부.
Phase 3 처방은 별도 cycle.

---

## Executive Summary

| 관점 | 1-2 문장 |
|------|---------|
| **Problem** | 외부 리뷰 "MERGE GO" 후 컨테이너 실측: 거짓 양성 12/12 (100%). threshold 0.6 무력 + 튜닝 처방 4종 전부 FAIL. 결함이 *threshold 가 아니라* state_vec=trigger_vec recall key 설계 자체에 있음. 회귀 방지선 미구축 시 *다음 외부 리뷰* 도 같은 판정 오류 반복 위험. |
| **Solution** | 결함을 *테스트로 고정* (RED): 무관 입력 거짓 양성 거부 + xfail strict. 측정 스크립트 `scripts/` 에 영구 커밋. CLAUDE.md 에 의사결정 기록. **코드 수정 없음** — 결함을 *증거화* 만. |
| **Function/UX Effect** | 사용자 체감 변화 없음. 단 다음 cycle 에서 처방 시도 시 *증명 가능한 baseline* 확보. xfail 이 XPASS 되는 순간 처방의 성공이 자동 표시됨. |
| **Core Value** | "PDCA 의 Check 가 외부 인용 아니라 *측정* 으로 작동해야 한다" 의 첫 실증. 리뷰 (LLM 출력) 와 실측 (e5 임베딩 numeric) 의 불일치를 영구 기록. |

---

## Context Anchor

| 키 | 값 |
|----|----|
| **WHY** | 외부 리뷰 "Full MERGE GO" 판정 vs 실측 100% FP. PDCA Check 가 LLM 평가에 의존하면 시스템 측정과 분리될 위험. 결함을 RED 테스트로 고정해 향후 처방의 GREEN 전환을 자동 검출. |
| **WHO** | HTP 개발자 (자신). 다음 cycle 에서 처방 시도 시 *증명 가능한 baseline* 필요. |
| **RISK** | (1) RED 를 "고쳐서 GREEN" 만들려는 충동 (지시서 §3-2 금지). (2) 금지 4건 위반 (threshold 변경 / recall key 변경 / MERGE / cos-L2 리팩토링). (3) Phase 3 처방 미리 손대기. |
| **SUCCESS** | (1) 회귀 보존 (HF skip 모드 304) (2) HF 모드 RED 1건 + xfail strict 1건 + PASS 1건 명확히 분리 (3) 측정 스크립트 2개 `scripts/` 커밋 (4) CLAUDE.md 의사결정 기록 (5) 금지 §1 위반 0건 |
| **SCOPE** | Phase 1·2 만 (지시서 §2·§3). **OUT**: Phase 3 처방 / threshold 조정 / recall key 변경 / cos-L2 리팩토링 / 외부 리뷰 GO 근거 MERGE. |

---

## 1. 외부 리뷰 판정 정정 — 컨테이너 실측 데이터

### 측정 1 — 거짓 양성 (anchor 6 / probe 14)

| 모드 | NEG 인데 HIT | TRUE_POS HIT |
|------|:------------:|:------------:|
| 현재 시스템 (절대 L2 < 0.6) | **12/12 (100%)** | 2/2 |
| query-prefix 적용 | 10/12 (83%) | 2/2 |

→ "중세 고딕 성당 구조", "김치 발효" 같은 완전 무관 입력도 충돌 해석 recall HIT.
→ 시연 1건 (eviction 2회) HIT 한 이유: threshold 너무 느슨해 *무엇이든* HIT.

### 측정 2 — 처방 비교 (튜닝 레이어 4종)

| 처방 | HARD_NEG FP | TRUE_POS TP | 판정 |
|------|:----------:|:----------:|:----:|
| Baseline (절대 cos≥0.82) | 6/6 | 2/2 | FAIL |
| P1 상대마진 (gap≥0.05) | 0/6 | 1/2 | FAIL |
| P2 분포컷 (cos≥0.867) | 1/6 | 1/2 | FAIL |
| P3 결합 | 0/6 | 1/2 | FAIL |

→ TRUE_POS cos(0.865~0.890) ↔ HARD_NEG cos(0.848~0.869) 분포 겹침 → 절대컷·마진·
분포컷 어느 자(尺) 로도 분리 불가. **신호 자체가 trigger 임베딩에 없음**.

### 측정 3 — 코드 리딩 발견 검증

| 발견 | 판정 | 근거 |
|------|:----:|------|
| A (cosine 선택 / L2 컷 지표 불일치) | **반증** | e5 = L2 정규화 출력 → corr(cos,L2) = **-0.997** 단조 일치 |
| B (`_try_recall_conflict` 가 `encode()` passage-prefix 사용) | **확인** | query-prefix 적용 시 FP 12→10 부분 개선 (완전 해결 아님) |

### 핵심 결론

> 결함 위치는 threshold 가 아니라 **recall key 설계** (외부 리뷰 §9-1.1
> 의 `state_vec = trigger_vec` 결정). 이 결정은 *htp-conflict-memory
> Bug 1 fix 시 도입* 됐으나 미검증 — xfail strict 로 명문화.

---

## 2. 절대 금지 (지시서 §1)

```
❌ CONFLICT_RECALL_MISMATCH_THRESHOLD 값 변경 / 동적 조정 함수 작성
❌ recall key 설계 변경 (interpretation-key / hybrid / 구조화)
❌ 외부 리뷰 "MERGE GO" 근거 master 머지 / PR 생성
❌ cosine/L2 정합 리팩토링 (발견 A 반증됨)
```

이 4가지를 건드리면 지난 3회 측정이 무의미.

---

## 3. Stage 분할

### Phase 1 (작업 2-1·2-2·2-3) — 회귀 방지선

| 작업 | 내용 | 산출 |
|------|------|------|
| 2-1 | 무관 충돌 거짓 양성 거부 테스트 3건 append | `tests/knowledge/test_conflict_memory.py` |
| 2-2 | trigger-key 미검증 xfail strict 1건 append | 동일 파일 |
| 2-3 | 측정 스크립트 2개 `scripts/` 로 git mv (로직 수정 금지) | `scripts/conflict_recall_fp_eval.py`, `scripts/conflict_recall_remedy_eval.py` |

### Phase 2 (작업 3-1·3-2·3-3) — 검증 게이트

| 작업 | 내용 | 합격 기준 |
|------|------|----------|
| 3-1 | HF_SKIP 모드 회귀 | 303 → 304 (비-HF 신규 1건 추가) |
| 3-2 | HF 모드 RED/xfail/PASS 분리 확인 | unrelated FP=FAIL, xfail strict=xfail, query_prefix=PASS |
| 3-3 | CLAUDE.md 의사결정 기록 append | 외부 리뷰 GO 정정 + 금지 4건 명문화 |

---

## 4. Success Criteria (지시서 §3·§4 기준)

| # | 기준 | 검증 방법 |
|---|------|----------|
| SC1 | 비-HF 회귀 304 (303 + sanity 1건) | `HTP_SKIP_HF_DOWNLOAD=1 pytest -q` |
| SC2 | HF 모드 unrelated FP 테스트 = RED | `pytest -rxX` 결과 FAIL 표시 |
| SC3 | trigger-key xfail strict = xfail | 동일 명령 결과 xfail 표시 |
| SC4 | query_prefix 테스트 = PASS | 동일 명령 결과 PASS |
| SC5 | 측정 스크립트 2개 `scripts/` 커밋 | `git log` 확인 |
| SC6 | CLAUDE.md 갱신 | diff 확인 |
| SC7 | 금지 §2 위반 0건 | 코드 변경 = 테스트 추가만 |

---

## 5. Risk + Mitigation

| Risk | 가능성 | 영향 | 완화 |
|------|:------:|:----:|------|
| RED 를 GREEN 으로 "수정" 충동 | 중 | 높음 | 지시서 §5 "RED/xfail 은 결함의 증거이지 버그가 아니다" 명시 — 절대 건드리지 않음 |
| xfail strict 가 XPASS 되어 회귀 깨짐 | 낮음 | 중 | 현재 단위 테스트로는 xfail (의도된 실패) 확정. 우연 XPASS 시 그 자체가 신호 — 보고 후 다음 cycle |
| 측정 스크립트 로직 수정 유혹 | 낮음 | 중 | 지시서 §2-3 "그대로 커밋" 엄수. git mv 만 |
| Phase 3 처방 미리 손대기 | 중 | 높음 | 본 Plan §2 금지 4건 + Phase 2 종료 시점 명확화 |

---

## 6. 진입 후 결정 없음

이 cycle 은 지시서가 모든 결정을 잠가놓음. 사용자 추가 확인 불필요. Do 즉시 진입.
