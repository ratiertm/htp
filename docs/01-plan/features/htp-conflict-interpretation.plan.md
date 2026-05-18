---
template: plan
feature: htp-conflict-interpretation
date: 2026-05-19
author: Mindbuild
project: HTP (Hub Topology Programming)
predecessor: sub-4 (LLMRegion + CostRouter.select_level) + Bridge Integration (CoherenceGate)
status: Draft (Architecture 옵션 확인 후 Design → Do)
---

# Plan — htp-conflict-interpretation

**한 줄**: sub-4 의 LLMRegion 과 Bridge 의 CoherenceGate 를 합쳐, escalate=True 시
LLMRegion 이 충돌을 자연어로 해석하게 만든다. "창의성의 라이브러리" 첫 실 사례.

---

## Executive Summary

| 관점 | 1-2 문장 |
|------|----------|
| **Problem** | sub-4 LLMRegion / Bridge CoherenceGate / sub-3 ingest 흐름이 모두 분리되어 사용자 가치 미발현. CoherenceGate 가 충돌 감지하면 `⚠ 충돌 감지` 만 출력되고 *왜* 인지는 침묵. |
| **Solution** | KnowledgeLoop.ingest 의 `coherence_info["escalate"] == True` 분기에서 LLMRegion 호출 → 충돌 entries (신규 + top-3 이웃) 를 prompt 로 구성 → 자연어 해석을 `KnowledgeEntry.interpretation` 에 저장 + CLI 출력. |
| **Function/UX Effect** | CLI ingest 가 충돌 감지 시 *자동으로* 두 관점의 차이를 설명하고 통합 가설을 제안. Mock 기본값으로 API 비용 사고 방지. |
| **Core Value** | RAG / LangChain 으로는 불가능한 HTP 고유 — 의미 충돌의 발견 + 해석 + 보존. |

---

## Context Anchor

| 키 | 값 |
|----|----|
| **WHY** | 인프라 (sub-3/4/5/Bridge) 가 분리되어 사용자 가치 미발현. "인프라는 충분하다. 만든 것을 연결하라" 원칙의 첫 적용. |
| **WHO** | HTP 사용자 — Vault 에 새 지식 추가 시 기존 지식과의 충돌이 *왜* 인지 즉시 알고 싶음. |
| **RISK** | (1) Anthropic API 비용 사고. (2) Mock 해석이 trivial 해 검증 어려움. (3) escalate=True 가 너무 많아 노이즈 폭증. (4) prompt template 품질이 결과 좌우. |
| **SUCCESS** | (1) 회귀 258 보존 (2) Mock 으로 동작 end-to-end (3) 실데이터 (Vault 일부) 에서 의미 있는 해석 1건 이상 (4) escalate 호출 빈도 cap 동작 (5) interpretation JSONL 영속화 |
| **SCOPE** | KnowledgeLoop.ingest 통합 + KnowledgeEntry.interpretation + CLI 출력 + Mock/실 API 양쪽 흐름 + 회귀 테스트. **OUT**: 해석 품질 평가 자동화 (별도 cycle), 다른 ExternalRegion 통합, vault 대량 검증. |

---

## 1. 진입 전 결정 사항 (사용자 확정)

| # | 결정 | 값 |
|---|------|-----|
| 1 | 충돌 해석 호출 시점 | **ingest 시 동기** (즉시 출력) |
| 2 | Mock 모드 default | **Mock default 안전** — CLI default 는 MockLLMNode, 실 API 는 `--llm-model` 명시 |
| 3 | 호출 빈도 제한 | **CostRouter cap + 수동 daily limit** — `max_interpretations_per_session` 추가 (default 20) |
| 4 | 해석 결과 저장 | **KnowledgeEntry 새 필드** `interpretation: str \| None` |

---

## 2. Stage 분할

### Stage 1 — Core 통합
- KnowledgeLoop 에 `conflict_interpreter: LLMRegion \| None = None` 인자 추가
- None 이면 자동으로 `LLMRegion("conflict_interpreter", specialty="reasoning", use_mock=True)` 생성 (안전 default)
- `_evaluate_coherence` 결과의 `escalate=True` 시 `_interpret_conflict` 호출
- `KnowledgeEntry.interpretation: str \| None = None` 신설
- `_interpretations_count` 카운터 + `max_interpretations` cap

### Stage 2 — Prompt template
- 충돌 entries (신규 + top-3 이웃) 를 자연어 prompt 로 구성
- 도메인별 source 명시 — "신규 텍스트 (뇌과학) vs 기존 텍스트 3건 (AI/인프라)"
- LLM 응답 파싱 — `{"interpretation": "...", "hypothesis": "..."}` JSON

### Stage 3 — CLI 출력
- `htp.knowledge ingest` 가 escalate 시 `⚠ 충돌 감지 + 💡 <interpretation>` 출력
- `htp.knowledge list` 가 interpretation 보유 entries 마크 표시

### Stage 4 — JSONL 영속화
- `KnowledgeStore.append/load` 가 `interpretation` 필드 round-trip
- 기존 jsonl backward-compat — 필드 없으면 `None`

### Stage 5 — 테스트 + 실데이터 검증
- 단위: Mock interpreter / 카운터 cap / 영속화
- 통합: escalate→해석→저장→재로드 end-to-end
- 실데이터 (수동): 사용자가 실 API 1회 실행 → 해석 품질 평가

---

## 3. Module Map

| Module | 위치 | 변경 | 줄수 |
|--------|------|------|-----:|
| M1 KnowledgeEntry.interpretation | `htp/knowledge/types.py` | +1 필드 | +3 |
| M2 KnowledgeLoop.ingest 분기 | `htp/knowledge/loop.py` | escalate=True 분기 + _interpret_conflict | +60 |
| M3 Prompt template | `htp/knowledge/conflict_prompt.py` 신규 | escalate prompt 구성 | +50 |
| M4 KnowledgeStore JSONL | `htp/knowledge/persistence.py` | interpretation 필드 영속화 | +10 |
| M5 CLI ingest 출력 | `htp/knowledge/cli/ingest.py` | interpretation 출력 추가 | +5 |
| M6 CLI list 마크 | `htp/knowledge/cli/list_cmd.py` | 💡 마크 표시 | +5 |
| M7 테스트 | `tests/knowledge/test_conflict_interpretation.py` | 신규 6-8 tests | +150 |

**총 소스 ~ +130줄, 테스트 ~ +150줄**.

---

## 4. Architecture 옵션 (Design 단계 확인 필요)

### Option A — Minimal DI
KnowledgeLoop 가 `conflict_interpreter: LLMRegion | None` 만 받음. None 이면 무동작.
사용자가 명시 인스턴스 넘기지 않으면 기능 off.

### Option B — Auto Mock default (Recommended)
None 시 자동으로 MockLLMRegion 생성. CLI 도 default 로 Mock. 실 API 는
`--llm-model claude-sonnet-4-6 --no-mock` 등 명시.

### Option C — Strategy Protocol
`ConflictInterpreter(Protocol)` 신설 + `LLMConflictInterpreter(ConflictInterpreter)` 구현.
향후 `RuleBasedConflictInterpreter` 등 확장 가능. **단** 단일 구현체만 있으면 과잉.

---

## 5. Success Criteria

| # | 기준 | 검증 |
|---|------|------|
| SC1 | 회귀 258 보존 | pytest 전체 통과 |
| SC2 | escalate=True 시 interpretation 생성 | 통합 테스트 |
| SC3 | Mock 모드 default 작동 | CLI 무인자 실행 |
| SC4 | KnowledgeEntry.interpretation JSONL 영속화 | 재로드 round-trip |
| SC5 | max_interpretations cap 동작 | 카운터 테스트 |
| SC6 | 실데이터 해석 1건 (수동) | 사용자 실 API 1회 실행 |

---

## 6. Risk + Mitigation

| Risk | 가능성 | 영향 | 완화 |
|------|:------:|:----:|------|
| 실 API 비용 사고 | 중 | 높음 | Mock default + cap + CostRouter.should_block |
| Mock 해석 trivial | 높음 | 중 | "interpretation: mock conflict between X and Y" 같은 placeholder 도 영속화 흐름 검증엔 충분 |
| escalate 폭증 | 중 | 중 | max_interpretations=20/session cap + `--no-interpretation` flag |
| Prompt 품질 | 높음 | 중 | Stage 2 prompt template 명시. 실 API 1회로 검증 |
| KnowledgeEntry 변경 회귀 | 낮음 | 높음 | dataclass `interpretation = None` default. 기존 jsonl 호환 |

---

## 7. 진입 후 Design 단계에서 결정

1. **Architecture 옵션** A/B/C 중 어느 것?
2. **Prompt template 위치** — 별도 파일 vs loop.py 내부?
3. **CLI flag 정책** — `--no-interpretation` vs `--llm-model None`?
4. **JSONL backward-compat** — 기존 entries (interpretation 없음) 로드 시 처리?

→ Design 문서에서 확정 후 Do 진입.
