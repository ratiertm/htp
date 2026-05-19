---
template: plan
feature: htp-conflict-memory
date: 2026-05-19
author: Mindbuild
predecessor: htp-conflict-interpretation + 양적 검증 (50건)
status: Draft (Architecture 옵션 확인 후 Design)
---

# Plan — htp-conflict-memory

**한 줄**: KnowledgeLoop 의 conflict interpretation 을 Memory L2 Episode 로 저장 +
다음 충돌 시 CA3 pattern completion 으로 recall. 통찰이 일회용 출력에서 *자산* 으로.

---

## Executive Summary

| 관점 | 1-2 문장 |
|------|----------|
| **Problem** | htp-conflict-interpretation 의 interpretation 이 entry 내부 필드로만 저장됨 — 같은 종류의 충돌이 반복되면 *매번 LLM 새로 호출*. 14s/call latency + 같은 통찰 반복 생성 낭비. |
| **Solution** | interpretation 을 임베딩화 (`encoder.encode(interp_text)`) → Episode 의 state_vec 으로 Memory L2 저장. 새 충돌 발생 시 CA3 pattern completion 으로 이전 해석 recall — "이전에 비슷한 충돌: [...]" CLI 출력. |
| **Function/UX Effect** | 두 번째 비슷한 충돌부터는 LLM 호출 없이 즉시 이전 통찰 노출. quality_hint 메타로 저품질 응답은 자동 강등. |
| **Core Value** | sub-5 (Memory) × sub-conflict-interpretation × Bridge 의 곱. "충돌을 발견하고, 해석하고, **기억하고, 재활용**하는" 완전 루프 — 창의성의 라이브러리 완성. |

---

## Context Anchor

| 키 | 값 |
|----|----|
| **WHY** | 양적 검증 50건 결과: 통찰 100% 응답 + 1/3 고품질, 그러나 매 충돌마다 14s LLM 호출은 낭비. 누적 자산화 + recall 필요. |
| **WHO** | HTP 사용자 — 비슷한 충돌이 반복되면 *이미 한 해석* 즉시 보고 싶음. LLM 호출 minimize. |
| **RISK** | (1) Memory L2 schema 변경 회귀. (2) recall mismatch 임계값 부적절. (3) quality_hint heuristic 과맞춤. (4) 14s latency 회피가 사용자 가치라 검증 필요. |
| **SUCCESS** | (1) 회귀 283 보존 (2) escalate=True 시 Episode 자동 저장 (3) recall 시 mismatch 작은 이전 interpretation 노출 (4) quality_hint 로 저품질 자동 강등 (5) 실 시연 — 2회차 충돌이 1회차 해석 recall 함 |
| **SCOPE** | Episode dataclass + SQL schema 확장 / KnowledgeLoop 통합 (MemorySystem DI) / recall CLI 출력 / quality_hint heuristic / 회귀 테스트 + 실 시연. **OUT**: prompt 개선 (별도 cycle), latency 자체 감소 (별도 cycle). |

---

## 1. 양적 검증 결과 반영 (5 함의)

| 함의 | Plan 반영 |
|------|----------|
| LLM 응답률 100% | 모든 escalate=True interpretation 저장 (사전 필터 안 함) |
| 1/3 고품질, 1/3 저품질 | `quality_hint` 메타 도입 — recall 시 best-match |
| conflict 값 quality 무관 | conflict 값을 confidence 필터로 사용 안 함 |
| JSON 파싱 불완전 | 현재 그대로 (text 필드 fallback). prompt 개선은 별도 cycle |
| 언어 혼용 | recall 시 그대로 노출 (그 자체로 가치). prompt 개선 별도 |

---

## 2. Stage 분할

### Stage 1 — Episode dataclass + SQL schema 확장

`htp/memory/types.py`:
- `Episode.interpretation_text: str = ""` 필드 추가

`htp/memory/episode_store.py`:
- SCHEMA 에 `interpretation_text TEXT DEFAULT ""` 추가
- `save()` 에 interpretation_text 반영
- `_row_to_episode` 에서 새 컬럼 매핑
- 기존 DB 마이그레이션 — `ALTER TABLE ADD COLUMN` (idempotent)

### Stage 2 — quality_hint heuristic

`htp/memory/quality_hint.py` 신규 (혹은 types.py):
```python
QUALITY_KEYWORDS = ["mechanism", "axis", "dimension", "scope", "layer",
                    "complementarity", "framing", "trade-off",
                    "메커니즘", "차원", "관점", "보완"]

def quality_hint(interp: str) -> float:
    if not interp: return 0.0
    text = interp.lower()
    hits = sum(1 for kw in QUALITY_KEYWORDS if kw in text)
    return min(1.0, hits / 3)
```

### Stage 3 — MemorySystem wrapper

`htp/memory/memory_system.py`:
- `save_conflict(interp_vec, new_text, partner_texts, interpretation, conflict, quality_hint)` 신규
- `recall_conflict(query_vec, top_k=3)` 신규 — interpretation_text 보유 Episode 만

### Stage 4 — KnowledgeLoop 통합

`htp/knowledge/loop.py`:
- `__init__(memory: MemorySystem | None = None)` 인자 추가
- None 이면 자동 `MemorySystem(memory_dir=".htp")`
- `_maybe_interpret_conflict` 후:
  - interpretation 이 None 아니면 `memory.save_conflict(...)`
- ingest 시작 시 recall 시도:
  - `memory.recall_conflict(vec_of_new_text)` 호출
  - top-1 mismatch 작으면 `recall_hint` 필드 채움

### Stage 5 — CLI 출력 + IngestResult 확장

```python
@dataclass
class IngestResult:
    ...
    coherence_info: dict | None = None
    recall_hint:    dict | None = None    # 신규
    #   {prev_interpretation: str, prev_trigger: str, mismatch: float, quality: float}
```

CLI ingest 출력:
```
$ ingest "..."
✓ saved (id=..., source=...)
  ⚠ 충돌 감지 (coherence=..., conflict=...)
  📚 이전에 비슷한 충돌이 있었습니다 (mismatch=0.12, quality=0.67):
     "주의 메커니즘은 국소적..." → "scope × temporal 2-axis 분해"
  💡 새 해석 (LLM): ...
```

### Stage 6 — 테스트 + 실 시연

- 단위: Episode 확장 / quality_hint / save_conflict / recall_conflict
- 통합: 2회 비슷한 충돌 ingest → 2회차에 recall_hint 채워짐
- 실 시연: ClaudeCliNode 로 4 도메인 ingest → 같은 도메인 충돌 재유발

---

## 3. Module Map

| Module | 위치 | 변경 | 줄수 |
|--------|------|------|-----:|
| M1 Episode.interpretation_text | `htp/memory/types.py` | +1 필드 | +3 |
| M2 SQL schema 확장 | `htp/memory/episode_store.py` | ALTER + save/load | +20 |
| M3 quality_hint | `htp/memory/quality_hint.py` 신규 | heuristic 함수 | +30 |
| M4 MemorySystem.save_conflict/recall_conflict | `htp/memory/memory_system.py` | wrapper | +60 |
| M5 KnowledgeLoop 통합 | `htp/knowledge/loop.py` | memory DI + recall 흐름 | +50 |
| M6 IngestResult.recall_hint | `htp/knowledge/loop.py` | +1 필드 | +3 |
| M7 CLI 출력 | `htp/knowledge/cli/ingest.py` | 📚 라인 | +5 |
| M8 테스트 | `tests/knowledge/test_conflict_memory.py` 신규 | 8-10 tests | +200 |

**총 소스 ~ +170줄, 테스트 ~ +200줄**.

---

## 4. Architecture 옵션 (Design 단계 확인 필요)

### Option A — Minimal DI
`KnowledgeLoop(memory: MemorySystem | None = None)`. None 이면 기능 off
(저장도 recall 도 안 함). 안전, backward-compat 강함.

### Option B — Auto-create default (Recommended, sub-conflict-interp 와 일관)
None 이면 자동으로 `MemorySystem(memory_dir=self.store.path.parent)` 생성.
CLI default 도 자동 활성. 사용자가 명시 `KnowledgeLoop(memory=None)` 안 하는 한
저장 + recall 모두 활성.

### Option C — Strategy Protocol
`ConflictMemory(Protocol)` + `EpisodeMemoryAdapter` 구현. 향후 다른 backend
(예: in-memory, redis) 가능. 단일 구현체만 있으면 과잉.

---

## 5. Success Criteria

| # | 기준 | 검증 |
|---|------|------|
| SC1 | 회귀 283 보존 | pytest 전체 |
| SC2 | escalate=True 시 Episode 저장 | 통합 테스트 |
| SC3 | 2회차 비슷한 충돌 시 recall_hint 채워짐 | 시연 + 테스트 |
| SC4 | quality_hint heuristic 동작 | 단위 |
| SC5 | Episode schema 마이그레이션 idempotent | 기존 DB 재실행 |
| SC6 | 실 시연 — ClaudeCliNode 로 2회 충돌 → recall 됨 | 사용자 명시 |

---

## 6. Risk + Mitigation

| Risk | 가능성 | 영향 | 완화 |
|------|:------:|:----:|------|
| 기존 SQLite DB 깨짐 | 중 | 높음 | ALTER TABLE IF NOT EXISTS COLUMN idempotent. 첫 schema check 시 컬럼 추가 |
| recall mismatch 임계값 부적절 | 중 | 중 | MemorySystem.CA1_MISMATCH_THRESHOLD (0.3) 재사용. recall_conflict 는 자체 임계 추가 가능 |
| quality_hint heuristic 과맞춤 | 낮음 | 낮음 | 키워드 list 보수적. 절반 이하면 0 — 노출 안 함 옵션 |
| Memory 자동 생성으로 디스크 사용 | 낮음 | 낮음 | tmp dir 명시 (CLI flag), 기본 `.htp/memory.db` 이미 사용 중 |

---

## 7. 진입 후 Design 단계 결정 사항

1. Architecture 옵션 A/B/C
2. recall mismatch threshold (CA1 기본 0.3 재사용 vs 별도)
3. `save_conflict` 의 winner 필드 값 — `"conflict_interpreter"` fixed? source 별?
4. CLI 출력에서 recall_hint 와 새 LLM 호출의 순서 (recall 우선 노출, 그 후 새 호출 결과)
