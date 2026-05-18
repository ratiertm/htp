---
template: design
feature: htp-conflict-interpretation
date: 2026-05-19
author: Mindbuild
project: HTP
plan_ref: docs/01-plan/features/htp-conflict-interpretation.plan.md
status: Confirmed (Architecture B + 별도 prompt + Migration 스크립트)
---

# Design — htp-conflict-interpretation

**Architecture**: B Auto Mock default
**Prompt 위치**: `htp/knowledge/conflict_prompt.py` (별도 파일)
**JSONL backward-compat**: Migration 스크립트 (`htp.knowledge.migrate --add-interpretation`)

---

## Context Anchor (Plan 인용)

| 키 | 값 |
|----|----|
| WHY | sub-3/4/5/Bridge 분리. CoherenceGate 가 충돌 감지만 하고 *왜* 인지 침묵. |
| WHO | HTP 사용자 — 충돌의 의미를 즉시 알고 싶음 |
| RISK | API 비용 / Mock trivial / escalate 폭증 / prompt 품질 / KnowledgeEntry 회귀 |
| SUCCESS | 6 SC — 회귀 보존 / escalate 동작 / Mock default / 영속화 / cap / 실데이터 1건 |

---

## 1. 데이터 흐름

```
ingest(text, source)
  │
  ├─ encoder.encode(text) → vec
  ├─ _update_signature(source, vec)
  ├─ neighbors = _find_neighbors(vec, top_k=5)
  ├─ coherence_info = _evaluate_coherence(vec, source, neighbors)
  │
  ├─ [신규] _maybe_interpret_conflict(text, source, neighbors, coherence_info)
  │   if coherence_info and coherence_info["escalate"] and self._can_interpret():
  │      prompt = build_conflict_prompt(new=(text,source), existing=[(n.text, n.source) ...])
  │      result = self.conflict_interpreter.run(prompt)   ← LLMRegion (Mock default)
  │      interpretation = result.get("interpretation") or result.get("text")
  │      self._interpretations_count += 1
  │   else:
  │      interpretation = None
  │
  └─ KnowledgeEntry(..., interpretation=interpretation)
     self._cache.append + self.store.append
```

### CLI 출력

```
$ htp.knowledge ingest "주의는 국소적이다" --source 뇌과학
✓ saved (id=abc12345, source=뇌과학)
  ⚠ 충돌 감지 (coherence=0.88, conflict=0.153)
     → 기존 지식과 모순될 수 있음
  💡 해석: AI 의 "Transformer attention 은 전역적" 과 모순. 그러나 scale 차이일
          수 있음. 두 관점 통합 시 multi-scale attention 가설 가능.
```

---

## 2. Module Map

### M1 KnowledgeEntry.interpretation 필드

`htp/knowledge/types.py`:

```python
@dataclass
class KnowledgeEntry:
    # 기존 필드 ...
    interpretation: "str | None" = None   # 신규 — CoherenceGate escalate 시 LLM 해석
```

### M2 conflict_prompt 별도 파일

`htp/knowledge/conflict_prompt.py` 신규:

```python
"""Conflict interpretation prompt 구성.

Design Ref: docs/02-design/features/htp-conflict-interpretation.design.md §2

KnowledgeLoop 가 escalate=True 시 호출. LLMRegion 이 처리하기 좋은
JSON-return 프롬프트 생성.
"""
from __future__ import annotations


SYSTEM_PROMPT = (
    "You are an analyst integrating cross-domain knowledge. "
    "When given a new statement that conflicts with existing knowledge, "
    "identify the precise nature of the conflict and propose a hypothesis "
    "that integrates both perspectives. "
    "Return JSON with keys 'interpretation' (1-2 sentences explaining the "
    "conflict and integration), 'hypothesis' (one synthesis idea)."
)


def build_conflict_prompt(
    new_text:      str,
    new_source:    str,
    existing:      "list[tuple[str, str]]",   # [(text, source), ...]
    coherence:     float,
    conflict:      float,
) -> str:
    """Conflict interpretation prompt 생성.

    LLMRegion 의 `run(data)` 가 받는 string 또는 dict — 여기서는 string 반환.
    """
    lines = [
        f"Conflict detected (coherence={coherence:.2f}, conflict={conflict:.2f}).",
        "",
        f"New statement ({new_source}):",
        f"  {new_text}",
        "",
        "Existing related knowledge:",
    ]
    for i, (text, source) in enumerate(existing, 1):
        lines.append(f"  {i}. [{source}] {text}")
    lines.append("")
    lines.append(
        "Explain the conflict precisely and propose an integration "
        "hypothesis. Return JSON."
    )
    return "\n".join(lines)


__all__ = ["SYSTEM_PROMPT", "build_conflict_prompt"]
```

### M3 KnowledgeLoop 통합

`htp/knowledge/loop.py`:

```python
from htp.llm.llm_region import LLMRegion
from .conflict_prompt    import build_conflict_prompt, SYSTEM_PROMPT


class KnowledgeLoop:
    def __init__(self,
                 encoder,
                 store=None,
                 ...,
                 conflict_interpreter: "LLMRegion | None" = None,
                 max_interpretations:  int = 20,
                 ):
        # ... 기존 ...

        # 신규: Architecture B — None 이면 자동 Mock 생성
        if conflict_interpreter is None:
            conflict_interpreter = LLMRegion(
                region_name="conflict_interpreter",
                specialty="reasoning",
                system=SYSTEM_PROMPT,
                use_mock=True,    # 안전 default
            )
        self.conflict_interpreter   = conflict_interpreter
        self.max_interpretations    = max_interpretations
        self._interpretations_count = 0

    def _can_interpret(self) -> bool:
        """cap + CostRouter.should_block 합쳐 호출 여부 판단."""
        if self._interpretations_count >= self.max_interpretations:
            return False
        router = getattr(self.conflict_interpreter, "router", None)
        if router is not None and router.should_block():
            return False
        return True

    def _maybe_interpret_conflict(
        self, text, source, neighbors, coherence_info,
    ) -> "str | None":
        if not coherence_info or not coherence_info.get("escalate"):
            return None
        if not self._can_interpret():
            return None

        existing = [
            (self._cache[n.entry_id].text, self._cache[n.entry_id].source)
            for n in neighbors[:3]
        ]
        prompt = build_conflict_prompt(
            new_text   = text,
            new_source = source,
            existing   = existing,
            coherence  = coherence_info["coherence"],
            conflict   = coherence_info["conflict"],
        )
        try:
            result = self.conflict_interpreter.run(prompt)
        except Exception as e:
            return f"(interpretation failed: {e})"

        self._interpretations_count += 1

        if isinstance(result, dict):
            return (
                result.get("interpretation")
                or result.get("text")
                or str(result)
            )
        return str(result)

    def ingest(self, text, source=""):
        # ... 기존 vec / signature / neighbors / coherence_info ...

        # 신규 분기 — escalate=True 시 해석
        interpretation = self._maybe_interpret_conflict(
            text, source, neighbors, coherence_info,
        )

        entry = KnowledgeEntry(
            text=text, vec=vec, source=source,
            timestamp=..., neighbors=..., conflict_count=...,
            interpretation=interpretation,   # 신규
        )
        # ...
```

### M4 KnowledgeStore JSONL round-trip

`htp/knowledge/persistence.py`:

```python
# append() 에 추가
"interpretation": getattr(entry, "interpretation", None),

# load_all 의 entry 생성에 추가
interpretation = rec.get("interpretation"),
```

### M5 CLI ingest 출력

`htp/knowledge/cli/ingest.py`:

```python
if ci is not None and ci["escalate"]:
    print(f"  ⚠ 충돌 감지 ...")
    if result.entry.interpretation:
        print(f"  💡 해석: {result.entry.interpretation}")
```

### M6 CLI list 마크

`htp/knowledge/cli/list_cmd.py`:

```python
marker = "💡 " if entry.interpretation else "   "
print(f"{marker}[{entry.id[:8]}] ({entry.source}) {preview}")
```

### M7 Migration 스크립트

`htp/knowledge/migrate.py`:

```python
def migrate_add_interpretation(path: Path) -> dict:
    """기존 jsonl 의 entry 에 interpretation=null 필드 명시.

    backward-compat 는 load 시 .get() 으로 이미 지원되지만, 명시적 추가가
    docs / 외부 도구 호환성에 안전.
    """
    # 백업 + interpretation 필드 없는 entries 에 null 추가
    # ...
    return {"migrated": N, "backup_path": backup_path}
```

CLI: `htp.knowledge migrate --add-interpretation`.

### M8 테스트

`tests/knowledge/test_conflict_interpretation.py`:

```python
def test_default_creates_mock_interpreter():
    """Architecture B — None 이면 자동 MockLLMRegion."""

def test_escalate_true_triggers_interpretation():
    """escalate=True 시 _interpret_conflict 호출 + entry.interpretation 채움."""

def test_escalate_false_skips_interpretation():
    """escalate=False 시 interpretation=None."""

def test_max_interpretations_cap():
    """max_interpretations=2 → 3번째 escalate 부터 interpretation=None."""

def test_jsonl_round_trip_with_interpretation():
    """ingest → save → load → interpretation 보존."""

def test_jsonl_legacy_entries_load_with_none():
    """interpretation 필드 없는 기존 jsonl 도 로드 가능 (None default)."""

def test_cli_ingest_prints_interpretation():
    """CLI 출력에 💡 라인 포함."""

def test_user_provided_interpreter_used():
    """사용자가 명시 LLMRegion 넘기면 그것 사용 (Mock 자동 생성 안 함)."""
```

---

## 3. Architecture B 흐름 정리

```
KnowledgeLoop(encoder=..., conflict_interpreter=None)   ← 무지정
   └→ __init__ 에서 자동 MockLLMRegion("conflict_interpreter", reasoning, use_mock=True)

KnowledgeLoop(encoder=..., conflict_interpreter=my_llm) ← 명시 (real API)
   └→ my_llm 그대로 사용 (Mock 생성 안 함)

KnowledgeLoop(encoder=..., conflict_interpreter=None,
              max_interpretations=0)                      ← 기능 사실상 off
   └→ _can_interpret() → False → 항상 skip
```

---

## 4. DAG

```
htp/knowledge/loop.py ──→ htp/llm/llm_region   (신규)
                       ──→ htp/knowledge/conflict_prompt
htp/knowledge/conflict_prompt.py ──→ (외부 의존 없음)
```

기존 DAG 룰 (`test_no_circular_deps.py`) 영향 없음 — `knowledge → llm` 은
이미 허용 (thalamus 와 동급). 신규 룰 불필요.

---

## 5. Risk + Mitigation 재확인

| Risk | Mitigation 적용 |
|------|----------------|
| API 비용 사고 | Mock default + max_interpretations=20 + CostRouter.should_block |
| Mock trivial | end-to-end 흐름 검증만 목적 — 실 API 1회 수동 검증으로 품질 평가 |
| escalate 폭증 | max_interpretations cap. CLI `--no-interpretation` (Design 시 추가) |
| KnowledgeEntry 회귀 | interpretation 필드 default=None. 기존 jsonl `.get()` fallback |

---

## 6. Implementation Guide

### Session A — Core 통합 (~1.5h)
1. M1 KnowledgeEntry.interpretation 필드 추가
2. M2 `conflict_prompt.py` 신규
3. M3 KnowledgeLoop.__init__ + _maybe_interpret_conflict + ingest 분기
4. M4 persistence round-trip (append + load_all)

### Session B — CLI + Migration (~30분)
5. M5 CLI ingest 출력
6. M6 CLI list 마크
7. M7 migration 스크립트 + CLI

### Session C — 테스트 + 검증 (~30분)
8. M8 단위 테스트 6-8건
9. 회귀 전체 통과 확인
10. (수동) 실 API 1회 시연

총 ~2.5h.

---

## 7. Success Criteria (Plan 인용)

| # | 기준 | 검증 방법 |
|---|------|----------|
| SC1 | 회귀 258 보존 | pytest 전체 |
| SC2 | escalate=True 시 interpretation 생성 | M8 단위 + 통합 |
| SC3 | Mock default 작동 | CLI 무인자 실행 |
| SC4 | JSONL 영속화 | round-trip 테스트 |
| SC5 | cap 동작 | max_interpretations 테스트 |
| SC6 | 실데이터 해석 1건 (수동) | 사용자 시연 |
