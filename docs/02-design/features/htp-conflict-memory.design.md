---
template: design
feature: htp-conflict-memory
date: 2026-05-19
plan_ref: docs/01-plan/features/htp-conflict-memory.plan.md
status: Confirmed (Architecture B + recall 우선 노출 + winner fixed)
---

# Design — htp-conflict-memory

**Architecture**: B Auto-create default
**Recall 순서**: recall 우선 → 그 후 새 LLM 호출 (둘 다 노출)
**Winner**: fixed `"conflict_interpreter"`

---

## 1. 데이터 흐름

```
ingest(text, source)
  │
  ├─ encoder.encode(text) → vec
  ├─ neighbors = _find_neighbors(vec, top_k=5)
  ├─ coherence_info = _evaluate_coherence(...)
  │
  ├─ [신규] recall_hint = _try_recall_conflict(vec)
  │   if memory and coherence_info["escalate"]:
  │      ctx = memory.recall_conflict(vec, top_k=3)
  │      if ctx.candidates and mismatch < THRESHOLD:
  │         recall_hint = {prev_interpretation, mismatch, quality}
  │
  ├─ interpretation = _maybe_interpret_conflict(...)
  │   (기존 — recall 과 무관하게 LLM 호출 시도)
  │
  ├─ [신규] _save_conflict_episode(vec, text, neighbors, interpretation)
  │   if memory and interpretation:
  │      interp_vec = encoder.encode(interpretation)
  │      memory.save_conflict(interp_vec, text, partner_texts,
  │                            interpretation, conflict_value, quality_hint)
  │
  └─ KnowledgeEntry(..., interpretation=interpretation)
     IngestResult(..., coherence_info, recall_hint)


CLI 출력:
  ✓ saved (id=..., source=...)
  ⚠ 충돌 감지 (coherence=0.88, conflict=0.15)
  📚 이전에 비슷한 충돌 (mismatch=0.12, quality=0.67):
     trigger: "주의 메커니즘은 국소적..."
     해석: "scope × temporal 2-axis 분해 ..."
  💡 새 해석: "이번 충돌은 ..."
```

---

## 2. Module 구현 상세

### M1: Episode.interpretation_text

```python
# htp/memory/types.py
@dataclass
class Episode:
    # ... 기존 13개 필드
    interpretation_text: str = ""    # htp-conflict-memory 신규
```

### M2: SQL schema 확장

```python
# htp/memory/episode_store.py
SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes (
    ... 기존 컬럼 ...,
    interpretation_text TEXT DEFAULT ''
);
"""

# __init__ 에 idempotent ALTER 추가 (기존 DB 호환)
def _ensure_schema(self):
    self._conn.executescript(SCHEMA)
    # 기존 테이블이 있으면 컬럼 추가 (PRAGMA 로 컬럼 존재 확인)
    cols = [r[1] for r in self._conn.execute("PRAGMA table_info(episodes)")]
    if "interpretation_text" not in cols:
        self._conn.execute(
            "ALTER TABLE episodes ADD COLUMN interpretation_text TEXT DEFAULT ''"
        )

def save(self, ep):
    # INSERT 에 interpretation_text 포함
    self._conn.execute(
        "INSERT INTO episodes (... 14 cols ..., interpretation_text) "
        "VALUES (..., ?)",
        (..., ep.interpretation_text),
    )

def _row_to_episode(self, row):
    # row[N] 에서 interpretation_text 매핑
```

### M3: quality_hint

```python
# htp/memory/quality_hint.py
QUALITY_KEYWORDS = [
    "mechanism", "axis", "dimension", "scope", "layer",
    "complementarity", "framing", "trade-off", "two-axis",
    "메커니즘", "차원", "관점", "보완", "축",
]

def quality_hint(interp: str) -> float:
    """Structural keyword count 기반 통찰 품질 추정 (0.0-1.0).

    양적 검증 결과: 고품질 응답에 다수 등장, 저품질엔 적음.
    완벽한 metric 아니나 best-match 정렬 기준으로 충분.
    """
    if not interp:
        return 0.0
    text = interp.lower()
    hits = sum(1 for kw in QUALITY_KEYWORDS if kw.lower() in text)
    return min(1.0, hits / 3.0)
```

### M4: MemorySystem 신규 메서드

```python
# htp/memory/memory_system.py
RECALL_CONFLICT_MISMATCH_THRESHOLD = 0.3   # CA1 기본값 재사용

def save_conflict(self,
                  interpretation_vec: torch.Tensor,
                  new_text:           str,
                  partner_texts:      list[str],
                  interpretation:     str,
                  conflict_score:     float,
                  ) -> str:
    """Conflict interpretation 을 Episode 로 저장.

    state_vec = interpretation 의 임베딩 (recall key).
    context   = new_text[:50] + partners 요약.
    score     = conflict_score (recall priority 가중용).
    winner    = fixed "conflict_interpreter".
    interpretation_text = 전체 본문.
    """
    from .quality_hint import quality_hint
    quality = quality_hint(interpretation)

    # context = trigger + partners 짧은 요약 (50자 cap)
    partner_summary = " / ".join(p[:15] for p in partner_texts[:2])
    context_str = f"{new_text[:25]} ↔ {partner_summary}"[:50]

    ep = Episode(
        step                = 0,
        winner              = "conflict_interpreter",
        action_type         = "interpret",
        score               = float(conflict_score),
        state_vec           = tensor_to_bytes(interpretation_vec),
        context             = context_str,
        novelty             = 1.0,        # 첫 저장 (이후 recall_count 증가)
        session_id          = self.session_id,
        interpretation_text = interpretation,
    )
    # quality_hint 는 별도 in-memory dict (schema 변경 회피, sub-3 의 conflict_by_episode 패턴)
    ep_id = self.l2.save(ep)
    self._quality_by_episode = getattr(self, "_quality_by_episode", {})
    self._quality_by_episode[ep_id] = quality
    return ep_id


def recall_conflict(self,
                    query_vec: torch.Tensor,
                    top_k:     int = 3,
                    ) -> "list[tuple[Episode, float]]":
    """이전 conflict interpretation 들 중 query_vec 과 가장 유사한 N개.

    반환: [(episode, quality_hint), ...] — quality 내림차순.
    EpisodeStore.search_similar 의 winner_filter="conflict_interpreter" 활용.
    """
    candidates = self.l2.search_similar(
        query_vec, top_k=top_k,
        winner_filter="conflict_interpreter",
    )
    if not candidates:
        return []

    qmap = getattr(self, "_quality_by_episode", {})
    scored = [(ep, qmap.get(ep.episode_id, 0.0)) for ep in candidates]
    # quality 내림차순 (동률은 cosine 순서 유지)
    scored.sort(key=lambda x: -x[1])
    return scored
```

### M5: KnowledgeLoop 통합

```python
# htp/knowledge/loop.py
from htp.memory.memory_system import MemorySystem

class KnowledgeLoop:
    def __init__(self, ..., memory: "MemorySystem | None" = None):
        # ... 기존 ...
        if memory is None:
            mem_dir = self.store.path.parent / "memory"
            memory = MemorySystem(memory_dir=mem_dir)
        self.memory = memory

    def _try_recall_conflict(self, query_vec):
        """이전 비슷한 충돌 해석 검색. recall_hint dict 또는 None."""
        if self.memory is None:
            return None
        import torch
        qv = torch.tensor(query_vec, dtype=torch.float32)
        results = self.memory.recall_conflict(qv, top_k=3)
        if not results:
            return None
        # 가장 가까운 (cosine 1st)
        best_ep, best_quality = results[0]
        prev_vec = torch.tensor(
            [float(x) for x in __import__("struct").unpack(
                f"{len(best_ep.state_vec)//4}f", best_ep.state_vec)],
            dtype=torch.float32,
        )
        mismatch = float((qv - prev_vec).norm())
        if mismatch >= self.memory.CA1_MISMATCH_THRESHOLD:
            return None
        return {
            "prev_interpretation": best_ep.interpretation_text,
            "prev_trigger":        best_ep.context,
            "mismatch":            mismatch,
            "quality":             best_quality,
            "episode_id":          best_ep.episode_id,
        }

    def _save_conflict_episode(self, vec, text, neighbors, interpretation):
        """interpretation 을 Episode 로 저장."""
        if self.memory is None or not interpretation:
            return
        import torch
        partner_texts = [self._cache[n.entry_id].text for n in neighbors[:3]]
        interp_vec_np = self.encoder.encode(interpretation)
        interp_vec = torch.tensor(interp_vec_np, dtype=torch.float32)
        self.memory.save_conflict(
            interpretation_vec = interp_vec,
            new_text           = text,
            partner_texts      = partner_texts,
            interpretation     = interpretation,
            conflict_score     = 0.0,   # placeholder; coherence_info 전달
        )

    def ingest(self, text, source=""):
        vec = self.encoder.encode(text)
        # ... 기존 흐름 ...
        coherence_info = self._evaluate_coherence(vec, source, neighbors)

        # 신규 — recall (escalate 시만)
        recall_hint = None
        if coherence_info and coherence_info["escalate"]:
            recall_hint = self._try_recall_conflict(vec)

        # 기존 — LLM 호출
        interpretation = self._maybe_interpret_conflict(...)

        # 신규 — save (성공한 경우)
        if interpretation:
            self._save_conflict_episode(vec, text, neighbors, interpretation)

        entry = KnowledgeEntry(...)
        return IngestResult(..., coherence_info=coherence_info,
                            recall_hint=recall_hint)
```

### M6: IngestResult.recall_hint

```python
@dataclass
class IngestResult:
    ...
    coherence_info: dict | None = None
    recall_hint:    dict | None = None    # 신규
```

### M7: CLI 출력

```python
# htp/knowledge/cli/ingest.py
if ci is not None and ci["escalate"]:
    print(f"  ⚠ 충돌 감지 ...")
    # recall 우선 노출
    if result.recall_hint:
        rh = result.recall_hint
        print(f"  📚 이전 유사 충돌 (mismatch={rh['mismatch']:.2f}, "
              f"quality={rh['quality']:.2f}):")
        print(f"     trigger: {rh['prev_trigger']}")
        print(f"     해석: {rh['prev_interpretation'][:180]}...")
    # 새 LLM 해석
    if result.entry.interpretation:
        print(f"  💡 새 해석: {result.entry.interpretation[:180]}...")
```

### M8: 테스트

```python
# tests/knowledge/test_conflict_memory.py
def test_episode_interpretation_text_field()
def test_episode_store_schema_migration_idempotent()   # 기존 DB ALTER OK
def test_quality_hint_keywords()
def test_quality_hint_empty_string()
def test_save_conflict_creates_episode_with_winner()
def test_recall_conflict_returns_quality_sorted()
def test_recall_conflict_empty_when_no_episodes()
def test_knowledge_loop_default_creates_memory()       # Architecture B
def test_knowledge_loop_save_on_escalate_with_interp()
def test_knowledge_loop_recall_hint_on_second_conflict()
```

---

## 3. DAG

```
htp/knowledge/loop.py ──→ htp/memory/memory_system   (신규)
                       ──→ htp/llm/llm_region        (기존)
htp/memory/memory_system.py ──→ htp/memory/quality_hint (신규)
htp/memory/quality_hint.py ──→ (외부 의존 없음)
```

기존 DAG 룰 영향: `knowledge → memory` 는 기존에 *금지* 였음 (test_no_circular_deps
의 forbidden = ("htp.runtime", "htp.memory") — knowledge 에서). **rule 갱신 필요**.

```python
# tests/unit/test_no_circular_deps.py
# Before: forbidden = ("htp.runtime", "htp.memory")
# After:  forbidden = ("htp.runtime",)   # memory 는 단방향 허용 (Bridge §6 와 동일 패턴)
```

memory 도 thalamus 처럼 *역방향만* 금지하면 됨. 신규 룰:
- `htp/memory/* → htp/knowledge/*` **영구 금지** (test 추가)

---

## 4. SC 매핑

| SC (Plan) | 구현 | 검증 |
|-----------|------|------|
| SC1 회귀 보존 | 모든 변경 backward-compat | pytest 전체 |
| SC2 escalate 시 Episode 저장 | `_save_conflict_episode` | `test_knowledge_loop_save_on_escalate_with_interp` |
| SC3 recall_hint 채워짐 | `_try_recall_conflict` | `test_knowledge_loop_recall_hint_on_second_conflict` |
| SC4 quality_hint heuristic | `quality_hint.py` | `test_quality_hint_keywords` |
| SC5 schema 마이그레이션 | `_ensure_schema` PRAGMA + ALTER | `test_episode_store_schema_migration_idempotent` |
| SC6 실 시연 | ClaudeCliNode + 2회 충돌 | 사용자 명시 |

---

## 5. Risk + Mitigation 재확인

| Risk | Mitigation |
|------|-----------|
| 기존 SQLite DB 깨짐 | `PRAGMA table_info` 로 컬럼 존재 확인 후 ALTER. Idempotent |
| recall mismatch 부적절 | CA1_MISMATCH_THRESHOLD (0.3) 재사용. 후속 cycle 튜닝 가능 |
| quality_hint 과맞춤 | 키워드 list 보수적 (양적 검증의 고품질 사례에서만 추출) |
| Memory 자동 생성 | `store.path.parent / "memory"` 명시 — 사용자 디렉토리 명시 가능 |
| 회귀 — knowledge → memory 신규 import | DAG 룰 갱신 + `test_memory_does_not_import_knowledge` 추가 |

---

## 6. Implementation Guide

### Session A — Memory 확장 (~45분)
1. M1 Episode.interpretation_text 필드
2. M2 SQL schema + idempotent ALTER + save/load
3. M3 quality_hint heuristic
4. M4 MemorySystem.save_conflict + recall_conflict

### Session B — KnowledgeLoop 통합 + CLI (~45분)
5. M5 KnowledgeLoop 의 memory DI + _try_recall_conflict + _save_conflict_episode
6. M6 IngestResult.recall_hint
7. M7 CLI ingest 출력 (📚 / 💡 2줄)
8. DAG 룰 갱신 (test_no_circular_deps)

### Session C — 테스트 + 실 시연 (~30분)
9. M8 단위/통합 테스트
10. 회귀 전체 통과
11. 실 시연 — ClaudeCliNode 로 2회 비슷한 충돌 → 2회차에 recall

총 ~2h.
