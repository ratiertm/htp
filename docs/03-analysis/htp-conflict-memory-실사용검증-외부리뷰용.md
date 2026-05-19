# HTP htp-conflict-memory — 실사용 검증 결과 (외부 리뷰용)

**작성일**: 2026-05-20
**대상**: 외부 LLM 리뷰 / 합의 검증
**범위**: htp-conflict-memory — interpretation 을 Episode 로 저장 + CA3 recall
**커밋**: `4cc348a` master

---

## 0. 한 줄 요약

> **"창의성의 라이브러리" 정의 완성** — 충돌 발견 (sub-3) → 해석 (sub-conflict-interp)
> → 기억 (이번 cycle) → 재활용. 같은 *eviction* 주제 2회 충돌 시연에서 2회차에
> 1회차 해석 즉시 노출 + 새 정밀 해석 추가. **6/6 SC strict PASS** (Match Rate 100%).

---

## 1. 배경 — 왜 이 cycle 인가

htp-conflict-interpretation 까지 완성된 흐름:

```
사용자 ingest "주의 메커니즘은 국소적이다" --source 뇌과학
  ↓
escalate=True (conflict=0.17)
  ↓
LLMRegion 호출 (claude -p, ~12-14s)
  ↓
💡 "Transformer self-attention 의 전역 가중 ... 충돌은 scope (local vs global) ×
    temporal (serial vs parallel)"
  ↓
KnowledgeEntry.interpretation 에 저장. **그러나 entry 내부 필드일 뿐 — 다음
충돌엔 무관**.
```

문제: 같은 종류의 *eviction* / *attention* 충돌이 반복되면 *매번 LLM 새 호출* —
14s 대기 + 같은 통찰 반복 생성 낭비.

해결 가설:

> interpretation 을 Episode 로 저장 (state_vec = trigger 의 text 임베딩) →
> 다음 비슷한 충돌 발생 시 vec 유사도로 이전 Episode 검색 → 이전 해석 즉시 표시
> + 새 LLM 해석은 별도 추가 노출.

이렇게 하면 interpretation 이 일회용 출력에서 **누적 자산** 으로 전환.

---

## 2. 진입 결정 (사용자 + 양적 검증 50건 데이터 기반)

| Decision | 결정 | 근거 |
|----------|------|------|
| Episode 저장 단위 | Episode 확장 (interpretation_text 필드) | 별도 store 보다 단일 store 일관성 |
| quality_hint 도입 | 도입 — recall best-match 필터 | 양적 검증: 통찰 품질 고/중/저 1/3씩 — 노이즈 33% 필터 필요 |
| Architecture | B Auto-create default | sub-conflict-interp 와 동일 패턴, onboarding 마찰 0 |
| recall 노출 순서 | recall 우선 → 그 후 새 LLM 호출 | 사용자 "이전엔 이랬다" + "이번엔 더 정밀" UX 공명 |
| winner 필드 값 | fixed "conflict_interpreter" | search_similar 의 winner_filter 단순 |

---

## 3. 양적 검증 (50건 cross-domain ingest)

`scripts/conflict_quant_eval.py` — 4 도메인 × 7 baseline + 50건 이질 ingest.

### 3-1. 통계

| 지표 | 값 |
|------|----|
| 총 ingest | 50 |
| escalate=True | **24 (48%)** |
| interpretation 생성 | **24/24 (100% 응답률)** |
| avg conflict | 0.123 |
| avg interp latency | 14,227 ms (~14s/call) |
| total elapsed | 516.5s (~8.6분) |

### 3-2. 통찰 품질 정성 분류

| 등급 | 케이스 | 비율 | 예 |
|------|:------:|:----:|----|
| 고 (mechanism mapping / 차원 분해) | 7 | 29% | "eviction ↔ pruning ↔ lateral inhibition" / "scope × temporal 2-axis" |
| 중 (layer/scope 보완) | 9 | 37% | "application layer vs CDN edge" / "strong vs eventual consistency" |
| 저 (서술적 / overlap 인정) | 8 | 33% | "substantively overlaps" / 단순 정의 설명 |

### 3-3. 핵심 발견

- conflict 값과 quality 무관 (고품질 avg 0.131 vs 저품질 avg 0.140)
- → **conflict 값으로 사전 confidence 필터 불가**
- → **모두 저장하되 recall 시 quality_hint best-match** 가 합리적

---

## 4. 구현

### 4-1. Episode 확장

```python
# htp/memory/types.py
@dataclass
class Episode:
    # ... 기존 13개 필드
    interpretation_text: str = ""    # 신규 — LLM 해석 전체 본문
```

### 4-2. SQL schema idempotent migration

```python
# htp/memory/episode_store.py
SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes (
    ... 기존 13개 컬럼 ...,
    interpretation_text TEXT DEFAULT ''
);
"""

def __init__(self, db_path):
    # ...
    cols = [r[1] for r in self._conn.execute("PRAGMA table_info(episodes)")]
    if "interpretation_text" not in cols:
        self._conn.execute(
            "ALTER TABLE episodes ADD COLUMN interpretation_text TEXT DEFAULT ''"
        )
```

→ 기존 DB 도 자동 마이그레이션. 첫 open 시 컬럼 존재 확인 후 ALTER.

### 4-3. quality_hint heuristic

```python
# htp/memory/quality_hint.py
QUALITY_KEYWORDS = [
    "mechanism", "axis", "dimension", "scope", "layer",
    "complementarity", "framing", "trade-off", ...,
    "메커니즘", "차원", "관점", "보완", "축", ...,   # 한국어
]   # 총 19개

def quality_hint(interpretation: str) -> float:
    """0.0 (키워드 0) → 1.0 (3개 이상). recall best-match 정렬용."""
    text = interpretation.lower()
    hits = sum(1 for kw in QUALITY_KEYWORDS if kw.lower() in text)
    return min(1.0, hits / 3.0)
```

### 4-4. MemorySystem.save_conflict / recall_conflict

```python
# htp/memory/memory_system.py
CONFLICT_RECALL_MISMATCH_THRESHOLD = 0.6   # 384-dim e5 기준 (cosine ~0.82)

def save_conflict(self, trigger_vec, new_text, partner_texts,
                  interpretation, conflict_score=0.0) -> str:
    """state_vec = trigger_vec (text 임베딩 — recall key)."""
    ep = Episode(
        winner="conflict_interpreter",
        action_type="interpret",
        score=float(conflict_score),
        state_vec=tensor_to_bytes(trigger_vec),
        context=f"{new_text[:25]} ↔ {partners_summary}"[:50],
        interpretation_text=interpretation,
    )
    ep_id = self.l2.save(ep)
    self._quality_by_episode[ep_id] = quality_hint(interpretation)
    return ep_id

def recall_conflict(self, query_vec, top_k=3):
    """winner_filter='conflict_interpreter' 로 검색.
    quality_hint 내림차순 정렬 (동률은 cosine 순서)."""
    candidates = self.l2.search_similar(
        query_vec, top_k=top_k,
        winner_filter="conflict_interpreter",
    )
    return sorted(
        [(ep, self._quality_by_episode.get(ep.episode_id, 0.0))
         for ep in candidates],
        key=lambda x: -x[1],
    )
```

### 4-5. KnowledgeLoop 통합

```python
# htp/knowledge/loop.py
class KnowledgeLoop:
    def __init__(self, ..., memory=None):
        # ...
        # Architecture B: None 이면 자동 MemorySystem
        if memory is None:
            memory = MemorySystem(memory_dir=self.store.path.parent / "memory")
        self.memory = memory

    def ingest(self, text, source=""):
        vec = self.encoder.encode(text)
        # ... coherence_info ...

        # 신규 — escalate 시 recall 먼저
        recall_hint = None
        if coherence_info and coherence_info.get("escalate"):
            recall_hint = self._try_recall_conflict(vec)

        # 기존 — LLM 호출 (skip marker 추가)
        interpretation = self._maybe_interpret_conflict(...)

        # 신규 — Episode 저장 (timeout/error 응답 자동 skip)
        if interpretation:
            self._save_conflict_episode(vec, text, neighbors, interpretation,
                                         conflict_value)

        return IngestResult(..., coherence_info=ci, recall_hint=recall_hint)

    _INTERPRETATION_SKIP_MARKERS = (
        "claude cli timeout", "claude cli rc=",
        "claude CLI not in PATH", "interpretation failed:",
        "cost_blocked",
    )
```

### 4-6. CLI 출력

```python
# htp/knowledge/cli/ingest.py
if ci["escalate"]:
    print(f"  ⚠ 충돌 감지 (...)")
    if result.recall_hint:
        rh = result.recall_hint
        print(f"  📚 이전 유사 충돌 (mismatch={rh['mismatch']:.2f}, "
              f"quality={rh['quality']:.2f}):")
        print(f"     trigger: {rh['prev_trigger']}")
        print(f"     해석: {rh['prev_interpretation'][:180]}...")
    if result.entry.interpretation:
        print(f"  💡 새 해석: {result.entry.interpretation[:180]}...")
```

---

## 5. 시연 결과 (v4 — warm-up + Redis + Kubernetes)

### 5-1. Raw 출력

```
# sandbox: /var/folders/.../htp-conflict-mem-v4-yawx01g4
# warm-up call...
  warm done in 12.6s, label=claude_cli_response

## 1회차 (Redis)
  [14.0s] escalate=True, recall_hint=None
  💡 해석: The conflict is categorical, not factual: the new statement describes
          a deterministic infrastructure cache (Redis LRU eviction, nginx load
          balancing) while exis...

## 2회차 (Kubernetes)
  [16.7s] escalate=True
  📚 RECALL HIT ✓✓✓  mismatch=0.530, quality=0.33
     trigger: Redis LRU 캐시 eviction 전략  ↔ 해마 CA3 패턴 완성 시냅 / 시냅스
     해석:    The conflict is categorical, not factual: the new statement
              describes a deterministic infrastructure cache (Redis LRU
              eviction, nginx load balancing) while existing ...
  💡 새 해석: 두 진술 모두 '인프라' 도메인의 'eviction' 메커니즘을 다루지만, 서로
            다른 추상 계층에서 작동한다 — Redis LRU는 메모리 캐시 항목을 접근
            시간 기반으로 축출하는 데이터 계층 정책인 반면, Kubernetes pod
            eviction은 노드 자원 압박(resource q...
```

### 5-2. 의미

같은 *eviction* 주제 2회 충돌에서:

- **1회차** (Redis): LLM 새 호출 → 정상 해석 → Memory Episode 저장
- **2회차** (Kubernetes): **이전 해석 즉시 노출** + **새 LLM 해석이 더 정밀한 비교 추가**

→ 사용자가 두 관점 *비교 가능*:
- 이전: "categorical, not factual" (도메인 카테고리 차이)
- 새: "Redis LRU 데이터 계층 vs Kubernetes pod 리소스 계층" (추상 계층 차이)

통찰이 일회용 출력에서 **누적 자산** 으로 전환. 두 번째 비슷한 충돌이 *비교 기반* 으로 작동.

### 5-3. 측정 지표

| 단계 | latency | 의미 |
|------|:------:|------|
| warm-up | 12.6s | claude CLI cold-start 흡수 |
| 1회차 ingest | 14.0s | escalate → 새 LLM 호출 → 저장 |
| 2회차 ingest | 16.7s | escalate → recall (즉시) + 새 LLM 호출 |

2회차 latency 가 1회차와 비슷 (16.7 vs 14.0) — recall 자체는 ms 단위, 새 LLM 호출
이 여전히 dominant. **latency 감소가 본질이 아님** — *누적 자산화 + 비교 노출* 이
핵심 가치.

---

## 6. 발견 + Fix (2 bugs — 시연으로만 발견)

### Bug 1: recall key 불일치 (v2 → v3)

**증상**: v2 시연에서 recall_hint=None.

**원인**: `save_conflict` 시 `interpretation` 임베딩 (LLM 응답의 vec) 저장 vs
`_try_recall_conflict` 시 `text` 임베딩 (사용자 ingest input 의 vec) 으로 검색.
**다른 벡터 공간 비교** → cosine 매칭 의미 없음.

**Fix**: `save_conflict` 의 첫 인자 `interpretation_vec` → **`trigger_vec`** (text
임베딩). 데이터 흐름 의미: "비슷한 *input* 이 들어오면 이전 해석 recall".

설계 단계 가정 ("interpretation 자체 임베딩이 recall key") 의 결함 — sub-cycle Plan
§7 에 명시했던 결정이 실제 흐름에서 부적합. **실 시연이 단위 테스트로는 잡히지 않을
설계 오류 발견**.

### Bug 2: threshold 너무 엄격 (v3 → v4)

**증상**: v3 시연에서 recall_hint 여전히 None.

**원인**: `CONFLICT_RECALL_MISMATCH_THRESHOLD = 0.3` 은 CA1 default 값으로 64-dim
sparse vec 기준. 384-dim e5 dense vec 에서:
- 일반 두 텍스트 cosine ~0.85 → L2 거리 ~0.55 → **0.3 미통과**

**Fix**: 0.3 → **0.6** (cosine ~0.82 ↔ "비슷한 도메인" 수준). v4 실측 mismatch=0.530
< 0.6 → HIT.

**근본 문제**: threshold 가 dim 의존적. 후속 cycle 에서 `encoder.dim` 기반 자동
조정 필요.

### Bug 3 부수적: claude CLI cold-start (v3 → v4)

**증상**: 매 시연 첫 호출이 120s timeout.

**관찰**: OAuth refresh 또는 CLI 프로세스 cold-start 환경적 특성. skip_marker 가
이미 작동해서 timeout 응답은 Episode 저장 안 됨 (노이즈 차단). 그러나 시연 자체는
warm-up 필요.

**해결**: v4 에서 `warm-up call` 1회 (`ClaudeCliNode("interp").run("ping")`) 추가.
12.6s 후 본 시연 정상 진행. Production 에서는 별도 cycle 로 자동 warm-up 검토.

---

## 7. 코드 변경 + DAG

### 7-1. 신규 파일 (10)

```
htp/memory/quality_hint.py           19 키워드 heuristic
tests/knowledge/test_conflict_memory.py  15 tests
scripts/conflict_quant_eval.py       양적 검증 50건
docs/01-plan/features/htp-conflict-memory.plan.md
docs/02-design/features/htp-conflict-memory.design.md
docs/03-analysis/htp-conflict-memory.analysis.md
docs/03-analysis/conflict_quant_summary.md
docs/03-analysis/conflict_quant_raw.jsonl
docs/04-report/htp-conflict-memory.report.md
docs/03-analysis/htp-conflict-memory-실사용검증-외부리뷰용.md  (이 문서)
```

### 7-2. 수정 파일 (6)

```
htp/memory/types.py                  Episode.interpretation_text 필드
htp/memory/episode_store.py          SCHEMA + ALTER + 14-col save
htp/memory/memory_system.py          save_conflict/recall_conflict + threshold 0.6
htp/knowledge/loop.py                memory DI + recall + save + SKIP_MARKERS
htp/knowledge/cli/ingest.py          📚 출력
tests/unit/test_no_circular_deps.py  DAG 룰 갱신
```

**소스 순증 ~256줄. 테스트 +250. 깨진 회귀 0건**.

### 7-3. DAG 변경

```
변경 전:
  knowledge → memory  ✗ 금지
  knowledge → thalamus ✓ 허용 (Bridge §6)

변경 후:
  knowledge → memory  ✓ 허용 (단방향, 이 cycle §3)
  knowledge → thalamus ✓ 허용
  knowledge → runtime ✗ 금지 (유일)

  memory → knowledge  ✗ 영구 금지 (역방향, 신규 룰)
  thalamus → knowledge ✗ 영구 금지 (기존)
```

`test_memory_does_not_import_knowledge` 추가 — 영구 보호.

---

## 8. 테스트 catalog

총 15 신규 (회귀 baseline 283 → 303 PASS):

```
tests/knowledge/test_conflict_memory.py (15):

  # M1: Episode.interpretation_text
  test_episode_interpretation_text_field_default_empty
  test_episode_with_interpretation_text

  # M2: SQL schema + 마이그레이션
  test_episode_store_save_and_load_with_interpretation
  test_episode_store_schema_migration_idempotent       ← Bug 2 의 ALTER 검증

  # M3: quality_hint
  test_quality_hint_empty_string
  test_quality_hint_zero_keywords
  test_quality_hint_high_keyword_count
  test_quality_hint_partial_keywords
  test_quality_hint_korean_keywords                    ← 한국어 키워드 5개

  # M4: MemorySystem
  test_save_conflict_creates_episode_with_winner
  test_recall_conflict_returns_sorted_by_quality       ← 양적 검증 결과 반영
  test_recall_conflict_empty_when_no_episodes

  # M5/M6: KnowledgeLoop 통합
  test_knowledge_loop_default_creates_memory           ← Architecture B 검증
  test_knowledge_loop_user_memory_used
  test_ingest_result_has_recall_hint_field

  # (E2E) — HF 의존
  test_recall_hint_on_second_similar_conflict   [skipped on HF unavailable]
```

---

## 9. 외부 리뷰 포커스

### 9-1. 설계 결정

1. **state_vec = trigger_vec (text 임베딩) 선택**. 사용자가 원래 명시한 "interpretation
   도 벡터화" 와 다른 방향. 진정한 의도는 *recall key 가 trigger 여야 한다* 였는지,
   아니면 양쪽 모두 저장해 hybrid recall (예: text 매칭 + interpretation 매칭) 이
   가능해야 하는가?

2. **CONFLICT_RECALL_MISMATCH_THRESHOLD = 0.6 hard-coded**. encoder dim 변경 시
   재조정 필요. `encoder.dim` 기반 자동 조정 함수가 더 robust 한가?
   ```python
   def _threshold_for_dim(dim):
       return 0.6 if dim >= 256 else 0.3   # 단순화 — 실측 표준화 필요
   ```

3. **quality_hint 19 키워드 heuristic**. 양적 검증 50건 기반이라 도메인 편향 위험.
   다른 도메인 (의료, 법률, 예술 등) 에서 키워드 재추출 필요한가? 또는 LLM-as-judge
   로 동적 측정?

4. **recall 우선 노출 → 그 후 새 LLM 호출 = 둘 다 항상 표시** 선택. 사용자가 비교
   가능하나 매 ingest 마다 14s LLM. 옵션: recall mismatch 가 매우 작으면 (예: < 0.3)
   새 LLM 호출 생략? Plan 단계 사용자 선택은 "둘 다 항상" 이었으나, threshold 추가
   가능성?

### 9-2. 검증 한계

5. **시연 1건 (eviction)** 으로 SC3 검증. 다른 도메인 쌍 (attention, learning,
   consistency 등) 에서 recall HIT 율은? 양적 검증은 사전 50건 — recall 측면은
   사후 양적 평가 미실시.

6. **Bug 1, 2 모두 시연으로만 발견**. 단위 테스트가 vec 공간 일치 / threshold
   적합성을 못 잡았음. 통합 테스트 (실 encoder + 실 LLM mock) 추가 필요한가?

7. **claude CLI cold-start** production 영향 미평가. warm-up 자동화 cycle 필요성?

### 9-3. 시스템 통합

8. **knowledge → memory 단방향 추가** 가 architecture 깔끔성에 영향. memory 가
   향후 다른 cycle (예: BrainRuntime 통합) 에서 추가 변경되면 knowledge 가 자동 영향
   받음. trade-off OK?

9. **Episode.winner 의 fixed string "conflict_interpreter"**. 향후 다른 ExternalRegion
   (SearchRegion, RAGRegion) 이 동일 패턴으로 Episode 저장하면 winner namespace
   충돌? (예: "search_recall", "rag_synthesis").

10. **recall hint 와 새 LLM 해석이 비슷한 내용일 때 사용자 인지 부담**. v4 시연에서
    1회차 해석 ("categorical, not factual") 과 2회차 새 해석 ("data layer vs resource
    layer") 이 *다른 관점* 이라 가치 있음. 그러나 항상 그렇진 않음 — 비슷한 해석이면
    노이즈. de-duplication 메커니즘 필요한가?

---

## 10. 후속 작업 (우선순위)

| 우선순위 | 항목 | 소요 |
|:--:|------|:----:|
| 1 | recall 양적 평가 (50건 후 recall HIT 율 측정) | ~1h |
| 2 | claude CLI cold-start 자동 warm-up — ClaudeCliNode 첫 호출 자동 ping | ~30분 |
| 3 | threshold encoder.dim 기반 자동 조정 | ~30분 |
| 4 | interpretation 다국어 일관성 (system_prompt 개선 — "same language as input") | ~30분 |
| 후순위 | JSON 파싱 강화 (```json 블록 처리) | ~30분 |
| 후순위 | Memory L3 통합 (recall 시 L3 pattern matching) | ~2h |

---

## 11. 결론

| 지표 | 값 |
|------|----|
| Plan SUCCESS | **6/6 strict** (Match Rate 100%) |
| 회귀 baseline | 283 → **303 PASS** (+20) |
| 신규 소스 | +256줄 |
| 신규 테스트 | +15 |
| 깨진 회귀 | **0건** |
| 발견 + fix | **2 bugs** (recall key, threshold) — 시연 발견 |
| 소요 | ~3.5h (양적 검증 0.5h + Plan/Design 1h + Do 1.5h + 시연 0.5h) |

### "창의성의 라이브러리" 정의 완성

```
                  발견                  해석                    기억           재활용
                ┌──────┐            ┌────────┐            ┌─────────┐    ┌──────────┐
ingest text ─→ │ Bridge │─→ escalate │ LLM    │─→ interp ─│Memory L2│─→ recall──→ CLI
              └ Coherence┘   =True  │ Region │            │Episode  │    │📚 + 💡  │
                                    └────────┘            └─────────┘    └──────────┘
                                                              ↑                ↓
                                                              └─ 다음 충돌 시 ─┘
                                                                 vec 유사도 검색
```

5 cycle (Bridge / sub-4 / conflict-interpretation / 양적 검증 / 이번 cycle) 누적이
만들어낸 *학습하는 지식 시스템*. RAG / LangChain 으로 불가능한 HTP 고유 가치.

### 다음 cycle 후보

§10 우선순위 4건. 사용자 결정에 따라 진행.
