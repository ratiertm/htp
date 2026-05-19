---
template: analysis
feature: htp-conflict-memory
date: 2026-05-19
predecessor: docs/02-design/features/htp-conflict-memory.design.md
---

# htp-conflict-memory — Check

## 1. SC 최종 상태

| SC | 기준 | 결과 | 상태 |
|----|------|------|:----:|
| SC1 | 회귀 283 보존 | **303 PASS** (+20) | ✅ |
| SC2 | escalate 시 Episode 저장 | 시연 — Memory Episodes 1건 (timeout 응답은 skip) | ✅ |
| SC3 | 2회차 recall_hint 채워짐 | **시연 PASS** mismatch=0.530 < 0.6 | ✅ |
| SC4 | quality_hint heuristic | 단위 5건 PASS | ✅ |
| SC5 | schema 마이그레이션 idempotent | `test_episode_store_schema_migration_idempotent` PASS | ✅ |
| SC6 | 실 시연 — 2회 충돌 → recall | **시연 v4 (warm-up + 1회차 + 2회차) PASS** | ✅ |

**Match Rate**: 6/6 strict = **100%**.

---

## 2. 실 시연 결과 (v4)

```
warm-up call             — 12.6s  (cold-start 흡수)
1회차 Redis              — 14.0s  escalate=True
                          💡 "conflict is categorical, not factual ..."
                          → Memory Episodes: 1
2회차 Kubernetes         — 16.7s  escalate=True
                          📚 RECALL HIT ✓ mismatch=0.530, quality=0.33
                             trigger: "Redis LRU 캐시 eviction 전략 ↔ 해마 CA3"
                             prev_interp: "The conflict is categorical..."
                          💡 새 해석: "두 진술 모두 '인프라' eviction 메커니즘이나
                                       추상 계층 다름 — Redis LRU 데이터 vs
                                       Kubernetes pod 리소스 계층"
```

**의미**: 같은 *eviction* 주제 2회 충돌에서
- 1회차: LLM 새 호출 (14s)
- 2회차: **이전 해석 즉시 표시 + 새 LLM 해석 추가 노출**

→ 사용자가 "이전 해석" + "이번 정밀 비교" 둘 다 봄. 통찰 누적 자산화 시연 완료.

---

## 3. 발견된 버그 + Fix (2건)

### Bug 1 — recall key 불일치

**증상**: v2 시연에서 recall_hint=None.

**원인**: `save_conflict` 시 `interpretation` 임베딩 저장 vs `_try_recall_conflict`
시 `text` 임베딩으로 검색. 다른 vec 공간 비교 → 매칭 안 됨.

**Fix**:
- `save_conflict` 의 첫 인자 `interpretation_vec` → `trigger_vec` (text 임베딩)
- `_save_conflict_episode` 에서 `encoder.encode(interpretation)` 대신 `vec` (text 임베딩) 사용
- design 의 "interpretation 자체 임베딩이 recall key" 가정 수정 — 실제로는 *다음 비슷한
  input* 이 들어왔을 때 매칭하는 게 자연스러움

### Bug 2 — CONFLICT_RECALL_MISMATCH_THRESHOLD 너무 엄격

**증상**: v2 → v3 시도에도 recall_hint=None.

**원인**: CA1 default 0.3 은 64-dim sparse vec 기준. 384-dim e5 dense 에서는
일반 cosine ~0.85 → L2 ~0.55 → 0.3 미통과.

**Fix**: 0.3 → **0.6** (cosine ~0.82 ↔ "비슷한 도메인" 수준). v4 시연에서 실측
mismatch=0.530 으로 HIT.

### Bug 3 — claude CLI cold start

**증상**: 매 시연 첫 호출이 timeout (120s 초과).

**관찰**: 시연 환경 특성. 해결책 2가지:
1. **warm-up call** — 시연 스크립트에 ping prompt 1회 추가 (v4 에서 적용, 12.6s 후 본 시연 정상)
2. **skip marker** — 이미 적용 — timeout 응답 (`"(claude cli timeout)"`) 은 Episode 저장 skip.
   1회차 timeout 시 노이즈 누적 방지.

Production 사용 시 `claude` CLI 의 OAuth refresh 등 환경적 — 별도 cycle 에서
관찰 후 결정 (현 시점 후순위).

---

## 4. 코드 변화

| 영역 | 변경 |
|------|------|
| `htp/memory/types.py` | Episode.interpretation_text 필드 (+3줄) |
| `htp/memory/episode_store.py` | SCHEMA + ALTER + save + _row_to_episode (+25줄) |
| `htp/memory/quality_hint.py` 신규 | 19개 키워드 + heuristic (+50줄) |
| `htp/memory/memory_system.py` | save_conflict + recall_conflict + threshold (+80줄) |
| `htp/knowledge/loop.py` | memory DI + _try_recall_conflict + _save_conflict_episode + SKIP_MARKERS (+90줄) |
| `htp/knowledge/cli/ingest.py` | 📚 출력 (+8줄) |
| `tests/knowledge/test_conflict_memory.py` 신규 | 15 tests (+250줄) |
| `tests/unit/test_no_circular_deps.py` | DAG 룰 갱신 (knowledge→memory 허용 + memory→knowledge 금지) |
| `docs/01-plan/features/htp-conflict-memory.plan.md` 신규 | |
| `docs/02-design/features/htp-conflict-memory.design.md` 신규 | |
| `docs/03-analysis/htp-conflict-memory.analysis.md` 신규 | 이 문서 |
| `docs/03-analysis/conflict_quant_summary.md` 신규 | 양적 검증 50건 |
| `docs/03-analysis/conflict_quant_raw.jsonl` 신규 | raw 데이터 |
| `scripts/conflict_quant_eval.py` 신규 | 양적 검증 스크립트 |

**소스 순증**: ~256줄. 테스트 +250. 문서 +5건.

---

## 5. Decision Record 검증

| Decision | 출처 | 구현 일치 |
|----------|------|:--------:|
| Episode 확장 (interpretation_text) | 사용자 확정 | ✓ |
| quality_hint 도입 | 사용자 확정 | ✓ |
| Architecture B (Auto-create) | 사용자 확정 | ✓ |
| recall 우선 → 그 후 새 LLM | 사용자 확정 | ✓ (CLI 출력 + IngestResult 흐름) |
| winner fixed "conflict_interpreter" | 사용자 확정 | ✓ |
| 양적 검증 후 B 진입 | 사용자 확정 | ✓ |

---

## 6. Gap

### Critical (0)

### Important (0)

### Minor

| Gap | 영향 | 권장 |
|-----|------|------|
| claude CLI cold start | 시연 환경 — production 영향 미상 | warm-up 또는 ClaudeCliNode 가 첫 호출 자동 ping 별도 cycle |
| CONFLICT_RECALL_MISMATCH_THRESHOLD 0.6 hard-coded | encoder dim 변화 시 재조정 필요 | encoder.dim 기반 자동 조정 별도 cycle |
| recall 시 trigger context 50자 cap | 표시 정보 부족 (예: `"Redis LRU 캐시 eviction 전략  ↔ 해마 CA3 패턴..."`) | KnowledgeEntry id 도 함께 저장 → 전체 본문 join 별도 cycle |
| quality_hint heuristic 19개 키워드 | 새 도메인 추가 시 확장 필요 | 양적 데이터 축적 후 자동 확장 별도 cycle |

---

## 7. 결론

**Match Rate 100% — Plan §SUCCESS 6/6 strict**.

핵심 성취:
1. **루프 완성** — 충돌 발견 (sub-3) → 해석 (sub-conflict-interp) → **기억** (이번 cycle) → 재활용. "창의성의 라이브러리" 의 정의가 완성됨.
2. **2 bugs 발견 + fix** — 양적 검증 + 실 시연 의 가치. 단위 테스트만으로는 못 잡았을 vec 공간 불일치 + threshold 부적절.
3. **사용자 의도 정확 반영** — quality_hint heuristic / recall 우선 노출 / Architecture B 모두 양적 검증 결과 직접 반영.

다음 cycle 후보:
1. **양적 평가** — Memory 누적 후 recall HIT rate / quality 분포 측정
2. **claude CLI cold-start 자동 warm-up** — Production 영향 확인 후
3. **CONFLICT_RECALL_MISMATCH_THRESHOLD encoder.dim 기반** — 384-dim 전용 0.6 generalize
4. **interpretation 결과 다국어 일관성** — system_prompt 개선 (English 우세 문제)
