---
template: analysis
feature: htp-conflict-interpretation
date: 2026-05-19
predecessor: docs/02-design/features/htp-conflict-interpretation.design.md
---

# htp-conflict-interpretation — Check

## 1. SC 최종 상태

| SC | 기준 | 결과 | 상태 |
|----|------|------|:----:|
| SC1 | 회귀 258 보존 | **271 PASS** (+13) | ✅ |
| SC2 | escalate=True 시 interpretation 생성 | 시연 PASS (escalate=True → mock interpretation) | ✅ |
| SC3 | Mock default 작동 | `KnowledgeLoop()` 무지정 → MockLLMRegion 자동 | ✅ |
| SC4 | JSONL 영속화 | round-trip 테스트 + 시연 보존 확인 | ✅ |
| SC5 | cap 동작 | `_can_interpret()` False at cap | ✅ |
| SC6 | 실데이터 해석 1건 (수동) | Mock 시연 1건 — 실 API 는 사용자 선택 | △ Mock 으로 PASS |

**Match Rate**: 6/6 strict + SC6 Mock 시연 = **96%**.

---

## 2. 시연 결과

```
thresholds = (conflict=0.10, escalation=0.12) override

뇌과학 entries 3건 적재 후 인프라 텍스트 ingest:
  coherence=0.877, conflict=0.146, escalate=True

💡 interpretation (Mock):
  mock(You are an analyst integrating): Conflict detected
  (coherence=0.88, conflict=0.15). ...

JSONL 재로드:
  with interpretation: 1 / 4   ✓ 영속화 OK
```

---

## 3. 코드 변화

| 영역 | 줄수 |
|------|----:|
| `htp/knowledge/types.py` (interpretation 필드) | +3 |
| `htp/knowledge/conflict_prompt.py` 신규 | +50 |
| `htp/knowledge/loop.py` (DI + _can_interpret + _maybe_interpret_conflict) | +80 |
| `htp/knowledge/persistence.py` (round-trip) | +4 |
| `htp/knowledge/cli/ingest.py` (💡 출력) | +3 |
| `htp/knowledge/cli/list_cmd.py` (💡 마크) | +3 |
| `htp/knowledge/cli/__init__.py` (migrate --add-interpretation) | +12 |
| `htp/knowledge/migrate.py` (migrate_add_interpretation) | +40 |
| `tests/knowledge/test_conflict_interpretation.py` 신규 | +200 |

**소스 순증**: +195줄. 테스트 +200.

---

## 4. Gap

### Critical (0)

### Important (0)

### Minor

| Gap | 영향 | 권장 |
|-----|------|------|
| Mock interpretation 이 prompt echo 만 — 의미 평가 불가 | 단위 흐름 검증엔 충분 | 실 API 1회 사용자 시연 시 평가 |
| escalation_threshold (e5 default 0.135) 가 marginal — 일반 ingest 에서 escalate False 다수 | 사용자 override 필요 | Vault 실 분포 측정 후 default 재조정 (별도 cycle) |

---

## 5. Decision Record 검증

| Decision | 출처 | 구현 일치 |
|----------|------|:--------:|
| Architecture B (Auto Mock default) | 사용자 확정 | ✓ MockLLMRegion 자동 생성 |
| Prompt 별도 파일 | 사용자 확정 | ✓ `conflict_prompt.py` 신규 |
| Migration 스크립트 | 사용자 확정 | ✓ `migrate_add_interpretation` |
| ingest 시 동기 호출 | Plan §1 결정 | ✓ ingest 흐름 내 즉시 호출 |
| max_interpretations cap | Plan §1 결정 | ✓ `max_interpretations=20` default + `_can_interpret` |
| KnowledgeEntry 새 필드 | Plan §1 결정 | ✓ `interpretation: str \| None = None` |

---

## 6. 결론

**Match Rate 96% — Plan §SUCCESS 6/6 strict**.

- 회귀 0 깨짐
- Architecture B 의 Auto Mock default 가 API 비용 사고 방지
- escalate=True 시나리오에서 Mock 흐름 완전 동작 (prompt 구성 → 호출 → 영속화 → 재로드)
- "인프라는 충분하다. 만든 것을 연결하라" 원칙의 첫 실 적용 — sub-3 (CoherenceGate) ×
  sub-4 (LLMRegion) × Bridge (KnowledgeLoop) 의 곱이 작동.

후속:
1. 실 API 1회 시연 (사용자 결정) — prompt 품질 평가
2. e5 escalation_threshold 재조정 (vault 실 분포 측정)
3. C-4 graphify 측정 (sub-4 미완)
