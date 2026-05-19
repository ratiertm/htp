# Conflict Interpretation — 양적 검증 결과 (50건)

**작성일**: 2026-05-19
**Raw**: `docs/03-analysis/conflict_quant_raw.jsonl`
**스크립트**: `scripts/conflict_quant_eval.py`
**Backend**: ClaudeCliNode (OAuth headless)

---

## 0. 한 줄 요약

> 50건 이질 ingest 중 escalate=True 24건 (48%) 모두 LLM 이 interpretation 생성.
> 통찰 품질은 고 29% / 중 37% / 저 33% — **모두 저장하되 recall 시 best-match
> 만 노출** 하는 것이 합리적. conflict 값 자체로는 quality 판별 불가.

---

## 1. 시나리오 + 환경

- 4 도메인 (뇌과학/AI/인프라/금융) × 7 baseline entries 적재 (28건)
- 50건 cross-domain ingest (의도적 이질)
- `EmbeddingBridge` (e5-small, 384-dim)
- `coherence_thresholds=(0.10, 0.12)` — escalate 빈도 보장 위해 낮춤
- LLMRegion + `ClaudeCliNode` (claude -p)
- `max_interpretations=100` (cap 충분)

## 2. 정량 통계

| 지표 | 값 |
|------|----|
| total ingest | 50 |
| escalate=True | **24 (48%)** |
| interpretation 생성 | **24 (100% of escalate)** |
| avg conflict | 0.123 |
| avg interp latency | 14,227 ms (~14s/call) |
| escalate=False 평균 ms | ~30 (LLM 호출 없음) |
| total elapsed | 516.5s (~8.6분) |

---

## 3. 통찰 품질 정성 분류 (24건)

### 고품질 — Mechanism mapping / 차원 분해 (7건, 29%)

| # | input | 통찰 |
|---|-------|------|
| 3 | self supervised learning | dopamine(외부적) vs SSL(데이터 구조 내부) — 학습 신호 *출처 차원* |
| 4 | hippocampal replay 깨어있을 때 | *시점(수면/각성) × 기능(공고화/시뮬레이션)* 2-axis |
| 5 | synaptic pruning | elimination ↔ strengthening complementarity |
| 6 | axonal myelination | white-matter speed vs synaptic strength (다른 layer) |
| 8 | Q-learning | RL paradigm vs supervised — sequential decision-making |
| 10 | multi armed bandit | exploration ↔ exploitation framing |
| 17 | market circuit breaker | 동음이의: 인프라 fault vs 금융 price halt |

### 중품질 — Layer/Scope 보완 (9건, 37%)

| # | input | 통찰 |
|---|-------|------|
| 7 | blue-green deployment | application 레이어 vs CDN edge 레이어 |
| 12 | Markowitz portfolio | implied vol 과 다른 시점/관점 |
| 13 | settlement T+2 | strong consistency (clearing) vs eventual (분산) |
| 14 | WAL | 단일 노드 durability vs 분산 consistency 강조점 |
| 16 | circuit breaker pattern | runtime fault isolation availability |
| 20 | amygdala fear | 음 vs 양 valence learning 보완 |
| 21 | market panic | behavioral contagion vs 정량 VaR |
| 22 | trader discipline | 개인 vs 시스템 레벨 risk |
| 24 | overnight risk | active position-sizing vs passive containment |

### 저품질 — 서술적 / overlap 인정 (8건, 33%)

| # | input | 응답 패턴 |
|---|-------|----------|
| 1 | 주의 메커니즘 국소적 | 단순 설명 (충돌 없음 강조) |
| 2 | attention 전역 연산 | softmax sparse 라고 부분 보완 |
| 9 | market making | "substantively overlaps with existing" — overlap만 |
| 11 | statistical arbitrage | pair trading 정의 설명 |
| 15 | limit order book | "not contradictory but adds layer" |
| 18 | trade reconciliation | 후처리 설명 |
| 19 | liquidity provider | 메커니즘 설명만 |
| 23 | algorithmic trading | rule vs human framing |

---

## 4. 핵심 발견

### 4-1. conflict 와 quality 의 상관 약함

| 등급 | avg conflict |
|------|:------------:|
| 고 (mechanism) | 0.131 |
| 중 (layer/scope) | 0.131 |
| 저 (서술적) | 0.140 |

→ conflict 값으로 quality 판별 불가. 사전 confidence 필터 부적합.

### 4-2. JSON 응답 형식 일관성 부족

24건 중 6건이 ` ```json {...} ``` ` 코드블럭으로 응답.
나머지는 plain text 또는 inline JSON. `_parse_response` 의 단순 `text.startswith("{")`
체크로는 코드블럭을 못 잡음 → 모두 `text` 필드로 fallback. 의미 손실은 없으나
파싱 안정성 필요.

### 4-3. 언어 혼용

한국어 ingest 도 LLM 응답이 60% 영어 (SYSTEM_PROMPT 가 영어). 사용자 UX 위해
"respond in the same language as input" 추가 권장.

### 4-4. Latency 분포

- escalate=False (26건): avg ~30 ms (LLM 호출 0)
- escalate=True (24건): avg 14,227 ms (claude -p subprocess)

대부분 ingest 는 escalate 안 됨 → latency 영향 작음. 충돌 시에만 14s 대기 (사용자
"해석 중..." 메시지로 wait 수용 가능).

---

## 5. B 설계 함의

### 5-1. 저장 정책

**모든 interpretation 저장. 사전 quality 필터 안 함.**

이유:
- conflict 값으로 quality 판별 불가
- 저품질 응답도 "이전엔 이런 해석을 했었다" 누적 자산 가치
- recall 시 best-match 노출이 사용자 체감 quality 결정

### 5-2. recall 시 quality_hint 메타

구조적 키워드 카운트:
- 고품질: `mechanism, axis, dimension, scope, layer, complementarity, framing` 다수
- 저품질: `overlap, similar, related, not contradictory` 다수

```python
QUALITY_KEYWORDS = ["mechanism", "axis", "dimension", "scope",
                    "layer", "complementarity", "framing",
                    "메커니즘", "차원", "관점"]

def quality_hint(interp: str) -> float:
    """0.0-1.0, 통찰성 자동 추정."""
    hits = sum(1 for kw in QUALITY_KEYWORDS if kw in interp.lower())
    return min(1.0, hits / 3)
```

recall 시 동률이면 `quality_hint` 높은 것 우선.

### 5-3. Episode 저장 단위

```python
# state_vec = encoder.encode(interpretation_text)   ← interpretation 자체 임베딩
# context   = new_text[:50]                          ← 충돌 trigger
# interpretation_text 는 새 필드로
# winner    = "conflict_interpreter"
# score     = conflict (recall priority 가중용)
```

### 5-4. Prompt 개선 후속

별도 micro-cycle 로:
- "respond in the same language as input"
- JSON 응답 강제 (코드블럭 회피)
- "if no real conflict, return empty hypothesis" (저품질 응답 자체 차단)

---

## 6. 결론

50건 데이터는 B (Memory ↔ KnowledgeLoop) 진입을 정당화합니다:

1. ✅ LLM 응답률 100% — Mock 흐름이 실 환경에서도 작동
2. ✅ 1/3 가 통찰 수준 — 누적 자산 가치 분명
3. ✅ 1/3 가 저품질 — recall 시 quality_hint 필터 필요
4. ✅ conflict 값으로 사전 필터 불가 — 모두 저장이 합리적
5. ⚠ JSON 파싱 / 언어 혼용 — 별도 prompt 개선 cycle (후순위)

B 설계는 위 5 함의를 반영해 진행.
