# htp-conflict-interpretation — 실 API 시연 결과

**작성일**: 2026-05-19
**대상**: SC6 (실데이터 1건) 정량 평가 — Mock 시연에 이은 실 LLM 호출 결과 분석.
**Backend**: `claude -p` subprocess (OAuth headless session) — `ClaudeCliNode` 신규.

---

## 0. 한 줄 요약

> **Mock interpretation (prompt echo) 와 실 LLM interpretation 이 질적으로 완전히 다름.
> 실 LLM 은 표면 vocabulary 차이 아래의 *구조적 유사성* 까지 발견 (eviction ↔ pruning,
> load balancing ↔ lateral inhibition). "창의성의 라이브러리" 가치 정성적으로 검증됨.**

---

## 1. Backend — `ClaudeCliNode`

ANTHROPIC_API_KEY 직접 사용 대신 Claude Code CLI 의 OAuth headless session 활용.

```python
class ClaudeCliNode:
    def run(self, data):
        prompt = self._build_prompt(data)
        proc = subprocess.run(
            ["claude", "-p", "--output-format", "text"],
            input=prompt,
            env=dict(os.environ) - {"ANTHROPIC_API_KEY"},   # OAuth 강제
            timeout=self.timeout_sec,
        )
        return self._parse_response(proc.stdout)
```

**장점**:
- API 키 관리 불필요 (Claude Code 구독 한도 활용)
- 별도 결제 분리 (Anthropic billing 의존 없음)
- `env -u ANTHROPIC_API_KEY` 로 OAuth session 강제

**단점**:
- subprocess 오버헤드 → latency ~12s/call (직접 API 는 ~3s)
- 토큰 사용량 / 비용 정확 측정 불가 (stderr 미파싱)

### LLMRegion drop-in 호환

`LLMRegion.__init__(llm_node=...)` 인자 신설 — `LLMNode` / `MockLLMNode` /
`ClaudeCliNode` 모두 동일 interface (`name/run/arun/_token_log/cost_report`).
사용자가 명시 인스턴스 넘기면 그것 사용, None 이면 기존 use_mock 분기.

---

## 2. 시연 환경

```
encoder           : EmbeddingBridge (intfloat/multilingual-e5-small, 384-dim)
coherence_thresholds: (0.10, 0.12)   ← Vault e5 분포에 맞춘 override
conflict_interpreter: LLMRegion(llm_node=ClaudeCliNode(...))
```

뇌과학 도메인 3 entries 누적 후 이질 ingest → escalate=True 유도.

---

## 3. 시나리오 1 — 도메인 충돌 (인프라 vs 뇌과학)

### 입력

```
ingest "Redis LRU 캐시 eviction 전략 nginx 로드밸런서 round robin gRPC service mesh"
       --source 인프라
```

### Coherence 측정

```
coherence = 0.879
conflict  = 0.144
escalate  = True
```

### Mock 응답 (이전 시연)

```
"mock(You are an analyst integrating): Conflict detected (coherence=0.88, conflict=0.15)."
```

→ prompt echo. 의미 없음.

### 실 LLM 응답 ⭐

> The conflict is **domain-categorical rather than factual**: an infrastructure
> statement (Redis LRU eviction, nginx round-robin, gRPC mesh) is being matched
> against neuroscience statements about Hebbian plasticity, CA3 pattern
> completion, and gamma-band binding, so there is no direct contradiction —
> only a low-coherence cross-domain juxtaposition where the surface
> vocabularies disagree while the **underlying control structures (eviction =
> pruning, load balancing = lateral inhibition, service mesh = recurrent
> connectivity)** are structurally analogous.

### 핵심 평가

- **"domain-categorical rather than factual"** — 충돌의 *유형* 정확 분류
- **3 개의 구조적 유사성 발견** — eviction↔pruning / load balancing↔lateral inhibition /
  service mesh↔recurrent connectivity. 이는 RAG / 단순 검색으로는 *불가능*.
- HTP 의 "cross-domain hub" 개념과 정확히 일치하는 통찰

---

## 4. 시나리오 2 — 의미적 모순 (Attention 메커니즘)

### 입력 시퀀스

```
ingest "Transformer 의 self attention 은 입력 전체를 한번에 참조하는 전역 메커니즘"
       --source AI
ingest "뇌의 주의 메커니즘은 국소적이며 한 번에 한 영역만 순차 처리한다"
       --source 뇌과학
```

### Coherence 측정

```
coherence = 0.850
conflict  = 0.174
escalate  = True
```

### 실 LLM 응답 ⭐

> The new statement claims attention is **local and strictly sequential** (one
> region at a time), which conflicts with the existing knowledge of **gamma-band
> oscillations binding distributed regions into unified conscious states** and
> with Transformer self-attention operating as a fully **global, parallel
> mechanism** over all inputs. The conflict is about **scope (local vs. global)
> and temporal mode (serial vs. parallel)** of attention.

### 핵심 평가

- **2 차원 분해** — scope (local/global) × temporal (serial/parallel)
- 신규 ingest 가 이전에 ingest 한 "gamma-band binding" 까지 *역참조* — top-3 이웃 prompt 가 효과 발휘
- 단순 contradiction 이 아니라 정밀한 *조건 차이* 식별
- 후속: 두 관점 통합 시 "multi-scale attention" / "energy-budgeted attention" 같은 가설 도출 가능

---

## 5. 호출 통계

| 지표 | 값 |
|------|----|
| 총 호출 | 3 (escalate=True 3건) |
| 총 latency | 37,106 ms |
| 평균 latency | ~12.4 s/call |
| `_interpretations_count` | 3 |

평균 latency ~12s 는 ingest UX 에 영향. 다음 옵션:
1. `claude -p` 의 `--output-format json` 활용해 더 빠른 모델 선택
2. 비동기 백그라운드 (Plan §1 §1 결정에서 *동기* 채택했으므로 후속 cycle)
3. CostRouter cap 으로 호출 빈도 자체 제한 (이미 적용 — max_interpretations=20)

---

## 6. Mock vs 실 LLM 정성 비교

| 측면 | Mock | 실 LLM (claude-cli) |
|------|------|---------------------|
| 의미 | prompt echo | 충돌의 본질 분석 |
| 구조적 통찰 | 없음 | eviction↔pruning 같은 cross-domain 유사성 발견 |
| 차원 분해 | 없음 | scope × temporal 2-axis 분해 |
| 통합 가설 | 없음 | "structurally analogous" / "multi-scale" 제안 |
| Latency | ~ms | ~12s |
| 비용 | 0 | Claude Code 구독 한도 |
| 개발 검증 가능 | ✓ end-to-end 흐름 | ✓ 의미 품질 |

---

## 7. SC6 최종 평가

**Plan §SUCCESS SC6 — 실데이터 해석 1건 (수동)** : **✅ PASS**.

- 시나리오 1, 2 모두 의미 있는 interpretation 생성
- prompt template (`build_conflict_prompt` + SYSTEM_PROMPT) 의 JSON-return 지시는
  CLI text mode 에서 plain English 로 응답되지만, dict 파싱 시 `text` 필드로 받음 —
  KnowledgeEntry.interpretation 에 그대로 저장 가능
- 의미 품질 평가: cross-domain 유사성 발견 + 차원 분해 — RAG 로는 불가능한 통찰

**htp-conflict-interpretation Match Rate 96% → 98%** (SC6 Mock → 실 PASS).

---

## 8. 후속 작업

| 우선순위 | 항목 |
|:--:|------|
| 1 | Memory ↔ KnowledgeLoop 연결 (interpretation 을 Episode 로 저장 → CA3 recall) |
| 2 | e5 escalation_threshold default 재조정 (Vault 실 분포 측정) |
| 3 | latency 개선 옵션 검토 (현 12s/call 이 ingest UX 에 영향) |
| 후순위 | C-4 graphify 측정 |

---

## 9. 산출물

| 파일 | 변경 |
|------|------|
| `htp/llm/claude_cli_node.py` 신규 | +150줄 |
| `htp/llm/llm_region.py` (llm_node 인자) | +10줄 |
| `htp/llm/__init__.py` | ClaudeCliNode export |
| `tests/regression/test_sub4_claude_cli_node.py` 신규 | +130줄, 12 tests |
| 이 문서 | 시연 결과 영구 보존 |

전체 회귀: 271 → **283 PASS** (+12 신규).

---

## 10. 결론

> sub-3 (CoherenceGate) × sub-4 (LLMRegion + ClaudeCliNode) × Bridge (KnowledgeLoop)
> 의 곱이 실 LLM 으로 의미 있는 통찰을 생성. "창의성의 라이브러리" 의 정성 가치
> 검증 완료. Mock 흐름은 개발 안전성, 실 LLM 흐름은 사용자 가치 — 두 모드 모두 작동.
