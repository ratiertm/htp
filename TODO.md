# HTP — TODO / Handoff

**Last updated**: 2026-05-19 (sub-4 완료 + 외부 리뷰 합의)
**Last commit**: `8063975` sub-4 외부 리뷰 리포트 (master, push 완료)
**Current state**: **sub-4 PDCA 종료 (91% Match Rate). 다음 cycle 방향 결정 완료** — LLMRegion ↔ CoherenceGate conflict 해석 연결.

다음 세션에서 이 파일부터 읽고 시작.

---

## 🎯 다음 cycle (1순위) — `htp-conflict-interpretation`

### 핵심 가설

> **CoherenceGate 가 감지한 충돌 (escalate=True) 을 LLMRegion 이 자연어로 해석해
> 새로운 통찰 가설을 제안한다. 이것이 "창의성의 라이브러리" 의 첫 실제 사례다.**

### 진입 명분

sub-3 (CoherenceGate) + sub-4 (LLMRegion) + Bridge (KnowledgeLoop 통합) 의 3 인프라가
이미 모두 검증됨. 그러나 분리되어 있어 사용자 가치 미발현. **둘을 합치면 다음:**

```
현재:
  $ htp knowledge ingest "주의는 국소적이다" --source 뇌과학
  ⚠ 충돌 감지 (escalate=True, conflict=0.153) — 끝

목표:
  $ htp knowledge ingest "주의는 국소적이다" --source 뇌과학
  ⚠ 충돌 감지 (escalate=True, conflict=0.153)
  💡 AI 의 "Transformer attention 은 전역적" 과 모순.
     그러나 scale 차이일 수 있음: 뇌는 뉴런 수준에서 국소적이나
     영역 간 통신은 전역적. Transformer 도 head 별로는 국소 패턴 학습.
     → 두 관점 통합 시 "multi-scale attention" 가설 가능.
```

### 구현 코어

`KnowledgeLoop.ingest()` 의 `_evaluate_coherence()` 결과에서 `escalate=True` 시
사용자 정의 `LLMRegion` (옵션) 을 호출. 충돌 entries (신규 + top-3 이웃) 를 prompt
로 구성. CostRouter.select_level 이 여기서 *처음으로 실제 의미*를 가짐 — conflict
해석은 complexity 높으므로 Level 3-4 선택.

### 작업 분할 (예상)

| Stage | 내용 | 소요 |
|-------|------|:----:|
| Plan | escalate=True 발견 시 호출 시점 / prompt 구조 / 비용 정책 결정 | ~30분 |
| Design | KnowledgeLoop 에 `conflict_interpreter: LLMRegion \| None` DI / Architecture 옵션 | ~30분 |
| Do | KnowledgeLoop.ingest + LLMRegion 통합 + prompt template + CLI 출력 | ~1.5h |
| Check | 실데이터 시나리오 검증 (이질 ingest → 해석 품질 평가) | ~30분 |
| 합계 | | **~3h** |

### 진입 전 결정 필요 (Plan 단계)

1. **충돌 해석 호출 시점**: ingest 시 동기 (즉시 출력) vs 비동기 (이후 query 시 조회)?
2. **Mock 모드 default**: 데모/테스트 시 API 키 없이 작동해야 함. CLI flag `--mock-llm`?
3. **호출 빈도 제한**: escalate 가 많아지면 비용 폭증. CostRouter pressure threshold 로 cap?
4. **해석 결과 저장 위치**: KnowledgeEntry 의 새 필드 vs 별도 store?

---

## ⏸️ 후순위 (실 필요 발생 시)

| 항목 | 사유 |
|------|------|
| **SearchRegion / RAGRegion 확장** | ExternalRegion 추상의 다양성 검증은 가치 있으나, "만들 수 있으니까 만들자" 는 인프라 정비의 늪. 실 사용처가 생기면 진행 |
| **sub-6 vector default 전환** (`routing_mode = "vector"`) | 사용자 체감 0. hybrid 모드가 잘 작동 중. 급할 이유 없음 |
| **CostRouter.select_level 임계값 튜닝** | 실 호출 데이터 부재. 연결 4 완료 후 데이터 수집되면 그때 |
| **C-4 graphify 정량 측정** | 30분 작업이나 sub-4 다음 cycle 중 끼워 진행 (별도 cycle 불필요) |

---

## 🌉 Bridge Integration (완료)

---

## 🌉 Bridge Integration (htp-bridge-integration) — 완료

설계서 `docs/02-design/features/htp-bridge-integration-design.md` 의 3 연결 모두 구현·검증:

| 연결 | 구현 위치 | 효과 |
|------|----------|------|
| §2 RegionSignature | `loop.ingest()` + `_update_signature()` | source 별 의미 중심 EMA 학습 |
| §3 CoherenceGate | `loop._evaluate_coherence()` + `IngestResult.coherence_info` | ingest 시 충돌 감지 |
| §4 VectorRouter | `loop.query(mode="routed")` + CLI `--mode compare` | 검색 노이즈 제거 |

테스트: +11 신규 (`tests/knowledge/test_bridge_s{1,2,3}_*.py`) + DAG 양방향 강제 +14 (thalamus→knowledge 금지). **222/222 PASS**.

### 가설 검증 결과 (EmbeddingBridge)

| Q | 결과 | 비고 |
|---|------|------|
| Q1 domain discrimination | **PASS** | 뇌과학 0.865 > 인프라 0.823 |
| Q2 coherence 모순/일관 | **부분** | 정성 PASS (일관 0.116 < 이질 0.152), 절대 threshold 미달 |
| Q3 VectorRouter 정밀도 | **PASS** | 3/3 케이스에서 routed 가 flat 노이즈 제거 |

→ design §9 "3개 중 2개 이상 → 가설 지지" **충족**. 시스템 A 의 가치 검증 완료.

### sub-4 (Stage 4+5) **완료** (2026-05-19)

`docs/02-design/features/htp-thalamus-car.sub-4.design.md` Architecture B 채택.
Session A→B→C 순차 진행, 회귀 0 깨짐.

| Session | 결과 |
|---------|------|
| A (M1+M2+M3) | ExternalRegion + LLMRegion + CostRouter.select_level (4-Level). 25 tests |
| B (M4+M5+M7) | LLMRegionRuntime archive 이동 + demo. 회귀 보존 |
| C (M6+M8) | PipelinedBrainRuntime + throughput 측정. 6 tests |

Match Rate **91%** — `docs/03-analysis/htp-thalamus-car.sub-4.analysis.md`.

**Throughput 실측** (PipelinedBrainRuntime vs AsyncBrainRuntime):
- N=4 → 1.95-2.00× / N=8 → 2.55-2.64× / N=16 → 2.65-2.67×
- Plan §SUCCESS 목표 1.5× 모두 큰 마진 초과.

C-1 (demo) ✓ / C-2 (LLMNode 내부 멤버) ✓ / C-3 (CostRouter 7-method 보존) ✓ /
C-4 (graphify isolated 50% 감소) — **△ 정량 측정 후속 cycle**.

### Bridge 후속 cycle — Q2 retune **완료** (2026-05-18)

측정 기반 encoder 분기 구현. 모든 항목 해결:

| encoder | 측정 (intra/inter) | retuned threshold | 효과 |
|---------|------------------|------------------|------|
| TfidfJLEncoder | 모두 ≈ 1.0 포화 | (0.5, 1.0) | escalation 비활성 — 노이즈 차단 |
| EmbeddingBridge | intra max=0.124, inter max=0.141 | (0.105, 0.135) | 일관/이질 strict 분리 |

`KnowledgeLoop(coherence_thresholds=(c,e))` 로 override 가능.

**Q2 재검증 결과** (EmbeddingBridge):
- 이질 conflict=0.153, escalate=**True** ✓
- 일관 conflict=0.107, escalate=**False** ✓
- design §9 Q2 strict 기준 충족 — Q1/Q2/Q3 **3/3 PASS**.

테스트 +5 (`test_bridge_q2_retune.py`): encoder default · 미지 encoder fallback · override.

---

## 📚 KnowledgeLoop UX 폴리시 후속 (별도 micro-cycle, 후순위)

sub-5 시나리오 D + Vault 실사용 발견 4건. 연결 4 (LLMRegion 해석) 이후 의미 있음:

| ID | 항목 | 소요 |
|----|------|:----:|
| I3 | relative ranking (top-k 내 min-max normalize) | 1-2h |
| I1 | recursive glob (서브디렉토리 ingest, `**/*.md` 기본화) | 30분 |
| I4 | Obsidian frontmatter tags 추출 → entry.tags 자동 매핑 | 1-2h |
| I2 | frontmatter strip (list/query preview 의 본문 우선 표시) | 1h |

→ `htp-knowledge-ux-polish` 묶음 (3-4h).

---

## 📊 현재 PDCA 진행 상태

```
htp-thalamus-car (9 Stage)
├─ sub-1 ✅  Stage 0 + 0.5 (Knowledge Loop MVP)        118 baseline
├─ sub-2 ✅  Stage 1 + 2 (Vector Routing + Hybrid)     140
├─ sub-3 ✅  Stage 3 (CoherenceGate + Memory)          148
├─ L2-sidequest ✅  CLI polish (batch/edit/filter/export) 172
├─ sub-5 ✅  Stage 6 EmbeddingBridge + merge plan     197  ← 현재
├─ sub-4 ⏳  Stage 4 + 5 (LLMRegion + Pipeline)
└─ sub-6 ⏳  Stage 7 (vector default)
```

---

## ✅ sub-5 완료 핵심 성과

### 본질 해결 (TF-IDF → Embedding)

| 검증 | Before | After |
|------|------:|-----:|
| Journal 한국어 매칭 | 0/5 | **5/5** |
| Paper 시나리오 D top-1 | 0/4 | **≥3/4** |
| Vault 99 entries top-1 | (미측정) | 4/5 |
| 회귀 깨짐 | - | **0건** |

### D1-D4 원칙 코드 검증 (영구 보호)

- D1 Frozen: `test_embedding_bridge_frozen_weights` + adversarial 2건
- D2 Protocol: isinstance(TextEncoder)
- D3 Fallback: TfidfJLEncoder 보존 + CLI `--encoder` 옵션
- D4 학습 분리: HTP centroid 학습 + embedding 불변 공존

### merge plan 4 작업 (Claude + Gemini 리뷰 합의)

1. Adversarial test (D1 위반 시도 방어) — 2건
2. e5 prefix (query/passage) — 1건
3. main merge (`a95c265`)
4. I5 confidence score (top-1 vs top-2 gap, threshold=0.01) — 4건

---

## ⚠️ 후속 검증 필요 사항

### 1. gap_threshold 0.01 의 적정성 재검토

실측 (Vault 99 entries):
- Hopfield (없음) gap=0.0051 → no_match ✓
- V-JEPA (있음)  gap=0.0042 → **false negative** ⚠
- HTP thalamus (있음) gap=0.0168 → match ✓

**Trade-off**: threshold 0.01 이 false negative (V-JEPA) 1건 만듦.
사용자가 `query_v2(gap_threshold=)` 로 조정 가능.
실사용 1-2주 후 분포 재측정 권장.

### 2. encoder_state.pkl 호환성

- TfidfJLEncoder state (sub-1 옵션 A-2): `.htp/encoder_state.pkl`
- EmbeddingBridge state (sub-5): 같은 파일명 — *충돌 가능*

```python
# 안전 패턴: encoder type 별 다른 경로
# .htp/encoder_state.tfidf.pkl
# .htp/encoder_state.embedding.pkl
```

후속 micro-fix (~30분).

---

## 📋 백로그 (별도 PDCA cycle)

### Phase Now (sub-5 직후 권장)

- `htp-knowledge-ux-polish` (I1+I2+I3+I4 통합, 3-4h)
- `htp-encoder-state-isolation` (encoder type 별 state 파일 분리)

### Phase Next (본선)

- **sub-4** Stage 4+5 (LLMRegion + Pipeline)
- **sub-6** Stage 7 (vector default 전환)

### Phase Backlog (별도)

- MED-3: 대시보드 BrainRuntime/Memory
- LOW-5: Friston B3 precision 시뮬레이션
- LOW-6: NGE split 파라미터
- LOW-7: Memory CUSUM × SWR 경계 분석
- `htp-region-signature-persistence` (sub-3 Gap #1)
- `htp-thalamus-coherence-lsh` (Plan §R2 N≥16 trigger)
- `htp-thalamus-async-pipeline` (sub-5 EmbeddingBridge 50ms+ 시)
- `htp-routing-tuning` (β/α/conflict threshold context 별 권장값)

---

## 🧪 테스트 baseline

```bash
source .venv/bin/activate
pytest tests/regression/ tests/unit/ tests/knowledge/ -q
# 기대: 197 passed (HF 캐시 warm 후 ~100s)
# 첫 실행 시 모델 다운로드 ~30s 추가
```

⚠️ 새 작업 직후 항상 통과 확인. 깨지면 즉시 롤백.

---

## 🔬 보존된 검증 데이터

`.htp/` (`.gitignore`):
- `knowledge_log.jsonl` — TF-IDF 시대 7 entries (sub-1 자료)
- `encoder_state.pkl` — TfidfJLEncoder fit 상태
- 백업: `.pre-optionA.jsonl`, `.optionA-inmemory.jsonl`

`archive/knowledge-test-papers/` (committed):
- 20 abstract (brain 6 / cogsci 4 / worldmodel 5 / ai 5)
- sub-5 Go/No-Go 검증 + 향후 재실험 자료

`tests/실사용 테스트.md`:
- S1-S5 (L2 sidequest 인수)
- S6-S7 (sub-5 EmbeddingBridge 검증)
- S8-S10 (L4 통합)

---

## 📁 핵심 문서

| 문서 | 경로 |
|------|------|
| **TODO (이 파일)** | `TODO.md` |
| 프로젝트 컨텍스트 | `CLAUDE.md` |
| sub-5 Plan | `docs/01-plan/features/htp-thalamus-car.sub-5.plan.md` |
| sub-5 Design | `docs/02-design/features/htp-thalamus-car.sub-5.design.md` |
| sub-5 Check | `docs/03-analysis/htp-thalamus-car.sub-5.analysis.md` |
| sub-5 Report | `docs/04-report/htp-thalamus-car.sub-5.report.md` |
| **외부 리뷰 리포트** | `docs/03-analysis/htp-sub5-실사용검증-외부리뷰용.md` |
| merge plan | `docs/01-plan/features/htp-sub5-merge-plan.md` |
| 실사용 시나리오 | `tests/실사용 테스트.md` |

---

## 🎯 다음 세션 시작 권장 순서

1. **이 TODO.md 먼저 읽기**
2. `git log --oneline -5` (최신: `ff4aff7`)
3. `pytest tests/regression/ tests/unit/ tests/knowledge/ -q` baseline 확인 (197/197)
4. **결정**: sub-4 본선 진입 vs `htp-knowledge-ux-polish` micro-cycle
   - sub-4 = LLMRegion (본선 진도)
   - ux-polish = 후속 5건 (I1/I2/I3/I4 + encoder_state 분리) 3-4h
5. 결정 시 `/pdca plan` 또는 `/pdca design`

---

## 💡 sub-5 핵심 교훈 — v4 Rev 1.3 5차 실증

> **"시나리오 D 의 정량 한계가 sub-5 우선순위 상향을 *주도*했다"**

- TF-IDF 의 본질 한계가 수치로 노출 (top-1 0/4)
- sub-4 보다 sub-5 가 사용자 가치 우선 — 우선순위 자체 재배치
- 사용자 데이터 (Journal/Vault) 로 본질 해결 *재검증*
- D1-D4 원칙으로 LLM 종속화 방지

다음 sub-cycle 도 같은 패턴 — **사용자 가치 우선 + 실사용 검증**.

---

## 🔗 외부 리뷰 받은 핵심 합의

Claude + Gemini 리뷰 결과 (merge plan §0):

| 항목 | 합의 |
|------|------|
| D1-D4 원칙 적절성 | 보호 충분 |
| 모델 선택 (e5-small) | 탁월한 타협점 |
| I5 confidence 최우선 후속 | 합의 |
| main merge | Go |
