# HTP — TODO / Handoff

**Last updated**: 2026-05-18
**Last commit**: `ff4aff7` sub-5 merge plan 작업 3 — I5 confidence (master, push 완료)
**Current state**: sub-5 (Stage 6 EmbeddingBridge) **완료 + merge 됨** — 본질 한계 해결

다음 세션에서 이 파일부터 읽고 시작.

---

## ⏭️ 즉시 진입 가능한 다음 작업

### A. sub-4 본선 진입 (Stage 4 + 5)

`htp-thalamus-car` plan §5 의 남은 본선 작업:
- **Stage 4**: ExternalRegion + LLMRegion 리팩토링
- **Stage 5**: PipelinedBrainRuntime

```bash
/pdca design htp-thalamus-car   # sub-4 scope
```

목표: 누적 197 → 210 (Plan §5 명시 89 + L2/sub-5 누적 가산)

### B. 후속 micro-cycle 5건 (sub-5 발견 사항)

merge plan §7 + 시나리오 D 발견 후속:

| 우선순위 | ID | 항목 | 소요 |
|:--:|----|------|:----:|
| 2 | **I3** | relative ranking (top-k 내 min-max normalize) | 1-2h |
| 3 | **I1** | recursive glob (서브디렉토리 ingest, `**/*.md` 기본화) | 30분 |
| 3 | **I4** | Obsidian frontmatter tags 추출 → entry.tags 자동 매핑 | 1-2h |
| 4 | **I2** | frontmatter strip (list/query preview 의 본문 우선 표시) | 1h |

→ 4건 묶음 `htp-knowledge-ux-polish` (3-4h) 또는 개별 cycle.

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
