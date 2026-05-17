# HTP — TODO / Handoff

**Last updated**: 2026-05-17 (Critical Gap #3 RESOLVED 반영)
**Last commit**: `0bb6aa3` Critical Gap #3 RESOLVED — encoder state 영속화 (옵션 A-2)
**Current PDCA**: `htp-thalamus-car` sub-1 완료 + Gap 해결 → **sub-2 진입 준비 완료**

다음 세션에서 이 파일부터 읽고 시작.

---

## ⏭️ 즉시 진입 가능한 다음 작업

### A. sub-2 시작 (Stage 1 + 2: Vector Routing + Hybrid)

```bash
/pdca design htp-thalamus-car   # sub-2 Design
# 또는
/pdca do htp-thalamus-car --scope stage-1
```

**Scope**:
- `htp/thalamus/signature.py` 신설 — `RegionSignature(centroid, count, lr=1/(n+1))`
- `htp/thalamus/core_cells.py` 에 `_gate_vector()` + `_gate_hybrid()` 추가
- `htp/thalamus/region_signal.py` 에 `region_signature` 필드 추가
- Dynamic threshold (`μ + β×σ`) 구현
- `routing_mode="tag"` 기본값 유지 (회귀 보호)

**Success Criteria**:
- 누적 테스트 65 → **78** (Stage 1 +10, Stage 2 +3)
- vector mode 가 tag mode 와 동등 또는 우위 (empty route 0건)
- α=0.1~0.9 변화 시 결과 연속적 (cosine of selected > 0.5)

**Stage 0.5 knowledge_log 활용**: `.htp/knowledge_log.jsonl` 의 7 entry 를 vector routing 테스트 데이터로 재활용. 같은 데이터가 sub-2/3/4 검증 전반에 쓰임.

---

## ✅ Critical Gap #3 RESOLVED (옵션 A-2 적용 완료, commit `0bb6aa3`)

**원래 현상**: `encoder.fit()` 재실행 시 `GaussianRandomProjection` 새 random matrix → 누적 cache vec 의미 공간 흔들림.

**추가 발견**: 인메모리 `_fitted` 플래그만으로는 부족. CLI 다중 호출 시 매번 새 KnowledgeLoop 인스턴스 → encoder._fitted=False 리셋 → 매번 fit 재실행.

**최종 해결**: **pickle 기반 영속화 (옵션 A-2)**
- `TfidfJLEncoder.save(path) / load(path)` — vocabulary + JL matrix + _fitted state
- `KnowledgeLoop` 생성자: `.htp/encoder_state.pkl` 자동 load
- 첫 ingest fit 직후 자동 save
- 다음 프로세스가 동일 state 복원 → 동일 임베딩 공간 영구 보장

**검증 결과**:
- 7-entry cross-process 재-encode: **7/7 diff = 0.00e+00** (완벽 동일)
- 영문 Go/No-Go (별도 디렉토리): brain↔ai 0.69 > brain↔infra cutoff ✅
- 전체 회귀: **118/118 PASS** (1.30s)
- 신규 unit test 3종 회귀 보호:
  - test_encoder_save_load_round_trip
  - test_loop_persists_encoder_across_instances (Gap #3 회귀 보호)
  - test_loop_encoder_state_file_created

**남은 trade-off**: 첫 fit 이후 새 어휘 영영 미반영. 첫 텍스트가 representative vocab 포함해야 함. **본질 해결은 sub-5 (Stage 6 EmbeddingBridge) — 사전학습 모델은 fit 불필요.**

---

## 📊 현재 PDCA 진행 상태

```
htp-thalamus-car (9 Stage, 6 sub-cycles)
├─ sub-1 ✅ [Plan ✅ Design ✅ Do ✅ Check ✅ 98%]   Stage 0 + 0.5
│  └─ Critical Gap #3 RESOLVED (commit 0bb6aa3, 옵션 A-2)
├─ sub-2 ⏳ pending                                  Stage 1 + 2 ← 진입 준비 완료
├─ sub-3 ⏳ pending                                  Stage 3 (CoherenceGate)
├─ sub-4 ⏳ pending                                  Stage 4 + 5 (LLMRegion + Pipeline)
├─ sub-5 ⏳ pending (experiment/embedding-bridge)    Stage 6
└─ sub-6 ⏳ pending                                  Stage 7 (vector default)
```

**완료된 이전 사이클**:
- `htp-phase2-integration` (이전, 커밋 `6be8746`)
- `htp-review-improvements` (이전, 커밋 `201f0f2` + 후속 작업 → `6291cbe`)

---

## 📋 백로그 (현 사이클 OUT-OF-SCOPE)

본 사이클 종료 후 별도 PDCA 사이클로 진행할 항목들:

### ⭐ NEW: `htp-knowledge-cli-polish` (L2 sidequest, sub-3 직후 권장)

**Trigger**: sub-3 (CoherenceGate) 완료 후 또는 sub-3 마지막 step 으로 흡수.
**소요**: 1-2일.
**효과**: L1 (가설 검증용) → **L2 (매일 쓰는 prototype)**.
**한계**: 여전히 TF-IDF — 한국어 의미 매칭 부정확은 sub-5 (EmbeddingBridge) 가 자동 해결 → L3.

**Acceptance Criteria** (5종 CLI 기능):

1. **Batch ingest** — 파일/디렉토리 일괄 처리
   ```bash
   python -m htp.knowledge ingest --file notes.md --source diary
   python -m htp.knowledge ingest --dir ~/Documents/Obsidian/Daily --source obs
   ```
2. **Stdin pipe**
   ```bash
   echo "오늘의 통찰..." | python -m htp.knowledge ingest --source 일기
   cat report.txt | python -m htp.knowledge ingest --source report
   ```
3. **Source / time / tag filter**
   ```bash
   python -m htp.knowledge query "패턴 인출" --source brain --since 2026-04
   python -m htp.knowledge discover --threshold 0.05 --since 2026-04
   ```
4. **Edit / delete / tag** (현재 append-only 한계 보완)
   ```bash
   python -m htp.knowledge delete --id 7
   python -m htp.knowledge tags --add "memory,distributed" --id 4
   ```
5. **Export**
   ```bash
   python -m htp.knowledge export --format markdown > knowledge.md
   python -m htp.knowledge export --format json --source brain > brain.json
   ```

**검증 시나리오**: `tests/실사용 테스트.md` 에 시드 작성 (사용자 매일 사용 use case 추가 가능).

**연관 follow-up 분기**:
- L3 (sub-5 EmbeddingBridge 완료 시 자동 도달) — 한국어 형태소 분석 + fit 불필요
- L4 (별도 cycle `htp-knowledge-integration`) — Obsidian / LLM / shell 통합

### Phase B (MED)
- **MED-3**: `static/index.html` 대시보드 BrainRuntime/Memory 반영
- ~~MED-4~~: LLMNode/CostRouter 사용 흐름 (htp-thalamus-car sub-4 에 흡수됨)

### Phase C (LOW)
- **LOW-5**: Friston B3 precision [0.1, 5.0] 5배 증폭 영향 시뮬레이션
- **LOW-6**: NGE split 파라미터 장기 시뮬레이션 (`maturity_calls` / `global_cooldown` / `max_gen_per_run`)
- **LOW-7**: Memory CUSUM × SWR threshold 0.5 경계 분석
- **LOW-8**: `compress_dim = 64` JL Lemma (N=1000 가정) 재검토

### Pre-existing
- `PruneStrategy` 가 `htp/__init__.py` 최상위 export 누락 (legacy 호환 경로 동작 중이라 무영향)

---

## 🔬 회귀 데이터 보존 (Stage 6 A/B 비교용)

다음 파일들은 `.htp/` 에 있고 `.gitignore` 됨 (의도적). **삭제 금지**:

```
.htp/knowledge_log.jsonl                 7 entries (옵션 A-2 재생성, 동일 임베딩 공간)
.htp/encoder_state.pkl                   pickle 영속화 (vocab + JL matrix, 1487 bytes)
.htp/knowledge_log.pre-optionA.jsonl     백업: 패치 전 (음수 mix 불일치)
.htp/knowledge_log.optionA-inmemory.jsonl 백업: 인메모리 옵션 A 만
.htp/knowledge_viz.png                   PCA 2D scatter + heatmap
.htp/knowledge_graph.html                graphify 인터랙티브 그래프
.htp/knowledge_graph.json                graphify 표준 포맷
```

**용도**: sub-5 (Stage 6 EmbeddingBridge) 완료 후 동일 7 entry 로 A/B 비교 — TF-IDF MVP vs sLLM 임베딩의 cross-language 매칭 품질 정량 측정.

**재현 시나리오** (필요시):
```bash
source .venv/bin/activate
rm -f .htp/knowledge_log.jsonl  # 초기화하려면

python -m htp.knowledge ingest --source brain \
    "content addressable memory pattern recall by content distributed representation"
python -m htp.knowledge ingest --source ai \
    "Hopfield network pattern recall by content energy minimization"
python -m htp.knowledge ingest --source infra \
    "Redis key value lookup database protocol"
python -m htp.knowledge discover --threshold 0.05

# 기대: brain↔ai 0.53 > brain↔infra 0.14
```

---

## 🧪 테스트 baseline (Critical Gap #3 RESOLVED 후)

```bash
source .venv/bin/activate
pytest tests/regression/ tests/unit/ tests/knowledge/ -q
# 기대: 118 passed (1.30s)
# - regression: 57
# - unit: 53 (config_isolation 15 + engine_di 16 + import_paths 10 + no_circular_deps 12)
# - knowledge: 8 (test_loop — 5 기존 + 3 신규 Gap #3 회귀 보호)
```

⚠️ sub-2 모든 step 직후 이 명령 통과해야 함. 깨지면 즉시 롤백.

---

## 📁 핵심 문서 경로

| 문서 | 경로 |
|------|------|
| **TODO (이 파일)** | `TODO.md` |
| 프로젝트 컨텍스트 | `CLAUDE.md` |
| 현재 사이클 Plan | `docs/01-plan/features/htp-thalamus-car.plan.md` (Rev 0.2) |
| 현재 사이클 Design (sub-1) | `docs/02-design/features/htp-thalamus-car.design.md` |
| sub-1 Check 보고 | `docs/03-analysis/htp-thalamus-car.analysis.md` |
| 설계서 원본 v4 | `htp_thalamus_car_design v4.md` |
| 이전 사이클 리뷰 | `docs/03-review/htp-project-review.md` |
| PDCA 상태 머신 | `.bkit/state/pdca-status.json` |
| Work log | `~/Documents/Obsidian Vault/ai4pkm-vault/Projects/htp/2026-05-17 work-log.md` |

---

## 🎯 다음 세션 시작 시 권장 순서

1. **이 TODO.md 먼저 읽기**
2. `git log --oneline -3` 으로 커밋 확인 (최신: `0bb6aa3` Critical Gap #3 RESOLVED)
3. `pytest tests/regression/ tests/unit/ tests/knowledge/ -q` 로 baseline 확인 (118/118)
4. `/pdca design htp-thalamus-car` (sub-2 Design 진입)
   - Stage 1: Vector Routing (RegionSignature + _gate_vector)
   - Stage 2: Hybrid (_gate_hybrid + α=0.1~0.9 연속성)
   - 누적 테스트 목표: 118 → 131 (Stage 1 +10, Stage 2 +3)
   - `.htp/knowledge_log.jsonl` 의 7 entry 를 vector routing 테스트 데이터로 재활용

---

## 💡 v4 Rev 1.3 "루프를 먼저 닫는다" 원칙 — 실증된 가치

오늘 발견한 Critical Gap #3 가 **8주 후 발견됐다면 Stage 1-7 전체 리워크** 위험이었음.
sub-1 직후 발견했기에 sub-2 진입 *전* 10분 패치로 해결 가능.
설계서가 v3→v4 진화하면서 도입한 이 원칙의 첫 실증 사례.

이후 sub-cycle 진행 시에도 동일 원칙 적용:
- 매 sub-cycle 직후 `discover --threshold 0.05` 등 *실사용 검증*
- 의도하지 않은 비결정성/순서 의존성 즉시 노출
- 후속 sub-cycle 에 반영
