# HTP — TODO / Handoff

**Last updated**: 2026-05-17
**Last commit**: `6291cbe` (origin/master push 완료)
**Current PDCA**: `htp-thalamus-car` sub-1 완료 → sub-2 진입 대기

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

## ⚠️ sub-2 진입 전 결정 필요: Critical Gap #3

**현상**: `encoder.fit()` 재실행 시 `GaussianRandomProjection` 이 새 random matrix 로 refit → 누적 cache vec 의 의미 공간이 흔들림.

**증거**: 한국어 cross-language 실험에서 brain↔infra 0.61 > brain↔ai 0.53 으로 역전 발생 (영문 단독 실험에선 정반대).

**3가지 해결 옵션**:

| 옵션 | 변경 | 장점 | 단점 |
|------|------|------|------|
| **A** | `encoder.fit()` 1회 호출 후 freeze. ingest 시 corpus 갱신 안 함 | 결정적, 안정적 | 새 어휘 미반영 |
| **B** | 매 ingest 마다 *cache 전체* re-encode | 현재 시도 중 | random_state 효과 미흡, 결과 여전히 흔들림 |
| **C** | sub-5 (Stage 6 EmbeddingBridge) 앞당김 | 본질 해결 | 일정 변경, sLLM 의존성 추가 |

**권장**: sub-2 진입 전 옵션 A 적용 (10분 작업). 이후 sub-5 에서 옵션 C 로 정식 해결.

```python
# 옵션 A 패치 예시 (htp/knowledge/loop.py)
def ingest(self, text: str, source: str = "") -> IngestResult:
    # AS-IS: 매번 fit
    # corpus = [e.text for e in self._cache] + [text]
    # self.encoder.fit(corpus)

    # TO-BE: 최초 1회만 fit
    if not self.encoder._fitted:
        corpus = [e.text for e in self._cache] + [text]
        self.encoder.fit(corpus)
    # ... rest unchanged
```

---

## 📊 현재 PDCA 진행 상태

```
htp-thalamus-car (9 Stage, 6 sub-cycles)
├─ sub-1 ✅ [Plan ✅ Design ✅ Do ✅ Check ✅ 98%]   Stage 0 + 0.5
├─ sub-2 ⏳ pending                                  Stage 1 + 2
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
.htp/knowledge_log.jsonl     7 entries (영문 3 + 한국어 3 + bilingual 1)
.htp/knowledge_viz.png       PCA 2D scatter + heatmap (현 상태 baseline)
.htp/knowledge_graph.html    graphify 인터랙티브 그래프
.htp/knowledge_graph.json    graphify 표준 포맷
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

## 🧪 테스트 baseline (sub-1 종료 시점)

```bash
source .venv/bin/activate
pytest tests/regression/ tests/unit/ tests/knowledge/ -q
# 기대: 115 passed
# - regression: 57 (이전 사이클 + 그 이전)
# - unit: 49 (test_config_isolation 15 + test_engine_di 16 + test_import_paths 10 + test_no_circular_deps 8)
# - knowledge: 5 (test_loop)
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
2. `git log --oneline -3` 으로 커밋 확인
3. `pytest tests/regression/ tests/unit/ tests/knowledge/ -q` 로 baseline 확인 (115/115)
4. **결정**: Critical Gap #3 옵션 A (10분 패치) 먼저 적용할지, 아니면 sub-2 진입하며 같이 처리할지
5. `/pdca design htp-thalamus-car` 또는 직접 `/pdca do htp-thalamus-car --scope stage-1`

---

## 💡 v4 Rev 1.3 "루프를 먼저 닫는다" 원칙 — 실증된 가치

오늘 발견한 Critical Gap #3 가 **8주 후 발견됐다면 Stage 1-7 전체 리워크** 위험이었음.
sub-1 직후 발견했기에 sub-2 진입 *전* 10분 패치로 해결 가능.
설계서가 v3→v4 진화하면서 도입한 이 원칙의 첫 실증 사례.

이후 sub-cycle 진행 시에도 동일 원칙 적용:
- 매 sub-cycle 직후 `discover --threshold 0.05` 등 *실사용 검증*
- 의도하지 않은 비결정성/순서 의존성 즉시 노출
- 후속 sub-cycle 에 반영
