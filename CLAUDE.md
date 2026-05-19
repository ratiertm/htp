# HTP (Hub Topology Programming) — Claude Code Context

이 파일은 Claude Code에서 프로젝트를 이어가기 위한 맥락 문서다.
**Phase 1–4 기본 구현 + Review Feedback Integration (LeCun·Friston·Memory 리뷰) 완료**
상태. 회귀 테스트 57/57 통과.

---

## 프로젝트 요약

뇌의 허브 노드 구조를 프로그래밍 패러다임으로 구현한 시스템.
핵심 아이디어: 개발자가 if/else로 라우팅을 설계하는 게 아니라,
데이터가 흐르면서 허브 구조가 창발하고, 노드가 생기고 죽고 분열한다.

생물학적 근거:
- 헤비안 학습 + Oja's Rule (시냅스 강화 + L2 정규화)
- 미세아교세포 가지치기 (시냅스 제거)
- 신경발생 (새 뉴런 생성 — 해마 치상회)
- 시상 게이팅 (Neuron 2024 — 압축·재구성·트리거)
- NRXN1 신호 → 피질 신경발생 유도 (bioRxiv 2025)
- Global Workspace Theory (PFC working memory + top-down)
- **Friston FEP**: precision-weighted gating (Turrigiano synaptic scaling)
- **해마 L2/L3 메모리**: CA3 pattern completion + CA1 mismatch detection
- **SWR consolidation**: Yang & Buzsáki 2024 (novelty × reward 기반 priority)

---

## 현재 구현 상태 (Phase 1–4 + Review Feedback + Review Improvements 완료)

### 파일 구조 (htp-review-improvements Session 1-3 반영, 2026-05-16)

```
htp/
├── __init__.py                          공개 API
├── core/                                [Phase 1 엔진 분리, DAG: torch만 의존]
│   ├── __init__.py                     PEP 562 lazy NGE loader + sub-config eager export
│   ├── config.py                       HubConfig + PruneConfig + ActivationConfig + HTPConfig facade
│   ├── weight_matrix.py                WeightMatrix (W[u][v] 단일 소유)
│   ├── hub_formation.py                HubFormationEngine (Oja + PageRank)
│   ├── pruning.py                      PruningEngine + PruneStrategy (4 strategies + hub_protect)
│   ├── activation.py                   ActivationEngine + Node + tag/terminal + FIRE_FLOOR
│   └── node_generation_engine.py        NodeGenerationEngine (split / sprout / interpolate)
├── runtime/                             [오케스트레이션, htp/core/* import 가능]
│   ├── htp_runtime.py                   HTPRuntime (≤250줄 — Step 7 SUCCESS) + re-exports
│   ├── _demo.py                        12/12 라우팅 데모 (Step 7 분리)
│   ├── region_runtime.py                HTPRuntime 확장 + precision proxy
│   ├── brain_runtime.py                 PFCRuntime + BrainRuntime + Memory 연동
│   ├── async_brain_runtime.py           비동기 실행
│   └── cortical_connections.py          Region 간 직접 약한 연결
├── thalamus/
│   ├── thalamus.py                      CoreCells+MatrixCells+NGE + JL 64-dim
│   ├── core_cells.py                    Sigmoidal Gate + Hebbian + Homeostatic + precision
│   ├── matrix_cells.py                  Lateral Inhibition + Softmax WTA (overload_bonus 파라미터)
│   ├── nge_trigger.py                   NRXN1 과부하 트리거
│   ├── region_signal.py                 RegionSignal (precision 포함) / ThalamusOutput / Action
│   └── top_down.py                      Softmax prior (temperature 파라미터)
├── memory/                              [Stage 5 신규]
│   ├── __init__.py
│   ├── types.py                         Episode / Pattern / MemoryContext + bytes 헬퍼
│   ├── episode_store.py                 L2 SQLite + SWR novelty × reward
│   ├── pattern_store.py                 L3 Online Hebbian EMA + Go-CLS + CA3
│   └── memory_system.py                 CA3-CA1 + CUSUM 트리거
├── llm/
│   ├── llm_node.py                      LLMNode / MockLLMNode
│   ├── llm_region_runtime.py            LLM 전용 RegionRuntime
│   └── cost_router.py                   API 비용 기반 라우팅
└── knowledge/                          [htp-thalamus-car sub-1: Stage 0.5 MVP]
    ├── __init__.py                     공개 export (TextEncoder, KnowledgeLoop, ...)
    ├── encoder.py                      TextEncoder Protocol + TfidfJLEncoder
    │                                   + save/load pickle 영속화 (Critical Gap #3)
    ├── loop.py                         KnowledgeLoop (ingest/query/discover) + 5 dataclass
    │                                   + 생성자 자동 load / fit 후 자동 save
    ├── persistence.py                  KnowledgeStore (JSONL append-only)
    └── __main__.py                     CLI (ingest/query/discover --threshold)

archive/deprecated_phase1/               [Stage 2-A1 정리]
├── hub_formation_engine.py              구 BCM-like
├── pruning_engine.py                    구 3전략 프루닝
└── activation_engine.py                 구 캐스케이드 엔진

tests/regression/                        [Stage 1 신규, 57개]
├── test_phase1_routing.py               12/12 시맨틱 라우팅
├── test_phase1_hub_formation.py         Oja, PageRank, uneven centrality
├── test_phase1_pruning.py               4전략 + 허브 보호
├── test_phase2_thalamus.py              Thalamus 체인, 64-dim state_vec, WTA
├── test_phase2_nge_split.py             NGE 구조
├── test_phase3_top_down.py              PFC + TopDown 생성
├── test_phase3_cortical_connections.py  CC 활성화
├── test_stage2_a3_homeostatic.py        CoreCells homeostatic + Hebbian 공존
├── test_stage2_a4_overload_bonus.py     MatrixCells 파라미터화
├── test_stage3_precision.py             RegionSignal / RegionRuntime precision
├── test_stage3_b4_softmax_prior.py      TopDownBias Softmax
├── test_stage5_memory.py                L2/L3/MemorySystem 단위
└── test_stage5_integration.py           BrainRuntime+Memory end-to-end

tests/unit/                              [Review Improvements + sub-1 신규, 53개]
├── test_engine_di.py                    Constructor DI 영구 검증 (HFE/PE/AE sub-config)
├── test_config_isolation.py             HTPConfig facade backward-compat 영구 보호 + 7 sub-config
├── test_import_paths.py                 4 import 경로 동일 객체 검증
└── test_no_circular_deps.py             DAG 강제 (htp/core ↔ htp/runtime + knowledge 단방향)

tests/knowledge/                         [htp-thalamus-car sub-1 신규, 8개]
└── test_loop.py                         ingest/query/discover + encoder 영속화 (Gap #3)
```

**현재 테스트 baseline: 222 = regression 57 + unit (DAG 양방향 강제 포함) + knowledge 30+** (~113s, EmbeddingBridge HF 캐시 warm 후)

## htp-conflict-memory recall — 측정 확정 (외부 리뷰 후속, 2026-05-20)

- 외부 리뷰 "Full MERGE GO" 판정 → 컨테이너 실측 결과 **NO-GO**
- 거짓 양성: 현재 시스템 100% (threshold 0.6 무력)
- 튜닝 레이어(threshold/margin/dist-cut) 전부 FAIL — 측정 2
- 결함 위치: threshold 아님. recall key 설계(§9-1.1 trigger-key)
- 발견 A(지표 불일치) 반증 / 발견 B(query-prefix) 확인
- trigger-key = 미검증 LOCK → xfail strict 로 명문화 (`test_trigger_key_recalls_same_conflict_different_surface`)
- 측정 스크립트: `scripts/conflict_recall_fp_eval.py`, `scripts/conflict_recall_remedy_eval.py`
- 3단계 처방(interpretation-key/hybrid/구조화) 측정 대기 중
- **금지** (지시서 §1): threshold 동적조정 / recall key 변경 / "MERGE GO" 근거 머지 / cos-L2 정합 리팩토링
- **차이 발견 + 해결** (2026-05-20 보강):
  - 지시서 §2-1 코드: `enc.encode_query()` + EASY_NEG 2건 → query-prefix 효과로 결함 우회 PASS
  - 시도 1 — KnowledgeLoop.ingest 실 경로 + HARD_NEG: 같은 도메인 연속 ingest 는 conflict<0.12 → escalate=False → `_try_recall_conflict` 자체 호출 안 됨 → 자동 우회
  - **시도 2 (성공)** — MemorySystem.recall_conflict 직접 호출 + passage prefix (encode) + HARD_NEG: **xfail strict 작동** (test_recall_conflict_hard_neg_via_memory_direct). 지시서 측정 1 HARD_NEG FP 6/6 (100%) 와 정확 일치
  - 교훈: 결함은 *escalate 분기* 가 아니라 *MemorySystem.recall_conflict* 자체. 통합 테스트는 escalate 분기 우회 가능 — 단위 테스트가 직접 검증해야
- **Phase 2.5 (2026-05-20)**: keyspace 공간문제 원인 = Y (패러다임 한계) — 단일 모델 raw cos 자동 판정 기준. 단 bge-m3 만 평균마진 +0.0168 (양수) + S1 분리도 0.545 — 부수 신호. 자세히: `docs/05-measure/phase2.5_keyspace_report.md`
- **Phase 2.6 (2026-05-20)**: bge-m3 + 처방 = P2 분포컷 / P3 결합 PASS (TP 2/2, HARD_NEG FP 1/6). **단 in-sample (TP=2 표본 + 같은 NEG 로 컷 산출).**
- **Phase 2.7 (2026-05-20) out-of-sample**: bge-m3 OOS eval **재현율 2/10 (20%) << 80%**. Phase 2.6 PASS 가 과적합이었음을 측정으로 확정. **Phase 3 규모 = 대 확정** (비-임베딩 키 신설 / LLM 라벨 추출). Phase 2.5 "대" 가 다른 경로로 부활. dense embedding 패러다임은 abstraction-level 매칭 불가 (HARD_NEG FP 0/6 완벽한데 TP 도 못 잡음 → 표면 어휘 매칭 한계). 자세히: `docs/05-measure/phase2.7_oos_report.md` (작성 예정)

**Bridge Integration (2026-05-18 신규)**: `htp/knowledge/loop.py` 가 `htp/thalamus` 의
RegionSignature / PairwiseCoherenceGate / VectorRouter 를 직접 사용 — 시스템 A↔B 단방향 연결.
- 연결 1 (§2): source 별 RegionSignature 가 ingest vec 으로 Hebbian EMA 학습.
- 연결 2 (§3): ingest 시 CoherenceGate.bind() 로 충돌 감지 → `IngestResult.coherence_info`.
- 연결 3 (§4): `query(mode="routed")` 가 VectorRouter 로 활성 source 선택 후 검색,
  CLI `--mode compare` 로 flat vs routed A/B 비교.
- DAG: `knowledge → thalamus` 단방향 허용, `thalamus → knowledge` 영구 금지
  (`test_thalamus_does_not_import_knowledge` 가 강제).
- EmbeddingBridge 재검증: Q1/Q3 PASS, Q2 부분 — 시스템 A 가치 검증 성공.

### DAG 의존 방향 (Review Improvements 강제)

```
htp/__init__.py
    ↓
htp/runtime/* ──→ htp/core/*  ──→ torch + dataclasses (단방향)
                    ↑
                node_generation_engine (예외, 향후 분리)
```

`htp/core/*` 는 `htp/runtime/*` 를 import 하지 못한다 (`test_no_circular_deps.py` 가 영구 검증).

### 핵심 클래스

```python
# Phase 1
WeightMatrix             # W[u][v] 단일 소유, 발화 이력 관리
HubFormationEngine       # Oja's Rule + PageRank hub 감지
PruningEngine            # decay / usage / redundancy / age (+ 허브 보호)
NodeGenerationEngine     # split / sprout / interpolate
ActivationEngine         # 캐스케이드 + 시맨틱 배제
HTPRuntime               # Phase 1 통합 오케스트레이터

# Phase 2 + Review Integration
RegionSignal             # + precision: float (Stage 3-B1)
RegionRuntime            # + _rate_history → precision proxy (Stage 3-B2)
CoreCells                # Hebbian + Homeostatic 이중 메커니즘 + precision-weighted gate
MatrixCells              # overload_bonus 파라미터화 (Stage 2-A4)
NGETrigger               # NRXN1 과부하 분열
Thalamus                 # compress_dim=64 (Stage 4)

# Phase 3
PFCRuntime               # Working memory deque(maxlen=7) + EMA + Cosine + Goal
BrainRuntime             # 최상위 — top-down loop + Memory 연동 (Stage 5-C3)
TopDownSignal/TopDownBias  # Softmax prior (Stage 3-B4)
CorticalConnections      # Region ↔ Region 직접 약한 연결

# Phase 4
LLMNode / MockLLMNode
LLMRegionRuntime
CostRouter
AsyncBrainRuntime

# Memory (Stage 5-C1 신규)
Episode / Pattern / MemoryContext
EpisodeStore             # L2 SQLite (WAL mode) + SWR novelty × reward
PatternStore             # L3 Online Hebbian EMA + Go-CLS + CA3 completion
MemorySystem             # CA3-CA1 recall + CUSUM 트리거 (novelty × reward)
```

### 데코레이터

```python
@rt.node              # 함수를 노드로 등록
@tag("success", ...)  # 시맨틱 라우팅 태그
@terminal             # 캐스케이드 종착점
```

---

## 검증된 동작 (회귀 테스트 57/57 통과)

### Phase 1

- 12/12 라우팅 정확도 (`success → to_cache`, `error → to_alert`)
- Oja's Rule 가중치 [0, 1] 유지
- PageRank 합 = 1, 분포 비균등성 확인
- 4전략 가지치기 작동 + 허브 보호

### Phase 2

- BrainRuntime 1 step → Action + TopDownSignal 생성
- RegionSignal 전 필드 정상 반환 (precision 포함)
- state_vec 64-dim (Stage 4)
- MatrixCells Softmax 확률 분포 (합 ≈ 1)

### Phase 3

- `deque(maxlen=7)` working memory 유지
- long_term_goals → TopDownSignal.biases 반영
- CorticalConnections 활성화 후 정상 동작

### Review Feedback

- **Homeostatic**: 과흥분 Region θ↑, 저활성 θ↓, Hebbian 과 공존
- **precision**: 안정 발화 (variance 작음) → precision ↑
- **Softmax prior**: 합 = 1 확률 분포, overlap=0 도 non-zero
- **Memory L2/L3**: save → recall → consolidation 사이클 동작
- **SWR**: priority = novelty × score ≥ 0.5 태깅
- **CA3 completion**: 노이즈 입력 → centroid 수렴
- **Go-CLS**: count ≥ 3 ∧ snr ≥ 1.5 → L3 패턴 승격

---

## Review Feedback Integration 기록

4개 리뷰 문서 (`design/htp_{lecun,friston,memory,multimodal}_review.md`) 의 피드백을 PDCA
사이클로 반영. Plan/Design: `docs/01-plan/`, `docs/02-design/`.

### Stage 완료 요약

| Stage | 내용 | 상태 |
|-------|------|------|
| 0 | CLAUDE.md 재작성 (실제 코드 상태 반영) | ✅ |
| 1 | 회귀 테스트 고정 (`pytest` + 24 base 테스트) | ✅ |
| 2 | LeCun A1–A4 (데드 코드 정리 / is_hub PageRank / Homeostatic / overload_bonus) | ✅ |
| 3 | Friston B1·B2·B4 (precision 필드 / variance proxy / Softmax prior); B3 는 A3 에서 선반영 | ✅ |
| 4 | `compress_dim` 8 → 64 | ✅ |
| 5 | `htp/memory/` 4파일 신규 + BrainRuntime 6곳 연동 | ✅ |
| 6 | CLAUDE.md 최종 갱신 (본 문서) | ✅ |

### Stage 0–5 중 발견 + 수정된 버그 (5종)

Stage 1 회귀 테스트 과정에서 drag-in 버그들이 발견되어 모두 수정됨. 이것들이 없었다면
Review Feedback 자체를 안전하게 반영할 수 없었음:

| # | 위치 | 증상 | 수정 |
|---|------|------|------|
| 1 | `htp_runtime.py:155` Laplacian | `W @ s` — 엣지 방향 반대로 전파 | `W.T @ s` + D_out/D_in 분리 정규화 |
| 2 | `htp_runtime.py:171` Oja index | `outer(fired, signal)` — post/pre 축 반대 | `outer(signal, fired)` + `(post²).unsqueeze(0)` |
| 3 | `htp_runtime.py:664` `_extract` | dict-value 문자열이 split 없이 한 덩어리 → tag 매칭 실패 | 각 값 공백 split → 개별 키워드 |
| 4 | `htp_runtime.py:192` PageRank | in-degree 정규화 (표준은 out-degree) | out-degree 정규화 수정 |
| 5 | `htp_runtime.py:192` PageRank | dangling 노드 rank 누설 | 표준 dangling 재분배 추가 |

증상: `htp_runtime.demo()` 의 공식 데모조차 parse 단계에서 캐스케이드 붕괴. 수정 후 즉각
정상 동작. "12/12 라우팅 정확도" 주장이 버그 수정 후에야 실제로 성립.

### Review 반영 요약 (Stage 2–5)

| 리뷰 지적 | 구현 |
|-----------|------|
| LeCun A1: Hebbian variant 불일치 | `core/hub_formation_engine.py` 를 `archive/deprecated_phase1/` 이동 (구 BCM 데드 코드 정리) |
| LeCun A2: Hub Detection 혼용 | `is_hub` 마스크를 PageRank 기반으로 (`pr * N > hub_pr_threshold=2.5`) |
| LeCun A3: Homeostatic 부재 | CoreCells `update()` 에 Turrigiano synaptic scaling 추가 — Hebbian 과 polarity 상반되는 이중 메커니즘 |
| LeCun A4: MatrixCells 하드코딩 | `overload_bonus` 생성자 인자 (기본 0.2) |
| Friston B1: precision 필드 없음 | `RegionSignal.precision: float = 1.0` |
| Friston B2: precision 계산 없음 | `RegionRuntime.collect_signal()` fire_rate variance 역수 proxy, clamp [0.1, 5.0] |
| Friston B3: Precision-weighted Gate 아님 | `CoreCells.gate()` 에서 `biased_score = precision × score + td_bias` |
| Friston B4: Jaccard → Softmax | `TopDownBias.compute()` 에서 `softmax(overlap_counts / T)`, `Σbiases = 1` |
| Memory: state_vec 8-dim 부족 | `compress_dim = 64` (JL Lemma k ≈ log(1000)/0.1²) |
| Memory: L2/L3 부재 | `htp/memory/` 4파일 신규 |
| Memory: SWR 태깅 | `priority = novelty × score ≥ 0.5` |
| Memory: CA3-CA1 | `complete()` α=0.7 혼합 + `mismatch` L2 거리 → recall 경로 분기 |
| Memory: Online Hebbian EMA | `lr = 1/(count+1)` centroid 점진 업데이트 (배치 K-Means 대신) |
| Memory: Go-CLS | `count ≥ 3 ∧ snr ≥ 1.5` 승격 조건 |

### 결정된 기본값 (Design §12)

| 항목 | 값 |
|------|-----|
| CoreCells `η_heb / η_hom` | 0.05 / 0.02 |
| `hub_pr_threshold` (PageRank × N 기준) | 2.5 |
| precision variance window | `deque(maxlen=10)` |
| precision clamp 범위 | `[0.1, 5.0]` |
| `CA1_MISMATCH_THRESHOLD` | 0.3 |
| SWR priority threshold | 0.5 |
| CA3 completion α | 0.7 |
| Go-CLS min_count / min_snr | 3 / 1.5 |
| Memory 경로 | `./.htp/memory.db`, `./.htp/patterns.json` (DI 가능) |
| 테스트 프레임워크 | `pytest` |

### 해소된 구조 질문 (v0.1 CLAUDE.md → 실제 코드 대조)

| 질문 | 답 |
|------|------|
| Core/Matrix 규칙 vs 학습? | 혼합 — Matrix 규칙, Core는 Hebbian + Homeostatic 이중 메커니즘 |
| Region 간 직접 통신? | YES — `cortical_connections.py` |
| PFC Working memory 크기? | 7 — `deque(maxlen=7)` |
| LLM 노드 비용? | `htp/llm/cost_router.py` |

---

## 향후 작업 (본 통합 이후)

### Phase 3 대상 (LeCun + Friston 중간 우선순위)

- **임베딩 기반 시맨틱 라우팅**: 현재 문자열 tag 매칭 → 의미 벡터 (LeCun #7)
- **Incremental PCA**: 고정 JL Projection → 학습형 압축 (LeCun #8)
- **Lateral Inhibition 국소화**: Global → 유사도 기반 (LeCun #5)
- **기능적 특화 분열**: NGE split 기준 정교화 (LeCun #6)

### Phase 4 대상 (Friston 장기)

- **Predictive Coding**: `PredictiveRegion` — 실제 예측 벡터 생성 + 오차 기반 precision
- **Active Inference**: PFC Variational Free Energy + Expected Free Energy 행동 선택

### 별도 Phase (Multimodal)

- **V-JEPA 방식 ModalEncoder**: LiDAR / Camera / Audio / IMU / Text
- **Cross-modal Fusion Tokens**: Le MuMo JEPA

---

## 테스트 실행

```bash
# venv 세팅 (최초 1회)
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 전체 회귀 테스트
pytest tests/regression/ -v

# 특정 Stage 테스트만
pytest tests/regression/test_stage5_memory.py -v
pytest -m regression
```

---

## 참고 문헌

- Oja (1982) — Oja's Rule / PCA 주성분 수렴
- Brin & Page (1998) — PageRank
- Turrigiano (2008) — Homeostatic synaptic scaling
- Thalamic contributions to consciousness. Neuron 2024
- Thalamic NRXN1-Mediated Neurogenesis. bioRxiv 2025
- Adult neurogenesis improves spatial information. Nature Comm 2024
- Synaptic pruning by microglia. Frontiers 2025
- Global Workspace Theory (Baars 1988, Dehaene 2003)
- Biased Competition (Desimone & Duncan 1995)
- Recurrent Independent Mechanisms (Goyal & Bengio 2021)
- Friston (2010) — Free Energy Principle
- Yang & Buzsáki (2024) — SWR novelty × reward
- Go-CLS Framework. Nature Neuroscience 2023
- LeCun et al. (2023) — V-JEPA, Le MuMo JEPA
- Hopfield (1982), Kanerva (1988) — Sparse Distributed Memory

---

## 작업 지시

현재 진행 중인 작업은 없음. 새로운 작업을 시작하려면:

```
/pdca status                       # 현재 상태 확인
/pdca plan <feature>               # 새 기능 계획
```

과거 통합 작업의 상세:

- `docs/01-plan/features/htp-phase2-integration.plan.md` — 전체 범위·근거
- `docs/02-design/features/htp-phase2-integration.design.md` — Stage별 상세 설계
- `design/htp_{lecun,friston,memory,multimodal}_review.md` — 원본 리뷰 문서들

Phase 1–4 + Memory 구현 맥락이 필요할 때는 `htp/` 트리 + `tests/regression/` 이 single
source of truth.
