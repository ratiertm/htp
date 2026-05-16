# HTP 프로젝트 검토 정리 (Review Brief)

기준 시점: 2026-05-16 · 커밋 `6be8746` · 회귀 테스트 57/57 통과 · 코드 ~4,406줄

---

## 1. 설계 의도 (Design Intent)

**한 줄 정의**: 뇌의 허브-토폴로지 원리를 프로그래밍 패러다임으로 구현 — "개발자가 if/else로 라우팅을 짜는 게 아니라, 데이터가 흐르며 허브가 창발하고 노드가 생기고 죽고 분열한다."

**4대 원칙** (`architecture/htp_architecture_design.md` §1.2)
1. 구조는 데이터가 만든다 — 사전 라우팅 설계 없음
2. 허브는 창발한다 — 자주 쓰이는 노드가 자동 승격
3. 네트워크는 살아있다 — 신경발생 / 가지치기
4. 판단은 위임한다 — LLM/Agent를 노드로 사용

**생물학적 근거** (CLAUDE.md 인용)
- Hebbian + Oja's Rule (시냅스 강화 + L2 정규화)
- 미세아교세포 가지치기 / NRXN1 신경발생 (bioRxiv 2025)
- 시상 게이팅 (Neuron 2024) / Global Workspace Theory
- Friston FEP (precision-weighted gating) / 해마 CA3-CA1 / SWR consolidation (Yang & Buzsáki 2024)

---

## 2. 아키텍처 (6-Layer)

```
입력 → Region Layer (N개 HTPRuntime)
     → Thalamus (CoreCells + MatrixCells + NGETrigger + TopDownBias)
     → PFCRuntime (WM=deque[7] + Attention + Goal Alignment)
     → Action  ⤴ Top-down Feedback Loop
                   ⤴ Cortical Connections (Region↔Region 약한 직접 연결)
                   ⤴ Memory L2/L3 (CA3-CA1 + SWR + Go-CLS)
```

| 계층 | 핵심 클래스 | 역할 | God-node 엣지 |
|------|------------|------|---------------|
| **Phase 1 (영역 내부)** | `HTPRuntime` = `WeightMatrix` + `HubFormationEngine` + `PruningEngine` + `ActivationEngine` + `NodeGenerationEngine` | 4-engine 통합 단일 피질 영역 | HTPRuntime: 37 / WeightMatrix: 34 |
| **Phase 2 (다중 영역 + 시상)** | `RegionRuntime` (HTPRuntime 상속) · `Thalamus`(`CoreCells`, `MatrixCells`, `NGETrigger`) · `RegionSignal` | Region → 시상 → 통합 게이팅 | **RegionRuntime: 59 (backbone)** · RegionSignal: 32 |
| **Phase 3 (PFC + Top-down)** | `PFCRuntime` (WM deque[7] + Scaled Dot-Product Attention) · `TopDownBias` (Softmax prior) · `CorticalConnections` | 장기목표 정렬 + 역방향 바이어스 | TopDownSignal: 32 / BrainRuntime: 37 |
| **Phase 4 (LLM-as-Node)** | `LLMNode` · `MockLLMNode` · `LLMRegionRuntime` · `CostRouter` (EMA 비용압박) · `AsyncBrainRuntime` (asyncio.gather) | LLM API 호출을 노드로 추상화 | (Community 6 격리도 높음) |
| **Phase 5 (Memory)** | `MemorySystem` (CA3-CA1 + CUSUM) · `EpisodeStore` (L2 SQLite + SWR) · `PatternStore` (L3 Hebbian EMA + Go-CLS) | 에피소드 저장 → 패턴 승격 | MemorySystem: 33 |
| **Config** | `HTPConfig` | 모든 구성요소 의존 | 41 엣지 ⚠️ (god-object 후보) |

---

## 3. 디자인 문서 (Review Feedback)

`design/htp_{lecun,friston,memory,multimodal}_review.md` 4개 리뷰가 PDCA로 코드에 반영됨.

### 3-A. 반영 완료 (Stage 1–5, 5종 버그 동시 수정)

| 영역 | 리뷰 지적 | 구현 결과 |
|------|----------|----------|
| **LeCun A1** | Hebbian variant 불일치 (BCM 데드 코드) | `archive/deprecated_phase1/` 이동 |
| **LeCun A2** | Hub Detection 혼용 | `is_hub` PageRank 기반 (`pr × N > 2.5`) |
| **LeCun A3** | Homeostatic 부재 | CoreCells에 Turrigiano synaptic scaling 추가 — Hebbian과 이중 메커니즘 공존 |
| **LeCun A4** | MatrixCells 하드코딩 | `overload_bonus` 파라미터화 (기본 0.2) |
| **Friston B1** | precision 필드 없음 | `RegionSignal.precision: float = 1.0` |
| **Friston B2** | precision 계산 부재 | `RegionRuntime.collect_signal()` fire_rate variance 역수 proxy, clamp [0.1, 5.0] |
| **Friston B3** | gate 가중치 없음 | `CoreCells.gate()` `biased_score = precision × score + td_bias` |
| **Friston B4** | Jaccard → Softmax | `TopDownBias.compute() = softmax(overlap/T)`, Σbiases=1 |
| **Memory** | state_vec 8-dim 부족 | JL Lemma → `compress_dim = 64` |
| **Memory** | L2/L3 부재 | `htp/memory/` 4파일 신규 (Episode/Pattern/MemorySystem/types) |
| **Memory** | SWR 태깅 | `priority = novelty × score ≥ 0.5` |
| **Memory** | CA3-CA1 | `complete()` α=0.7 혼합 + `mismatch` L2 거리 분기 |
| **Memory** | Online Hebbian EMA | `lr = 1/(count+1)` centroid 점진 업데이트 |
| **Memory** | Go-CLS 승격 | `count ≥ 3 ∧ snr ≥ 1.5` → L3 승격 |

### 3-B. 부수 산물: 회귀 테스트 중 발견한 5종 버그 (단순 typo가 아닌 알고리즘 버그)

1. Laplacian: `W @ s` → `W.T @ s` (엣지 방향 반대 전파)
2. Oja index: `outer(fired, signal)` → `outer(signal, fired)` (post/pre 축 반대)
3. `_extract` dict-value split 누락 → tag 매칭 전부 실패
4. PageRank: in-degree → out-degree 정규화 (표준)
5. PageRank: dangling 노드 rank 누설 → 표준 재분배 추가

**의미**: 리뷰 반영 자체보다, *공식 데모(`htp_runtime.demo()`)조차 캐스케이드 붕괴 상태였던 것을 회귀 테스트가 발견했다*는 것이 더 큰 가치. "12/12 라우팅 정확도"는 버그 수정 후에야 진짜로 성립.

### 3-C. 미반영 (향후 Phase 3-4 후보)

- 임베딩 기반 시맨틱 라우팅 (현재 문자열 tag 매칭)
- Incremental PCA (현재 JL 고정 압축)
- Lateral Inhibition 국소화 (현재 Global)
- Predictive Coding `PredictiveRegion` + Active Inference (Friston 장기)
- V-JEPA 방식 멀티모달 ModalEncoder

---

## 4. 구현 현황

### 4-A. 파일 구조 (4,406줄)

```
htp/
├── __init__.py                  공개 API 93줄 (Phase 1-5 통합 export)
├── core/node_generation_engine  689줄 — split/sprout/interpolate
├── runtime/
│   ├── htp_runtime.py            967줄 — WeightMatrix + 3 engines + 데코레이터
│   ├── brain_runtime.py          412줄 — PFC + BrainRuntime + Memory 연동
│   ├── region_runtime.py         188줄 — HTPRuntime 확장 + precision proxy
│   ├── cortical_connections.py    91줄 — Region↔Region 약한 연결
│   └── async_brain_runtime.py    127줄 — asyncio.gather + SLA
├── thalamus/                     727줄 (6 파일)
├── memory/                       663줄 (4 파일)
└── llm/                          421줄 (4 파일)

tests/regression/                 57 테스트 / 14 파일
archive/deprecated_phase1/        BCM 변형 데드코드 보존 (Stage 2-A1)
```

### 4-B. 공개 API 표면 (`htp/__init__.py`)

Phase별 깔끔히 분리된 export — 5개 Phase × 평균 6개 심볼.

### 4-C. 회귀 테스트 매트릭스 (57/57 통과)

| Phase | 테스트 | 검증 대상 |
|-------|--------|----------|
| Phase 1 | routing(12) + hub_formation + pruning | 12/12 라우팅, Oja 경계, PageRank 합=1, 4전략 + 허브 보호 |
| Phase 2 | thalamus + nge_split | RegionSignal 전 필드, state_vec=64-dim, Softmax 합≈1 |
| Phase 3 | top_down + cortical_connections | WM maxlen=7, biases 반영, CC 정상 동작 |
| Stage 2-A3 | homeostatic | 과흥분→θ↑, 저활성→θ↓, Hebbian 공존 |
| Stage 2-A4 | overload_bonus | 파라미터 0/기본/높음 분기 |
| Stage 3-B1/B2 | precision | 안정 발화 → precision↑, clamp [0.1, 5.0] |
| Stage 3-B4 | softmax_prior | 합=1, overlap=0도 non-zero, temperature 효과 |
| Stage 5 | memory + integration | L2 SWR, L3 Go-CLS, CA3 completion, CUSUM 트리거 |

---

## 5. 검토 관점에서의 강점 / 약점 / 위험

### 강점

1. **이론 → 구현 매핑이 단순 어휘적이 아니라 수식적** (Oja, PageRank, Softmax, JL Lemma, Hebbian EMA, CUSUM, SWR priority 모두 회귀 테스트로 검증)
2. **회귀 테스트가 진짜로 작동** — 리뷰 반영 과정에서 5종 알고리즘 버그를 추가 발견
3. **Phase별 공개 API 분리**가 깔끔 (`__init__.py` 5개 섹션)
4. **Deprecated 코드 보존** (`archive/`) — 의사결정 추적 가능
5. **단일 소유 원칙** — `WeightMatrix` 하나가 모든 시냅스, 3 엔진은 참조만

### 약점 / 코드 냄새

1. **`RegionRuntime` 59 엣지** — graphify 분석상 betweenness centrality 0.166 (전체 최고). "Region IS-A HTPRuntime"이라는 한 줄 상속이 6 커뮤니티를 연결시키는 backbone 역할. 의도된 설계지만 단위 테스트가 거의 풀 스택을 띄워야 함 (test_stage3_precision.py)
2. **`HTPConfig` 41 엣지** — 모든 구성요소가 단일 Config에 의존. DI 리팩토링 후보
3. **`htp_runtime.py` 967줄** — Phase 1 4 엔진이 한 파일에 모여 있음. 추후 분리 후보
4. **`static/index.html` 대시보드**는 Phase 1 `HTPRuntime`만 노출 — Phase 2-5(BrainRuntime, Memory) 미반영
5. **`LLMNode`/`CostRouter`는 그래프상 isolated 경향** (176개 isolated 노드 중 다수) — 실제 사용 흐름이 약함

### 위험 (Reviewer가 확인해야 할 것)

1. **Friston B3 precision-weighted gate**: `biased_score = precision × score + td_bias`에서 precision이 [0.1, 5.0]으로 5배까지 증폭 가능. winner 결정에 미치는 영향 시뮬레이션 필요
2. **NGE split immature 쿨다운** (`maturity_calls`, `global_cooldown`, `max_gen_per_run=1`) 파라미터의 안정성은 단위 테스트만 있고 장기 시뮬레이션 부재
3. **Memory consolidation 트리거** (CUSUM novelty × reward)이 SWR priority threshold 0.5와 어떻게 상호작용하는지 통합 테스트는 있으나 경계 분석 부족
4. **`compress_dim = 64`** 가 JL Lemma `k ≈ log(N)/ε²` 기준이지만 N=1000 가정. Region 수가 늘어나면 재검토 필요

---

## 6. 미해결 설계 질문 (architecture §9, 일부 해소됨)

| 질문 | 현 상태 |
|------|--------|
| Thalamus 자체가 학습해야 하는가? | **부분 해소** — CoreCells는 Hebbian + Homeostatic 이중 학습, MatrixCells는 여전히 규칙 |
| Region 간 직접 통신 허용? | **해소** — `CorticalConnections` (Phase 3) |
| PFC WM 크기? | **해소** — `deque(maxlen=7)` (Miller 1956) |
| LLM 비용 vs 성능 트레이드오프? | **해소** — `CostRouter` EMA 압박 + 모델 다운그레이드 |
| 4D 시각화 = 개발도구? | **미해소** — `static/index.html`만 존재, Phase 1만 노출 |

---

## 7. 검토자가 바로 확인할 수 있는 진입점

```bash
# 1) 전체 테스트 (1분)
source .venv/bin/activate && pytest tests/regression/ -v

# 2) 데모 (1 step end-to-end)
python -c "from htp import BrainRuntime; print(BrainRuntime().run('hello'))"
python -m htp.runtime.htp_runtime  # Phase 1 routing demo

# 3) 핵심 설계서
docs/01-plan/features/htp-phase2-integration.plan.md      # 전체 범위·근거
docs/02-design/features/htp-phase2-integration.design.md  # Stage별 상세
design/htp_{lecun,friston,memory,multimodal}_review.md    # 원본 리뷰

# 4) 인터랙티브 의존 그래프
open graphify-out/graph.html   # 718 노드 · 27 커뮤니티 · god/surprise 표시
graphify-out/GRAPH_REPORT.md   # 감사 가능한 보고서
```

---

## 부록 A. Knowledge Graph 요약 (graphify-out/)

- **규모**: 58 파일 · 47,511 단어 → **718 노드, 1,741 엣지, 27 커뮤니티**
- **추출 신뢰도**: EXTRACTED 53% · INFERRED 47% · AMBIGUOUS 0%
- **토큰 절감**: 5.8× (corpus 직접 읽기 대비)

### God Nodes Top 10

| # | 노드 | 엣지 | 의미 |
|---|------|-----|------|
| 1 | `RegionRuntime` | 59 | 진짜 backbone — Region·Thalamus·Memory를 잇는 단일 피질 영역 추상 |
| 2 | `HTPConfig` | 41 | 설정 의존성이 모든 곳에 퍼짐 (코드 냄새 후보) |
| 3 | `HTPRuntime` | 37 | Phase 1 오케스트레이터 |
| 4 | `BrainRuntime` | 37 | 최상위 — top-down loop + Memory 연동 |
| 5 | `WeightMatrix` | 34 | 단일 소유 시냅스 행렬 |
| 6 | `MemorySystem` | 33 | CA3-CA1 + CUSUM 트리거 |
| 7 | `TopDownSignal` | 32 | PFC → Region 게이팅 신호 |
| 8 | `RegionSignal` | 32 | Region → Thalamus 통신 단위 (precision 포함) |
| 9 | `HubFormationEngine` | 32 | PageRank 허브 감지 |
| 10 | `PruningEngine` | 31 | 4-strategy 가지치기 |

### Surprising Connections

1. **`NGE hub_split` ↔ `LeCun #6 functional specialization split (2-means)`** — 구현이 LeCun 비평 #6과 의미적으로 거의 동일. LeCun 리뷰가 NGE 분열 전략의 사후 정당화로 작용한 흔적.
2. **`old hub_formation_engine (BCM-like)` ↔ `LeCun #1 Oja's Rule unification`** — Deprecated 코드가 LeCun #1이 비판한 정확한 그 BCM 변형. archive 결정이 리뷰 결과로 사후 검증됨.
3. **`HTP Architecture Design v0.1` → `HTPRuntime v2 (4-engine)`** [rationale_for] — design 문서가 실제 4-engine 구현의 근거.
4. **`HTP 3D Dashboard frontend (static/index.html)` → `HTPRuntime`** — 프론트엔드가 Phase 1 런타임만 참조 (Phase 2-3 BrainRuntime을 노출하지 않음 → 대시보드 갱신 필요).

### Hyperedges (구조적 그룹)

- **Friston FEP 구현 그룹** (7 노드): B1 precision field · B2 dynamic · B3 gate · B4 softmax · VFE · PredictiveRegion · EFE
- **LeCun 8-item 비평 세트** (8 노드): Oja unification, PageRank hub, Homeostatic, Overload bonus, Local inhibition, Functional split, Embedding routing, Incremental PCA
- **Memory L2/L3 컴포넌트** (Episode + Pattern + MemorySystem + types)
- **Thalamus 합성** (Core + Matrix + NGE + TopDown)
- **HTP 4-engine 런타임** (WeightMatrix + HFE + PE + AE)
- **Phase 1 / Phase 2-3 / Stage 2-3-5 회귀 테스트 슈트** (57개 테스트 그룹)

### Knowledge Gap

- **176개 isolated 노드** — LLMNode/CostRouter/일부 dataclass가 그래프에 약하게만 연결됨. 실제 사용처 부족 또는 문서화 미흡.

---

## 부록 B. RegionRuntime backbone 분석 (왜 6 커뮤니티를 연결하는가)

**Short answer**: `RegionRuntime`은 *unit of cortical work*. 모든 아키텍처 개념이 "Region이 무엇이거나, 포함하거나, 생성하거나, 감싸는 대상"으로 모델링됐기 때문에, 결국 Region을 거치게 됨. 59 엣지, betweenness 0.166 (그래프 전체 최고).

### 6 커뮤니티 브리지 메커니즘

| 커뮤니티 | 연결 방식 | 그래프 증거 |
|---------|---------|-----------|
| **C1 — BrainRuntime + Integration (home)** | `inherits HTPRuntime` + `BrainRuntime` 내부 `{region_name: RegionRuntime}` 레지스트리 | 32 neighbors — `implements HTPRuntime`, `uses PFCRuntime`, `uses HTPConfig` |
| **C0 — Activation Engine + NGE** | Region이 자신의 `NodeGenerationEngine`을 소유 — 신경발생이 *Region별* 발생 | `uses NodeGenerationEngine`, `method .run()`, `method ._ensure_built()` |
| **C4 — CoreCells + Thalamus Gating** | `collect_signal()` → `RegionSignal` 생성; `NGETrigger`와 데이터 공유 | `method .collect_signal()`, `uses RegionSignal`, `shares_data_with NGETrigger` |
| **C6 — CostRouter + LLM** | `LLMRegionRuntime`이 `RegionRuntime`을 **상속** — LLM은 `.run()`만 다른 Region | `implements LLMRegionRuntime`, `semantically_similar_to CostRouter` |
| **C9 — HTP Runtime + Cortical Connections** | `region_runtime.py`가 runtime 패키지에 위치; TopDown signal + NGE Trigger 모듈 사용 | `contains region_runtime.py`, `uses Top-Down Signal`, `uses NGE Trigger` |
| **C2 — Activation Internals (tests)** | Stage 3-B1/B2 precision 테스트가 `RegionRuntime._entropy_concentration()` + `.collect_signal()` 타겟 | `calls test_region_runtime_precision_clamped()` |

### 의도된 설계 vs 우연한 결합

1. **`RegionRuntime inherits HTPRuntime`** — 하나의 상속 엣지가 4-engine을 자동 상속시켜 커뮤니티 0, 8, 9 동시 연결.
2. **`BrainRuntime`이 N개 `RegionRuntime` 합성** — `regions: {region_name: RegionRuntime}` 딕셔너리가 Thalamus(C4), TopDown(C12), NGE Trigger(C4) 모두에서 참조됨.
3. **`collect_signal()`이 Region→Thalamus 컨트랙트** — `RegionSignal`(precision 포함) 생산. C4와 C16(RegionSignal+Precision)이 모두 Region에 의존.
4. **`LLMRegionRuntime` 상속** — LLM API 호출을 "또 하나의 Region"으로 추상화 → C6 부착.

### Verdict

- **좋은 설계**: cortical region을 합성 가능한 단위로 추상화. HTPRuntime 상속은 정당화됨 (Region IS-A 4-engine runtime), 나머지는 composition.
- **위험**: 59 엣지 = god-object 후보. Region을 단위 테스트하려면 HTPRuntime + Thalamus + NGE + Memory + Top-Down을 거의 다 띄워야 함.
- **재확인 필요**: `LLMRegionRuntime` 상속 — LLM-as-Region이 깔끔한 추상인지, leaky abstraction인지 (예: LLM 노드에 PageRank 허브 형성이 정말 필요한가?)
