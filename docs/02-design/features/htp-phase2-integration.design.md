---
template: design
version: 1.2
feature: htp-phase2-integration
date: 2026-04-14
author: Mindbuild
project: HTP (Hub Topology Programming)
status: Draft
---

# HTP Review Feedback Integration — Design Document

> **Summary**: Plan v0.3에서 확정된 Stage 0~6의 각 수정 포인트를 파일:라인 수준으로 명세한다.
>
> **Project**: HTP
> **Version**: Phase 1–4 구현 완료 → Review Feedback 통합
> **Author**: Mindbuild
> **Date**: 2026-04-14
> **Status**: Draft
> **Planning Doc**: [htp-phase2-integration.plan.md](../../01-plan/features/htp-phase2-integration.plan.md)

---

## 1. Overview

### 1.1 Design Goals

1. **리뷰 지적사항 8건(L1–L4, F1–F4) + Memory(C1–C4)를 최소 침습 수정으로 반영** — Phase 1–4 회귀 없음
2. **anti-homeostatic Hebbian + homeostatic 안정화의 이중 메커니즘**을 수학적으로 균형 있게 결합
3. **precision-weighted gating** 도입으로 Friston FEP 원칙을 기존 Sigmoidal Gate에 직교 합성
4. **L2(SQLite) + L3(Online Hebbian EMA)** 메모리 시스템 구축, CA3-CA1 양방향 recall 복원
5. **state_vec 8→64 차원 전환**의 파급효과(코사인 임계값, 메모리 저장 포맷) 일괄 관리

### 1.2 Design Principles

- **국소 수정 우선**: 대부분 단일 파일 수정. 크로스 컷 수정은 Stage 4(차원 전환), Stage 5(Memory 연동)에 집중
- **리뷰 원문 수식 충실 재현**: Oja's Rule, PageRank, Softmax prior, novelty×reward SWR 등 논문 수식 그대로 구현
- **이중 메커니즘 직교성**: 기존 구현(Hebbian 강화, Jaccard bias 등)을 덮어쓰지 않고 **새 term과 선형/곱 결합**하여 공존
- **테스트 우선**: 각 Stage 시작 전 회귀 테스트가 **통과 상태**여야 진입

---

## 2. Architecture

### 2.1 수정 대상 컴포넌트 맵

```
htp/
├── core/
│   └── hub_formation_engine.py          [Stage 2-A1] 제거 또는 데모 격리
├── runtime/
│   ├── htp_runtime.py                   [Stage 2-A2] is_hub PageRank 전환 (L.176-177)
│   ├── region_runtime.py                [Stage 3-B2] precision 계산 추가 (L.99-136)
│   └── brain_runtime.py                 [Stage 5-C3] memory 연동 6곳
├── thalamus/
│   ├── region_signal.py                 [Stage 3-B1] precision 필드 (L.23-31)
│   ├── core_cells.py                    [Stage 2-A3 + Stage 3-B3] homeostatic + precision gate (L.105-132)
│   ├── matrix_cells.py                  [Stage 2-A4] overload_bonus 파라미터화 (L.44-55, 75)
│   ├── thalamus.py                      [Stage 4-C2] compress_dim 8→64 (L.52)
│   └── top_down.py                      [Stage 3-B4] Softmax prior (L.79-80)
└── memory/                              [Stage 5-C1] 신규
    ├── __init__.py
    ├── types.py                         Episode, Pattern, MemoryContext
    ├── episode_store.py                 L2 SQLite + SWR 태깅
    ├── pattern_store.py                 L3 Online Hebbian + Go-CLS + CA3
    └── memory_system.py                 CA3-CA1 통합 + CUSUM 트리거
```

### 2.2 Data Flow (Stage 5 완료 후)

```
┌─ BrainRuntime.run(data) ───────────────────────────────────┐
│                                                            │
│ ① (step>1) memory.recall(prev_state_vec)                  │
│    └→ CA3 complete → CA1 mismatch → rec_winner             │
│    └→ inject into TopDownSignal (bias boost)               │
│                                                            │
│ ② for region in regions: region.run(data)                  │
│    └→ collect_signal() returns RegionSignal(+precision)    │
│                                                            │
│ ③ thalamus.step(data, top_down=last_td+mem_hint)          │
│    └ CoreCells.gate(): precision×(score+td_bias) + theta_adapt  │
│    └ MatrixCells.compete(): +overload_bonus param          │
│    └ JL compress → 64-dim state_vec                        │
│                                                            │
│ ④ pfc.decide(thal_out) → (Action, TopDownSignal)          │
│    └ TopDownBias.compute(): Softmax(overlap_counts)        │
│                                                            │
│ ⑤ memory.save(thal_out, action, ctx, score)                │
│    └ novelty = 1 - L3.match_confidence(state_vec)          │
│                                                            │
│ ⑥ for region: if cusum_S > cusum_h:                        │
│      memory.on_overload(region_id)                         │
│      └ tag_swr → L3.consolidate (Online Hebbian EMA)       │
│      region._cusum_S = 0                                   │
│                                                            │
│ ⑦ suppression + cortical_connections + winner.result       │
└────────────────────────────────────────────────────────────┘
```

### 2.3 Dependencies

| Stage | 선행 의존 | 산출물 |
|-------|-----------|--------|
| 0 CLAUDE.md 갱신 | — | 재작성된 CLAUDE.md |
| 1 회귀 테스트 고정 | — | `tests/regression/` 에 Phase 1–4 시나리오 고정 |
| 2 LeCun A1–A4 | Stage 1 | Oja 단일화, is_hub PageRank, homeostatic, MatrixCells 파라미터화 |
| 3 Friston B1–B4 | Stage 2-A3 | precision 신호 체인, Softmax prior |
| 4 차원 전환 C2 | Stage 3 | `compress_dim=64`, 영향 임계값 재조정 |
| 5 Memory C1·C3·C4 | Stage 4 | `htp/memory/` 4파일 + BrainRuntime 6곳 |
| 6 통합 테스트 + CLAUDE.md 최종 갱신 | Stage 5 | 최종 PR |

---

## 3. Data Model

### 3.1 RegionSignal 확장 (B1)

```python
# htp/thalamus/region_signal.py (현재 L.23-31)
@dataclass
class RegionSignal:
    region_id:    str
    hub_strength: float
    fire_rate:    float
    top_hubs:     list
    overload:     bool
    output_vec:   torch.Tensor
    precision:    float = 1.0  # [NEW] Friston precision. 1.0=중립, 높을수록 신뢰 ↑
```

**precision 의미**: 해당 Region의 예측 일관성. gating에서 score를 amplification하는 스케일러.

### 3.2 Memory 데이터 구조 (C1 — design 문서 원본 충실)

**Episode** (L2, `htp/memory/types.py`):

| 필드 | 타입 | 설명 |
|------|------|------|
| `episode_id` | str (UUID) | PK |
| `step` | int | BrainRuntime._step |
| `winner` | str | 이긴 Region |
| `action_type` | str | "execute" \| "inhibit" |
| `score` | float | PFC combined score |
| `state_vec` | bytes | 64-dim float32 blob |
| `context` | str | 입력 요약 50자 |
| `outcome` | str \| None | 사후 "success"/"fail" |
| `recall_count` | int | CA1 재활성화 횟수 |
| `novelty` | float | 1 - L3 매칭 신뢰도 |
| `swr_tagged` | bool | consolidation 대상 |
| `session_id` | str | |
| `timestamp` | float | |

**Pattern** (L3, `htp/memory/types.py`):

| 필드 | 타입 | 설명 |
|------|------|------|
| `pattern_id` | str | |
| `centroid_vec` | bytes | 64-dim EMA 중심 |
| `best_winner` | str | 다수 성공 Region |
| `success_rate` | float | |
| `episode_count` | int | |
| `winner_dist` | dict[str,int] | Region별 성공 횟수 |
| `snr` | float | μ(scores)/σ(scores) |
| `generalize_ok` | bool | count≥3 ∧ snr≥1.5 |
| `updated_at` | float | |

**MemoryContext** (recall 반환):

```python
@dataclass
class MemoryContext:
    completed_vec:   torch.Tensor   # CA3 완성본
    mismatch:        float          # CA1 불일치 L2 거리
    candidates:      list[Episode]
    recommendation:  str | None     # CA1 best_winner
    confidence:      float
    pattern:         Pattern | None
    is_novel:        bool           # mismatch >= 0.3
```

### 3.3 SQLite Schema (C1)

```sql
CREATE TABLE IF NOT EXISTS episodes (
    episode_id   TEXT PRIMARY KEY,
    step         INTEGER,
    winner       TEXT,
    action_type  TEXT,
    score        REAL,
    state_vec    BLOB,
    context      TEXT,
    outcome      TEXT,
    recall_count INTEGER DEFAULT 0,
    novelty      REAL    DEFAULT 1.0,
    swr_tagged   BOOLEAN DEFAULT FALSE,
    session_id   TEXT,
    timestamp    REAL
);
CREATE INDEX idx_winner    ON episodes(winner);
CREATE INDEX idx_swr       ON episodes(swr_tagged);
CREATE INDEX idx_timestamp ON episodes(timestamp);
CREATE INDEX idx_outcome   ON episodes(outcome);
```

Pattern은 `htp_patterns.json` (JSON 직렬화).

**파일 경로 정책 (Plan §8 질문 2 해결)**: `BrainRuntime(memory_dir: Path = Path(".htp"))` 의존성 주입. 기본값 `./.htp/memory.db`·`./.htp/patterns.json`. `.gitignore`에 `.htp/` 추가.

---

## 4. 핵심 알고리즘 상세

### 4.1 A2 — is_hub PageRank 전환

**기존** (`htp_runtime.py:176-177`):
```python
in_str      = self.wm.W.sum(dim=0)
self.is_hub = in_str > self.cfg.hub_threshold
```

**변경**:
```python
pr          = self.pagerank()          # [N], 합=1
# PageRank는 1/N 중심 분포이므로 임계값을 상대 스케일로 재정의
self.is_hub = pr > self.cfg.hub_pr_threshold  # 기본값 2.5 / N
```

**신규 config** (`HTPConfig`):
```python
hub_pr_threshold: float = 2.5  # PageRank 값 × N 기준 (즉 1/N 대비 배수)
```

런타임에서는 `pr > self.cfg.hub_pr_threshold / self.cfg.n_nodes` 로 비교하여 노드 수에 독립.

**사이드 이펙트**:
- `top_hubs()` (L.213-217) 이미 PageRank 사용 중 → 변경 없음
- `fire_count` 허브 승격 이벤트 로깅(L.180-185)도 그대로 동작
- `PruningEngine._hub_mask()` (L.263-266)는 `self.is_hub`를 참조하므로 자동 반영

### 4.2 A3 + B3 CoreCells 이중 메커니즘 (Plan §8 질문 1 해결)

**기존** (`core_cells.py`):
- `gate()` L.105-110: `biased_score = score + td_weight × td_bias × strength`
- `update()` L.124-132: `theta_bias[rid] -= η × win_history[rid]` (anti-homeostatic)

**설계 결정 — 이중 메커니즘 공존**:

```python
# gate() 수정
def gate(self, signals, top_down=None):
    ...
    for rid, score in normalized.items():
        precision    = precision_map.get(rid, 1.0)          # [NEW B3]
        biased_score = precision * score + td_biases.get(rid, 0.0)
        eff_theta    = self.theta + self._theta_bias.get(rid, 0.0)
        gated[rid]   = sigmoid(self.beta * (biased_score - eff_theta))
    return GatingMask(scores=gated)
```

```python
# update() 수정 — Hebbian(강화) + Homeostatic(안정화) 병존
def update(self, winner_id, all_ids, fire_rates: dict[str, float]):
    TARGET_RATE = 0.1
    for rid in all_ids:
        # (1) Hebbian 승리 EMA — 기존 유지
        win  = 1.0 if rid == winner_id else 0.0
        self._win_history[rid] = 0.1 * win + 0.9 * self._win_history.get(rid, 0.0)

    for rid in all_ids:
        hebbian_term    = -self._eta_heb * self._win_history[rid]         # 승자 θ ↓ (이기기 쉬움)
        homeo_term      =  self._eta_hom * (fire_rates.get(rid, 0) - TARGET_RATE)  # 과흥분 θ ↑
        bias            = self._theta_bias.get(rid, 0.0) + hebbian_term + homeo_term
        self._theta_bias[rid] = max(-0.2, min(0.2, bias))
```

**신규 생성자 인자**: `eta_heb: float = 0.05` (기존 eta), `eta_hom: float = 0.02`, `target_rate: float = 0.1`

**균형 원리**:
- Hebbian term: 최근 승자 → θ ↓ (매 스텝 ~0.005)
- Homeostatic term: fire_rate 과열 (>0.1) → θ ↑
- 정상 상태에서는 `η_hom × (r − 0.1) ≈ η_heb × win_ema` 근처에서 균형
- clamp [-0.2, +0.2]로 폭주 방지 유지

**precision 파이프라인**:
- `gate(signals, top_down)` → 내부에서 `precision_map = {s.region_id: s.precision for s in signals}` 자동 추출
- 기본값 1.0이므로 B2 완료 전에도 회귀 없음

### 4.3 B2 RegionRuntime precision 계산

**위치**: `region_runtime.py:99-136` `collect_signal()` 말미에 추가.

**전략**: 예측 벡터 보존 없이 최소 침습 — 최근 fire_rate 이동 분산의 역수를 precision proxy로 사용.

```python
# collect_signal() 말미에 추가
recent_rates = [self._rate_history[-k:]]   # self._rate_history: deque(maxlen=10)
self._rate_history.append(fire_rate)
if len(self._rate_history) >= 3:
    variance  = statistics.pvariance(self._rate_history)
    precision = 1.0 / (variance + 0.01)       # [0.5, 100] 스케일
    precision = min(max(precision, 0.1), 5.0) # clamp
else:
    precision = 1.0

return RegionSignal(..., precision=precision)
```

**근거**: 안정적으로 발화하는 Region은 variance 작음 → precision 높음 → gate에서 amplification. 불안정한 Region은 precision 낮음 → gate 약화.

**추후 확장(Phase 3)**: 실제 예측 벡터 `predicted_fire_rate`를 보존하고 `(actual - predicted)² 역수`로 교체.

### 4.4 B4 TopDownBias Softmax 전환

**기존** (`top_down.py:79-80`):
```python
overlap = goal_set & region_tags
biases[rid] = len(overlap) / max(len(goal_set), 1)
```

**변경**:
```python
import torch
overlap_counts = {rid: len(goal_set & region_tags[rid]) for rid in regions}
counts_tensor  = torch.tensor([overlap_counts[rid] for rid in regions], dtype=torch.float32)
temperature    = 1.0
probs          = torch.softmax(counts_tensor / temperature, dim=0)
biases         = {rid: float(p) for rid, p in zip(regions.keys(), probs)}
```

**수학적 성질**:
- overlap이 0인 Region도 `exp(0)/Σexp(·) > 0`의 최소 확률 할당 (Jaccard는 0)
- 합=1 보장 → Friston VFE 계산 시 적법한 확률 prior로 사용 가능
- temperature 인자로 sharpness 조절 (기본 1.0, 0에 가까울수록 hard argmax)

### 4.5 A4 MatrixCells 파라미터화

**기존** (`matrix_cells.py:44-55, 75`):
```python
def __init__(self, temperature=1.0, lateral_w=0.15, lateral_iter=3):
    ...

raw = torch.tensor([
    gating.scores.get(sig.region_id, 0.0) + (0.2 if sig.overload else 0.0)
    for sig in signals
], ...)
```

**변경**:
```python
def __init__(self, temperature=1.0, lateral_w=0.15, lateral_iter=3,
             overload_bonus: float = 0.2):
    ...
    self.overload_bonus = overload_bonus

raw = torch.tensor([
    gating.scores.get(sig.region_id, 0.0) + (self.overload_bonus if sig.overload else 0.0)
    for sig in signals
], ...)
```

### 4.6 Memory — CA3-CA1 Recall

설계 문서 원본(`design/htp_memory_design_final.md` §5) 수식 그대로 구현:

```
CA3 pattern completion:
  best_pat = argmax_{p ∈ patterns} cos(state_vec, p.centroid)
  if best_sim ≥ 0.75:
    completed_vec = 0.7 × state_vec + 0.3 × best_pat.centroid
  else:
    completed_vec = state_vec (그대로)

CA1 mismatch:
  mismatch = ||state_vec - completed_vec||₂
  is_novel = mismatch ≥ 0.3

CA1 가치:
  best_episode = argmax_{e ∈ candidates} e.score × log(e.recall_count + 2)
  recommendation = best_episode.winner
```

**candidates 선택**:
- `is_novel == False`: `l2.search_similar(state_vec, top_k=10, winner_filter=best_pat.best_winner)`
- `is_novel == True`: `l2.search_similar(state_vec, top_k=5)` (필터 없음)

### 4.7 Memory — SWR + Online Hebbian EMA

**SWR 태깅**:
```
priority   = novelty × score
swr_tagged = priority >= 0.5
```

**Online Hebbian EMA 패턴 업데이트**:
```
lr            = 1 / (pattern.episode_count + 1)
new_centroid  = (1 - lr) × old_centroid + lr × episode.state_vec
pattern.episode_count += 1
pattern.winner_dist[ep.winner] += 1 (if ep.outcome == "success")
pattern.snr   = μ(scores) / σ(scores)
```

**Go-CLS 패턴 승격 조건** (버퍼 → L3):
```
count ≥ 3  ∧  snr ≥ 1.5  ∧  winner_dist 중 성공 outcome 존재
```

### 4.8 BrainRuntime 6곳 연동 (C3)

`brain_runtime.py:255-296` `run()` 메서드에 6 삽입점:

| # | 위치 (라인) | 삽입 내용 |
|---|-------------|-----------|
| ① | L.262 직전 | `mem_ctx = memory.recall(self._last_state_vec) if self._step > 1 and self._last_state_vec is not None else None` |
| ② | L.272 td 전달 전 | `if mem_ctx and not mem_ctx.is_novel: self._last_td = self._inject_memory_hint(self._last_td, mem_ctx.recommendation, mem_ctx.confidence)` |
| ③ | L.272 이후 (thal_out 생성 후) | `self._last_state_vec = thal_out.state_vec.clone()` |
| ④ | L.275 이후 (action 결정 후) | `score = self._extract_score(action); memory.save(thal_out, action, str(data)[:50], score)` |
| ⑤ | L.283 이후 | `for name, region in self.regions.items(): if region._cusum_S > region._cusum_h: memory.on_overload(name); region._cusum_S = 0.0` |
| ⑥ | `__init__` L.231 | `self.memory = MemorySystem(memory_dir=memory_dir)` |

신규 헬퍼:
- `_inject_memory_hint(td, recommended, confidence)` — td의 `biases[recommended]`를 `max(existing, confidence × 0.5)`로 boost
- `_extract_score(action)` — `action.reason` 파싱 (`score=0.723` 정규식), 실패 시 0.5

---

## 5. Stage 별 구현 순서 (Plan §9 반영)

| Stage | 내용 | 영향 파일 수 | LOC 추정 |
|-------|------|-------------:|---------:|
| 0 | CLAUDE.md 재작성 | 1 | ~150 |
| 1 | 회귀 테스트 고정 (`tests/regression/`) | 신규 5~8개 | ~300 |
| 2 | L1 제거, L2 PageRank, L3 homeostatic, L4 파라미터화 | 4 | ~80 |
| 3 | B1 필드, B2 precision, B3 gate, B4 softmax | 4 | ~100 |
| 4 | `compress_dim=64` + 영향 임계값 재조정 | 2~3 | ~20 |
| 5 | Memory 4파일 신규 + BrainRuntime 연동 | 5 | ~600 |
| 6 | 통합 테스트 + CLAUDE.md 최종 | — | ~100 |

---

## 6. Error Handling

Python 연구 프로젝트 특성상 REST API 에러 모델은 비적용. 대신:

| 상황 | 처리 |
|------|------|
| Memory SQLite lock (async 환경) | `check_same_thread=False` + `PRAGMA journal_mode=WAL` |
| Pattern JSON 로드 실패 | try/except → 빈 dict로 초기화 |
| precision NaN (variance=0) | clamp `[0.1, 5.0]` |
| L3 centroid 차원 불일치 | 스킵 후 경고 로그 |
| state_vec 8→64 전환 후 과거 에피소드 혼재 | `search_similar()`에서 `vec.shape[0] != state_vec.shape[0]` 스킵 (이미 구현) |

---

## 7. Security Considerations

- Memory SQLite는 로컬 파일. 경로는 DI로 제한 (`memory_dir` 필수 인자로 전환 권장)
- LLM Node 호출 로그에 API 키 누출 방지 (기존 `htp/llm/` 모듈 책임, 본 작업 범위 아님)
- `eval`/`pickle` 사용 금지 — centroid는 `float32 numpy → tobytes()`만 사용

---

## 8. Test Plan

### 8.1 회귀 테스트 (Stage 1 선행)

| 시나리오 | 검증 |
|----------|------|
| Phase 1 HFE demo | 100 step 후 top 5 허브 재현 (노드 0~7 포함) |
| 12/12 라우팅 | `success → to_cache`, `error → to_alert` 전수 통과 |
| 허브 분열 (30회 데이터 후 classify 분열) | NGE split 이벤트 1건 이상 |
| 3가지 prune 전략 | decay/usage/redundancy 각 통계 0 초과 |
| Phase 3 top-down | `long_term_goals=["success","cache"]` → 다음 스텝 td.biases["memory"] > 0 |
| BrainRuntime 1 step | action.type ∈ {"execute","inhibit"}, td_signal 생성 |

### 8.2 신규 단위 테스트

**Stage 2**:
- A2: 허브 노드가 PageRank 상위 k 랭크 내 들어오는지
- A3: 과흥분 Region의 theta_bias가 +방향 이동하는지
- A4: `overload_bonus=0.0` 설정 시 과부하 보너스 사라지는지

**Stage 3**:
- B1: `precision=2.0` 설정 시 gate score 2배 근접
- B2: 안정 Region (variance 작음)의 precision > 불안정 Region
- B4: `biases.values()` 합 ≈ 1.0

**Stage 5 Memory**:
- L2: save → search_similar 왕복 정확도
- L3: 3회 유사 에피소드 투입 시 패턴 1개 승격
- CA3: 노이즈 추가한 입력이 원래 centroid에 수렴
- CA1: mismatch < 0.3 일 때 recommendation 반환
- SWR: `novelty=0.8, score=0.8` → priority=0.64 → tagged
- CUSUM → consolidation: region.cusum 과부하 시뮬 시 `on_overload` 호출 확인

### 8.3 통합 시나리오 (Stage 6)

```
setup: 2개 region, BrainRuntime, memory_dir=tmp
step 1~50: 동일 패턴 입력 반복
expect:
  - L2 에피소드 50개
  - L3 패턴 1개 이상 승격
  - 마지막 10 step의 action.winner 일치율 ≥ 80%
  - memory.recall() → recommendation == 과반 승자
```

---

## 9. Clean Architecture (Python 적응)

웹 계층 없음. Python 레이어 매핑:

| Layer | 책임 | 위치 |
|-------|------|------|
| **Domain** | 순수 데이터 타입 | `htp/thalamus/region_signal.py`, `htp/memory/types.py` |
| **Engine** | 수학·알고리즘 | `htp/core/*`, `htp/thalamus/{core_cells,matrix_cells,top_down}.py`, `htp/memory/{episode_store,pattern_store}.py` |
| **Runtime** | 오케스트레이션 | `htp/runtime/*`, `htp/thalamus/thalamus.py`, `htp/memory/memory_system.py` |
| **Adapter** | 외부 시스템 | `htp/llm/*` |

**Dependency 규칙**:
- Domain → 의존 없음 (pure dataclass, torch.Tensor 예외)
- Engine → Domain만
- Runtime → Engine + Domain
- Adapter → Engine + Domain

---

## 10. Coding Convention

| 항목 | 규칙 |
|------|------|
| 포맷터 | `ruff format` (라인 100자) |
| 린터 | `ruff check` (pyflakes + pycodestyle + isort) |
| 타입 | `from __future__ import annotations`, dataclass 전면 사용 |
| 타입 힌트 | 신규 코드 100%, 기존 수정부 필수 |
| 네이밍 | 함수 `snake_case`, 클래스 `PascalCase`, 상수 `UPPER_SNAKE` |
| 주석 언어 | 생물학·수학 근거는 한글 OK, 코드 docstring은 영어 혼용 허용 (기존 스타일 준수) |
| import 순서 | stdlib → torch/외부 → `htp.*` (기존 파일 스타일 따름) |

---

## 11. Implementation Guide

### 11.1 Stage 2 예시 — A2 is_hub PageRank 전환

**파일**: `htp/runtime/htp_runtime.py`

**수정 전**:
```python
# L.174-177
prev_hub = self.is_hub.clone()
in_str   = self.wm.W.sum(dim=0)
self.is_hub = in_str > self.cfg.hub_threshold
```

**수정 후**:
```python
prev_hub = self.is_hub.clone()
pr_scores = self.pagerank()           # [N] 합=1
pr_scaled = pr_scores * self.cfg.n_nodes   # 1/N 중심 제거 → 1.0 기준
self.is_hub = pr_scaled > self.cfg.hub_pr_threshold
```

**HTPConfig 추가** (`htp_runtime.py:45` 근처):
```python
hub_pr_threshold: float = 2.5   # PageRank 기준 (1/N 대비 배수)
```

**Deprecation 주석**: 기존 `hub_threshold` 필드는 `core/hub_formation_engine.py`만 사용 → Stage 2-A1 처리 후 삭제 고려.

### 11.2 Stage 5 예시 — MemorySystem.recall()

```python
# htp/memory/memory_system.py
def recall(self, state_vec: torch.Tensor) -> MemoryContext:
    completed_vec, pattern = self.l3.complete(state_vec)
    mismatch = (state_vec - completed_vec).norm().item()

    if mismatch < self.CA1_MISMATCH_THRESHOLD:  # 0.3
        candidates = self.l2.search_similar(
            state_vec, top_k=10,
            winner_filter=pattern.best_winner if pattern else None,
        )
    else:
        candidates = self.l2.search_similar(state_vec, top_k=5)

    recommendation, confidence = None, 0.0
    if candidates:
        for ep in candidates:
            self.l2.increment_recall(ep.episode_id)
        best = max(candidates,
                   key=lambda e: e.score * math.log(e.recall_count + 2))
        recommendation, confidence = best.winner, best.score

    return MemoryContext(
        completed_vec=completed_vec,
        mismatch=mismatch,
        candidates=candidates,
        recommendation=recommendation,
        confidence=confidence,
        pattern=pattern,
        is_novel=mismatch >= self.CA1_MISMATCH_THRESHOLD,
    )
```

---

## 12. 남은 Design 레벨 결정 (Do 진입 전 최종 확정)

1. **CoreCells η_heb vs η_hom 기본값**: 제안 0.05 / 0.02 — Stage 2 구현 시 회귀 테스트로 조정
2. **PageRank hub threshold 2.5**: Stage 1 회귀 테스트에서 Phase 1 허브(0~7)가 여전히 is_hub=True 되는지 확인. 안 되면 2.0 또는 3.0으로 재조정
3. **precision variance 윈도우 크기**: 제안 `deque(maxlen=10)` — Region 빌드 시 추가
4. **MemorySystem.CA1_MISMATCH_THRESHOLD**: 설계 문서 0.3 채택. 64-dim 기준이므로 L2 norm 분포 확인 후 조정 가능
5. **Memory 파일 경로 기본값**: `./.htp/memory.db`, `./.htp/patterns.json` (Plan §8 #2 질문 해결)
6. **CLAUDE.md 갱신 시점**: Stage 0(초기 정정) + Stage 6(최종 결과) 2회 (Plan §8 #3 질문 해결)

---

## Version History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 0.1 | 2026-04-14 | Plan v0.3 기반 초안 — 8개 리뷰 지적 + Memory 구현 설계 | Mindbuild |
