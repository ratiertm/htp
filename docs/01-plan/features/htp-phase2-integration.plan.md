---
template: plan
version: 1.4
feature: htp-phase2-integration
date: 2026-04-14
author: Mindbuild
project: HTP (Hub Topology Programming)
status: Draft
---

# HTP Review Feedback Integration — Planning Document

> **Summary**: Phase 1–4 기본 구현이 완료된 현 상태에 `design/` 폴더 4개 리뷰 문서(LeCun·Friston·Memory·Multimodal)의 피드백을 반영한다.
>
> **Project**: HTP
> **Version**: Phase 1–4 구현 완료 → Review Feedback 통합
> **Author**: Mindbuild
> **Date**: 2026-04-14
> **Status**: Draft (v1.4 — 리뷰 지적사항 ↔ 실제 코드 정밀 대조 반영)

---

## 0. 코드 인벤토리 결과 (Plan 전제 수정)

`htp/__init__.py` 기반 실측:

| 항목 | CLAUDE.md 기술 | **실제 코드** |
|------|----------------|--------------|
| Phase 상태 | "Phase 1 완료, Phase 2 시작" | **Phase 1·2·3·4 모두 이미 구현됨** |
| Thalamus 체인 | "구현 필요" | `htp/thalamus/` 전체 존재 (Thalamus/CoreCells/MatrixCells/NGETrigger/TopDown) |
| PFCRuntime | "구현 필요" | `brain_runtime.py:46` 구현됨 (`deque(maxlen=7)` 확정) |
| BrainRuntime | "구현 필요" | `brain_runtime.py:212` 구현됨 |
| CorticalConnections | 미해결 질문 | 이미 구현됨 — **Region 직접 통신 허용 확정** |
| LLM-as-Node | "Phase 4 장기 목표" | `htp/llm/` 이미 구현 (`LLMNode`, `LLMRegionRuntime`, `CostRouter`) |
| Async | 언급 없음 | `AsyncBrainRuntime` 존재 |
| `compress_dim` | — | **`8`** 확정 (`thalamus.py:52`) — Memory 설계 "8→64" 변경 대상 |

### 0.1 리뷰 지적 ↔ 실제 코드 정밀 대조 (v1.4 추가)

| # | 리뷰 지적 | 실제 코드 위치·내용 | 판정 |
|---|-----------|---------------------|------|
| L1 | Hebbian variant 불일치 (BCM vs Oja) | `core/hub_formation_engine.py:105` = BCM-like `delta = lr × coact × (1-W)`<br>`runtime/htp_runtime.py:163-171` = **이미 Oja's Rule** (`__init__.py`는 runtime 쪽을 export) | ⚠️ **부분 오해** — `core/hub_formation_engine.py`는 사실상 **데드 코드**. 해야 할 일은 "두 파일 통일"이 아니라 **구 파일 제거 또는 데모 전용 격리** |
| L2 | Hub Detection 혼용 (sum vs PageRank) | `htp_runtime.py:176-177` `is_hub = in_str > threshold` (**sum**)<br>`htp_runtime.py:192-217` = PageRank 함수<br>`htp_runtime.py:213-217` `top_hubs()`만 PageRank | ✅ **정확** — `is_hub` 마스크 결정이 여전히 sum. 수정 대상 = 이 한 줄 |
| L3 | CoreCells homeostatic 없음 | `core_cells.py:116-132` `update()`에 Hebbian adaptive θ 有. 그러나 **방향은 anti-homeostatic**: `theta_bias -= η × win_ema` (자주 이기면 θ↓ → 더 이기기 쉬움). 생물학적 homeostatic은 반대 polarity (과흥분 → θ↑ 안정화) | ✅ **정확** — 현재 Hebbian 강화와 별개로 homeostatic term 추가 필요 (polarity 상반되는 **두 메커니즘 공존**이 정답) |
| L4 | MatrixCells 하드코딩 | `matrix_cells.py:75` `0.2 if sig.overload else 0.0` | ✅ **정확** |
| F1 | RegionSignal precision 없음 | `region_signal.py:23-31` 필드 없음 | ✅ **정확** |
| F2 | RegionRuntime precision 없음 | `region_runtime.py` 예측·precision 로직 없음 | ✅ **정확** |
| F3 | CoreCells Precision-weighted 아님 | `core_cells.py:105-110` `biased_score = score + td_bias` (precision 없음) | ✅ **정확** |
| F4 | TopDownBias Jaccard → Softmax | `top_down.py:79-80` `len(overlap) / len(goal_set)` Jaccard-like | ✅ **정확** |
| Mem | state_vec 8-dim / L2·L3 없음 / SWR·CA3·Go-CLS 없음 | `thalamus.py:52` `compress_dim=8`, `htp/memory/` 없음 | ✅ **정확** |

### 0.2 미해결된 CLAUDE.md 질문 (v1.4 정정)

- **"Core/Matrix 규칙 vs 학습?"** — v1.3은 "혼합 — CoreCells Hebbian 有"라고 적었으나 이는 **불완전**. 정확한 기술은:
  - MatrixCells = **규칙 기반** (Lateral Inhibition + Softmax, 학습 없음)
  - CoreCells = **anti-homeostatic Hebbian** (현재). L3 지적에 따라 **homeostatic term 추가 필요** → 결과적으로 "Hebbian 강화 + Homeostatic 안정화 이중 메커니즘"이 되어야 함

**결론**: 본 작업은 "Phase 2 신규 구현"이 **아니라**, 기존 Phase 1–4 코드에 대한 **국소 수정(L1~L4, F1~F4) + Memory 신규 모듈 추가**다.

---

## 1. Overview

### 1.1 Purpose

Phase 1–4 기본 구현(단일 HTPRuntime → Thalamus → BrainRuntime → LLM/Async)이 이미 완료된 상태다. 다만 구현 과정에서 다음과 같은 **수학적·생물학적 엄밀성 이슈**가 4개 리뷰 문서로 지적되었다:

- Hebbian variant가 `hub_formation_engine.py`와 `htp_runtime.py`에서 불일치
- Hub Detection 로직 두 벌 혼용
- Friston FEP 원칙 부재 (precision·예측 오차·EFE 없음)
- 메모리 시스템 부재 (L1 working memory만 있고 L2/L3 없음)
- `state_vec` 8-dim 표현력 부족
- 단일 모달만 지원

본 Plan은 네 리뷰의 **Phase 2 (즉시 수정) 범위**를 통합하여 기존 코드를 리팩토링·보강하고, 신규 `memory/` 모듈을 추가하는 것이다.

### 1.2 Background

4개 설계 문서가 제시하는 비판과 수정 방향:

| 문서 | 핵심 지적 | 주요 수정 |
|------|-----------|-----------|
| `htp_lecun_review.md` | Hebbian variant 불일치, Hub Detection 혼용, homeostatic 부재 | Oja's Rule 통일, PageRank 통일, homeostatic term |
| `htp_friston_review.md` | 반응 기계 (예측·precision 부재), 고정 임계값 | precision 필드, VFE 기반 결정, EFE 행동 선택 |
| `htp_memory_design_final.md` | L2/L3 메모리 부재, state_vec 8-dim 부족 | 64-dim, EpisodeStore(SQLite), PatternStore, CA3-CA1 |
| `htp_multimodal_design.md` | 단일 모달만 지원 | ModalEncoder 계층, Fusion Tokens, V-JEPA 방식 |

네 문서 모두 Phase 2 (즉시 수정) / Phase 3 / Phase 4 로드맵을 제시하며, **Phase 2 수정사항은 우선순위가 명확하고 서로 독립적이거나 호환 가능**하다.

### 1.3 Related Documents

- 설계 원본: `design/htp_lecun_review.md`, `design/htp_friston_review.md`, `design/htp_memory_design_final.md`, `design/htp_multimodal_design.md`
- 프로젝트 컨텍스트: `CLAUDE.md`
- 아키텍처: `architecture/` (기존), `htp_architecture_design.md`

---

## 2. Scope

### 2.1 In Scope (Phase 2 전용)

**A. LeCun 즉시 수정 4종** (v1.4 수정범위 재정의)
- [ ] A1. `htp/core/hub_formation_engine.py` **제거 또는 데모 격리** — 이미 `runtime/htp_runtime.py`가 Oja's Rule 사용, 구 BCM 파일은 데드 코드. `__init__.py` export 경로 확인 후 정리
- [ ] A2. `htp/runtime/htp_runtime.py:176-177` — `is_hub` 마스크를 `in_str > threshold` (sum 기반)에서 **PageRank 점수 기반**으로 변경. threshold도 PageRank 분포 스케일에 맞게 재조정
- [ ] A3. `htp/thalamus/core_cells.py` — **Homeostatic term 추가** (기존 anti-homeostatic Hebbian과 병존). 수식: `theta_bias = -η_heb × win_ema + η_hom × (fire_rate - target_rate)`. 두 term의 크기 균형은 Design에서 확정
- [ ] A4. `htp/thalamus/matrix_cells.py:75` — `0.2` 하드코딩 제거, `overload_bonus` 생성자 인자화 (기본값 0.2 유지)

**B. Friston 즉시 수정 4종** (v1.4 위치 명시)
- [ ] B1. `htp/thalamus/region_signal.py:23-31` — `RegionSignal`에 `precision: float` 필드 추가 (default 1.0)
- [ ] B2. `htp/runtime/region_runtime.py:99-136` `collect_signal()` — precision 동적 계산 (예측 오차 역수). 예측 벡터 보존이 선행 필요 → 최소 구현은 "지난 fire_rate vs 현재 fire_rate 일관성"을 precision proxy로 사용
- [ ] B3. `htp/thalamus/core_cells.py:105-110` — Precision-weighted Gate: `biased_score = precision × score + td_bias` (A3와 **같은 파일** 수정이므로 병합 커밋). precision이 gating amplitude scaler 역할
- [ ] B4. `htp/thalamus/top_down.py:79-80` — `biases[rid] = softmax(overlap_counts)[rid]` 로 전환. Jaccard `overlap/goal_size` 대신 모든 Region의 overlap을 softmax 정규화해 확률 분포 생성

**C. Memory System 전체 (LeCun 검토 반영판)**
- [ ] C1. `htp/memory/` 신설 — `types.py`, `episode_store.py` (SQLite), `pattern_store.py` (JSON), `memory_system.py`
- [ ] C2. `thalamus.py` — `compress_dim` 8 → **64** 변경
- [ ] C3. `brain_runtime.py` — recall/save/on_overload 연동 (6곳 삽입)
- [ ] C4. SWR 태깅 (`novelty × reward`), CA3 완성 + CA1 불일치, Online Hebbian EMA, Go-CLS SNR 조건

**D. Multimodal — 본 Plan에서 제외 (Phase 3로 연기, §2.3 참조)**

### 2.2 Out of Scope (Phase 3·4로 연기)

- LeCun #5 Lateral Inhibition 국소화, #6 기능적 특화 분열, #7 임베딩 시맨틱 라우팅, #8 Incremental PCA
- Friston #5 PFC VFE 전면 교체, #6 PredictiveRegion, #7 EFE 행동 선택
- Multimodal Cross-modal Attention (Le MuMo JEPA), LiDAR/Camera/Audio 인코더
- LLM-as-Node (Phase 4 장기 목표)
- `CLAUDE.md` 원본의 Thalamus 설계(CoreCells/MatrixCells/NGETrigger 신규 작성) — 이미 `htp/thalamus/` 존재하므로 **현황 파악 후 수정 vs 재작성 결정 필요**

### 2.3 전제 확인 완료 (§0 참조)

코드 인벤토리 결과 모든 전제가 §0에서 해소됨. 남은 결정 사항은 §8 참조.

**범위 D (Multimodal) 결정: Phase 3로 연기** — 본 Plan에서 제외한다. 이유:
- Memory(C)만 해도 신규 4파일 + `BrainRuntime` 6곳 수정으로 규모 큼
- Multimodal은 `htp/multimodal/` 전체 + ModalEncoder 5종 + Thalamus Fusion Token 계층이 독립 작업

---

## 3. Requirements

### 3.1 Functional Requirements

| ID | Requirement | Priority | Status |
|----|-------------|----------|--------|
| FR-01 | Oja's Rule 단일 구현으로 헤비안 학습 통일 (두 파일 동일 수식) | High | Pending |
| FR-02 | PageRank 기반 허브 감지 단일 구현 | High | Pending |
| FR-03 | CoreCells가 homeostatic term + precision-weighted gate 모두 반영 | High | Pending |
| FR-04 | `RegionSignal.precision` 필드가 모든 Region에서 동적 계산되어 Thalamus로 전달 | High | Pending |
| FR-05 | TopDownBias가 Softmax 확률 분포 기반 prior로 동작 | Medium | Pending |
| FR-06 | `MemorySystem` L2/L3 구현 — Episode 저장·조회, 패턴 통합, CA3-CA1 recall | High | Pending |
| FR-07 | `state_vec` 64-dim 일관 적용 (Thalamus·Memory·save/recall 모두) | High | Pending |
| FR-08 | CUSUM overload → `memory.on_overload()` 자동 트리거 | Medium | Pending |
| FR-09 | (옵션 D) `ModalMatrix` + `ModalEncoder` 추상 계층 및 최소 2개 인코더 | Low | Pending |

### 3.2 Non-Functional Requirements

| Category | Criteria | Measurement |
|----------|----------|-------------|
| Correctness | Phase 1 검증 시나리오(12/12 라우팅, 허브 분열) 회귀 없음 | 기존 테스트 스위트 통과 |
| Performance | step latency < 2× Phase 1 기준 (precision·memory 오버헤드 허용 범위) | micro-bench `brain_runtime.step()` |
| Memory footprint | L2 SQLite 1000 에피소드 < 10MB | 디스크 사용량 측정 |
| Biological fidelity | 수정사항마다 참조 논문 근거 명시 | 코드 docstring·주석에서 확인 |
| Testability | 각 엔진 단위 테스트, 통합은 최소 시나리오 1개 | pytest |

---

## 4. Success Criteria

### 4.1 Definition of Done

- [ ] Scope A·B·C 9개 수정사항 전부 구현
- [ ] `htp/memory/` 모듈 신설 및 `BrainRuntime` 연동
- [ ] Phase 1 회귀 테스트 통과 (라우팅 정확도 유지, 허브 분열 재현)
- [ ] 새 기능 통합 테스트: recall→top-down hint→결정→save→consolidation 1사이클
- [ ] `CLAUDE.md` Phase 2 섹션을 **실제 구현 결과**로 갱신
- [ ] `design/` 문서와의 Gap 분석 ≥ 90%

### 4.2 Quality Criteria

- [ ] 모든 수식에 참조 논문 또는 수학적 근거 주석
- [ ] 타입 힌트 완전 (mypy strict 통과 목표)
- [ ] 단위 테스트 커버리지 ≥ 70% (신규 모듈 기준)
- [ ] import 순환 없음

---

## 5. Risks and Mitigation

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| CoreCells가 A3(homeostatic) + B3(precision gate) 동시 수정 → 설계 충돌 | High | Medium | Design 단계에서 두 수식 결합 형태 수학적으로 먼저 확정 |
| 8→64-dim 변경이 기존 허브 로직·코사인 임계값과 충돌 | High | Medium | 차원 전환 전 `compress_dim` 전수 grep, 임계값 파라미터화 |
| `htp/thalamus/` 기존 코드와 설계 문서가 명칭·구조 불일치 | Medium | High | Design 전 **실제 코드 인벤토리** 먼저 작성 |
| 범위 D(Multimodal) 포함 시 Phase 2가 비대해짐 | Medium | High | 기본값 **Phase 3 연기**, 요구사항 확정 후 재평가 |
| Memory L2 SQLite 동시성 이슈 (async runtime 존재) | Medium | Medium | `async_brain_runtime.py`에서 thread-safe 접근 검증 |
| Oja's Rule 전환이 Phase 1 검증 시나리오(30회 분열)를 깨뜨림 | High | Low | 기존 시나리오를 회귀 테스트로 고정 후 전환 |

---

## 6. Architecture Considerations

### 6.1 Project Level

| Level | Selected |
|-------|:--------:|
| Starter | ☐ |
| Dynamic | ☐ |
| **Research / Custom Python** | ☑ |
| Enterprise | ☐ |

> 일반적인 웹 프로젝트 레벨이 적용되지 않는 **연구 성격의 Python 프로젝트**. bkit 템플릿의 웹 프레임워크/BaaS 관련 항목은 비적용.

### 6.2 Key Architectural Decisions

| Decision | Options | Selected | Rationale |
|----------|---------|----------|-----------|
| Hebbian 규칙 | BCM / Oja / Hebbian 원형 | **Oja's Rule** | L2 norm 자동 보존, PCA 주성분 수렴, 수학적 엄밀성 (LeCun #1) |
| Hub 감지 | strength 합산 / PageRank / Eigenvector centrality | **PageRank** | 전역 중요도 반영, 표준화된 단일 구현 (LeCun #2) |
| Region→Thalamus 신호 | 단일 수치(hub_strength) / precision 포함 구조체 | **precision 포함** | Friston FEP 원칙 (Friston #1·#2) |
| L2 저장소 | SQLite / JSON / in-memory | **SQLite** | 에피소드 수 증가 시 쿼리 성능, 인덱스 지원 |
| L3 저장소 | SQLite / JSON | **JSON** | 패턴 수 적고, 구조 단순, 사람이 읽기 쉬움 |
| state_vec 차원 | 8 / 32 / 64 / 128 | **64** | JL Lemma `k ≈ log(1000)/0.1² ≈ 64`, 해마 sparse 표현 |
| 패턴 승격 방식 | 배치 K-Means / Online Hebbian EMA | **Online Hebbian EMA** | 뇌는 전체 재계산 안 함 (LeCun Memory 검토) |
| Top-down prior | Jaccard 유사도 / Softmax 확률 | **Softmax** | 확률적 prior로 VFE와 수학적 호환 (Friston #4) |

### 6.3 Folder Structure

```
htp/
├── core/                         (기존)
│   ├── hub_formation_engine.py   ← Oja 통일 수정 (A1)
│   ├── pruning_engine.py
│   └── node_generation_engine.py
├── runtime/                      (기존)
│   ├── htp_runtime.py            ← PageRank 통일 (A2)
│   ├── brain_runtime.py          ← memory 연동 6곳 (C3)
│   ├── async_brain_runtime.py
│   ├── activation_engine.py
│   ├── cortical_connections.py
│   └── region_runtime.py         ← precision 계산 (B2)
├── thalamus/                     (기존)
│   ├── thalamus.py               ← compress_dim 64 (C2)
│   ├── core_cells.py             ← homeostatic + precision gate (A3+B3)
│   ├── matrix_cells.py           ← 파라미터화 (A4)
│   ├── nge_trigger.py
│   ├── region_signal.py          ← precision 필드 (B1), modal_matrix (D3 옵션)
│   └── top_down.py               ← Softmax prior (B4)
├── memory/                       (신규 C1)
│   ├── __init__.py
│   ├── types.py                  ← Episode, Pattern, MemoryContext
│   ├── episode_store.py          ← L2 SQLite + SWR 태깅
│   ├── pattern_store.py          ← L3 Online Hebbian + Go-CLS + CA3
│   └── memory_system.py          ← CA3-CA1 통합 + CUSUM 트리거
├── multimodal/                   (옵션 D1)
│   ├── __init__.py
│   └── modal_encoder.py
└── llm/                          (기존 — 변경 없음)
```

---

## 7. Convention Prerequisites

### 7.1 Existing

- [x] `CLAUDE.md` 존재 (Phase 1 상태·Phase 2 목표 기술)
- [ ] `docs/01-plan/conventions.md` 없음 — 본 Plan 이후 생성 권장
- [x] `requirements.txt` 존재 (torch 포함 확인 필요)
- [ ] ESLint/Prettier — 해당 없음 (Python)

### 7.2 Python Convention 확정 필요

| Category | Current | To Define | Priority |
|----------|---------|-----------|:--------:|
| 포맷터 | 미지정 | `ruff format` 또는 `black` | Medium |
| 린터 | 미지정 | `ruff check` | Medium |
| 타입 체커 | 미지정 | `mypy` (strict 권장) | Medium |
| 테스트 | 미지정 | `pytest` | High |
| 네이밍 | Phase 1 코드 스타일 혼재 | snake_case 함수·PascalCase 클래스 확정 | High |
| import 순서 | 혼재 | stdlib → 3rd party → local (isort 호환) | Low |

### 7.3 Environment Variables

현재 단계 없음. LLM Node 활용 시 `ANTHROPIC_API_KEY` 등 추가 (Phase 4).

---

## 8. 미해결 설계 질문 (Design 단계로 이월)

코드 인벤토리로 해소된 질문은 §0에서 처리됨. 남은 3개:

1. **CoreCells A3(homeostatic) + B3(precision gate) 결합식**
   - 현재 `core_cells.py`는 이미 `biased_score + theta_bias` Hebbian 학습 포함
   - homeostatic term과 precision 가중치를 어느 위치에 어떻게 합성할지 — 덧셈/곱셈/중첩?
   - Design 단계 첫 번째 과제

2. **Memory 파일 경로 정책**
   - 옵션 A: 프로젝트 루트 `htp_memory.db` / `htp_patterns.json` (설계 문서 기본값)
   - 옵션 B: `./data/` 또는 `./.htp/` 서브 디렉터리 (gitignore 용이)
   - 옵션 C: `BrainRuntime(memory_dir=...)` 주입 (테스트 용이)
   - 권장: **C (의존성 주입) + 기본값 B**

3. **CLAUDE.md 업데이트 정책** (Stage 0)
   - 옵션 A: 본 Plan 완료 시 일괄 업데이트
   - 옵션 B: Stage별 완료마다 섹션 업데이트
   - 권장: **A** (한 번에 Phase 1–4 + Review 통합 결과로 재작성)

---

## 9. Implementation Order (Design에서 확정)

제안 순서 (의존성 기반):

```
[Stage 0] CLAUDE.md 실제 상태로 갱신 + 본 Plan 질문 #3 결정
  └ Phase 1–4 실제 구현 반영, 남은 미해결 질문 정리

[Stage 1] 회귀 테스트 고정
  └ Phase 1–4 기존 동작(12/12 라우팅, 허브 분열, PFC decide, top-down) pytest 화

[Stage 2] LeCun 즉시 수정 (순서 중요)
  1. A1 Oja's Rule — 단일 파일 수정, 영향 국소
  2. A2 PageRank  — 단일 파일 수정
  3. A4 MatrixCells 파라미터화 — 단독

[Stage 3] Friston 신호 체인
  4. B1 RegionSignal.precision 필드
  5. B2 RegionRuntime precision 계산
  6. (병합 수정) A3 + B3 CoreCells
  7. B4 TopDownBias Softmax

[Stage 4] 차원 전환
  8. C2 compress_dim 8→64 + 영향받는 임계값 조정

[Stage 5] Memory System
  9. C1 htp/memory/ 신설 (types → episode_store → pattern_store → memory_system)
  10. C3 BrainRuntime 6곳 연동
  11. C4 SWR/Go-CLS/CA3-CA1 검증

[Stage 6] 통합 테스트 + CLAUDE.md 최종 갱신
```

---

## 10. Next Steps

1. [ ] 본 Plan 사용자 검토·승인
2. [ ] 현재 `htp/thalamus/`·`runtime/` 실제 코드 인벤토리 (Design 전 필수)
3. [ ] §8 미해결 질문 1·5·6번 결정
4. [ ] `/pdca design htp-phase2-integration` 으로 Design 문서 착수
5. [ ] Design에서 Stage별 상세 설계·수식 결합식 확정 후 `/pdca do`

---

## Version History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 0.1 | 2026-04-14 | 4개 design 문서 통합 초안 | Mindbuild |
| 0.2 | 2026-04-14 | 코드 인벤토리 반영: Phase 1–4 이미 구현됨 확인, 해소된 질문 §0/§8 정리, Multimodal(D) Phase 3 연기 확정, Stage 0(CLAUDE.md 갱신) 추가 | Mindbuild |
| 0.3 | 2026-04-14 | 리뷰 지적 ↔ 코드 정밀 대조(§0.1) 추가. A1을 "통일"에서 "데드 코드 제거"로 정정, A2 수정 위치 `htp_runtime.py:176-177` 명시, A3을 "homeostatic+Hebbian 이중 메커니즘"으로 재정의, B1–B4 파일:라인 명시 | Mindbuild |
