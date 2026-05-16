---
template: plan
version: 1.3
feature: htp-thalamus-car
date: 2026-05-16
author: Mindbuild
project: HTP (Hub Topology Programming)
status: Draft
---

# htp-thalamus-car Planning Document

> **Summary**: Thalamus를 Content-Addressable Parallel Router로 재설계. 태그 매칭(주소 기반)을 벡터 유사도(내용 기반)로 전환하고, 다중 Region 응답의 temporal binding을 추가하며, LLM 통합 경계를 재정의한다. 설계서 `htp_thalamus_car_design v4.md` (**Rev 1.3 — Knowledge Loop MVP 삽입**) 전체 **9 Stage**(0, 0.5, 1-7)를 PDCA로 집행.
>
> **Project**: HTP
> **Version**: post-`201f0f2` (htp-review-improvements 완료 후 토대 위에 적층)
> **Author**: Mindbuild
> **Date**: 2026-05-17
> **Status**: Draft (Rev 0.2 — 설계서 v4 반영, Stage 0.5 Knowledge Loop MVP 추가)
> **선행 문서**:
> - `htp_thalamus_car_design v4.md` (설계서, Rev 1.3)
> - `docs/03-review/htp-project-review.md` (§3-C 미반영 항목 2건 해소 대상)
> - `htp_memory_design_final.md`, `htp_friston_review.md`, `htp_multimodal_design.md`

---

## Executive Summary

| Perspective | Content |
|-------------|---------|
| **Problem** | (G1) Thalamus 라우팅이 문자열 태그 매칭(주소 기반) — "붉은 둥근 과일"은 "vision" 태그가 없으면 시각 Region 도달 불가. (G2) 다중 Region 응답의 temporal binding 부재 — A가 "사과"로 B가 "토마토"로 해석해도 불일치 미감지. (G3) LLMRegionRuntime이 RegionRuntime을 상속해 PageRank/Hebbian/NGE를 불필요하게 끌어옴 → graphify상 176 isolated 노드 다수 발생. **(G4 v4 신규) 설계의 실효성을 검증할 실사용 루프 부재** — 현재 4,406줄은 "신경 회로 기판"이지 "지식을 넣고 통찰을 꺼내는 도구"가 아님. |
| **Solution** | Stage 0-7 **9단계** 진행 (0.5 추가): (0) HTPConfig 4개 sub-config 분리 → **(0.5) Knowledge Loop MVP — 텍스트 입력→벡터 저장→관계 발견→텍스트 출력 최소 루프 완주** → (1) RegionSignature + CoreCells vector mode → (2) Hybrid 검증 → (3) CoherenceGate + Memory novelty 연동 → (4) ExternalRegion + LLMRegion 리팩토링 → (5) PipelinedBrainRuntime → (6) EmbeddingBridge 실험 브랜치 → (7) vector default 전환. 각 Stage Go/No-Go 기준 명시. `routing_mode = "tag"` 기본값으로 회귀 57/57 보호. |
| **Function/UX Effect** | **Stage 0.5 직후 CLI 도구 사용 가능** (`python -m htp.knowledge ingest/query/discover`) + content-addressable routing + 복수 Region 동시 라우팅 + conflict 자동 감지 → SWR novelty 증폭 + LLM 그래프 본체 합류 + Local 처리 비율 70%로 API 비용 ~70% 절감 (목표). 신규 테스트 본선 **27-30개** + 실험 4개. |
| **Core Value** | 뇌의 세 가지 핵심 메커니즘 (분산 표상·시냅스 가소성·시간적 바인딩) 중 부재했던 #3(시간적 바인딩)과 #2의 표면적 구현(태그 매칭)을 진정한 내용 기반으로 전환. **"매일 쓰는 도구"가 되어야 아키텍처가 가치를 가진다는 원칙으로 Stage 0.5에서 루프를 먼저 닫고 이후 Stage를 실사용 피드백으로 교정.** HTP 4대 원칙 ("구조는 데이터가 만든다", "허브는 창발한다", "판단은 위임한다")의 실제 작동 메커니즘 정렬. |

---

## Context Anchor

| Key | Value |
|-----|-------|
| **WHY** | 태그 매칭이 HTP의 4대 원칙과 모순됨("구조가 데이터에서 창발"이라면서 사전 라벨링된 태그로 라우팅). Phase 3-4 확장(임베딩 라우팅·Predictive Coding) 전 토대 정리 필수. **v4 추가**: 실사용 루프가 없으면 Stage 7까지 진행 후 "실제로는 다르게 동작해야 했다"를 발견할 위험. |
| **WHO** | HTP 개발자 본인 + 향후 Region 추가 컨트리뷰터 + **Stage 0.5 이후 본인이 "매일 쓰는 도구" 사용자**. 사용자 데이터(`from htp import …`)는 routing_mode 기본값 보존으로 무영향. |
| **RISK** | (1) 회귀 57/57 깨짐 → 즉시 롤백. (2) CoherenceGate O(N²) 스케일링 (N≥16 LSH 전환 임계). (3) RegionSignature 냉시작 (centroid 영벡터). (4) EmbeddingBridge sLLM 의존성. (5) vec↔prompt 변환 품질이 LLMRegion 실용성 좌우. **(6 v4 신규) Stage 0.5 MVP의 TF-IDF+JL 임베딩이 cross-domain similarity 발견 못 할 위험 — discovered 검증 시나리오(뇌과학-AI > 뇌과학-인프라) 실패 시 Stage 1-7 전제 흔들림. 조기 발견이 핵심**. |
| **SUCCESS** | **(0 v4 신규) Stage 0.5 Go/No-Go: 뇌과학/AI/인프라 3 source 입력 후 cross-domain discover() 작동 — 뇌과학-AI 유사도 > 뇌과학-인프라 유사도**. (1) 회귀 57/57 + 신규 본선 **27-30개** + 실험 4개 모두 통과 (2) Stage 4 후 graphify isolated 노드 수 감소 (3) Stage 5 후 throughput ≥ 1.5× 순차 (4) Stage 7 전환 시 routing 정확도가 tag mode 동등 또는 우위 (5) Level 1-2 처리 비율 70% (실현 시 API 비용 -70%). |
| **SCOPE** | Stage 0-7 **9 Stage** 전체 (0, 0.5, 1-7). 단 Stage 6은 `experiment/embedding-bridge` 브랜치 분리, 본선 머지는 Go/No-Go 통과 시에만. 미래 확장점(§8) — Multi-head attention, ModalEncoder, Continuous token, PredictiveRegion, Incremental PCA, Lateral Inhibition — 모두 OUT-OF-SCOPE. **Stage 0.5 의도적 배제**: 고품질 임베딩(Stage 6), LLM 통찰 생성(Stage 4), 웹 스크래핑/PDF 파싱(별도 도구), 시각화 UI(Stage 5+), 자동 수면 공고화(후순위). |

---

## 1. Overview

### 1.1 Purpose

설계서 §0-2 3대 갭 + **v4 추가 G4** 해소:
- **G1**: 시맨틱 라우팅 부재 → CoreCells tag→vector 전환
- **G2**: Temporal binding 없음 → CoherenceGate 도입
- **G3**: LLMRegion 그래프 고립 → ExternalRegion 추상 + 4-Level CostRouter + EmbeddingBridge
- **G4** (v4 신규): 실사용 루프 부재 → **Stage 0.5 Knowledge Loop MVP** — 텍스트 ingest/query/discover CLI로 "매일 쓰는 도구" 진입

### 1.2 Background

이전 PDCA 사이클 `htp-review-improvements`(`201f0f2`)에서 god-file/god-object를 해소하며 **DAG 강제** 토대를 만들었다. 본 사이클은 그 토대 위에서 **routing 알고리즘 자체**를 교체한다. 즉, 이전 사이클이 *구조* 를 정리했다면 본 사이클은 *의미* 를 채운다.

**v4 (Rev 1.3) 핵심 변경**: 설계서가 v3 → v4로 진화하면서 Stage 0과 Stage 1 사이에 **Stage 0.5 Knowledge Loop MVP**가 삽입되었다. 그 이유는 "아무리 정교한 라우팅·CoherenceGate가 있어도 실제 지식을 넣고 꺼내보지 않으면 설계 실효성을 검증할 수 없다"는 인식 — Stage 1-7을 진행한 후 "실제로는 다르게 동작해야 했다"를 발견하는 것보다, 최소 루프를 닫고 매일 쓰면서 나오는 피드백으로 이후 Stage를 교정하는 것이 합리적이라는 결정. 이것이 v3에서 v4로의 결정적 차이다.

### 1.3 Related Documents

- 설계서: `htp_thalamus_car_design v3.md` (856줄, Rev 1.2)
- 선행 PDCA: `docs/01-plan/features/htp-review-improvements.plan.md` (HTPConfig sub-config 분리 토대)
- 리뷰: `docs/03-review/htp-project-review.md` §3-C 미반영, §5 약점-4(LLMNode 고립)
- 신경과학: 설계서 §0-1 (분산 표상·Hebb's rule·gamma binding 3 조건)

---

## 2. Scope

### 2.1 In Scope — Stage 0–7 전체 (9 Stage)

#### Stage 0: HTPConfig sub-config 분리 (최우선, 토대)
- [ ] `RoutingConfig`, `CoherenceConfig`, `LLMBridgeConfig`, `PipelineConfig` 4개 신설
- [ ] 기존 HTPConfig facade와 통합 (이전 사이클 `htp/core/config.py` 확장)
- [ ] 기존 flat kwarg/attribute 접근 hosting 호환 유지 (deprecated warning 동반)

#### Stage 0.5: Knowledge Loop MVP — **v4 신규 핵심** (G4 해소)
- [ ] `htp/knowledge/` 패키지 신설 — `loop.py`, `encoder.py`, `__init__.py`
- [ ] `TextEncoder` Protocol 정의 — `str → np.ndarray[64]`. `KnowledgeLoop`/`LLMRegion`/`RegionSignature` 모두에서 공유
- [ ] MVP `TextEncoder` 구현 — TF-IDF + JL random projection (또는 feature hashing)
- [ ] `KnowledgeLoop` 클래스 — `ingest(text, source)` / `query(question)` / `discover()` 3 메서드
- [ ] `KnowledgeEntry`, `IngestResult`, `QueryResult`, `Discovery` dataclass
- [ ] CLI: `python -m htp.knowledge ingest|query|discover`
- [ ] `htp/knowledge/__main__.py` — argparse CLI 디스패처

#### Stage 1: RegionSignature + CoreCells vector mode
- [ ] `htp/thalamus/signature.py` 신설 — `RegionSignature(centroid, count, lr=1/(n+1))`
- [ ] **`RegionSignature`는 Stage 0.5의 `TextEncoder` Protocol을 사용해 텍스트 입력도 지원**
- [ ] `CoreCells._gate_vector()` 추가 — content-addressable routing
- [ ] `RegionSignal.region_signature` 필드 추가
- [ ] Dynamic threshold (μ + β×σ) 구현
- [ ] `routing_mode="tag"` 기본값 유지
- [ ] **Stage 0.5의 knowledge_log를 vector routing 테스트 입력으로 활용**

#### Stage 2: Hybrid routing 검증
- [ ] `_gate_hybrid()` 구현 — `alpha × vec_score + (1-alpha) × tag_score`
- [ ] α를 0.1~0.9 변화시키며 라우팅 결과 연속성 검증

#### Stage 3: CoherenceGate + Memory 연동
- [ ] `htp/thalamus/coherence.py` 신설 — `CoherenceGate.bind()` pairwise coherence + conflict detection
- [ ] `htp/thalamus/types.py` — `BoundResponse` dataclass 신설
- [ ] BrainRuntime에 CoherenceGate 삽입 (Region 응답 수집 후, PFC 전달 전)
- [ ] `MemorySystem.swr_priority` 확장 — `novelty × reward × (1 + conflict_magnitude)`

#### Stage 4: ExternalRegion + LLMRegion 리팩토링 (MED-4 흡수)
- [ ] `htp/runtime/external_region.py` 신설 — `ExternalRegion` 추상 클래스 (RegionRuntime 비상속)
- [ ] `htp/llm/llm_region.py` 신설 — `LLMRegion(ExternalRegion)`
- [ ] `vec_to_prompt` / `prompt_to_vec` 단기 전략 (dim-tag 매핑)
- [ ] 기존 `LLMRegionRuntime`을 `archive/deprecated_phase4/`로 이동
- [ ] `CostRouter` 4-Level 의사결정 트리 확장 (`select_level`)
- [ ] **MED-4 차이점 보완** (§3 별도 참조)

#### Stage 5: PipelinedBrainRuntime
- [ ] `htp/runtime/pipelined_brain.py` 신설 — L3 파이프라인 병렬성
- [ ] 기존 `AsyncBrainRuntime` 보존 (대체 아닌 추가)

#### Stage 6: EmbeddingBridge (실험 브랜치)
- [ ] **브랜치 `experiment/embedding-bridge` 생성은 Stage 6 진입 시점에**
- [ ] `htp/llm/embedding_bridge.py` 신설 — sLLM 양방향 프로젝션
- [ ] `tests/experimental/` 디렉토리 신설 (본선 CI 영향 차단)
- [ ] sLLM 모델 선택 (`BAAI/bge-small-ko-v1.5` 기본)

#### Stage 7: vector default 전환
- [ ] `RoutingConfig.mode` 기본값을 `"vector"`로 전환
- [ ] tag 관련 코드를 `archive/`로 이동
- [ ] CLAUDE.md / 프로젝트 리뷰 갱신 (§3-C "임베딩 기반 시맨틱 라우팅" 해소 표기)

### 2.2 Out of Scope

설계서 §8 미래 확장점 — 모두 별도 PDCA 사이클:
- Multi-head attention 라우팅 (what/where pathway 분리)
- ModalEncoder 통합 (`htp_multimodal_design.md` V-JEPA)
- Continuous token (변환 제거, 로컬 sLLM 필수)
- Predictive Coding `PredictiveRegion`
- Incremental PCA (Region 수 N > 1000 도달 시)
- Lateral Inhibition 국소화

이전 사이클 백로그 중 미흡수 항목 (별도 사이클):
- MED-3: 대시보드 BrainRuntime/Memory 반영 (orthogonal)
- LOW-5: Friston B3 precision 5배 증폭 시뮬레이션 (orthogonal)
- LOW-6: NGE split 파라미터 장기 시뮬레이션 (orthogonal)
- LOW-7: Memory CUSUM × SWR threshold 경계 분석 (부분적 관련 — Stage 3에서 SWR 확장하지만 경계 분석은 아님)
- LOW-8: `compress_dim=64` 재검토 (orthogonal — 본 사이클은 64 유지)

---

## 3. MED-4 ↔ G3 차이점 보완 검토

> User 지시: "세부 구현 내용과 디자인 비교하여 검토해서 차이점을 보완"

### 3.1 비교 매트릭스

| 차원 | MED-4 원본 의도 | 설계서 G3 | 차이 / 보완 필요 |
|------|--------------|----------|--------------|
| **진단** | "사용 흐름이 약함" (모호) | "RegionRuntime 상속이 잘못된 추상화" (구체) | G3가 더 정확. ✓ 흡수 |
| **해법 범위** | "사용 흐름 강화 OR 미사용 코드 정리" | LLMRegion + ExternalRegion + 4-Level CostRouter + EmbeddingBridge | G3가 훨씬 큼. ✓ 흡수 |
| **미사용 코드 정리** | 명시 안 됨 | `LLMRegionRuntime` → `archive/deprecated_phase4/` 이동 | G3가 명시. ✓ 흡수 |
| **graphify isolated 감소 검증** | 명시 안 됨 | Stage 4 Go/No-Go에 "graphify 재실행하여 isolated 노드 수 감소 확인" | G3가 명시. ✓ 흡수 |
| **실제 사용처(데모) 생성** | "사용 흐름 강화" 의도에 포함 | 명시 안 됨 ⚠️ | **보완 필요**: Stage 4에 LLMRegion 데모 추가 |
| **`LLMNode` (HTP Node 아닌 별도 클래스) 처리** | 명시 안 됨 | 설계서 §3-2에서 `LLMRegion.process()`가 `RegionSignal` 반환 — 그러나 기존 `LLMNode`(`htp/llm/llm_node.py:29`)의 위치 명시 안 됨 ⚠️ | **보완 필요**: `LLMNode`를 `LLMRegion` 내부 멤버로 유지 또는 통합 |
| **기존 `CostRouter` 인터페이스와의 충돌** | 명시 안 됨 | 4-Level 확장 — 그러나 기존 `routing_score()`/`should_block()`/`PRESSURE_BLOCK` 와의 통합 방안 명시 안 됨 ⚠️ | **보완 필요**: Stage 4 plan에 기존 4-method 보존 + `select_level()` 추가 명시 |
| **graphify 측정 기준** | 명시 안 됨 (모호) | "isolated 노드 수 감소" — 그러나 *얼마나* 감소해야 Go인지 미정의 ⚠️ | **보완 필요**: "LLM 관련 isolated 노드 ≥ 50% 감소" 같은 정량 기준 |

### 3.2 Stage 4 보완 작업 (4건)

설계서 §3에 없는 **4건의 추가 작업**을 Stage 4 안에 흡수:

| 보완 # | 작업 | 위치 | 측정 기준 |
|-------|------|------|---------|
| C-1 | `LLMRegion` 실제 사용 데모 추가 | `htp/runtime/_demo.py` 또는 신규 `examples/llm_region_demo.py` | 데모가 `routing_mode="vector"` 에서 정상 라우팅되어야 함 |
| C-2 | 기존 `LLMNode` 클래스 처리 정책 | Plan 문서에 결정 명시 — 옵션 A: `LLMRegion` 내부 멤버, 옵션 B: deprecated 후 archive | 결정 후 Stage 4 Do 단계에서 적용 |
| C-3 | `CostRouter` 기존 인터페이스 보존 | `htp/llm/cost_router.py` — `update`/`pressure`/`status`/`suggest_model`/`routing_score`/`should_block`/`report` 7-method 유지 + `select_level` 추가 | 회귀 테스트에서 기존 호출 깨지지 않아야 함 |
| C-4 | graphify isolated 감소 정량 기준 | Stage 4 Go/No-Go | LLM 관련 (llm_*, cost_router_*) isolated 노드 50% 이상 감소 |

### 3.3 Stage 4 진입 전 확정 사항

C-2 (LLMNode 처리 정책)는 Stage 4 시작 전 결정 필요. Design 단계에서 옵션 비교 후 사용자 확인.

---

## 4. Requirements

### 4.1 Functional Requirements

| ID | Stage | Requirement | Priority | Status |
|----|-------|-------------|----------|--------|
| FR-01 | 0 | `RoutingConfig` dataclass 신설 — mode/alpha/threshold_beta/warmup_steps | High | Pending |
| FR-02 | 0 | `CoherenceConfig` dataclass 신설 — conflict_threshold/agreement_threshold/novelty_boost/lsh_transition_n | High | Pending |
| FR-03 | 0 | `LLMBridgeConfig` dataclass 신설 — embedding_model/embed_dim/cost_level_thresholds | High | Pending |
| FR-04 | 0 | `PipelineConfig` dataclass 신설 — buffer_size | High | Pending |
| FR-05 | 0 | 기존 HTPConfig flat kwarg/attribute 접근 호환 유지 (deprecated warning 동반) | High | Pending |
| **FR-05.1** | **0.5** | **`htp/knowledge/` 패키지 신설 — `loop.py`, `encoder.py`, `__init__.py`, `__main__.py`** | **High** | **Pending** |
| **FR-05.2** | **0.5** | **`TextEncoder` Protocol — `encode(str) → np.ndarray[64]`. `KnowledgeLoop`/`LLMRegion`/`RegionSignature` 공유 인터페이스** | **High** | **Pending** |
| **FR-05.3** | **0.5** | **MVP `TextEncoder` 구현 — TF-IDF + JL random projection (sklearn) 또는 feature hashing** | **High** | **Pending** |
| **FR-05.4** | **0.5** | **`KnowledgeLoop` 클래스 — `ingest(text, source)` / `query(question)` / `discover()` + `knowledge_log` 누적** | **High** | **Pending** |
| **FR-05.5** | **0.5** | **`KnowledgeEntry` / `IngestResult` / `QueryResult` / `Discovery` dataclass** | **High** | **Pending** |
| **FR-05.6** | **0.5** | **CLI: `python -m htp.knowledge ingest --source <src> "<text>"`, `query "<q>"`, `discover`** | **High** | **Pending** |
| FR-06 | 1 | `RegionSignature` 클래스 — `centroid (64-dim) + count + update(input_vec) + similarity(query_vec)` | High | Pending |
| FR-07 | 1 | `RegionSignal.region_signature` 필드 추가 | High | Pending |
| FR-08 | 1 | `CoreCells._gate_vector()` — content-addressable routing + dynamic threshold | High | Pending |
| FR-09 | 1 | `routing_mode="tag"` 기본값 유지 (회귀 보호) | High | Pending |
| FR-10 | 2 | `CoreCells._gate_hybrid()` — `alpha × vec_score + (1-alpha) × tag_score` | High | Pending |
| FR-11 | 2 | α 0.1→0.9 변화 시 선택 Region 집합 변화 연속적 (cosine sim of selected set > 0.5) | Med | Pending |
| FR-12 | 3 | `CoherenceGate.bind()` — pairwise coherence + conflict detection + precision-weighted fusion | High | Pending |
| FR-13 | 3 | `BoundResponse` dataclass — `responses/coherence/conflict/escalate_to_pfc/fused_vec` | High | Pending |
| FR-14 | 3 | BrainRuntime에 CoherenceGate 삽입 (Region 응답 후, PFC 전) | High | Pending |
| FR-15 | 3 | `MemorySystem.swr_priority` 확장 — `novelty × reward × (1 + conflict_magnitude)` | Med | Pending |
| FR-16 | 4 | `ExternalRegion` 추상 클래스 — RegionRuntime 비상속, `process()/update_signature()` 프로토콜 | High | Pending |
| FR-17 | 4 | `LLMRegion(ExternalRegion)` — `vec_to_prompt`/`prompt_to_vec` (dim-tag 매핑) | High | Pending |
| FR-18 | 4 | `CostRouter.select_level()` — 4-Level 의사결정 (Local/sLLM/API 소형/API 대형) | High | Pending |
| FR-19 | 4 | 기존 `CostRouter` 7-method 보존 (C-3 보완) | High | Pending |
| FR-20 | 4 | `LLMRegionRuntime` → `archive/deprecated_phase4/` 이동 | High | Pending |
| FR-21 | 4 | `LLMRegion` 사용 데모 추가 (C-1 보완) | Med | Pending |
| FR-22 | 4 | `LLMNode` 처리 정책 결정 + 적용 (C-2 보완) | Med | Pending |
| FR-23 | 5 | `PipelinedBrainRuntime` 신설 — Thalamus/Region/Coherence/PFC 4-stage 파이프라인 | Med | Pending |
| FR-24 | 5 | `AsyncBrainRuntime` 보존 (대체 아닌 추가) | High | Pending |
| FR-25 | 6 | `EmbeddingBridge` 신설 — sLLM 양방향 프로젝션 + 온라인 학습 | Med | Pending |
| FR-26 | 6 | `tests/experimental/` 디렉토리 신설 + 본선 CI 격리 | High | Pending |
| FR-27 | 7 | `RoutingConfig.mode` 기본값 `"vector"` 전환 | Low | Pending |
| FR-28 | 7 | tag 관련 코드 `archive/` 이동 | Low | Pending |

### 4.2 Non-Functional Requirements

| Category | Criteria | Measurement |
|----------|----------|-------------|
| 회귀 안전 | 기존 57/57 + 이전 사이클 unit 46개 모두 통과 매 Stage 직후 | `pytest tests/regression/ tests/unit/` |
| 본선 신규 테스트 | **27–30개 추가 (Stage별 누적 60→65→75→78→84→87→89)** | `pytest tests/unit/`, `tests/thalamus/`, `tests/knowledge/` |
| 실험 테스트 | 4개 격리 (`tests/experimental/`) | 본선 CI 영향 없음 |
| **Stage 0.5 실용성** | **CLI 1회 라운드트립 통과 (ingest → query → discover)** | **수동 + `tests/knowledge/test_loop.py` 5개** |
| **Stage 0.5 cross-domain 발견** | **뇌과학-AI 유사도 > 뇌과학-인프라 유사도 (3 source 시나리오)** | **`test_loop_discover_cross_domain`** |
| Throughput (Stage 5) | 연속 입력 10개 기준 ≥ 1.5× 순차 | 벤치마크 스크립트 |
| Coherence 정확도 (Stage 3) | 의도적 conflict 10건 중 ≥ 9건 감지, 정합 10건 중 false positive ≤ 1건 | `test_coherence_*` |
| Cost level 분포 (Stage 4+6) | Level 1-2 처리 비율 ≥ 70% (실제 워크로드) | `CostRouter.report()` 누적 |
| graphify isolated 감소 (Stage 4) | LLM 관련 isolated 노드 ≥ 50% 감소 | `graphify` 재실행 비교 |

---

## 5. Success Criteria

### 5.1 Definition of Done

- [ ] FR-01 ~ FR-28 + **FR-05.1 ~ FR-05.6** 모두 구현
- [ ] 매 Stage 직후 회귀 57+46 = 103 테스트 통과 유지
- [ ] 본선 신규 **27-30개** + 실험 4개 모두 통과
- [ ] **Stage 0.5 cross-domain 발견 시나리오 (뇌과학/AI/인프라) Go**
- [ ] **Stage 0.5 CLI 1회 완주 — 개발자가 "실제로 써보고 싶다" 느낌 보고**
- [ ] Stage 4 후 graphify isolated 노드 (LLM 관련) ≥ 50% 감소
- [ ] Stage 5 후 PipelinedBrainRuntime throughput ≥ 1.5× 순차
- [ ] Stage 7 전환 시 실제 샘플 20+개에서 vector mode가 tag mode 동등 또는 우위
- [ ] CLAUDE.md 파일 구조 트리 갱신 (각 Stage 종료 시 또는 일괄)
- [ ] `docs/03-review/htp-project-review.md` §3-C 항목 해소 표기

### 5.2 Quality Criteria

- [ ] HTPConfig 직접 엣지 graphify상 41 → ~35 감소 (Stage 0 후)
- [ ] `BoundResponse.coherence` 평균이 신경과학 직관과 정합 (정합 응답 ≥ 0.7, conflict ≤ 0.3)
- [ ] Level 1-2 처리 비율 ≥ 70% (Stage 4+6 후, 실제 워크로드 기준)
- [ ] vec↔prompt roundtrip cosine > 0.8 (Stage 6, dim-tag 대비 우위)
- [ ] PDCA gap analysis match rate ≥ 90% (Check phase 기준)

---

## 6. Risks and Mitigation

설계서 §7의 7건 + 본 사이클 특유 위험 보완:

| 위험 | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| 회귀 깨짐 | High | Low | `routing_mode="tag"` 기본값 유지 + 매 Stage 직후 pytest 강제 |
| **Stage 0.5 MVP TF-IDF가 cross-domain similarity 발견 못 함** | **High** | **Med** | **No-Go 시 (a) feature hashing 차원 조정 (b) 간이 word2vec 평균 벡터 fallback. Stage 1-7 전제가 흔들리므로 조기 발견 자체가 가치. 최악의 경우 Stage 0.5를 EmbeddingBridge로 직접 진입 (Stage 6 앞당김)** |
| **TextEncoder Protocol이 향후 교체 시 깨짐** | **Med** | **Low** | **Protocol 정의를 `htp/knowledge/encoder.py`에 격리하여 KnowledgeLoop/LLMRegion/RegionSignature가 동일 인터페이스 사용. 교체는 Stage 6 EmbeddingBridge에서 단일 함수 교체로 끝남** |
| RegionSignature 냉시작 (centroid 영벡터) | Med | High | `RoutingConfig.warmup_steps=10` — 최초 N회 tag mode 강제. **Stage 0.5의 knowledge_log를 RegionSignature 초기 데이터로 사용 가능** |
| Dynamic threshold 불안정 (N=2-3) | Med | Med | N < 4이면 threshold 고정값 0.3 fallback |
| CoherenceGate O(N²) | Low | Low (N≥16 도달 시) | `lsh_transition_n=16` 임계 — ModalEncoder 통합과 동시 LSH 도입 |
| HTPConfig 비대화 | High | Med | **Stage 0 최우선** — 4 sub-config 분리로 직접 엣지 ~35로 감소 |
| vec↔prompt 변환 품질 | High | High | 3단계 전략 (dim-tag → EmbeddingBridge → continuous). Level 1-2 70% 목표로 비용 hedging |
| EmbeddingBridge 모델 의존성 | Med | Med | 실험 브랜치 분리. roundtrip cosine > 0.8 + dim-tag 대비 우위 미달 시 본선 머지 안 함 |
| 기존 테스트 깨짐 (routing_mode default 변경) | High | Low | tag 기본값 유지 + Stage 7에서만 전환, 그것도 Go/No-Go 통과 시 |
| **추가**: MED-4와 G3의 차이점 보완 누락 | Med | Med | **§3 별도 검토 + 4건 보완 작업 Stage 4 흡수** |
| **추가**: LLMRegion 데모 부재 → 실사용 검증 안 됨 | Med | Med | **C-1: Stage 4에 데모 의무화** |
| **추가**: `LLMNode` (기존 클래스) 처리 정책 미정 | Med | High | **C-2: Stage 4 Design 단계에서 사용자 확인 필수** |
| **추가**: Stage가 8개로 길다 — context 손실 위험 | High | Med | Stage별 commit + Plan/Design/Do/Check/Report 사이클 단위 단축 |

---

## 7. Impact Analysis

### 7.1 Changed Resources

| Resource | Type | Change | Stage |
|----------|------|--------|:----:|
| `htp/core/config.py` | Source | 4 sub-config 추가 (이전 사이클 facade 확장) | 0 |
| **`htp/knowledge/__init__.py`** | **New** | **패키지 진입점, export 정의** | **0.5** |
| **`htp/knowledge/encoder.py`** | **New** | **`TextEncoder` Protocol + MVP TF-IDF+JL 구현** | **0.5** |
| **`htp/knowledge/loop.py`** | **New** | **`KnowledgeLoop` + dataclass (`KnowledgeEntry`, `IngestResult`, `QueryResult`, `Discovery`)** | **0.5** |
| **`htp/knowledge/__main__.py`** | **New** | **argparse CLI (`python -m htp.knowledge ingest|query|discover`)** | **0.5** |
| **`tests/knowledge/`** | **New dir** | **5 본선 테스트 (loop_ingest_basic, query_neighbor, discover_cross_domain, text_encoder_interface, empty_state)** | **0.5** |
| **`requirements.txt`** | **Source** | **`scikit-learn` 추가 (TF-IDF 용도)** | **0.5** |
| `htp/thalamus/signature.py` | New | `RegionSignature` (Stage 0.5 `TextEncoder` Protocol 재사용) | 1 |
| `htp/thalamus/core_cells.py` | Source | `_gate_vector` / `_gate_hybrid` 추가, 기존 `gate()` 분기 | 1, 2 |
| `htp/thalamus/region_signal.py` | Source | `region_signature` 필드 추가 | 1 |
| `htp/thalamus/coherence.py` | New | `CoherenceGate` | 3 |
| `htp/thalamus/types.py` | New 또는 region_signal.py 확장 | `BoundResponse` | 3 |
| `htp/runtime/brain_runtime.py` | Source | CoherenceGate 삽입, top-down feedback 조정 | 3 |
| `htp/memory/memory_system.py` | Source | `swr_priority` conflict 반영 | 3 |
| `htp/runtime/external_region.py` | New | `ExternalRegion` 추상 클래스 | 4 |
| `htp/llm/llm_region.py` | New | `LLMRegion` (LLMRegionRuntime 대체) | 4 |
| `htp/llm/llm_region_runtime.py` | Move | → `archive/deprecated_phase4/` | 4 |
| `htp/llm/cost_router.py` | Source | 4-Level `select_level` 추가, 기존 7-method 보존 | 4 |
| `htp/llm/llm_node.py` | TBD | C-2 결정에 따라 (옵션 A: LLMRegion 내부, B: archive) | 4 |
| `htp/runtime/_demo.py` 또는 `examples/llm_region_demo.py` | New 또는 확장 | C-1 LLMRegion 데모 | 4 |
| `htp/runtime/pipelined_brain.py` | New | `PipelinedBrainRuntime` | 5 |
| `htp/llm/embedding_bridge.py` | New (실험 브랜치) | `EmbeddingBridge` | 6 |
| `tests/experimental/` | New dir | 본선 CI 격리 | 6 |
| `requirements.txt` | Source (실험 브랜치) | `sentence-transformers` 추가 | 6 |
| `CLAUDE.md` | Doc | 파일 구조 트리 + DAG + routing_mode 기본값 갱신 | 7 |

### 7.2 Current Consumers

| Resource | Operation | Path | Impact |
|----------|-----------|------|--------|
| `CoreCells.gate()` | call | `htp/thalamus/thalamus.py`, `htp/runtime/brain_runtime.py` | None — 기본 mode="tag" 유지 |
| `LLMRegionRuntime` | import | `htp/__init__.py`, 외부 사용자 코드 | re-export 유지 (deprecated warning) |
| `CostRouter` | import + call | `htp/llm/llm_region_runtime.py` (archive 이동 후 `llm_region.py`) | 7-method 인터페이스 보존 |
| `MemorySystem.swr_priority` | call | `htp/memory/memory_system.py` 내부 | 확장 (기본 동작 동일, conflict=0 시 곱셈 1.0) |
| `BrainRuntime.run()` | call | 사용자 코드 | CoherenceGate가 내부에서 작동, 외부 인터페이스 동일 |

### 7.3 Verification

- [ ] 모든 consumer가 변경 후에도 정상 작동 (Stage별 회귀 + unit + import_paths)
- [ ] `routing_mode="tag"` 일 때 모든 회귀 통과
- [ ] LLMRegion 데모가 `routing_mode="vector"` 에서 정상 라우팅

---

## 8. Architecture Considerations

### 8.1 Project Level

Research / Library (이전 사이클과 동일). bkit 표준 3-level 외.

### 8.2 Key Architectural Decisions

| Decision | Options | Selected | Rationale |
|----------|---------|----------|-----------|
| Stage 순서 | (a) Config 마지막 (b) Config 처음 | **(b) Config 처음 (Stage 0)** | 토대 흔들리면 이후 전체 흔들림 |
| **루프 폐쇄 시점 (v4)** | **(a) Stage 7 후 (b) Stage 0 직후** | **(b) Stage 0.5에서 조기 폐쇄** | **"매일 쓰는 도구가 되어야 가치"라는 원칙. 실사용 피드백으로 Stage 1-7 교정** |
| **MVP TextEncoder** | **(a) sentence-transformer (Stage 6 의존) (b) TF-IDF+JL (의도적 조잡)** | **(b) TF-IDF+JL** | **루프 폐쇄가 목적, 품질은 Stage 6에서. 의존성 최소화 (scikit-learn만)** |
| **TextEncoder 인터페이스** | **(a) 각 모듈별 구현 (b) 공유 Protocol** | **(b) 공유 Protocol** | **KnowledgeLoop/LLMRegion/RegionSignature가 동일 인터페이스. Stage 6 교체 시 한 곳만** |
| Vector mode 전환 | (a) Big-bang (b) 점진적 | **(b) 점진: tag → hybrid → vector** | 회귀 보호 + 마이그레이션 안전 |
| Threshold 방식 | (a) 고정 (b) Dynamic μ+β×σ | **(b) Dynamic** | Region 수 적응 + 자극 명확도 적응 |
| Conflict 처리 | (a) 자동 fusion (b) PFC 에스컬레이션 | **둘 다** — coherent → fusion, conflict → PFC | 신경과학 정합 |
| LLM 상속 vs 프로토콜 | (a) RegionRuntime 상속 (현재) (b) ExternalRegion 프로토콜 | **(b) 프로토콜** | "LLM에 PageRank 필요한가" 문제 해소 |
| vec↔prompt 변환 | (a) 즉시 EmbeddingBridge (b) 단기 dim-tag → 중기 Bridge | **(b) 3단계** | LLMRegion 인터페이스 확정이 우선, 품질은 점진 |
| Pipeline | (a) AsyncBrainRuntime 대체 (b) 별도 PipelinedBrainRuntime | **(b) 별도** | 기존 인터페이스 보존 |
| Stage 6 브랜치 | (a) 본선 (b) 실험 브랜치 | **(b) 실험 — Stage 6 진입 시 생성** | sLLM 의존성 격리, 본선 진행 속도 보호 |
| Stage 6 본선 머지 | (a) Stage 6 완료 즉시 (b) Go/No-Go 통과 시에만 | **(b) Go/No-Go** | 실험 결과가 dim-tag 대비 우위가 없으면 본선 머지 안 함 |

### 8.3 Layered Structure (Stage 7 완료 후 목표)

```
htp/
├── __init__.py                       (공개 API 표면 — 무변경)
├── core/
│   ├── config.py                     +RoutingConfig/CoherenceConfig/LLMBridgeConfig/PipelineConfig
│   ├── weight_matrix.py              (변경 없음)
│   ├── hub_formation.py              (변경 없음)
│   ├── pruning.py                    (변경 없음)
│   ├── activation.py                 (변경 없음)
│   └── node_generation_engine.py     (변경 없음)
├── knowledge/                        ★★ NEW (Stage 0.5) — Knowledge Loop MVP
│   ├── __init__.py
│   ├── encoder.py                    TextEncoder Protocol + MVP TF-IDF+JL
│   ├── loop.py                       KnowledgeLoop + dataclass
│   └── __main__.py                   CLI dispatcher (argparse)
├── runtime/
│   ├── htp_runtime.py                (변경 없음)
│   ├── _demo.py                      ← LLMRegion 데모 추가 (C-1)
│   ├── region_runtime.py             (변경 없음)
│   ├── brain_runtime.py              CoherenceGate 삽입
│   ├── pipelined_brain.py            ★ NEW (Stage 5)
│   ├── external_region.py            ★ NEW (Stage 4)
│   ├── cortical_connections.py       (변경 없음)
│   └── async_brain_runtime.py        (보존)
├── thalamus/
│   ├── thalamus.py                   (변경 없음 — Region 호출만)
│   ├── core_cells.py                 _gate_vector/_gate_hybrid 추가
│   ├── signature.py                  ★ NEW (Stage 1) — TextEncoder Protocol 재사용
│   ├── coherence.py                  ★ NEW (Stage 3)
│   ├── matrix_cells.py               (변경 없음)
│   ├── nge_trigger.py                (변경 없음)
│   ├── region_signal.py              +region_signature 필드, +BoundResponse
│   └── top_down.py                   (변경 없음)
├── memory/
│   └── memory_system.py              swr_priority conflict 반영
└── llm/
    ├── llm_region.py                 ★ NEW (Stage 4) — TextEncoder Protocol 재사용
    ├── cost_router.py                +select_level (기존 7-method 보존)
    ├── llm_node.py                   C-2 결정에 따라
    └── embedding_bridge.py           ★ NEW (Stage 6, 실험 브랜치) — TextEncoder 구현 교체

archive/deprecated_phase4/
    └── llm_region_runtime.py         (Stage 4에서 이동)

tests/
├── regression/                       57개 (변경 없음, 매 Stage 통과)
├── unit/                             46개 (이전 사이클, 변경 없음)
├── knowledge/                        ★★ NEW (Stage 0.5) — 5개
│   └── test_loop.py                  5개 (ingest_basic, query_neighbor,
│                                          discover_cross_domain,
│                                          text_encoder_interface, empty_state)
├── thalamus/                         ★ NEW — 본선 신규 22-25개
│   ├── test_signature.py             4개 (Stage 1)
│   ├── test_car.py                   6개 (Stage 1)
│   ├── test_hybrid.py                3개 (Stage 2)
│   ├── test_coherence.py             6개 (Stage 3)
│   ├── test_external_region.py       3개 (Stage 4)
│   └── test_pipeline.py              2개 (Stage 5)
└── experimental/                     ★ NEW — 실험 4개 (Stage 6, 브랜치 한정)
    └── test_embedding_bridge.py      4개
```

**핵심 설계 결정**: `TextEncoder` Protocol은 `htp/knowledge/encoder.py`에 단일 정의되고, `KnowledgeLoop`/`RegionSignature`/`LLMRegion` 모두 이 Protocol 의존성으로 사용한다. Stage 6 EmbeddingBridge가 완성되면 단일 구현 교체로 전 시스템 품질이 동시에 향상된다.

---

## 9. Stage 일정 (소요 기간만, 마감일 없음) — v4 갱신

| Stage | 소요 (예상) | 누적 테스트 | 비고 |
|-------|----------|----------|------|
| 0 — Config 분리 | ~0.5주 | 60 | 이전 사이클 facade 확장이라 가벼움 |
| **0.5 — Knowledge Loop MVP** | **~0.5–1주** | **65** | **TF-IDF+JL + CLI + 5 테스트. cross-domain discover 시나리오 통과가 핵심** |
| 1 — Signature + CAR | ~1.5주 | **75** | 신경과학 검증 시간 포함. **Stage 0.5 knowledge_log를 테스트 데이터로 활용** |
| 2 — Hybrid 검증 | ~0.5주 | **78** | α 스위프 + 데모 비교 |
| 3 — CoherenceGate + Memory | ~1.5주 | **84** | conflict detection 신중 설계 |
| 4 — ExternalRegion + LLMRegion (+MED-4 보완 4건) | ~2주 | **87** | 가장 큰 단일 Stage. **Stage 0.5 discover() 결과를 LLMRegion이 자연어 해석** |
| 5 — Pipeline | ~1주 | **89** | asyncio 검증 + 벤치마크 |
| 6 — EmbeddingBridge *(실험 브랜치)* | ~1주 | 별도 4개 | sLLM 의존성 설치 + roundtrip 검증. **Stage 0.5 TF-IDF vs Bridge A/B 비교** |
| 7 — Vector default 전환 | ~0.5주 | 89 재실행 | tag 코드 archive 이동 |

총 소요 예상: **~9–9.5주** (Stage 0.5 추가, Stage 6 본선 머지 포함 시). **마감일은 적지 않음**.

**v4 Stage 진행 원칙**:
1. **토대 먼저** — Config (Stage 0)
2. **루프를 먼저 닫는다** (v4 신규) — Knowledge Loop MVP (Stage 0.5)
3. **본선/실험 분리** — Stage 6은 별도 브랜치
4. **각 Stage Go/No-Go** — 테스트 개수 + 제품적 성공 기준

---

## 10. Convention Prerequisites

### 10.1 Existing

- [x] CLAUDE.md
- [x] `tests/regression/` (57) + `tests/unit/` (46)
- [x] DAG 규칙 (이전 사이클): `htp/core/` ← `htp/runtime/`
- [x] HTPConfig facade (`htp/core/config.py`)

### 10.2 To Enforce in This Cycle

| Category | Enforce |
|----------|---------|
| DAG 확장 | `htp/thalamus/` + **`htp/knowledge/`** 도 `htp/runtime/` 미참조 (단방향 추가) — `test_no_circular_deps.py` 확장 |
| Config 추가 패턴 | 신규 파라미터는 항상 sub-config에 (`HTPConfig` 직접 추가 금지) |
| Test 위치 | **`tests/knowledge/` (Stage 0.5)**, `tests/thalamus/` (Stage 1+) 본선, `tests/experimental/` (Stage 6) |
| Branch 정책 | Stage 6은 `experiment/embedding-bridge` 분기, 본선 머지는 Go/No-Go 통과 시에만 |
| **TextEncoder Protocol 단일 정의** | **`htp/knowledge/encoder.py`에만 정의. 다른 모듈은 import만. 구현 교체 시 한 곳만 변경** |

---

## 11. Next Steps

1. [ ] `/pdca design htp-thalamus-car` — 3 아키텍처 옵션 비교 (전체 Stage 0-7 9 단계 진행 방식)
2. [ ] Design Checkpoint 3에서 Stage 운영 방식 선택 (예: Stage 단위 PDCA mini-cycle vs 일괄)
3. [ ] Design 후 `/pdca do htp-thalamus-car --scope stage-0,stage-0.5` (토대 + 루프 폐쇄 우선)
4. [ ] **Stage 0.5 Go/No-Go 통과 확인 후 Stage 1 진입**. No-Go 시 TextEncoder 구현 재검토
5. [ ] Stage 7 완료 시 `/pdca report` + 프로젝트 리뷰 §3-C 해소 표기

---

## Version History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 0.1 | 2026-05-16 | Initial — 설계서 v3 (Rev 1.2) 전체 8 Stage 흡수, MED-4 vs G3 차이점 보완 4건 추가 | Mindbuild |
| **0.2** | **2026-05-17** | **설계서 v4 (Rev 1.3) 반영 — Stage 0.5 Knowledge Loop MVP 삽입. G4(실사용 루프 부재) 갭 추가, FR-05.1~05.6 6건 추가, `htp/knowledge/` 패키지 + `TextEncoder` Protocol 통합, 본선 테스트 22-25→27-30개, 누적 60→65→75→78→84→87→89, 일정 ~8.5주→~9-9.5주** | **Mindbuild** |
