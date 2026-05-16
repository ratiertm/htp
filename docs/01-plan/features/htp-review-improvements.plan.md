---
template: plan
version: 1.3
feature: htp-review-improvements
date: 2026-05-16
author: Mindbuild
project: HTP (Hub Topology Programming)
status: Draft
---

# htp-review-improvements Planning Document

> **Summary**: `docs/03-review/htp-project-review.md`에서 발견된 코드 약점·위험을 PDCA로 개선. Phase A에서 HIGH 2건(HTPConfig god-object DI, htp_runtime.py 967줄 분할)만 끊고, MED/LOW는 백로그로.
>
> **Project**: HTP
> **Version**: post-`6be8746` (Phase 1-4 + Review Feedback 완료, 회귀 57/57)
> **Author**: Mindbuild
> **Date**: 2026-05-16
> **Status**: Draft

---

## Executive Summary

| Perspective | Content |
|-------------|---------|
| **Problem** | `RegionRuntime` 59 엣지·`HTPConfig` 41 엣지·`htp_runtime.py` 967줄. graphify 분석상 god-object/god-file 후보. 단위 테스트가 거의 풀 스택을 요구함 (test_stage3_precision이 그 예). 향후 Phase 3-4 확장(임베딩 라우팅·Predictive Coding) 시 결합도가 발목을 잡을 위험. |
| **Solution** | HTPConfig를 sub-config로 분리(`HubConfig`/`PruneConfig`/`NGEConfig`/`ActivationConfig`)하고 각 엔진은 자기 sub-config만 받는 **Constructor Injection**으로 전환. Phase 1 4 엔진은 `htp/core/{weight_matrix,hub_formation,pruning,activation}.py`로 파일 분리. `htp_runtime.py`는 `HTPRuntime` 오케스트레이터 + 데코레이터(~200줄)만 남김. 공개 API(`from htp import …`)는 100% 유지. |
| **Function/UX Effect** | 사용자 코드 무변경. 단위 테스트가 엔진별로 독립 가능 → 새 엔진·새 Region 타입 추가 시 진입 비용 절감. `LLMRegionRuntime`이 PageRank 허브 형성을 정말로 필요로 하는지 같은 의문도 sub-config 토글로 검증 가능. |
| **Core Value** | "Region IS-A HTPRuntime" 상속이 부른 god-backbone 위험을 *공개 API를 깨지 않고* 내부 구조로만 해소. 회귀 테스트 57/57은 절대 깨지 않는 것이 제약. |

---

## Context Anchor

| Key | Value |
|-----|-------|
| **WHY** | god-object/god-file이 향후 Phase 확장과 단위 테스트성을 막는 구조적 부채. 지금 갚지 않으면 임베딩 라우팅·Predictive Coding 도입 시 폭증. |
| **WHO** | HTP 개발자 본인(향후 컨트리뷰터 포함). 사용자(`from htp import …`)는 무영향. |
| **RISK** | 회귀 테스트 57/57 깨짐 → 즉시 롤백. import 경로 변경이 외부 `static/index.html`·`server.py`에 누설되면 안 됨. |
| **SUCCESS** | (1) 회귀 57/57 + 신규 unit ~20개 모두 통과 (2) `htp_runtime.py` ≤250줄 (3) HTPConfig 직접 참조 위치 50% 이상 감소 (4) `from htp import …` 모든 심볼 동일하게 import 가능. |
| **SCOPE** | Phase A = HIGH 1·2만. Phase B(MED 3·4)·Phase C(LOW 5-8)는 백로그 항목으로 등록만. |

---

## 1. Overview

### 1.1 Purpose

리뷰 산출물(`docs/03-review/htp-project-review.md` §5 "약점/코드 냄새")에서 식별된 구조적 부채를 갚는다. 새 기능 추가가 아니라 **내부 구조 리팩토링**으로, 외부 동작은 100% 동일해야 한다.

### 1.2 Background

`6be8746` 커밋으로 LeCun·Friston·Memory 리뷰 반영 + 5종 알고리즘 버그 수정이 완료된 직후, graphify 기반 코드 그래프 분석(718 노드·1,741 엣지)에서 다음이 드러남:

- `RegionRuntime` 59 엣지 (betweenness 0.166, 그래프 최고)
- `HTPConfig` 41 엣지 (god-object 후보)
- `htp_runtime.py` 967줄 단일 파일에 Phase 1 4 엔진 + WeightMatrix + decorator + demo
- 단위 테스트 `test_stage3_precision.py`가 `precision` 필드 한 줄 검증을 위해 거의 풀 스택을 띄움

### 1.3 Related Documents

- 리뷰 원본: `docs/03-review/htp-project-review.md` §5 (강점/약점/위험)
- graphify 보고: `graphify-out/GRAPH_REPORT.md` (god nodes · isolated nodes · hyperedges)
- 직전 PDCA: `docs/01-plan/features/htp-phase2-integration.plan.md` (선행 작업 — 본 작업의 제약 조건 베이스라인)
- 아키텍처 원본: `architecture/htp_architecture_design.md`

---

## 2. Scope

### 2.1 In Scope (Phase A)

- [ ] **HIGH-1**: `HTPConfig`를 sub-config(`HubConfig`/`PruneConfig`/`NGEConfig`/`ActivationConfig`)로 분리. `HTPConfig`는 sub-config들을 묶는 thin facade로 유지.
- [ ] **HIGH-1**: 각 엔진(`HubFormationEngine`/`PruningEngine`/`NodeGenerationEngine`/`ActivationEngine`) 생성자가 sub-config만 받도록 변경. `HTPRuntime`이 `HTPConfig`에서 sub-config를 꺼내 주입.
- [ ] **HIGH-2**: `htp/core/` 디렉토리 확장 — `weight_matrix.py` · `hub_formation.py` · `pruning.py` · `activation.py` 파일로 분리. 기존 `htp/core/node_generation_engine.py`는 위치 유지.
- [ ] **HIGH-2**: `htp/runtime/htp_runtime.py`에는 `HTPRuntime` 클래스 + 데코레이터(`tag`, `terminal`) + `Node`/`RunResult` dataclass + demo만 남김 (목표 ≤250줄).
- [ ] **호환성**: `htp/__init__.py` 공개 API 93줄 100% 유지. `from htp import WeightMatrix, HubFormationEngine, …` 모두 동일하게 import 가능 (`htp/runtime/htp_runtime.py`에서 re-export).
- [ ] **테스트**: 기존 회귀 57/57 통과 유지 + 신규 unit test ~20개 추가 (`tests/unit/`).

### 2.2 Out of Scope (백로그 → Phase B/C)

**Phase B (MED, 별도 PDCA 사이클)**
- 대시보드(`static/index.html`) BrainRuntime/Memory 반영
- LLMNode/CostRouter 사용 흐름 강화 또는 미사용 코드 정리

**Phase C (LOW, 별도 PDCA 사이클)**
- Friston B3 precision [0.1, 5.0] 5배 증폭 영향 시뮬레이션
- NGE split 파라미터 장기 시뮬레이션 (maturity_calls / global_cooldown / max_gen_per_run)
- Memory CUSUM × SWR threshold 0.5 경계 분석
- `compress_dim = 64` JL Lemma 가정 (N=1000) 재검토

---

## 3. Requirements

### 3.1 Functional Requirements

| ID | Requirement | Priority | Status |
|----|-------------|----------|--------|
| FR-01 | `HubConfig`/`PruneConfig`/`NGEConfig`/`ActivationConfig` 4개 dataclass 신설. 기본값은 현재 `HTPConfig` 값을 그대로 옮김 | High | Pending |
| FR-02 | `HTPConfig`는 위 4개를 필드로 포함하는 facade. `HTPConfig()` 호출은 4개 sub-config 기본값을 자동 생성 | High | Pending |
| FR-03 | `HubFormationEngine.__init__(wm, cfg: HubConfig)` 형태로 변경. 다른 엔진도 동일 패턴 | High | Pending |
| FR-04 | `HTPRuntime`이 sub-config를 꺼내 각 엔진에 분배. 외부 호출(`HTPRuntime(HTPConfig())`)은 무변경 | High | Pending |
| FR-05 | `htp/core/weight_matrix.py` 신설. `WeightMatrix` 이동 | High | Pending |
| FR-06 | `htp/core/hub_formation.py` 신설. `HubFormationEngine` 이동 | High | Pending |
| FR-07 | `htp/core/pruning.py` 신설. `PruningEngine` + `PruneStrategy` 이동 | High | Pending |
| FR-08 | `htp/core/activation.py` 신설. `ActivationEngine` + 데코레이터(`tag`, `terminal`) 이동 | High | Pending |
| FR-09 | `htp/runtime/htp_runtime.py`는 `HTPRuntime`/`Node`/`RunResult`/`HTPConfig`/`FIRE_FLOOR` + demo만 유지 | High | Pending |
| FR-10 | 기존 `from htp.runtime.htp_runtime import WeightMatrix, HubFormationEngine, …` 경로도 동작 (re-export) | High | Pending |
| FR-11 | `tests/unit/` 디렉토리 신설. ~20개 단위 테스트로 sub-config 독립 동작 / 엔진 독립 생성 / 순환 의존 부재 검증 | High | Pending |

### 3.2 Non-Functional Requirements

| Category | Criteria | Measurement Method |
|----------|----------|-------------------|
| 회귀 안전 | 회귀 테스트 57/57 통과 유지 | `pytest tests/regression/ -v` |
| 가독성 | `htp/runtime/htp_runtime.py` ≤ 250줄 | `wc -l` |
| 결합도 감소 | `HTPConfig` 직접 참조 위치 50% 이상 감소 | `grep -r "HTPConfig" htp/ \| wc -l` (전후 비교) |
| 단위 테스트 가능성 | 각 엔진을 sub-config만으로 독립 생성 가능 | `tests/unit/test_*.py` 신규 ~20개 |
| 공개 API 호환 | `htp/__init__.py` export 모두 import 가능 | `python -c "from htp import *"` |

---

## 4. Success Criteria

### 4.1 Definition of Done

- [ ] FR-01 ~ FR-11 모두 구현 완료
- [ ] 회귀 테스트 57/57 통과 유지
- [ ] 신규 unit test ~20개 모두 통과
- [ ] `htp/runtime/htp_runtime.py` 250줄 이하
- [ ] `HTPConfig` 참조 50% 이상 감소
- [ ] `python -c "from htp import *"` 무에러
- [ ] `server.py` 무수정으로 동작
- [ ] CLAUDE.md 갱신 (파일 구조 트리)

### 4.2 Quality Criteria

- [ ] 기존 import 경로 모두 동작 (`from htp.runtime.htp_runtime import WeightMatrix` 같은 경로도 호환)
- [ ] 순환 import 없음 (`python -c "import htp"` 깨끗)
- [ ] 신규 파일은 모두 ≤ 300줄
- [ ] PDCA gap analysis match rate ≥ 90%

---

## 5. Risks and Mitigation

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| 회귀 테스트 깨짐 | High | Medium | 매 파일 이동마다 `pytest tests/regression/` 즉시 실행. 깨지면 그 단계에서 중단·롤백 |
| 순환 import 발생 (`htp/core/activation.py` ↔ `htp/runtime/htp_runtime.py`) | High | High | `htp/core/`는 `htp/runtime/`을 import 금지 원칙. `HTPRuntime`만 모든 `htp/core/*` import 가능 (DAG 강제) |
| `server.py` / `static/index.html`이 내부 경로에 의존했을 가능성 | Medium | Low | Phase A 시작 직후 `grep -r "htp_runtime" server.py static/`로 의존 매핑 확인 |
| Sub-config 호환성 (`HTPConfig(hub_pr_threshold=3.0)` 같은 기존 사용 코드) | Medium | Medium | `HTPConfig`에 `__init__` 키워드 호환 레이어 추가 (옛 키워드 → `HubConfig`로 위임) |
| 작업 범위 폭주 (refactor 중 "이것도 고치자" 욕구) | Medium | High | Phase A 외 항목은 무조건 백로그(Out of Scope §2.2)로. 발견 즉시 `docs/01-plan/backlog.md`에 기록만 |

---

## 6. Impact Analysis

### 6.1 Changed Resources

| Resource | Type | Change Description |
|----------|------|--------------------|
| `htp/runtime/htp_runtime.py` | Source file (967줄) | 4 엔진을 `htp/core/`로 이동. HTPRuntime + 데코레이터만 유지. re-export 추가 |
| `htp/core/__init__.py` | Module init | 새 엔진 export 추가 |
| `htp/core/weight_matrix.py` | New file | `WeightMatrix` 이동 |
| `htp/core/hub_formation.py` | New file | `HubFormationEngine` 이동 |
| `htp/core/pruning.py` | New file | `PruningEngine` + `PruneStrategy` 이동 |
| `htp/core/activation.py` | New file | `ActivationEngine` + `tag` + `terminal` + `Node` 이동 |
| `HTPConfig` (dataclass) | API | sub-config 4개 분리, facade 호환 레이어 추가 |
| `tests/unit/` | New directory | ~20개 단위 테스트 신설 |
| `CLAUDE.md` | Doc | 파일 구조 트리 갱신 |

### 6.2 Current Consumers

| Resource | Operation | Code Path | Impact |
|----------|-----------|-----------|--------|
| `WeightMatrix` | import | `htp/__init__.py`, `htp/runtime/region_runtime.py`, tests | None (re-export 유지) |
| `HubFormationEngine` | import | `htp/__init__.py`, tests | None (re-export 유지) |
| `PruningEngine`/`PruneStrategy` | import | `htp/__init__.py`, tests | None (re-export 유지) |
| `ActivationEngine` | import | `htp/__init__.py` (via `htp/runtime/htp_runtime.py`) | None (re-export 유지) |
| `tag`/`terminal` 데코레이터 | import | `htp/__init__.py`, test files | None (re-export 유지) |
| `HTPConfig` | constructor | `HTPRuntime(HTPConfig(...))`, tests | Compat layer로 키워드 호환 유지 |
| `HTPConfig.hub_pr_threshold` | attribute access | `HubFormationEngine.step()` | Sub-config로 이전. facade가 `__getattr__`로 위임 |
| `htp_runtime.demo()` | function | CLI demo | None (위치 유지) |

### 6.3 Verification

- [ ] 모든 consumer는 위 표에서 verified — re-export로 무변경 동작 확인
- [ ] auth/permission 무관 (라이브러리 코드)
- [ ] `HTPConfig` 키워드 호환 레이어 단위 테스트 추가 (FR-11에 포함)

---

## 7. Architecture Considerations

### 7.1 Project Level Selection

| Level | Characteristics | Recommended For | Selected |
|-------|-----------------|-----------------|:--------:|
| Starter | Simple structure | Static sites | ☐ |
| Dynamic | Feature-based modules, BaaS | Web apps | ☐ |
| **Research / Library** | scientific Python package | HTP (this project) | ☑ |

> **Note**: HTP는 Web/SaaS가 아닌 연구용 Python 라이브러리. bkit 표준 3-level에는 정확히 매치되지 않음. Enterprise에 가까우나 microservices/DI framework는 미사용.

### 7.2 Key Architectural Decisions

| Decision | Options | Selected | Rationale |
|----------|---------|----------|-----------|
| DI 방식 | Manual constructor / DI framework (`dependency-injector`) / Service locator | **Manual Constructor Injection** | 외부 라이브러리 의존 불필요. dataclass + 명시적 주입으로 충분 |
| Config 구조 | 단일 dataclass / Sub-config 분리 / Pydantic Settings | **Sub-config 4개 + Facade 유지** | 사용자 답변 반영. 호환성 + 결합도 둘 다 해소 |
| 파일 위치 | `htp/engines/` 신설 / `htp/core/` 확장 / `htp/runtime/` 분할 | **`htp/core/` 확장** | 기존 `htp/core/node_generation_engine.py`와 의미 통일 ("core engines") |
| 공개 API | Breaking change 허용 / 100% 유지 | **100% 유지** | 사용자 답변 반영. `htp/__init__.py` re-export |
| 테스트 전략 | 기존 회귀만 / 계약 검증 ~5개 / 전면적 unit ~20개 | **전면적 unit ~20개** | 사용자 답변 반영. 향후 Phase 확장 시 안전망 |
| Dependency Direction | `htp/runtime/` → `htp/core/` (단방향, DAG) | **단방향 강제** | 순환 import 위험 차단 |

### 7.3 Layered Structure

```
htp/
├── __init__.py                       공개 API 표면 (변경 없음)
├── core/                             ← Phase 1 엔진 + dataclass (DI 의존성 없음)
│   ├── __init__.py
│   ├── config.py                     ★ NEW — HubConfig/PruneConfig/NGEConfig/ActivationConfig
│   ├── weight_matrix.py              ★ NEW — WeightMatrix (htp_runtime.py에서 이동)
│   ├── hub_formation.py              ★ NEW — HubFormationEngine
│   ├── pruning.py                    ★ NEW — PruningEngine + PruneStrategy
│   ├── activation.py                 ★ NEW — ActivationEngine + Node + tag/terminal
│   └── node_generation_engine.py    (위치 유지)
├── runtime/                          ← 오케스트레이션 (htp/core/ 만 import)
│   ├── htp_runtime.py                ≤250줄 — HTPRuntime + HTPConfig facade + demo + re-exports
│   ├── region_runtime.py
│   ├── brain_runtime.py
│   ├── cortical_connections.py
│   └── async_brain_runtime.py
├── thalamus/                         (변경 없음)
├── memory/                           (변경 없음)
└── llm/                              (변경 없음)

tests/
├── regression/                       57개 (변경 없음, 모두 통과 유지)
└── unit/                             ★ NEW — ~20개 단위 테스트
    ├── test_config_isolation.py      sub-config 독립 / facade 호환
    ├── test_engine_di.py             각 엔진을 sub-config만으로 생성
    ├── test_import_paths.py          공개 API + 기존 import 경로 모두 동작
    └── test_no_circular_deps.py      htp/core/ → htp/runtime/ import 금지
```

---

## 8. Convention Prerequisites

### 8.1 Existing Project Conventions

- [x] `CLAUDE.md` 핵심 구조·결정 정리됨
- [ ] `docs/01-plan/conventions.md` 미존재 (본 사이클 산출물 아님)
- [x] `requirements.txt` (pytest 포함)

### 8.2 Conventions to Enforce in This Cycle

| Category | Current | Enforced in Phase A |
|----------|---------|---------------------|
| Import direction | 암묵적 | **`htp/core/` → `htp/runtime/` import 금지 (DAG 강제)** |
| Engine constructor | `(wm, cfg: HTPConfig)` | `(wm, cfg: <SubConfig>)` |
| Re-export | 없음 | `htp/runtime/htp_runtime.py`가 `htp/core/*`의 심볼을 그대로 재공개 |
| Test layout | `tests/regression/`만 | `tests/regression/` + `tests/unit/` 분리 |

### 8.3 Environment Variables Needed

해당 없음. 라이브러리 코드 변경뿐.

### 8.4 Pipeline Integration

해당 없음 — bkit 9-phase pipeline은 Web 프로젝트용. HTP는 PDCA만 적용.

---

## 9. Next Steps

1. [ ] `/pdca design htp-review-improvements` — 3가지 아키텍처 옵션 비교(Minimal/Clean/Pragmatic)
2. [ ] Design 승인 후 `/pdca do htp-review-improvements`
3. [ ] Phase A 완료 후 `/pdca analyze htp-review-improvements` — 회귀 + 신규 unit + match rate 검증
4. [ ] match rate ≥90%면 `/pdca report` → 완료. 미달이면 `/pdca iterate`
5. [ ] Phase B(MED) / Phase C(LOW)는 별도 PDCA 사이클로

---

## Version History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 0.1 | 2026-05-16 | Initial draft — 리뷰 §5 약점 8건 중 HIGH 2건으로 범위 한정 | Mindbuild |
