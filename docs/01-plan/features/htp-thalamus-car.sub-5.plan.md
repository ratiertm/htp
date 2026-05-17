---
template: plan
version: 1.0
feature: htp-thalamus-car
sub_cycle: sub-5 (Stage 6 EmbeddingBridge — 실험 브랜치)
date: 2026-05-17
author: Mindbuild
project: HTP (Hub Topology Programming)
status: Draft
branch: experiment/embedding-bridge
---

# htp-thalamus-car sub-5 Plan — EmbeddingBridge (Stage 6)

> **Summary**: 시나리오 D (20 paper / 4 도메인) 에서 TF-IDF + JL 인코더가 query top-1 0/4 의 정량 한계를 노출. 본질 해결을 위해 **사전학습 임베딩 모델** 을 `TextEncoder` Protocol 의 추가 구현체 (`EmbeddingBridge`) 로 도입. **D1-D4 design 원칙으로 LLM 종속 방지** — HTP 의 brain-like 구조 (Hub/Memory/Coherence) 는 그대로 자체 학습 유지.
>
> **Project**: HTP
> **Predecessor**: L2 sidequest (commit `ca27b0b`, baseline 172) + sub-3 (Stage 3 CoherenceGate)
> **Branch**: `experiment/embedding-bridge` (Plan §SCOPE 명시 — Go/No-Go 통과 시 main merge)
> **Selected Architecture**: TBD (Design 단계 — Option B/C 후보)

---

## Executive Summary

| Perspective | Content |
|-------------|---------|
| **Problem** | sub-1~L2 sidequest 의 TF-IDF + JL 인코더는 *데이터 양/품질로 해결 안 되는* 본질적 한계 — 20 paper / 4 도메인 / 영문 학술 abstract 200+ 단어 환경에서도 **query top-1 0/4, discover 정확도 38%**. 학술 공통 어휘 ("framework", "model", "approach") 가 신호를 덮고, JL random projection 이 의미를 보존 못함. 한국어는 형태소 분석 부재로 더 심각. |
| **Solution** | **EmbeddingBridge**: 사전학습 sentence embedding 모델 (예: BGE / E5) 을 `TextEncoder` Protocol 의 추가 구현체로 도입. **D1-D4 원칙으로 LLM 종속 방지** — D1 모델 frozen / D2 Protocol 호환 (TfidfJLEncoder ↔ EmbeddingBridge 1:1 교체) / D3 optional fallback (오프라인 환경) / D4 학습은 HTP 구조에서만 (RegionSignature/CA3/Hub). |
| **Function/UX Effect** | Query top-1 정확도 0/4 → **3-4/4 목표**. Discover 합리적 매칭 38% → **70%+**. 한국어 의미 매칭 (sub-1 의 "기억은"/"기억이" 분리 문제) 자동 해결. cross-language hub (영문↔한국어) 진정한 실현. 처리 시간: encode 50-200ms (TF-IDF 1ms 대비). |
| **Core Value** | HTP 가 "사전학습 representation 위에 brain-like 구조를 얹는" 위치 정립 — 시각피질(진화로 hardcoded) + 해마(경험학습) 의 생물학적 분업 비유 자연. v4 Rev 1.3 "루프를 먼저 닫는다" 원칙의 5차 진화: 본질 한계를 코드 검증으로 정량 발견 → 즉시 본질 해결로 점프. |

---

## Context Anchor

| Key | Value |
|-----|-------|
| **WHY** | TF-IDF + JL 이 본질적 한계 (시나리오 D 정량 증명: top-1 0/4). 더 이상 데이터로 개선 불가. 본질 해결 우선순위 = sub-4 (LLMRegion) > sub-5 였으나 **sub-5 를 sub-4 보다 앞당김**. |
| **WHO** | HTP 개발자 + Knowledge Loop 사용자. **D2 (Protocol 호환) 으로 기존 호출자 무영향** — TextEncoder 인터페이스 그대로. |
| **RISK** | (R1) 모델 다운로드 크기 100-500MB / (R2) inference 시간 50-200ms / (R3) 모델 의존성 (sentence-transformers / transformers) / (R4) **HTP 의 LLM 종속화 위험** — D1-D4 원칙으로 방지 / (R5) experiment 브랜치 main merge 시점 결정 / (R6) RegionSignature 의 dim 변경 (64 → 384/768) backward-compat |
| **SUCCESS** | (1) Query top-1 정확도 ≥ 3/4 (시나리오 D 재현) (2) Discover 합리적 매칭 ≥ 70% (3) 회귀 172/172 유지 (TfidfJLEncoder fallback) (4) 한국어 의미 매칭 PASS (sub-1 의 "기억은" ↔ "기억이" similarity ≥ 0.5) (5) cross-language hub PASS (영문 "attention" ↔ 한국어 "어텐션" similarity ≥ 0.5) |
| **SCOPE** | Stage 6 만 (`experiment/embedding-bridge` 브랜치). Go/No-Go 통과 시 main merge. **D1-D4 원칙 명문화 필수**. sub-4 (LLMRegion) 은 sub-5 후로 미룸. |

---

## 1. Background — 왜 sub-5 를 앞당기는가

### 1.1 시나리오 D 의 정량 증거 (직전 세션)

| 데이터 | 결과 |
|--------|------|
| 7-entry sub-1 (영문+한국어 mix) | brain↔ai 0.53 > brain↔infra 0.14 (Go/No-Go PASS) |
| Journal 9-entry (순수 한국어) | 형태소 부재로 의미 매칭 0건 |
| 6-paper 시나리오 A (영문 학술) | discover 직관 어긋남 (brain↔worldmodel > brain↔cogsci) |
| **20-paper 시나리오 D (4 도메인)** | **query top-1 0/4, discover 정확도 38%** |

→ **TF-IDF + JL 은 데이터로 해결 안 됨이 정량 증명됨**.

### 1.2 sub-4 (LLMRegion) 보다 sub-5 가 우선인 이유

- sub-4 (LLMRegion) 는 *LLM 호출* 비용 큼 + Knowledge Loop 일상 사용에 즉시 도움 안 됨
- sub-5 (EmbeddingBridge) 는 *기존 사용자 흐름* 의 representation 만 교체 — 모든 다운스트림 (query/discover/list/export) 즉시 품질 향상
- v4 Rev 1.3 원칙: 매일 쓰는 도구 우선

### 1.3 4 원칙 (D1-D4) 의 의미

| ID | 원칙 | 의미 |
|----|------|------|
| **D1** | Frozen 사용 | 임베딩 모델 weights 고정. fine-tune 금지 → "외부 도구" 위치 유지 |
| **D2** | Protocol 호환 | `TextEncoder` 그대로 (encode/fit/dim/save/load). TfidfJLEncoder ↔ EmbeddingBridge 1:1 교체 |
| **D3** | Optional fallback | TfidfJLEncoder 도 유지 — 오프라인/저비용 환경 지원. `KnowledgeLoop(encoder=...)` 선택 자유 |
| **D4** | 학습은 위에서만 | RegionSignature.centroid, CA3 patterns, hub 형성, NGE split — 모두 사용자 데이터로 학습. 임베딩 모델은 input feature extraction 만 |

→ **HTP 가 LLM 단일 거대 모델로 *변하지 않고*, brain-like 구조 위에 LLM 의 representation 능력을 빌리는 위치 유지**.

---

## 2. Requirements (FR)

| ID | Stage | Requirement | Priority | Status |
|----|------|-------------|---------|------|
| **FR-01** | Bridge | `htp/knowledge/embedding_bridge.py` 신설 — `EmbeddingBridge(TextEncoder)` 구현체 | High | Pending |
| **FR-02** | Bridge | sentence-transformers 또는 transformers 기반 model loading (frozen) — **D1 명시** | High | Pending |
| **FR-03** | Bridge | `encode(text) → np.ndarray[dim]` Protocol 준수 — fit() 은 no-op (이미 사전학습됨) — **D2 명시** | High | Pending |
| **FR-04** | Bridge | save/load 도 Protocol 준수 (model 경로 + dim 메타데이터만 pickle) | High | Pending |
| **FR-05** | Bridge | 한국어 + 영문 multilingual model 지원 (예: BGE-m3 또는 multilingual-e5) | High | Pending |
| **FR-06** | Bridge | model 자동 다운로드 + 캐시 (`.htp/models/`) | Med | Pending |
| **FR-07** | Bridge | CPU 추론 default (GPU 옵션) | Med | Pending |
| **FR-08** | Compat | `KnowledgeLoop` 가 EmbeddingBridge 받아도 기존 흐름 그대로 — **D2 핵심** | High | Pending |
| **FR-09** | Compat | TfidfJLEncoder 도 유지 — fallback 가능 — **D3 명시** | High | Pending |
| **FR-10** | Compat | `RegionSignature.dim` dynamic (64 / 384 / 768 등) — backward-compat | Med | Pending |
| **FR-11** | Constraint | HTP 구조 (RegionSignature/CA3/Hub/NGE) 의 학습 로직은 무변경 — **D4 명시** | High | Pending |
| **FR-12** | Constraint | 임베딩 모델 weights freeze — fine-tune 금지 — **D1 명시** | High | Pending |
| **FR-13** | Test | 시나리오 D 재현: query top-1 ≥ 3/4 | High | Pending |
| **FR-14** | Test | 한국어 의미 매칭 PASS ("기억은" ↔ "기억이" similarity ≥ 0.5) | High | Pending |
| **FR-15** | Test | cross-language hub PASS ("attention" ↔ "어텐션" ≥ 0.5) | Med | Pending |
| **FR-16** | Test | 회귀 172/172 유지 (TfidfJLEncoder fallback path) | High | Pending |
| **FR-17** | DAG | `embedding_bridge.py` 는 sentence-transformers/transformers 외부 라이브러리만 — htp.runtime/thalamus/memory 미참조 (sub-1 DAG 규칙 유지) | High | Pending |

---

## 3. Design Constraints (D1-D4 명시)

| Constraint | Detail |
|-----------|--------|
| **D1 Frozen** | `model.eval()` + `torch.no_grad()` 강제. 어떤 training loop 도 EmbeddingBridge 의 weights 변경 안 함. test 가 명시적 검증 (`test_embedding_bridge_frozen_weights`) |
| **D2 Protocol** | `TextEncoder` Protocol 완전 준수 — `dim` / `encode(text)` / `fit(corpus)` (no-op) / `save(path)` / `load(path)`. 외부 호출자 무변경 |
| **D3 Fallback** | TfidfJLEncoder 보존. `KnowledgeLoop(encoder=...)` 가 자유 선택. CLI `--encoder` 옵션 추가 가능 (옵셔널) |
| **D4 학습 분리** | 임베딩 모델 → input feature 만. HTP 구조 (RegionSignature.update / CA3.completion / Hub 창발 / NGE split) → 사용자 데이터로 *자체 학습 유지*. test 가 검증 (`test_region_signature_learns_post_embedding`) |

---

## 4. Success Criteria

### 4.1 정량 (회귀 + 신규)

| 지표 | Before (sidequest 완료) | After (sub-5 목표) |
|------|------------------------|--------------------|
| 총 테스트 | 172 | **180-185** (+8-13) |
| 회귀 깨짐 | 0 | **0** |
| Query top-1 정확도 (시나리오 D) | 0/4 | **≥ 3/4** |
| Discover 합리적 매칭 (20 paper) | 3/8 = 38% | **≥ 6/8 = 75%** |
| 한국어 의미 매칭 | 0건 | **PASS** |
| Cross-language hub | 부분 (영문 술어 섞일 때만) | **진정한 PASS** |

### 4.2 정성 (Go/No-Go for main merge)

- **Go**: 시나리오 D 재현에서 query top-1 ≥ 3/4 + discover 직관 매칭 ≥ 75% + 한국어 PASS
- **No-Go**: top-1 < 3/4 또는 회귀 깨짐 → 별도 모델 선택 재검토 또는 experiment 브랜치 보류

---

## 5. Implementation Sketch

### 5.1 File map

```
htp/knowledge/
├── embedding_bridge.py     [신규] EmbeddingBridge(TextEncoder) — Frozen 사전학습 모델
├── encoder.py              무변경 (TfidfJLEncoder 보존 — D3)
└── ...

tests/knowledge/
├── test_embedding_bridge.py    [신규] D1-D4 검증
└── test_session_e_scenarios.py [신규] 시나리오 D 재현 + 한국어 + cross-language
```

### 5.2 모델 후보 평가

| 모델 | 크기 | dim | 다국어 | 한국어 품질 | 비고 |
|------|-----|----:|:----:|:--------:|------|
| `BAAI/bge-m3` | 567MB | 1024 | ✅ | 중 | 강력하나 큼 |
| `intfloat/multilingual-e5-small` | 118MB | 384 | ✅ | 중 | **추천 default** |
| `BAAI/bge-small-en-v1.5` | 133MB | 384 | ❌ | ❌ | 영문 only 시 |
| `paraphrase-multilingual-MiniLM-L12-v2` | 117MB | 384 | ✅ | 하 | 작지만 한국어 약함 |

**Design 단계 결정**: 기본 `intfloat/multilingual-e5-small` (118MB, 384-dim, 한국어 지원). 사용자가 `model_name` 인자로 다른 모델 선택 가능.

### 5.3 Session Plan

| Session | Scope | 누적 테스트 | 소요 |
|---------|-------|----------|------|
| **session-1** | EmbeddingBridge 구현 + D1/D2/D3 검증 테스트 | 172 → 178 | ~3h |
| **session-2** | 시나리오 D 재현 + 한국어 + cross-language 시나리오 검증 | 178 → 183 | ~3h |
| **session-3** | RegionSignature dim 동적 호환 + D4 검증 + Go/No-Go 보고 | 183 → 185 | ~2h |

총 ~8h. **`experiment/embedding-bridge` 브랜치에서 진행, Go 통과 시 main merge.**

---

## 6. Risks

| ID | Risk | Severity | Mitigation |
|----|------|---------|----------|
| R1 | sentence-transformers 의존성 추가 (~100MB) | Med | requirements 에 명시 + D3 fallback 으로 미설치 환경 대응 |
| R2 | model 다운로드 시간 (첫 호출 시 ~30s) | Low | `.htp/models/` 캐시 + 첫 사용 시 안내 메시지 |
| R3 | CPU inference 50-200ms/text | Med | TF-IDF (1ms) 대비 느리나 운영 가능 — batch ingest 시 amortize |
| R4 | **HTP 의 LLM 종속화** | **High** | **D1-D4 원칙 강제 + test 검증** (`test_embedding_bridge_frozen_weights` 등) |
| R5 | RegionSignature.dim 64 → 384 변경 시 backward-compat | Med | FR-10 — dim 인자 dynamic. 기존 RegionSignature(dim=64) 와 신규 RegionSignature(dim=384) 공존 |
| R6 | experiment 브랜치 main merge 결정 — Go/No-Go 기준 명확화 | Low | §4.2 정량 기준 + git tag 로 분리 |
| R7 | 한국어 모델 품질이 영문보다 낮음 | Med | multilingual model 사용 + 한국어 특화 모델 fallback 옵션 (`BAAI/bge-small-ko-v1.5`) |

---

## 7. Out-of-Scope (sub-5)

- LLM full reasoning (chain-of-thought, multi-step generation) — **sub-4 (LLMRegion) 로 분리**
- 임베딩 모델 fine-tuning — **D1 위반**, 본 사이클 금지
- GPU optimization / batch inference speedup — sub-7 또는 별도 cycle
- HTP 핵심 알고리즘 (Hub/Memory/NGE) 수정 — **D4 위반**, 본 사이클 무변경
- ModalEncoder (image/audio/multimodal) — Plan §SCOPE multimodal cycle
- Embedding 모델 자동 평가 (사용 패턴 기반 모델 자동 추천) — 후속 cycle

---

## 8. Decisions to Defer (sub-decision 필요 — Design 단계)

1. **Default 모델 선택**: `multilingual-e5-small` (권장) vs `bge-m3` (강력) vs 영문 only `bge-small-en`
2. **Architecture**: Option A (단일 파일 embedding_bridge.py) vs Option B (embedding/ 패키지 + multiple model adapters)
3. **dim 호환 정책**: dynamic dim vs 64-dim projection 강제
4. **CLI 확장**: `--encoder embedding|tfidf` 옵션 추가 여부
5. **캐시 정책**: HuggingFace 기본 캐시 vs `.htp/models/` 명시적 캐시
6. **D1 검증 방식**: weights hash 비교 vs `torch.set_grad_enabled(False)` 강제

---

## 9. Next Steps

1. **브랜치 분리**: `git checkout -b experiment/embedding-bridge`
2. **Design 진입**: `/pdca design htp-thalamus-car` (sub-5 scope) — 6 sub-decision 결정 + 3-architecture 옵션
3. **Do**: Session 3분할 (Bridge 구현 / 시나리오 검증 / D4 통합)
4. **Check**: Go/No-Go 정량 측정
5. **Report**: Go 시 main merge, No-Go 시 모델 재검토
