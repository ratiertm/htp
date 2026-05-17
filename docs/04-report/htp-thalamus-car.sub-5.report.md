---
template: report
feature: htp-thalamus-car
sub_cycle: sub-5 (Stage 6 EmbeddingBridge)
date: 2026-05-18
author: Mindbuild
status: Completed (Go for main merge)
match_rate: 99%
branch: experiment/embedding-bridge
---

# htp-thalamus-car sub-5 Completion Report

## Executive Summary

| Perspective | Content |
|-------------|---------|
| **Problem** | sub-1~L2 sidequest 의 TF-IDF + JL 인코더가 *데이터로 해결 안 되는 본질적 한계*. 20-paper / 4 domain / 영문 학술 abstract 환경에서도 query top-1 0/4. 순수 한국어는 형태소 부재로 0건 매칭. |
| **Solution** | **EmbeddingBridge** — 사전학습 multilingual sentence embedding (multilingual-e5-small, 384-dim) 을 `TextEncoder` Protocol 의 추가 구현체로 도입. **D1-D4 design 원칙** 으로 HTP 의 LLM 종속화 방지. |
| **Function/UX Effect** | 시나리오 D query top-1 0/4 → **PASS** (≥ 3/4). 한국어 의미 매칭 PASS ("기억은" ↔ "기억이"). Cross-language hub PASS (영문 "attention" ↔ 한국어 "어텐션"). CLI `--encoder embedding` 옵션. |
| **Core Value** | HTP 가 "사전학습 representation + brain-like 구조 자체학습" 의 명확한 위치 확립. 시각피질(진화 사전학습) + 해마(경험학습) 의 생물학적 분업 비유 실현. **D1-D4 가 test 로 영구 보호** — 후속 어떤 변경도 D1 (freeze) / D4 (학습 분리) 위반 시 test 실패. v4 Rev 1.3 원칙 5차 실증. |

### 1.3 Value Delivered (4 perspectives, 실측 기준)

| Perspective | Planned | Delivered | Δ |
|-------------|---------|-----------|---|
| Test count | 172 → 185 (+13) | 172 → **189 (+17)** | **+4 초과** |
| Match Rate | ≥ 90% | **99%** | +9pp |
| Plan FR 충족 | 17/17 | **17/17** | 100% |
| Go/No-Go 항목 | 5/5 | **5/5** | 100% |
| D1-D4 원칙 검증 | 4/4 | **4/4** | 100% |

---

## Context Anchor

| Key | Value |
|-----|-------|
| **WHY** | TF-IDF + JL 본질 한계 정량 증명 → 사전학습 임베딩 필수 |
| **WHO** | HTP 개발자 + Knowledge Loop 사용자 (D2/D3 로 무영향) |
| **RISK** | R4 (LLM 종속화) — D1-D4 강제로 방지 ✅ |
| **SUCCESS** | Go/No-Go 5/5 PASS |
| **SCOPE** | Stage 6 EmbeddingBridge, experiment/embedding-bridge → main merge |

---

## 2. Plan → Design → Implementation 일관성

### 2.1 Decision Record Chain

```
[Plan]   Trigger: 시나리오 D 정량 한계 (top-1 0/4)
   ↓     D1-D4 원칙으로 LLM 종속화 방지
   ↓
[Design] Option B Modular + 6 sub-decision 결정
   ↓       모델: multilingual-e5-small / Adapter: STAdapter
   ↓       dim Dynamic / CLI --encoder / D1 3중 검증
   ↓
[Do]     Session 3분할
   ↓       session-1 9eb7af8  Bridge + STAdapter + D1/D2/D3
   ↓       session-2 ad06a36  Go/No-Go 5/5 PASS
   ↓       session-3 b60dd40  D4 + dim 동적
   ↓
[Check]  Match Rate 99%, Gap 1건 (Low deprecation, 즉시 fix)
   ↓
[Report] (현 문서) + main merge 권장
```

### 2.2 Key Decisions & Outcomes

| Decision | Followed? | Outcome |
|----------|:---:|---------|
| Architecture Option B Modular | ✅ | `embedding/` 패키지 (router/, coherence/ 패턴 일관) |
| Default 모델 multilingual-e5-small | ✅ | 118MB, 384-dim, 한국어 PASS |
| BaseEmbeddingModel Protocol | ✅ | STAdapter / HFAdapter 다형성 토대 |
| dim Dynamic | ✅ | RegionSignature(dim=64) ↔ (dim=384) 공존 |
| CLI `--encoder` 옵션 | ✅ | tfidf default (회귀 보호) + embedding 옵션 |
| D1 3중 검증 (eval+grad+no_grad) | ✅ | weights hash test 로 영구 보호 |
| HF 기본 캐시 | ✅ | 별도 코드 없이 OS 표준 |
| D4 학습 분리 | ✅ | RegionSignature update 학습 + embedding freeze 공존 |

---

## 3. Plan Success Criteria Final Status

### Plan FR-01~17 (Plan §2)

| ID | Criterion | Status |
|----|-----------|:---:|
| FR-01 | embedding_bridge.py 신설 | ✅ Met |
| FR-02 | sentence-transformers frozen | ✅ Met |
| FR-03 | Protocol 준수 (encode/fit no-op) | ✅ Met |
| FR-04 | save/load metadata pickle | ✅ Met |
| FR-05 | multilingual model 지원 | ✅ Met |
| FR-06 | 자동 다운로드 + 캐시 | ✅ Met (HF default) |
| FR-07 | CPU 추론 default | ✅ Met |
| FR-08 | KnowledgeLoop 호환 무변경 | ✅ Met |
| FR-09 | TfidfJLEncoder 유지 (D3) | ✅ Met |
| FR-10 | RegionSignature dim 동적 | ✅ Met |
| FR-11 | HTP 구조 학습 무변경 (D4) | ✅ Met |
| FR-12 | weights freeze (D1) | ✅ Met |
| FR-13 | 시나리오 D top-1 ≥ 3/4 | ✅ Met |
| FR-14 | 한국어 ≥ 0.5 | ✅ Met |
| FR-15 | cross-language ≥ 0.5 | ✅ Met |
| FR-16 | 회귀 172 유지 | ✅ Met (189/189) |
| FR-17 | DAG 단방향 | ✅ Met (rglob 자동 검사) |

**Overall Success Rate: 17/17 (100%)**

### Go/No-Go (Plan §1.3)

| 항목 | 목표 | 결과 |
|------|------|:----:|
| Query top-1 정확도 | ≥ 3/4 | ✅ |
| Discover 합리적 매칭 | ≥ 6/8 또는 강력 ≥ 3/8 | ✅ (강력 ≥ 3) |
| 한국어 의미 매칭 | ≥ 0.5 | ✅ |
| Cross-language hub | 평균 ≥ 0.5 | ✅ |
| 회귀 보호 | 0건 깨짐 | ✅ |

**Go 5/5 — main merge 권장.**

---

## 4. Delivered Artifacts

### 4.1 신규 파일 (8)

| 파일 | LoC | 책임 |
|------|----:|------|
| `htp/knowledge/embedding/__init__.py` | 26 | 공개 export |
| `htp/knowledge/embedding/base.py` | 33 | BaseEmbeddingModel Protocol |
| `htp/knowledge/embedding/st_adapter.py` | 76 | sentence-transformers 어댑터 (D1 frozen) |
| `htp/knowledge/embedding/bridge.py` | 123 | EmbeddingBridge (TextEncoder 어댑터) |
| `htp/knowledge/cli/_common.py` | 30 | encoder 선택 헬퍼 (D3) |
| `tests/knowledge/test_sub5_session1_bridge.py` | 158 | D1/D2/D3 검증 6 tests |
| `tests/knowledge/test_sub5_session2_scenarios.py` | 180 | Go/No-Go 5 tests |
| `tests/knowledge/test_sub5_session3_d4_dim.py` | 95 | D4 + dim 동적 2 tests |
| `docs/02-design/features/htp-thalamus-car.sub-5.design.md` | 307 | Design |
| `docs/03-analysis/htp-thalamus-car.sub-5.analysis.md` | ~250 | Check 결과 |

### 4.2 수정 파일 (3)

| 파일 | 변경 | 내용 |
|------|------|------|
| `htp/thalamus/signature.py` | +3 LoC | dim 자동 추론 (centroid shape 우선) |
| `htp/knowledge/cli/*.py` (5 파일) | +30 LoC | encoder 인자 args.encoder 기반 |
| `htp/knowledge/cli/__init__.py` | +7 LoC | `--encoder tfidf|embedding` 옵션 |
| `requirements.txt` | +1 LoC | sentence-transformers>=2.7 |

### 4.3 commits (5)

```
e4c58fa sub-5 Plan + 20 archive paper abstracts
3d87f57 sub-5 Design — Option B Modular + 6 sub-decision
9eb7af8 sub-5 session-1  Bridge + STAdapter + D1/D2/D3
ad06a36 sub-5 session-2  Go/No-Go 5/5 PASS
b60dd40 sub-5 session-3  RegionSignature dim 동적 + D4
```

### 4.4 외부 의존성

- `sentence-transformers>=2.7` (transformers/torch 자동) — Plan FR-02
- 모델 다운로드 (첫 1회): `intfloat/multilingual-e5-small` ~118MB
- 캐시: `~/.cache/huggingface/` (OS 표준)

---

## 5. Metrics

### 5.1 Volume

| 지표 | 값 |
|------|---:|
| 신규 LoC (htp/) | 288 |
| 수정 LoC | ~41 |
| 신규 테스트 LoC | 433 |
| Design 문서 LoC | 307 |
| Analysis 문서 LoC | ~250 |
| 총 변경 LoC | ~1,319 |

### 5.2 Quality

| 지표 | Before (L2 sidequest 완료) | After (sub-5 완료) |
|------|------:|------:|
| 총 테스트 | 172 | **189** (+17) |
| 회귀 깨짐 | 0 | **0** |
| 실행 시간 | 1.38s | 56s (embedding test 포함, 캐시 후 ~30s) |
| Match Rate | 100% (L2) | **99%** (sub-5) |
| Public API 호환성 | 100% | **100%** |
| Critical Gap | 0 | **0** |
| 시나리오 D top-1 | 0/4 | **≥3/4** |
| 한국어 매칭 | 0건 | **PASS** |

---

## 6. Lessons Learned

### 6.1 잘된 점

1. **D1-D4 원칙의 코드 검증** — 단순 문서 원칙이 아닌 *실행 가능한 test* 로 박았음. 후속 어떤 변경도 D1 위반 시 `test_embedding_bridge_frozen_weights` 가 실패하여 PR 차단.
2. **시나리오 D 의 정량 트리거** — sub-5 진입 결정이 *추상적 기대* 가 아닌 *측정된 한계* 에 근거. 0/4 → 3/4+ 의 본질 해결이 명확.
3. **OCP 패턴 3중 일관성** — sub-2 router/, sub-3 coherence/, sub-5 embedding/ 모두 Strategy/Adapter 패턴. *프로젝트 레벨* OCP 일관성 정착.
4. **2 session 분할 → 3 session 으로 자연스러운 분할** — D4 검증이 별도 session 가치 큼 (HTP-LLM 경계의 핵심).

### 6.2 개선 여지

1. **sentence-transformers 5.x deprecation warning** — `get_sentence_embedding_dimension` → `get_embedding_dimension`. 본 보고서 시점에 1줄 fix 적용.
2. **모델 다운로드 첫 1회 느림 (~30s)** — 사용자 안내 메시지 부재. CLI 첫 사용 시 `Downloading model...` 표시 권장 (후속 micro-patch).
3. **dim 자동 추론 시 에러 메시지 부재** — 차원 불일치 query 시 ValueError 메시지가 dim 디버깅 hint 부족. 후속 보강.

### 6.3 v4 Rev 1.3 "루프를 먼저 닫는다" 원칙의 5차 실증

| 차수 | 시점 | 발견/예방 |
|-----|-----|---------|
| 1차 | sub-1 직후 | Critical Gap #3 (encoder.fit refit) 24h 내 발견 |
| 2차 | sub-1 cross-language 실험 | "언어가 hub" 가설 부분 실증 |
| 3차 | sub-2 | β sweep 메트릭 코드 내장 |
| 4차 | sub-3 | conflict_magnitude 학습 신호 자동 누적 |
| **5차** | **sub-5 (현 사이클)** | **시나리오 D top-1 0/4 의 정량 한계가 sub-5 진입 우선순위 상향을 *주도* (sub-4 보다 앞당김)** |

원칙의 진화: 매 sub-cycle 가 *후속 사이클의 우선순위 결정 근거* 를 코드에 남김. sub-5 의 D1-D4 검증 test 가 sub-6/sub-4 진입 시 안전성 보장.

---

## 7. Next Cycle Recommendations

### 7.1 즉시 — main merge 결정

```bash
git checkout master
git merge experiment/embedding-bridge --no-ff
git push origin master
```

merge 사유: Go/No-Go 5/5 + 회귀 0 + D1-D4 모두 검증.

### 7.2 다음 sub-cycle 후보

- **sub-6 (Stage 7 vector default 전환)** — sub-5 의 EmbeddingBridge 가 toggle 의 기본값으로 — but conservative
- **sub-4 (Stage 4 + 5 LLMRegion + Pipeline)** — sub-5 후로 미뤄졌음. 이제 진입 가능
- **별도 cycle `htp-region-signature-persistence`** — sub-3 Gap #1 (centroid 영속화)
- **별도 cycle `htp-knowledge-batch-fit-vocab`** — 시나리오 A 첫 실험에서 발견한 첫 fit 시점 vocab 풍부화

---

## 8. Sign-off

| 항목 | 결과 |
|------|------|
| **Plan SC (FR-01~17)** | 17/17 (100%) |
| **Go/No-Go** | 5/5 (100%) |
| **D1-D4 원칙 검증** | 4/4 |
| **Decision Record** | 7/7 따름 |
| **회귀 보호** | 0건 깨짐 |
| **Match Rate** | **99%** |
| **PDCA 단계** | Plan → Design (Option B) → Do (3 sessions) → Check (sub-5.analysis) → **Report (현 문서)** |

**sub-5 PDCA 완료 ✅ — main merge 권장**.
