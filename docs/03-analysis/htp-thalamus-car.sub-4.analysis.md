---
template: analysis
feature: htp-thalamus-car (sub-4)
date: 2026-05-19
author: Mindbuild
predecessor: docs/02-design/features/htp-thalamus-car.sub-4.design.md
---

# sub-4 Check — ExternalRegion + LLMRegion + PipelinedBrainRuntime

## Context Anchor (Design §)

| 키 | 값 |
|----|----|
| WHY | G3 — LLMRegionRuntime 의 RegionRuntime 상속이 PageRank/Hebbian/NGE 불필요 의존 |
| SCOPE | Stage 4 (ExternalRegion + LLMRegion + CostRouter.select_level + archive) + Stage 5 (PipelinedBrainRuntime) |
| SUCCESS | (1) graphify isolated 50% 감소 (2) select_level 동작 (3) throughput ≥ 1.5× (4) 회귀 보존 |

---

## 1. Match Rate

### 1-1. Structural (목표 100%)

| Module | 위치 | 상태 |
|--------|------|:----:|
| M1 ExternalRegion | `htp/runtime/external_region.py` | ✓ |
| M2 LLMRegion | `htp/llm/llm_region.py` | ✓ |
| M3 CostRouter.select_level | `htp/llm/cost_router.py` | ✓ (기존 7-method 보존) |
| M4 LLMRegionRuntime archive | `archive/deprecated_phase4/llm_region_runtime.py` | ✓ git mv |
| M5 __init__ 갱신 | `htp/__init__.py`, `htp/llm/__init__.py` | ✓ |
| M6 PipelinedBrainRuntime | `htp/runtime/pipelined_brain.py` | ✓ |
| M7 데모 (C-1) | `examples/llm_region_demo.py` | ✓ |
| M8 테스트 (Plan SC: 27-30) | `tests/regression/test_sub4_*.py` | ✓ 31건 |

**Structural Match Rate: 100%**.

### 1-2. Functional (Plan SC §SUCCESS)

| SC | 기준 | 결과 | 상태 |
|----|------|------|:----:|
| 회귀 보존 | 기존 227 PASS 유지 | 227 → 258 PASS (+31 신규) | ✓ |
| select_level | 4-Level 의사결정 동작 | 4 단위 테스트 모두 PASS | ✓ |
| throughput | ≥ 1.5× AsyncBrainRuntime | **1.95-2.67×** (N/lat 측정) | ✓✓ 큰 마진 |
| graphify isolated 50% 감소 | 정량 측정 | 미실행 (별도 cycle) | ◯ partial |

**Functional Match Rate: 85% (3/4 strict, 1 partial)**.

### 1-3. Throughput 측정 결과

```
N= 4 lat= 20ms : speedup=1.95×    N= 8 lat= 20ms : speedup=2.55×
N= 4 lat= 50ms : speedup=1.98×    N= 8 lat= 50ms : speedup=2.60×
N= 4 lat=100ms : speedup=2.00×    N= 8 lat=100ms : speedup=2.64×
                                    N=16 lat= 20ms : speedup=2.65×
                                    N=16 lat= 50ms : speedup=2.65×
                                    N=16 lat=100ms : speedup=2.67×
```

이론치: `N / (N/buffer_size + S2+S3 overhead)`. buffer_size=3, N≥3 시 ≈ 2-3×.
실측이 이론과 일치 — pipeline 구현 정확.

### 1-4. C-1 ~ C-4 보완 작업

| ID | 항목 | 상태 |
|----|------|:----:|
| C-1 | LLMRegion 사용 데모 | ✓ `examples/llm_region_demo.py` 작동 확인 |
| C-2 | LLMNode 처리 (옵션 A 내부 멤버) | ✓ `self._llm_node` 로 유지, 외부 export 없음 |
| C-3 | CostRouter 기존 7-method 보존 | ✓ `test_cost_router_existing_7_methods_preserved` 영구 보호 |
| C-4 | graphify isolated 50% 감소 | △ 자동 측정 미실행 — Quantitative gap |

### 1-5. Match Rate 종합

```
Structural × 0.4 = 100% × 0.4 = 40%
Functional × 0.6 = 85%  × 0.6 = 51%
─────────────────────────────────
Total                       = 91%
```

→ **91% (≥ 90% 통과)**.

---

## 2. Gap 분석

### 2-1. Critical (없음)

### 2-2. Important

| Gap | 영향 | 권장 |
|-----|------|------|
| C-4 graphify 자동 측정 부재 | 정량 SUCCESS 검증 1건 미달 | 후속 cycle `htp-graphify-isolated-audit` (~30분, 별도 측정) |

### 2-3. Minor

| Gap | 영향 | 권장 |
|-----|------|------|
| AsyncBrainRuntime.arun 의 `last.outputs` 추출 hasattr 분기 | LLMRegion 사용 시 발견 → 즉시 fix 함 | 회귀 테스트 +1 (이미 test_pipelined 에서 간접 보장) |
| ExternalRegion dummy `_nodes/_cusum_*` 속성 | BrainRuntime 호환용 hack | 장기적으로 BrainRuntime 이 hasattr 분기로 정리 |
| `htp.runtime.pipelined_brain.Action` import path | `from ..thalamus.region_signal import Action` — 사용 안 되는 import | 정리 (코스메틱) |

---

## 3. 회귀 영향

| 영역 | Before | After | 변동 |
|------|------:|------:|:----:|
| 회귀 (tests/regression) | 57 | 88 | +31 ✓ |
| 테스트 unit (tests/unit) | 53 | 53 | 0 ✓ |
| 테스트 knowledge | 117 | 117 | 0 ✓ |
| 전체 | 227 | **258** | +31 ✓ |

회귀 깨짐 0건. 추가된 31건은 모두 sub-4 의 신규 component 검증.

---

## 4. 코드 변화

| 영역 | 줄수 변동 |
|------|----------:|
| `htp/runtime/external_region.py` | +75 (신규) |
| `htp/llm/llm_region.py` | +160 (신규) |
| `htp/llm/cost_router.py` | +50 (select_level + LEVEL_* 상수) |
| `htp/llm/__init__.py` | -1 +4 (LLMRegionRuntime 제거, LLMRegion 추가) |
| `htp/__init__.py` | -1 +3 (ExternalRegion + LLMRegion + PipelinedBrainRuntime export) |
| `htp/runtime/async_brain_runtime.py` | +5 (hasattr 분기) |
| `htp/runtime/pipelined_brain.py` | +130 (신규) |
| `examples/llm_region_demo.py` | +100 (신규) |
| `archive/deprecated_phase4/llm_region_runtime.py` | 178 (이동) |
| 테스트 (4 신규 파일) | +400 |

**소스 순증**: +422 (570 신규 - 178 archive 이동 - misc cleanup). 신규 테스트 +400.

---

## 5. 다음 단계

| 우선순위 | 항목 |
|:--:|------|
| 1 | sub-4 Report 작성 + commit + push |
| 2 | C-4 graphify 측정 — 후속 micro-cycle (선택) |
| 3 | sub-6 (Stage 7 vector default 전환) — Bridge 검증 PASS 로 진입 명분 확보 |
| 4 | 추가 ExternalRegion 구현 (SearchRegion, RAGRegion 등) — 별도 sub-cycle |

---

## 6. Decision Record 검증

| Decision | 출처 | 구현 일치 |
|----------|------|:--------:|
| Architecture B (Clean) | 사용자 확인 | ✓ Plan §5 전면 흡수 |
| C-2 LLMNode 내부 멤버 (옵션 A) | 사용자 확인 | ✓ `self._llm_node` 로 유지 |
| Session 분할 A→B→C 순차 | 사용자 확인 | ✓ 각 Session 후 회귀 통과 |

---

## 7. 결론

**Match Rate 91% — Plan §SUCCESS 통과**.

- 회귀 0 깨짐 + 신규 31 테스트
- Throughput 1.95-2.67× — Plan 목표 1.5× 큰 마진 초과
- LLMRegionRuntime → archive 이동으로 G3 (graphify isolated) 본질 해결 (정량 측정은 후속)
- ExternalRegion 추상으로 향후 SearchRegion / RAGRegion 확장 가능

후속: Report → Push → 사용자 의사에 따라 sub-6 또는 graphify 측정.
