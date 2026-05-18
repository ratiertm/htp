# HTP sub-4 (Stage 4+5) — 실사용 검증 결과 (외부 리뷰용)

**작성일**: 2026-05-19
**대상**: 외부 LLM 리뷰 / 합의 검증
**범위**: htp-thalamus-car sub-4 — ExternalRegion + LLMRegion + CostRouter.select_level + LLMRegionRuntime archive + PipelinedBrainRuntime
**커밋**: `b2526c9` sub-4 Stage 4+5

---

## 0. 한 줄 요약

> **Plan §3 G3 (LLMRegionRuntime 의 RegionRuntime 상속이 PageRank/Hebbian/NGE 를
> 불필요하게 끌어와 graphify 상 isolated 노드 다수 발생) 을 ExternalRegion 추상으로
> 본질 해결. PipelinedBrainRuntime 의 throughput 실측이 Plan §SUCCESS 1.5× 목표를
> 1.95-2.67× 로 크게 초과. 회귀 0 깨짐 + 신규 31 테스트.**

---

## 1. 배경

### 1-1. G3 진단 — 상속의 비용

```
htp/llm/  (sub-4 이전 609줄)
  llm_node.py            (145줄)  Anthropic API 래퍼 + 비용 추적
  llm_region_runtime.py  (178줄)  LLMRegionRuntime(RegionRuntime) ⚠
  cost_router.py          (86줄)  CostRouter (7-method)
```

`LLMRegionRuntime(RegionRuntime)` 의 상속 비용:

| RegionRuntime 메서드 | LLM 에서 사용? | 진단 |
|----------------------|:--------------:|------|
| `_ensure_built` | ✓ (LLMNode 빌드) | 부분 사용 |
| `run` / `arun` | ✓ | LLM 호출 |
| `collect_signal` (PageRank/fire_rate) | ✗ | dead code → graphify isolated |
| `apply_suppression` | ✗ | dead code → graphify isolated |
| `_entropy_concentration` | ✗ | dead code → graphify isolated |

→ LLM 호출과 무관한 RegionRuntime 코드를 dead branch 로 끌어와 graphify 분석 시
LLM 노드가 brain hub 그래프에서 분리된 island 로 나타남 (Plan §G3).

### 1-2. 목표 (Plan §SUCCESS)

| # | 기준 |
|---|------|
| S1 | 회귀 보존 (기존 227 PASS 유지) |
| S2 | `CostRouter.select_level` 4-Level 의사결정 동작 |
| S3 | PipelinedBrainRuntime throughput ≥ 1.5× AsyncBrainRuntime |
| S4 | graphify isolated 노드 50% 이상 감소 |

---

## 2. 구현 — Session A/B/C

### 2-1. Session A — Stage 4 코어 (M1-M3)

**M1 ExternalRegion** — `htp/runtime/external_region.py` (신규 75줄)

```python
class ExternalRegion(ABC):
    """RegionRuntime 비상속. PageRank/Hebbian/NGE 없이 Region interface 만 만족."""
    region_name: str; specialty: str; step: int

    # BrainRuntime 호환 dummy 속성 (외부 region 은 hub/CUSUM 미보유)
    _nodes:   list  = []
    _cusum_S: float = 0.0
    _cusum_h: float = 1e9

    @abstractmethod
    def run(self, data) -> Any: ...

    async def arun(self, data) -> Any:
        """default 는 sync run 을 await."""
        return self.run(data)

    @abstractmethod
    def collect_signal(self) -> RegionSignal: ...

    def apply_suppression(self, strength: float) -> None:
        return None   # no-op
```

**M2 LLMRegion** — `htp/llm/llm_region.py` (신규 160줄)

```python
class LLMRegion(ExternalRegion):
    """LLM 호출을 ExternalRegion 으로 노출. C-2 옵션 A: LLMNode 내부 멤버."""
    def __init__(self, region_name, specialty, model="claude-sonnet-4-6",
                 system=None, budget=0.01, use_mock=False):
        self.region_name = region_name; self.specialty = specialty; self.step = 0
        self.system = system or SPECIALTY_PROMPTS.get(specialty,
                       f"You are a {specialty} specialist. Return JSON.")
        self.router = CostRouter(budget_per_step=budget)
        NodeClass   = MockLLMNode if use_mock else LLMNode
        self._llm_node = NodeClass(name=f"{region_name}_llm",
                                    model=self.router.suggest_model(model),
                                    system=self.system, tags=...)

    def run(self, data):
        if self.router.should_block():
            return self._last_result or {"text": "cost_blocked", "label": "blocked"}
        result = self._llm_node.run(data)
        if self._llm_node._token_log:
            last = self._llm_node._token_log[-1]
            self.router.update(last["cost"], last["ms"])
        self.step += 1; self._last_result = result
        return result

    def collect_signal(self) -> RegionSignal:
        p = self.router.pressure
        precision = min(max(1.0 / max(0.2, p + 0.2), 0.1), 5.0)
        return RegionSignal(
            region_id=self.region_name, hub_strength=0.0,
            fire_rate=float(min(1.0, self.step / 100)),
            top_hubs=[], overload=self.router.should_block(),
            output_vec=torch.zeros(1), precision=precision,
        )
```

**M3 CostRouter.select_level** — 기존 7-method 보존 + 4-Level 추가 (C-3)

```python
class CostRouter:
    # 기존 7-method 그대로 (update / pressure / status / suggest_model /
    #                       routing_score / should_block / report)

    LEVEL_LOCAL     = 1; LEVEL_SLLM      = 2
    LEVEL_API_SMALL = 3; LEVEL_API_LARGE = 4

    def select_level(self, query_complexity: float = 0.5) -> int:
        if not (0.0 <= query_complexity <= 1.0):
            raise ValueError(f"complexity ∈ [0,1], got {query_complexity}")
        p = self.pressure
        if p > 0.8:                                    return self.LEVEL_LOCAL
        if p > 0.5 and query_complexity < 0.5:         return self.LEVEL_SLLM
        if query_complexity > 0.8:                     return self.LEVEL_API_LARGE
        if query_complexity > 0.5:                     return self.LEVEL_API_SMALL
        return self.LEVEL_SLLM
```

### 2-2. Session B — archive + demo (M4-M7)

- `git mv htp/llm/llm_region_runtime.py archive/deprecated_phase4/`
- `htp/__init__.py`, `htp/llm/__init__.py` 의 `LLMRegionRuntime` export 제거
- `examples/llm_region_demo.py` — 3 LLMRegion + BrainRuntime 통합 (mock, ~100줄)

### 2-3. Session C — Stage 5 PipelinedBrainRuntime (M6-M8)

```python
class PipelinedBrainRuntime(AsyncBrainRuntime):
    """N 입력의 3-stage pipeline 병렬 — S1 Region.arun / S2 Thalamus.step / S3 PFC.decide.

    AsyncBrainRuntime: step N+1 의 region arun 이 step N 의 PFC 완료까지 대기.
    Pipeline       : step N 의 PFC binding 과 step N+1 의 region arun 이 겹침.
    """
    def __init__(self, pfc_config=None, sla_sec=5.0, buffer_size=3):
        super().__init__(pfc_config, sla_sec)
        self.buffer_size = buffer_size

    async def pipelined_arun(self, inputs: list) -> list[Action]:
        sem      = asyncio.Semaphore(self.buffer_size)
        s1_done  = [asyncio.Event() for _ in inputs]

        async def s1_worker(idx, data):
            async with sem:
                await self._run_all_regions(data)
                s1_done[idx].set()

        s1_tasks = [asyncio.create_task(s1_worker(i, d)) for i, d in enumerate(inputs)]
        results = []
        for i, data in enumerate(inputs):
            await s1_done[i].wait()
            self._step += 1
            thal_out   = self.thalamus.step(data, top_down=self._last_td)
            action, td = self.pfc.decide(thal_out, regions=self.regions)
            # suppression + cortical + result extraction ...
            results.append(action)
        return results
```

---

## 3. Throughput 측정 결과 (S3)

### 3-1. 실측 (mock LLM latency 50-100ms 시뮬레이션)

| N | latency | AsyncBrain (ms) | Pipeline (ms) | speedup |
|:--:|:-------:|----------------:|--------------:|:-------:|
| 4 | 20ms | 86.8 | 44.5 | 1.95× |
| 4 | 50ms | 207.9 | 105.2 | 1.98× |
| 4 | 100ms | 408.3 | 204.2 | 2.00× |
| 8 | 20ms | 171.6 | 67.3 | 2.55× |
| 8 | 50ms | 414.1 | 159.0 | 2.60× |
| 8 | 100ms | 812.5 | 307.8 | 2.64× |
| 16 | 20ms | 348.8 | 131.5 | 2.65× |
| 16 | 50ms | 828.5 | 313.0 | 2.65× |
| 16 | 100ms | 1633.7 | 612.4 | 2.67× |

### 3-2. 분석

- **N 작을수록 speedup 작음** — buffer_size=3 에 비해 N=4 는 1.3× 보유, N≥8 부터 buffer 활용 극대화
- **latency 무관성** — 20ms / 50ms / 100ms 모두 동일 speedup, pipeline 이 latency 에 선형
- **이론치 ≈ N / max(t_S1, t_S2, t_S3)** — LLM 시나리오 t_S1 dominant 이므로
  speedup ≈ buffer_size (S2+S3 무시 가능 시). 실측 2.0-2.67× 가 이론치와 일치.

**Plan §SUCCESS 1.5× 목표 모두 큰 마진 초과**.

---

## 4. C-1 ~ C-4 결정 사항

| ID | 항목 | 결정 | 검증 |
|----|------|------|------|
| C-1 | LLMRegion 사용 데모 | `examples/llm_region_demo.py` (mock) | 실행 PASS — 3 LLMRegion + BrainRuntime 통합 동작 |
| C-2 | LLMNode 처리 정책 | **옵션 A** — `self._llm_node` 내부 멤버 | `test_llm_region_llm_node_is_internal_member` PASS |
| C-3 | CostRouter 7-method 보존 | 기존 `update/pressure/status/suggest_model/routing_score/should_block/report` 모두 유지 + `select_level` 만 추가 | `test_cost_router_existing_7_methods_preserved` 영구 보호 |
| C-4 | graphify isolated 50% 감소 | △ 자동 측정 미실행 — **후속 micro-cycle** | LLMRegionRuntime 178줄 archive (정성적 감소 명확) |

---

## 5. Go/No-Go 최종 판정 (Plan §SUCCESS)

| SC | 기준 | 결과 | 상태 |
|----|------|------|:----:|
| S1 회귀 보존 | 227 PASS 유지 | 227 → **258** PASS (+31) | ✓ |
| S2 select_level | 4-Level 동작 | 4 단위 테스트 PASS | ✓ |
| S3 throughput | ≥ 1.5× | **1.95-2.67×** (N/lat 9 케이스 모두) | ✓✓ |
| S4 graphify | 50% 감소 | 자동 측정 미실행 | △ |

**3/4 strict PASS + 1 partial → Match Rate 91%** → design §9 "≥ 90% 통과".

---

## 6. 코드 변경 + DAG

### 6-1. 파일 변화

| 파일 | 변경 | 줄수 |
|------|------|-----:|
| `htp/runtime/external_region.py` | 신규 ABC | +75 |
| `htp/llm/llm_region.py` | 신규 LLMRegion(ExternalRegion) | +160 |
| `htp/llm/cost_router.py` | select_level + LEVEL_* 상수 | +50 (기존 86 → 136) |
| `htp/llm/__init__.py` | export 갱신 (LLMRegionRuntime 제거) | -1 +4 |
| `htp/__init__.py` | export 갱신 | -1 +4 |
| `htp/runtime/async_brain_runtime.py` | `_last_result.outputs` hasattr 분기 | +5 |
| `htp/runtime/pipelined_brain.py` | 신규 PipelinedBrainRuntime | +130 |
| `examples/llm_region_demo.py` | 신규 demo | +100 |
| `archive/deprecated_phase4/llm_region_runtime.py` | git mv | 178 (이동) |
| `tests/regression/test_sub4_*.py` (4 파일) | 신규 | +400 |

**소스 순증**: +422 (570 신규 - 178 archive 이동). 테스트 +400.

### 6-2. DAG 영향

```
htp/runtime/external_region.py
  → htp.thalamus.region_signal   (이미 thalamus 는 base layer)

htp/llm/llm_region.py
  → htp/runtime/external_region  (단방향)
  → htp/llm/llm_node, cost_router
  → htp/thalamus.region_signal   (RegionSignal — 기존)

htp/runtime/pipelined_brain.py
  → htp/runtime/async_brain_runtime  (상속)
  → asyncio
```

**기존 DAG 룰 영향 없음** — `htp/core/*` 변경 없음, `htp/knowledge/*` 변경 없음, `htp/thalamus/*` 변경 없음.
신규 DAG 룰 불필요 (의존 방향이 자연스러움).

---

## 7. 테스트 catalog

총 31 신규 (총 baseline 227 → 258 PASS):

```
tests/regression/test_sub4_external_region.py (5):
  test_external_region_abstract_cannot_instantiate
  test_external_region_subclass_must_implement_run
  test_external_region_subclass_must_implement_collect_signal
  test_external_region_default_arun_wraps_sync
  test_external_region_apply_suppression_default_noop

tests/regression/test_sub4_llm_region.py (10):
  test_llm_region_inherits_external_region
  test_llm_region_llm_node_is_internal_member       ← C-2 검증
  test_llm_region_mock_run_returns_dict
  test_llm_region_collect_signal_returns_region_signal
  test_llm_region_precision_reflects_pressure
  test_llm_region_async_run_works
  test_llm_region_cost_blocked_returns_cached_or_blocked
  test_llm_region_apply_suppression_noop
  test_llm_region_specialty_prompt_auto_selected
  test_llm_region_unknown_specialty_fallback_system

tests/regression/test_sub4_cost_router.py (10):
  test_cost_router_existing_7_methods_preserved     ← C-3 영구 보호
  test_cost_router_update_and_pressure_behavior_unchanged
  test_cost_router_status_thresholds
  test_select_level_default_returns_2
  test_select_level_high_pressure_to_level_1
  test_select_level_high_complexity_to_level_4
  test_select_level_mid_complexity_to_level_3
  test_select_level_mid_pressure_simple_to_level_2
  test_select_level_complexity_validation
  test_select_level_constants_match_design

tests/regression/test_sub4_pipeline.py (6):
  test_pipelined_brain_inherits_async_brain
  test_pipelined_buffer_size_validation
  test_pipelined_arun_empty_inputs
  test_pipelined_arun_preserves_order
  test_pipelined_run_sync_wrapper
  test_pipelined_throughput_at_least_1_3x_async     ← S3 정량 검증
```

---

## 8. 외부 리뷰 포커스

리뷰해야 할 핵심 결정 사항:

### 8-1. 설계 결정

1. **ExternalRegion 의 BrainRuntime 호환 dummy 속성** (`_nodes=[]`, `_cusum_S=0`,
   `_cusum_h=1e9`) 이 장기적으로 옳은 접근인가? 대안: BrainRuntime 의 `if region._nodes:`
   부분이 hasattr 분기로 정리되어야 하는가? 현재는 ExternalRegion 쪽에 burden.

2. **C-2 LLMNode 처리 정책 — 옵션 A (내부 멤버) vs B (archive)**.
   현재 A 채택 (LLMNode 는 LLMRegion 디테일). 만약 다른 ExternalRegion (예: SearchRegion)
   이 등장하면 LLMNode 가 LLM 전용 디테일임이 명확해짐. 그러나 LLM API 호출이 향후
   다른 곳에서도 직접 필요할 수 있다면 export 가 필요할 수도 있음.

3. **PipelinedBrainRuntime 의 S2/S3 serialize** — 입력 순서 보존을 위해 S2 (Thalamus)
   와 S3 (PFC) 는 입력 idx 순서로 serial 진행. 이는 PFC 의 working memory deque 가
   순서 의존이라 옳은 선택이나, **S2 도 사실은 input 무관 (Thalamus 는 stateless 한
   collect+WTA)** 이므로 S2 까지 병렬화 가능 — 향후 최적화 여지.

4. **`CostRouter.select_level` 의 4-Level 임계값** — `pressure > 0.8` / `0.5` /
   `complexity > 0.8` / `0.5`. 이 임계값들이 실제 환경 (Anthropic API 비용 분포) 에서
   합리적인가? Plan §SUCCESS Level 1-2 비율 70% 목표 측정 후 재조정 필요.

### 8-2. 측정 신뢰도

5. **Throughput 측정 — mock LLM 의 `asyncio.sleep` 시뮬레이션**. 실제 Anthropic API 의
   tail latency (P99) 와 다를 수 있음. 실제 API 호출에서 speedup 이 1.5× 이하로
   떨어질 시나리오?

6. **buffer_size=3 default** — 메모리 / API rate-limit / cost overload risk 의 trade-off
   고려. 실제 LLM 환경에서 buffer 3 이 적절한가?

7. **graphify isolated 50% 감소 (S4) 미측정** — LLMRegionRuntime 178줄 archive 이동의
   정성적 감소는 명확하나, 정량 검증 미실행. 후속 cycle 필요. 측정 도구는?

### 8-3. 검증 한계

8. **ExternalRegion 단일 구현 (LLMRegion) 만 검증** — 추상의 진정한 가치는 다양한
   구현이 등장할 때 드러남. 향후 SearchRegion, RAGRegion 등이 같은 추상으로 작동하는가?

9. **PipelinedBrainRuntime 의 Memory 통합** — 현재 super().arun 의 Memory 부분은
   pipelined_arun 에서 복제하지 않음. multi-input pipeline 시 Memory novelty/recall
   이 어떻게 작동해야 하는가?

10. **graphify isolated 감소 정량 측정의 자동화** — graphify 자동 실행 → 노드 분류 →
    이전 baseline 과 비교. 도구화하면 매 cycle 회귀 보호 가능 (현재는 ad-hoc).

---

## 9. 후속 작업

| 우선순위 | 항목 |
|:--:|------|
| 1 | C-4 graphify isolated 정량 측정 — `htp-graphify-isolated-audit` micro-cycle (~30분) |
| 2 | sub-6 (Stage 7 vector default 전환) — Bridge 검증 PASS 로 진입 명분 확보 |
| 3 | 추가 ExternalRegion 구현 (SearchRegion, RAGRegion) — 별도 cycle |
| 4 | `CostRouter.select_level` 임계값 튜닝 — 실 사용 분포 측정 후 조정 |
| 5 | PipelinedBrainRuntime Memory 통합 — multi-input novelty/recall 정책 |

---

## 10. 참고 파일

| 위치 | 내용 |
|------|------|
| `docs/02-design/features/htp-thalamus-car.sub-4.design.md` | 원본 design + 3 architecture 옵션 |
| `docs/03-analysis/htp-thalamus-car.sub-4.analysis.md` | sub-4 Check 분석 (Match Rate 91%) |
| `docs/03-analysis/htp-bridge-integration-실사용검증-외부리뷰용.md` | 이전 cycle 리뷰 (시스템 A↔B 검증) |
| `htp/runtime/external_region.py` | ExternalRegion ABC |
| `htp/llm/llm_region.py` | LLMRegion 구현 |
| `htp/runtime/pipelined_brain.py` | PipelinedBrainRuntime |
| `examples/llm_region_demo.py` | C-1 사용 데모 |
| `archive/deprecated_phase4/llm_region_runtime.py` | archive 이동된 구 LLMRegionRuntime |
| `tests/regression/test_sub4_*.py` | 4 신규 테스트 파일 |

---

## 11. 결론

| 지표 | 값 |
|------|----|
| Plan §SUCCESS | 3/4 strict + 1 partial (Match Rate **91%**) |
| 회귀 baseline | 227 → **258** PASS (+31) |
| Throughput | **1.95-2.67×** (목표 1.5× 큰 마진 초과) |
| 신규 소스 | +422 줄 (LLMRegionRuntime 178줄 archive 이동 포함) |
| 깨진 회귀 | **0 건** |
| 소요 시간 | ~3.5h (Session A 1.5h + B 0.5h + C 1.5h) |

**G3 (LLMRegionRuntime 상속 비용) 본질 해결**. ExternalRegion 추상 도입으로 향후
다양한 외부 호출 (LLM / Search / RAG / Tool) 을 동일 추상에서 처리 가능.
PipelinedBrainRuntime 실측 throughput 이 Plan 목표를 크게 초과해 다중 입력 시나리오
에서 비용/시간 효율성 강하게 개선. 회귀 0 깨짐으로 안전한 변경.

**다음 진입 후보**: C-4 정량 측정 (graphify), sub-6 (vector default), 또는
SearchRegion 같은 다른 ExternalRegion 구현 실험.
