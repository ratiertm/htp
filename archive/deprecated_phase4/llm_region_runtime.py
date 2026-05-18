"""
LLMRegionRuntime  —  LLM을 HTP 노드로 동작시키는 RegionRuntime
================================================================

specialty -> system prompt 자동 변환.
CostRouter로 모델 다운그레이드 및 차단 관리.
arun() 비동기 지원.

SPECIALTY_PROMPTS:
  사전 정의된 전문 영역별 system prompt.
  미정의 specialty는 기본 JSON 반환 프롬프트.
"""

from __future__ import annotations

import time
from typing import Any

from ..runtime.region_runtime import RegionRuntime
from ..runtime.htp_runtime    import Node, RunResult
from .llm_node                import LLMNode, MockLLMNode
from .cost_router             import CostRouter


SPECIALTY_PROMPTS: dict[str, str] = {
    "language":  (
        "Parse input, extract intent and entities as JSON "
        "with 'intent', 'entities', 'text' keys."
    ),
    "code":      (
        "Analyze code, find issues, suggest fixes. "
        "Return JSON with 'analysis', 'issues', 'suggestions'."
    ),
    "memory":    (
        "Retrieve and synthesize context. "
        "Return JSON with 'recalled', 'relevance', 'summary'."
    ),
    "emotion":   (
        "Assess emotional tone. "
        "Return JSON with 'sentiment', 'intensity', 'context'."
    ),
    "reasoning": (
        "Reason step by step. "
        "Return JSON with 'steps', 'conclusion', 'confidence'."
    ),
}


class LLMRegionRuntime(RegionRuntime):
    """
    LLM API를 HTP 노드로 사용하는 Region.

    Parameters
    ----------
    region_name : Region 식별자
    specialty   : SPECIALTY_PROMPTS 키 또는 자유 문자열
    model       : Anthropic 모델 ID
    system      : 커스텀 system prompt (None이면 specialty 자동 선택)
    budget      : 스텝당 허용 비용 ($)
    use_mock    : True면 API 호출 없이 MockLLMNode 사용
    """

    def __init__(
        self,
        region_name: str,
        specialty: str,
        model: str = "claude-sonnet-4-6",
        system: str | None = None,
        budget: float = 0.01,
        use_mock: bool = False,
        config=None,
        gen_config=None,
    ):
        super().__init__(region_name, specialty, config, gen_config)
        self.model    = model
        self.system   = system or SPECIALTY_PROMPTS.get(
            specialty, f"You are a {specialty} specialist. Return JSON."
        )
        self.router   = CostRouter(budget_per_step=budget)
        self.use_mock = use_mock

        self._llm_node: LLMNode | None = None
        self._last_result: RunResult | None = None  # timeout 캐시

    # ── 빌드 ─────────────────────────────────────────

    def _ensure_built(self):
        super()._ensure_built()
        if self._llm_node is None:
            self._build_llm_node()

    def _build_llm_node(self):
        NodeClass = MockLLMNode if self.use_mock else LLMNode
        self._llm_node = NodeClass(
            name=f"{self.region_name}_llm",
            model=self.router.suggest_model(self.model),
            system=self.system,
            tags=set(self.specialty.replace("_", " ").split()),
        )

        # HTP 노드로 등록
        fn = self._make_llm_fn()
        fn._htp_tags = self._llm_node.tags
        node = Node(fn=fn, node_id=self._node_count)
        self._nodes.append(node)
        self._name_map[fn.__name__] = node
        self._node_count += 1
        fn._htp_node = node

    def _make_llm_fn(self):
        llm    = self._llm_node
        router = self.router

        def llm_node_fn(data):
            if router.should_block():
                return {"text": "cost_blocked", "label": "blocked"}
            result = llm.run(data)
            if llm._token_log:
                last = llm._token_log[-1]
                router.update(last["cost"], last["ms"])
            return result

        llm_node_fn.__name__ = f"{self.region_name}_llm"
        return llm_node_fn

    # ── 비동기 실행 ───────────────────────────────────

    async def arun(self, data: Any) -> RunResult:
        """비동기 LLM 호출. AsyncBrainRuntime에서 사용."""
        self._ensure_built()

        if self.router.should_block():
            return self._cached_result_or_empty(data)

        t0 = time.perf_counter()
        try:
            result_data = await self._llm_node.arun(data)
            elapsed     = (time.perf_counter() - t0) * 1000

            if self._llm_node._token_log:
                last = self._llm_node._token_log[-1]
                self.router.update(last["cost"], elapsed)

            self._step += 1
            rr = RunResult(
                input_data=data,
                route_path=[],
                outputs={self.region_name: result_data},
                hub_ids=[],
                pruned={},
                total_ms=elapsed,
            )
            self._last_result = rr
            return rr

        except Exception as e:
            print(f"  [LLMRegionRuntime:{self.region_name}] error: {e}")
            return self._cached_result_or_empty(data)

    def _cached_result_or_empty(self, data: Any) -> RunResult:
        if self._last_result is not None:
            return self._last_result
        return RunResult(
            input_data=data,
            route_path=[],
            outputs={},
            hub_ids=[],
            pruned={},
            total_ms=0.0,
        )

    # ── 보고 ─────────────────────────────────────────

    def cost_report(self) -> str:
        lines = [f"  [{self.region_name}] {self.router.report()}"]
        if self._llm_node:
            lines.append(self._llm_node.cost_report())
        return "\n".join(lines)
