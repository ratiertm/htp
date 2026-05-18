"""
LLMRegion — ExternalRegion 의 LLM 구현.

Design Ref: docs/02-design/features/htp-thalamus-car.sub-4.design.md §3-2
Plan SC: FR-17 (LLMRegion(ExternalRegion)) + C-2 (LLMNode 는 내부 멤버)

기존 LLMRegionRuntime 의 LLM 호출 로직을 흡수하되 RegionRuntime 비상속.
PageRank/Hebbian/NGE dependency 가 사라져 graphify 상 isolated 감소.

LLMNode 는 self._llm_node 내부 멤버 (C-2 옵션 A) — 외부 import 노출 안 함.
"""
from __future__ import annotations

import time
from typing import Any

import torch

from htp.runtime.external_region import ExternalRegion
from htp.thalamus.region_signal  import RegionSignal
from .llm_node                   import LLMNode, MockLLMNode
from .cost_router                import CostRouter


# 기존 llm_region_runtime.py 의 SPECIALTY_PROMPTS 그대로 재사용
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


class LLMRegion(ExternalRegion):
    """LLM 호출을 ExternalRegion 으로 노출.

    Parameters
    ----------
    region_name : Region 식별자
    specialty   : SPECIALTY_PROMPTS 키 또는 자유 문자열
    model       : Anthropic 모델 ID
    system      : 커스텀 system prompt (None 이면 specialty 자동)
    budget      : 스텝당 허용 비용 ($)
    use_mock    : True 면 MockLLMNode (API 키 없이 사용 가능)
    """

    def __init__(
        self,
        region_name: str,
        specialty:   str,
        model:       str   = "claude-sonnet-4-6",
        system:      "str | None" = None,
        budget:      float = 0.01,
        use_mock:    bool  = False,
        llm_node:    "object | None" = None,
    ):
        """
        llm_node: override — 명시 시 그 인스턴스를 self._llm_node 로 사용.
          LLMNode / MockLLMNode / ClaudeCliNode 등 동일 interface (name/run/arun/
          _token_log/cost_report) 만족하면 됨. None 이면 use_mock 에 따라
          기존 LLMNode/MockLLMNode 자동 생성.
        """
        self.region_name = region_name
        self.specialty   = specialty
        self.step        = 0

        self.model    = model
        self.system   = system or SPECIALTY_PROMPTS.get(
            specialty, f"You are a {specialty} specialist. Return JSON.",
        )
        self.router   = CostRouter(budget_per_step=budget)
        self.use_mock = use_mock

        if llm_node is not None:
            self._llm_node = llm_node
        else:
            NodeClass = MockLLMNode if use_mock else LLMNode
            self._llm_node = NodeClass(
                name        = f"{region_name}_llm",
                model       = self.router.suggest_model(self.model),
                system      = self.system,
                tags        = set(specialty.replace("_", " ").split()),
            )
        self._last_result: "dict | None" = None

    # ── ExternalRegion interface ─────────────────────────

    def run(self, data: Any) -> dict:
        """동기 LLM 호출. cost block 시 cached 또는 blocked dict 반환."""
        if self.router.should_block():
            return self._last_result or {
                "text": "cost_blocked", "label": "blocked",
            }
        result = self._llm_node.run(data)
        if self._llm_node._token_log:
            last = self._llm_node._token_log[-1]
            self.router.update(last["cost"], last["ms"])
        self.step += 1
        self._last_result = result
        return result

    async def arun(self, data: Any) -> dict:
        """비동기 LLM 호출."""
        if self.router.should_block():
            return self._last_result or {
                "text": "cost_blocked", "label": "blocked",
            }
        t0 = time.perf_counter()
        try:
            result  = await self._llm_node.arun(data)
            elapsed = (time.perf_counter() - t0) * 1000
            if self._llm_node._token_log:
                last = self._llm_node._token_log[-1]
                self.router.update(last["cost"], elapsed)
            self.step += 1
            self._last_result = result
            return result
        except Exception as e:
            # 외부 API 실패 시 마지막 성공 결과 또는 blocked 반환 (회복 가능 동작)
            print(f"  [LLMRegion:{self.region_name}] error: {e}")
            return self._last_result or {
                "text": f"error: {e}", "label": "error",
            }

    def collect_signal(self) -> RegionSignal:
        """CostRouter pressure 를 precision 의 역수로 환산.

        - precision  : pressure 0 → 5.0, pressure 1 → 0.5 (clamp [0.1, 5.0])
        - fire_rate  : self.step / 100 (clamp 1.0) — 호출 빈도 근사
        - overload   : router.should_block()
        - output_vec : LLM 응답의 의미 vec 없으면 placeholder.
                       EmbeddingBridge 통합 (Stage 6) 시 prompt_to_vec 사용.
        """
        p = self.router.pressure   # property
        # precision: pressure 가 0 (저비용) 일수록 신뢰도 ↑. clamp [0.1, 5.0]
        precision = min(max(1.0 / max(0.2, p + 0.2), 0.1), 5.0)
        return RegionSignal(
            region_id    = self.region_name,
            hub_strength = 0.0,                         # 외부 region 은 hub 없음
            fire_rate    = float(min(1.0, self.step / 100)),
            top_hubs     = [],
            overload     = self.router.should_block(),
            output_vec   = torch.zeros(1),              # placeholder
            precision    = precision,
        )

    # apply_suppression 은 ExternalRegion default (no-op) 사용

    # ── 보고 ─────────────────────────────────────────────

    def cost_report(self) -> str:
        return (
            f"  [{self.region_name}] {self.router.report()}\n"
            f"{self._llm_node.cost_report()}"
        )


__all__ = ["LLMRegion", "SPECIALTY_PROMPTS"]
