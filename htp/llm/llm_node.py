"""
LLMNode  —  Anthropic API 동기/비동기 래핑 + 비용 추적
===========================================================

Token Cost Model:
  price (claude-sonnet-4-6):
    p_in  = 3e-6  $/token
    p_out = 15e-6 $/token

  cost_per_call = input_tokens × p_in + output_tokens × p_out

MockLLMNode:
  API 키 없을 때 단위 테스트용.
"""

from __future__ import annotations

import json
import time


PRICING: dict[str, dict[str, float]] = {
    "claude-sonnet-4-6":        {"in": 3e-6,    "out": 15e-6},
    "claude-haiku-4-5-20251001":{"in": 0.8e-6,  "out": 4e-6},
    "claude-opus-4-6":          {"in": 15e-6,   "out": 75e-6},
}


class LLMNode:
    """
    HTP 노드로 동작하는 LLM 래퍼.

    - run(data)   : 동기 호출
    - arun(data)  : 비동기 호출
    - cost_report(): 비용/지연 요약
    """

    def __init__(
        self,
        name: str,
        model: str,
        system: str,
        tags: set | None = None,
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ):
        self.name        = name
        self.model       = model
        self.system      = system
        self.tags        = tags or set()
        self.temperature = temperature
        self.max_tokens  = max_tokens

        # 추적
        self.call_count  = 0
        self.total_cost  = 0.0
        self.total_ms    = 0.0
        self._token_log: list[dict] = []  # 최근 10개

    # ── 내부 헬퍼 ────────────────────────────────────

    def _format_prompt(self, data) -> str:
        if isinstance(data, dict):
            return "\n".join(f"{k}: {v}" for k, v in data.items())
        return str(data)

    def _parse_response(self, response) -> dict:
        text = response.content[0].text
        try:
            return json.loads(text)
        except Exception:
            return {"text": text, "label": "llm_response"}

    def _track(self, usage, elapsed_ms: float):
        p    = PRICING.get(self.model, PRICING["claude-sonnet-4-6"])
        cost = usage.input_tokens * p["in"] + usage.output_tokens * p["out"]
        self.call_count  += 1
        self.total_cost  += cost
        self.total_ms    += elapsed_ms
        entry = {
            "in":   usage.input_tokens,
            "out":  usage.output_tokens,
            "cost": cost,
            "ms":   elapsed_ms,
        }
        self._token_log = (self._token_log + [entry])[-10:]

    # ── 공개 인터페이스 ───────────────────────────────

    def run(self, data) -> dict:
        """동기 Anthropic API 호출."""
        import anthropic
        client = anthropic.Anthropic()
        t0 = time.perf_counter()
        resp = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=self.system,
            messages=[{"role": "user", "content": self._format_prompt(data)}],
        )
        self._track(resp.usage, (time.perf_counter() - t0) * 1000)
        return self._parse_response(resp)

    async def arun(self, data) -> dict:
        """비동기 Anthropic API 호출."""
        import anthropic
        client = anthropic.AsyncAnthropic()
        t0 = time.perf_counter()
        resp = await client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=self.system,
            messages=[{"role": "user", "content": self._format_prompt(data)}],
        )
        self._track(resp.usage, (time.perf_counter() - t0) * 1000)
        return self._parse_response(resp)

    def cost_report(self) -> str:
        avg_ms   = self.total_ms   / max(self.call_count, 1)
        avg_cost = self.total_cost / max(self.call_count, 1)
        return (
            f"    LLMNode({self.name})  model={self.model}\n"
            f"      calls={self.call_count}  "
            f"total_cost=${self.total_cost:.6f}  "
            f"avg_cost=${avg_cost:.6f}  "
            f"avg_ms={avg_ms:.1f}"
        )


class MockLLMNode(LLMNode):
    """
    API 키 없을 때 단위 테스트용 Mock.
    실제 API 호출 없이 즉시 반환.
    """

    def run(self, data) -> dict:
        return {
            "text":  f"mock({self.system[:30]}): {str(data)[:50]}",
            "label": "mock_response",
        }

    async def arun(self, data) -> dict:
        return self.run(data)
