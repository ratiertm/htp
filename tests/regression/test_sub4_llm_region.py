"""sub-4 Stage 4 — LLMRegion(ExternalRegion) 검증.

Design Ref: docs/02-design/features/htp-thalamus-car.sub-4.design.md §3-2
Plan SC: FR-17 + C-2 (LLMNode 내부 멤버 옵션 A)

MockLLMNode 사용 — API 키 불필요.
"""
from __future__ import annotations

import asyncio
import torch

from htp.runtime.external_region import ExternalRegion
from htp.llm.llm_region          import LLMRegion, SPECIALTY_PROMPTS
from htp.llm.llm_node            import MockLLMNode
from htp.thalamus.region_signal  import RegionSignal


def test_llm_region_inherits_external_region():
    """LLMRegion 은 ExternalRegion 의 하위 클래스."""
    r = LLMRegion("test", specialty="language", use_mock=True)
    assert isinstance(r, ExternalRegion)


def test_llm_region_llm_node_is_internal_member():
    """C-2 옵션 A: LLMNode 는 self._llm_node 내부 멤버."""
    r = LLMRegion("test", specialty="language", use_mock=True)
    assert hasattr(r, "_llm_node")
    assert isinstance(r._llm_node, MockLLMNode)


def test_llm_region_mock_run_returns_dict():
    """MockLLMRegion.run() 은 dict 반환 + step 증가."""
    r = LLMRegion("test", specialty="language", use_mock=True)
    assert r.step == 0
    result = r.run("hello")
    assert isinstance(result, dict)
    assert "text" in result
    assert r.step == 1


def test_llm_region_collect_signal_returns_region_signal():
    """collect_signal 은 RegionSignal — Thalamus 입력."""
    r = LLMRegion("test", specialty="language", use_mock=True)
    sig = r.collect_signal()
    assert isinstance(sig, RegionSignal)
    assert sig.region_id == "test"
    assert sig.hub_strength == 0.0          # 외부 region 은 hub 없음
    assert sig.top_hubs == []
    assert 0.1 <= sig.precision <= 5.0      # precision clamp


def test_llm_region_precision_reflects_pressure():
    """CostRouter pressure 가 높을수록 precision 낮음."""
    r = LLMRegion("test", specialty="language", use_mock=True)
    sig_low = r.collect_signal()
    p_low = sig_low.precision

    # 인위적으로 ema_cost 를 끌어올려 pressure 증가
    r.router._ema_cost = r.router.budget * 2.0   # pressure ≈ 2.0
    sig_high = r.collect_signal()
    p_high = sig_high.precision

    assert p_high < p_low, (
        f"pressure 증가 시 precision 감소 기대. low={p_low}, high={p_high}"
    )


def test_llm_region_async_run_works():
    """arun (async) — MockLLMNode 도 비동기 호출 지원."""
    r = LLMRegion("test", specialty="language", use_mock=True)
    result = asyncio.run(r.arun("hello async"))
    assert isinstance(result, dict)
    assert r.step == 1


def test_llm_region_cost_blocked_returns_cached_or_blocked():
    """should_block 시 cached 또는 blocked dict 반환 — 호출 안 함."""
    r = LLMRegion("test", specialty="language", use_mock=True)
    # 첫 호출로 _last_result 채움
    first = r.run("first")
    # 압박 임계 초과
    r.router._ema_cost = r.router.budget * 5.0   # pressure ≈ 5.0 > PRESSURE_BLOCK(2.0)
    assert r.router.should_block()
    blocked = r.run("blocked")
    # cached (first) 또는 blocked 둘 다 acceptable
    assert blocked == first or blocked.get("label") == "blocked"


def test_llm_region_apply_suppression_noop():
    """LLMRegion 은 apply_suppression default 사용 (no-op)."""
    r = LLMRegion("test", specialty="language", use_mock=True)
    r.apply_suppression(0.5)   # 예외 없음
    # 상태 변경 없음
    assert r.step == 0


def test_llm_region_specialty_prompt_auto_selected():
    """specialty 가 SPECIALTY_PROMPTS 키면 자동 system prompt 선택."""
    r = LLMRegion("lang", specialty="language", use_mock=True)
    assert r.system == SPECIALTY_PROMPTS["language"]


def test_llm_region_unknown_specialty_fallback_system():
    """미지 specialty 는 fallback system prompt."""
    r = LLMRegion("custom", specialty="quantum_physics", use_mock=True)
    assert "quantum_physics" in r.system
    assert "specialist" in r.system
