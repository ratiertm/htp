"""sub-4 Stage 4 — CostRouter.select_level + 기존 7-method 보존 검증.

Design Ref: docs/02-design/features/htp-thalamus-car.sub-4.design.md §3-3
Plan SC: FR-18 + C-3 (기존 7-method 보존)
"""
from __future__ import annotations

import pytest

from htp.llm.cost_router import CostRouter


# ── C-3 회귀: 기존 7-method 보존 ─────────────────────

def test_cost_router_existing_7_methods_preserved():
    """기존 update/pressure/status/suggest_model/routing_score/should_block/report 모두 존재."""
    r = CostRouter(budget_per_step=0.01)
    assert callable(r.update)
    # property 3
    _ = r.pressure
    _ = r.status
    # method 4
    assert callable(r.suggest_model)
    assert callable(r.routing_score)
    assert callable(r.should_block)
    assert callable(r.report)


def test_cost_router_update_and_pressure_behavior_unchanged():
    """기존 EMA 동작 보존."""
    r = CostRouter(budget_per_step=0.001)
    assert r.pressure == 0.0   # 초기
    r.update(cost=0.0005, latency_ms=100.0)
    # ema_cost = 0.3 * 0.0005 + 0.7 * 0 = 0.00015
    # pressure = 0.00015 / 0.001 = 0.15
    assert 0.1 < r.pressure < 0.2


def test_cost_router_status_thresholds():
    """status threshold 보존: normal/warn/overload."""
    r = CostRouter(budget_per_step=0.01)
    r._ema_cost = 0.003
    assert r.status == "normal"
    r._ema_cost = 0.007
    assert r.status == "warn"
    r._ema_cost = 0.015
    assert r.status == "overload"


# ── FR-18: select_level 4-Level ──────────────────────

def test_select_level_default_returns_2():
    """초기 (pressure=0) + default complexity=0.5 → LEVEL_SLLM (2)."""
    r = CostRouter()
    level = r.select_level()
    assert level == r.LEVEL_SLLM == 2


def test_select_level_high_pressure_to_level_1():
    """비용 극압박 (pressure > 0.8) → LEVEL_LOCAL (1)."""
    r = CostRouter(budget_per_step=0.001)
    r._ema_cost = 0.001    # pressure ≈ 1.0 > 0.8
    assert r.select_level(query_complexity=0.5) == r.LEVEL_LOCAL == 1


def test_select_level_high_complexity_to_level_4():
    """복잡 쿼리 (complexity > 0.8) + 저압박 → LEVEL_API_LARGE (4)."""
    r = CostRouter()
    assert r.select_level(query_complexity=0.9) == r.LEVEL_API_LARGE == 4


def test_select_level_mid_complexity_to_level_3():
    """중간 쿼리 (0.5 < complexity ≤ 0.8) + 저압박 → LEVEL_API_SMALL (3)."""
    r = CostRouter()
    assert r.select_level(query_complexity=0.6) == r.LEVEL_API_SMALL == 3
    assert r.select_level(query_complexity=0.8) == r.LEVEL_API_SMALL == 3


def test_select_level_mid_pressure_simple_to_level_2():
    """중간 압박 (0.5 < p ≤ 0.8) + 단순 쿼리 → LEVEL_SLLM (2)."""
    r = CostRouter(budget_per_step=0.001)
    r._ema_cost = 0.0006   # pressure ≈ 0.6
    assert r.select_level(query_complexity=0.3) == r.LEVEL_SLLM == 2


def test_select_level_complexity_validation():
    """complexity 범위 외 입력 → ValueError."""
    r = CostRouter()
    with pytest.raises(ValueError):
        r.select_level(query_complexity=-0.1)
    with pytest.raises(ValueError):
        r.select_level(query_complexity=1.1)


def test_select_level_constants_match_design():
    """LEVEL_* 상수가 design §3-3 과 일치 (1-4)."""
    r = CostRouter()
    assert r.LEVEL_LOCAL     == 1
    assert r.LEVEL_SLLM      == 2
    assert r.LEVEL_API_SMALL == 3
    assert r.LEVEL_API_LARGE == 4
