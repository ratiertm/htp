"""
Phase 3 회귀 — Top-down feedback loop + PFC 결정.

검증:
  - long_term_goals 설정 시 다음 스텝 TopDownSignal.biases에 반영
  - PFC working_memory deque(maxlen=7) 유지
  - TopDownBias가 Jaccard-like 계산 (Stage 3-B4에서 Softmax로 변경 예정)
"""
from __future__ import annotations

import pytest


@pytest.mark.regression
def test_pfc_working_memory_maxlen_7(two_region_brain):
    """PFC working_memory는 deque(maxlen=7) — 8회 실행 후에도 7개만 유지."""
    brain = two_region_brain

    for i in range(10):
        brain.run({"label": "text", "i": i})

    assert len(brain.pfc.working_memory) == 7, \
        f"maxlen=7 기대, 실제 {len(brain.pfc.working_memory)}"


@pytest.mark.regression
def test_top_down_signal_generated_each_step(two_region_brain):
    """매 step마다 TopDownSignal이 생성되어 _last_td에 저장."""
    brain = two_region_brain

    brain.run({"label": "text"})

    assert brain._last_td is not None, "TopDownSignal 미생성"
    assert hasattr(brain._last_td, "biases")
    assert hasattr(brain._last_td, "strength")
    assert 0.0 <= brain._last_td.strength <= 1.0


@pytest.mark.regression
def test_long_term_goals_affect_top_down_biases(two_region_brain):
    """
    long_term_goals=["cache"] 설정 시 cache tag를 가진 Region의 bias > 0.

    Stage 3-B4 변경 후에도 "cache"가 overlap_count>0이면 softmax에서 non-zero 확률.
    """
    brain = two_region_brain
    brain.pfc.long_term_goals = ["cache", "store"]

    brain.run({"label": "text"})

    biases = brain._last_td.biases
    # memory Region은 "cache" specialty — bias > 0 기대
    assert biases.get("memory", 0.0) > 0.0, \
        f"memory Region bias=0, biases={biases}"


@pytest.mark.regression
def test_action_reason_contains_score(two_region_brain):
    """PFC.decide()가 생성한 action.reason에 score= 문자열 포함 (save() score 파싱 근거)."""
    brain = two_region_brain

    action = brain.run({"label": "text"})

    assert "score=" in action.reason, f"reason에 score= 없음: {action.reason!r}"
