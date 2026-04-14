"""
Stage 2-A4 — MatrixCells overload_bonus 파라미터화 회귀 테스트.
"""
from __future__ import annotations

import pytest
import torch

from htp import MatrixCells, GatingMask, RegionSignal


def _sig(name, *, overload=False, fire_rate=0.1):
    return RegionSignal(
        region_id=name,
        hub_strength=0.5,
        fire_rate=fire_rate,
        top_hubs=[],
        overload=overload,
        output_vec=torch.zeros(4),
    )


@pytest.mark.regression
def test_overload_bonus_default_is_0_2():
    """기본값 0.2 — 하위 호환 확인."""
    mc = MatrixCells()
    assert mc.overload_bonus == 0.2


@pytest.mark.regression
def test_overload_bonus_zero_disables_overload_advantage():
    """overload_bonus=0.0 이면 과부하 Region 의 승리 우위가 사라져야."""
    mc = MatrixCells(overload_bonus=0.0)
    gating = GatingMask(scores={"A": 0.5, "B": 0.5})
    signals = [_sig("A", overload=True), _sig("B", overload=False)]

    result = mc.compete(signals, gating)

    # gating 점수가 같으므로 보너스 없으면 두 Region 확률이 거의 같아야
    assert abs(result.all_scores["A"] - result.all_scores["B"]) < 1e-3, \
        f"overload_bonus=0.0 인데 확률이 다름: {result.all_scores}"


@pytest.mark.regression
def test_overload_bonus_high_forces_overload_winner():
    """overload_bonus=1.0 이면 과부하 Region 이 거의 확정 승자."""
    mc = MatrixCells(overload_bonus=1.0, temperature=0.5)
    gating = GatingMask(scores={"A": 0.3, "B": 0.4})  # B 가 gating 우위
    signals = [_sig("A", overload=True), _sig("B", overload=False)]

    result = mc.compete(signals, gating)

    # overload 가 gating 열세를 뒤집어 A 가 승자
    assert result.winner_id == "A", \
        f"큰 overload_bonus 에도 A 승자 아님: winner={result.winner_id}, scores={result.all_scores}"
