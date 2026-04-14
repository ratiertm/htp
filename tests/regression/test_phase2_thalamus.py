"""
Phase 2 회귀 — Thalamus 체인.

검증:
  - BrainRuntime 1 step 실행 → Action 생성 + TopDownSignal 생성
  - RegionSignal이 모든 Region에서 수집됨
  - Thalamus가 JL 압축으로 8-dim state_vec 출력 (Stage 4 전까지)
  - MatrixCells WTA 승자 결정
"""
from __future__ import annotations

import pytest
import torch


@pytest.mark.regression
def test_brain_runtime_single_step(two_region_brain):
    """BrainRuntime.run() 1회 → Action 반환, winner ∈ 등록된 Region."""
    brain = two_region_brain

    action = brain.run({"label": "text", "payload": "hello"})

    assert action.type in ("execute", "inhibit"), f"unknown action.type: {action.type}"
    assert action.winner in brain.regions, f"winner가 미등록 Region: {action.winner}"


@pytest.mark.regression
def test_region_signal_fields(two_region_brain):
    """각 Region이 RegionSignal 전 필드를 정상 반환."""
    brain = two_region_brain

    # 1회 실행 후 signal 수집
    brain.run({"label": "text"})

    for region in brain.regions.values():
        sig = region.collect_signal()
        assert isinstance(sig.region_id, str)
        assert isinstance(sig.hub_strength, float)
        assert 0.0 <= sig.fire_rate <= 1.0, f"fire_rate 범위 이탈: {sig.fire_rate}"
        assert isinstance(sig.top_hubs, list)
        assert isinstance(sig.overload, bool)
        assert isinstance(sig.output_vec, torch.Tensor)


@pytest.mark.regression
def test_thalamus_state_vec_dim_is_64(two_region_brain):
    """
    Stage 4 — compress_dim=64. 해마 place-cell sparse 표현 근거.
    JL Lemma: k ≈ log(1000)/0.1² ≈ 64.
    """
    brain = two_region_brain
    brain.run({"label": "text"})

    assert len(brain.pfc.working_memory) >= 1
    state_vec = brain.pfc.working_memory[-1].state_vec
    assert state_vec.shape[0] == 64, \
        f"Stage 4 이후 64-dim 기대, 실제 {state_vec.shape[0]}"


@pytest.mark.regression
def test_matrix_cells_softmax_is_probability(two_region_brain):
    """MatrixCells all_scores는 확률 분포 — 합 ≈ 1."""
    brain = two_region_brain
    brain._ensure_thalamus()
    brain.run({"label": "text"})

    # 내부 호출로 CompetitionResult 생성
    signals = [r.collect_signal() for r in brain.regions.values()]
    gating = brain.thalamus.core.gate(signals)
    comp = brain.thalamus.matrix.compete(signals, gating)

    total = sum(comp.all_scores.values())
    assert abs(total - 1.0) < 1e-3, f"Softmax 합 {total} != 1"
    assert comp.winner_id in comp.all_scores
    assert comp.winner_score == max(comp.all_scores.values())
