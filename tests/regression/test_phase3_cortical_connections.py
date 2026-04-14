"""
Phase 3 회귀 — CorticalConnections (Region 간 직접 약한 연결).
"""
from __future__ import annotations

import pytest


@pytest.mark.regression
def test_enable_cortical_connections_returns_object(two_region_brain):
    """enable_cortical_connections() 호출 시 CorticalConnections 객체 반환."""
    brain = two_region_brain

    cc = brain.enable_cortical_connections()

    assert cc is not None
    assert brain._cc is cc


@pytest.mark.regression
def test_brain_run_with_cortical_connections(two_region_brain):
    """CorticalConnections 활성화 상태에서 BrainRuntime.run() 정상 동작."""
    brain = two_region_brain
    brain.enable_cortical_connections()

    action = brain.run({"label": "text"})

    assert action is not None
    assert action.type in ("execute", "inhibit")
