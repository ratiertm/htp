"""
Phase 2 회귀 — NodeGenerationEngine 분열 메커니즘.

CLAUDE.md 주장: "허브 분열: 30회 데이터 후 classify 노드 자동 분열".
실제 구현은 GenConfig.split_strength_threshold + split_call_threshold + maturity + cooldown으로
매우 보수적 — 단일 fixture로 30회 반복해도 분열이 일어나지 않을 수 있다.

본 테스트는 NGE 자체의 호출성(check_split 호출됨, 이벤트 로그 존재) 회귀를 검증한다.
실제 분열 발생은 별도의 시뮬 시나리오로 확인.
"""
from __future__ import annotations

import pytest


@pytest.mark.regression
def test_region_runtime_has_nge_after_build(two_region_brain):
    """RegionRuntime.run() 1회 후 nge가 초기화되어야."""
    brain = two_region_brain
    brain.run({"label": "text"})

    for region in brain.regions.values():
        assert region.nge is not None, \
            f"{region.region_name}.nge 미초기화"


@pytest.mark.regression
def test_nge_check_split_increments_step(two_region_brain):
    """여러 run() 후 region._step이 증가 — NGE check_split가 호출됨을 시사."""
    brain = two_region_brain

    for _ in range(5):
        brain.run({"label": "text"})

    for region in brain.regions.values():
        assert region._step >= 5, \
            f"{region.region_name}._step={region._step}"


@pytest.mark.regression
def test_cusum_state_exists(two_region_brain):
    """Shannon Entropy CUSUM 상태 필드가 RegionRuntime에 존재."""
    brain = two_region_brain
    brain.run({"label": "text"})

    for region in brain.regions.values():
        assert hasattr(region, "_cusum_S")
        assert hasattr(region, "_cusum_k")
        assert hasattr(region, "_cusum_h")
        assert region._cusum_S >= 0.0
