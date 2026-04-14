"""
Stage 3-B1+B2 — RegionSignal.precision + RegionRuntime 동적 계산 회귀 테스트.
"""
from __future__ import annotations

import pytest

from htp import HTPConfig, RegionRuntime, tag, terminal


@pytest.mark.regression
def test_region_signal_has_precision_field():
    """RegionSignal dataclass 에 precision 필드가 존재."""
    from htp.thalamus.region_signal import RegionSignal
    import torch

    sig = RegionSignal(
        region_id="x", hub_strength=0.5, fire_rate=0.1, top_hubs=[],
        overload=False, output_vec=torch.zeros(4),
    )
    assert sig.precision == 1.0, "default precision=1.0 기대"


@pytest.mark.regression
def test_region_runtime_precision_clamped():
    """precision 값이 [0.1, 5.0] clamp 범위 내."""
    r = RegionRuntime("test", "dummy", config=HTPConfig(threshold=0.35))

    @r.node
    @terminal
    @tag("x")
    def node_a(data):
        return data

    for _ in range(5):
        r.run({"label": "x"})

    sig = r.collect_signal()
    assert 0.1 <= sig.precision <= 5.0, f"precision clamp 벗어남: {sig.precision}"


@pytest.mark.regression
def test_region_runtime_precision_default_before_history():
    """히스토리 3개 미만일 때 precision=1.0 (중립)."""
    r = RegionRuntime("test", "dummy", config=HTPConfig(threshold=0.35))

    @r.node
    @terminal
    @tag("x")
    def node_a(data): return data

    r.run({"label": "x"})  # 1회만 실행 — history < 3
    sig = r.collect_signal()
    assert sig.precision == 1.0, f"초기 precision 1.0 기대, 실제 {sig.precision}"


@pytest.mark.regression
def test_stable_fire_rate_yields_high_precision():
    """
    동일 입력 반복 + 매 step collect_signal 호출 시 precision ↑.
    실제 BrainRuntime 루프의 사용 패턴 (Thalamus.step() 이 매 스텝 collect_signal 호출).
    """
    r = RegionRuntime("test", "dummy", config=HTPConfig(threshold=0.35))

    @r.node
    @terminal
    @tag("x")
    def node_a(data): return data

    for _ in range(10):
        r.run({"label": "x"})
        r.collect_signal()   # history 적재 (실제 Thalamus 호출 패턴 재현)

    sig = r.collect_signal()
    # 안정 발화 (variance → 0) → precision 이 중립 1.0 을 초과
    assert sig.precision > 1.0, \
        f"안정 발화에도 precision 이 낮음: {sig.precision}"
