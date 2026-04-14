"""
Stage 3-B4 — TopDownBias Softmax prior 회귀 테스트.
"""
from __future__ import annotations

import pytest

from htp import HTPConfig, RegionRuntime, TopDownBias, tag, terminal


@pytest.fixture
def two_regions():
    lang = RegionRuntime("language", "text_processing",
                         config=HTPConfig(threshold=0.35))
    mem = RegionRuntime("memory", "cache_storage",
                        config=HTPConfig(threshold=0.35))

    @lang.node
    @terminal
    @tag("parse")
    def parse(data): return data

    @mem.node
    @terminal
    @tag("cache")
    def store(data): return data

    return {"language": lang, "memory": mem}


@pytest.mark.regression
def test_bias_is_probability_distribution(two_regions):
    """biases.values() 합이 1.0 이어야 (softmax 분포)."""
    td = TopDownBias().compute(
        goals=["cache", "store"],
        regions=two_regions,
        step=1,
    )
    total = sum(td.biases.values())
    assert abs(total - 1.0) < 1e-6, f"분포 합 {total} != 1.0"


@pytest.mark.regression
def test_zero_overlap_region_still_gets_nonzero_prob(two_regions):
    """overlap=0 Region 도 softmax 에서는 0 이 아닌 최소 확률을 받아야 (Jaccard 대비 개선)."""
    td = TopDownBias().compute(
        goals=["cache"],
        regions=two_regions,
        step=1,
    )
    # language Region 은 cache 와 overlap 0 이지만 exp(0)=1 로 여전히 확률 > 0
    assert td.biases["language"] > 0.0, \
        f"overlap=0 Region 에 확률 0 할당 (Jaccard 회귀): {td.biases}"


@pytest.mark.regression
def test_higher_overlap_yields_higher_prob(two_regions):
    """overlap 큰 Region 이 더 높은 확률을 받아야."""
    td = TopDownBias().compute(
        goals=["cache", "store"],  # memory Region 과 완벽 매칭
        regions=two_regions,
        step=1,
    )
    assert td.biases["memory"] > td.biases["language"], \
        f"overlap 우위 Region 이 더 낮은 확률: {td.biases}"


@pytest.mark.regression
def test_temperature_affects_sharpness(two_regions):
    """낮은 temperature 일수록 더 sharp (argmax 에 확률 집중)."""
    sharp = TopDownBias(temperature=0.2).compute(
        goals=["cache"], regions=two_regions, step=1,
    )
    soft = TopDownBias(temperature=5.0).compute(
        goals=["cache"], regions=two_regions, step=1,
    )
    assert sharp.biases["memory"] > soft.biases["memory"], \
        f"sharp temperature 가 더 집중되지 않음: sharp={sharp.biases}, soft={soft.biases}"


@pytest.mark.regression
def test_empty_goals_returns_zero_strength(two_regions):
    """goals=[] 이면 strength=0 으로 top-down 무효."""
    td = TopDownBias().compute(goals=[], regions=two_regions, step=1)
    assert td.strength == 0.0
    assert td.biases == {}
