"""
Stage 1 Vector Routing — foundation tests (M1 + M2 + M3 + M5).

Design Ref: docs/02-design/features/htp-thalamus-car_sub-2_design v1.md §5.2
Plan SC: FR-06, FR-08, FR-09 + Review #2/#3 (sub-2)

stage-1-foundation 범위 (118 → 125):
- M1 RegionSignature  : init / update EMA / similarity cold start  (+3)
- M2 RouterStrategy   : Protocol 준수                                (+1)
- M3 TagRouter        : 기존 hub_strength 로직 동등성 / 빈 regions    (+2)
- M5 RegionSignal     : region_signature 필드                        (+1)

stage-1-vector / stage-2-hybrid 분은 후속 세션에서 추가.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from htp.thalamus.signature        import RegionSignature
from htp.thalamus.region_signal    import RegionSignal
from htp.thalamus.router           import RouterStrategy, RoutingScore, TagRouter


# ══════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════

def _make_signal(rid: str, hub: float, fire: float,
                 signature: "RegionSignature | None" = None) -> RegionSignal:
    return RegionSignal(
        region_id    = rid,
        hub_strength = hub,
        fire_rate    = fire,
        top_hubs     = [],
        overload     = False,
        output_vec   = torch.zeros(8),
        region_signature = signature,
    )


# ══════════════════════════════════════════════════════════
# M1 — RegionSignature  (+3)
# ══════════════════════════════════════════════════════════

def test_signature_init_zero_centroid():
    """기본 init 은 dim=64 영벡터, count=0 (냉시작 마커)."""
    sig = RegionSignature()
    assert sig.centroid.shape == (64,)
    assert float(np.linalg.norm(sig.centroid)) == 0.0
    assert sig.count == 0
    assert sig.dim == 64


def test_signature_update_ema():
    """lr = 1 / (count+1) 점진 평균 — 무한대 수렴 시 centroid → 평균(input)."""
    sig = RegionSignature(dim=4)
    v1 = np.array([1.0, 0.0, 0.0, 0.0])
    v2 = np.array([0.0, 1.0, 0.0, 0.0])
    v3 = np.array([0.0, 0.0, 1.0, 0.0])

    sig.update(v1)
    # count=0 → lr=1 → centroid = v1
    np.testing.assert_allclose(sig.centroid, v1, rtol=1e-10)
    assert sig.count == 1

    sig.update(v2)
    # count=1 → lr=1/2 → centroid = 0.5*v1 + 0.5*v2
    np.testing.assert_allclose(sig.centroid, 0.5 * (v1 + v2), rtol=1e-10)
    assert sig.count == 2

    sig.update(v3)
    # count=2 → lr=1/3 → centroid = (2/3)*prev + (1/3)*v3 = (v1+v2+v3)/3
    np.testing.assert_allclose(sig.centroid, (v1 + v2 + v3) / 3.0, rtol=1e-10)
    assert sig.count == 3


def test_signature_similarity_cold_start():
    """centroid=0 (count=0) 시 similarity 항상 0.0 — 냉시작 보호 (Review #3 토대)."""
    sig = RegionSignature(dim=4)
    query = np.array([1.0, 1.0, 1.0, 1.0])
    assert sig.similarity(query) == 0.0

    # 영벡터 query 도 0.0
    zero_query = np.zeros(4)
    sig.update(np.array([1.0, 0.0, 0.0, 0.0]))   # 이제 centroid != 0
    assert sig.similarity(zero_query) == 0.0


# ══════════════════════════════════════════════════════════
# M2 — RouterStrategy Protocol  (+1)
# ══════════════════════════════════════════════════════════

def test_router_strategy_protocol_compliance():
    """runtime_checkable Protocol — TagRouter 가 isinstance 통과."""
    tag = TagRouter()
    assert isinstance(tag, RouterStrategy)
    assert tag.mode == "tag"

    # RoutingScore dataclass 기본 동작
    rs = RoutingScore(region_id="r1", score=0.5)
    assert rs.region_id == "r1"
    assert rs.score == 0.5
    assert rs.breakdown == {}   # default_factory dict


# ══════════════════════════════════════════════════════════
# M3 — TagRouter  (+2)
# ══════════════════════════════════════════════════════════

def test_tag_router_regression_equivalence():
    """기존 CoreCells.gate() 의 raw = hub_strength × (1 + fire_rate)
    L1 정규화 결과와 1:1 동등.

    회귀 보호 핵심: 기존 12/12 routing 패턴이 TagRouter 단독으로 재현되어야 함.
    """
    signals = [
        _make_signal("brain",  hub=0.6, fire=0.5),  # raw = 0.6 × 1.5 = 0.90
        _make_signal("ai",     hub=0.3, fire=0.2),  # raw = 0.3 × 1.2 = 0.36
        _make_signal("infra",  hub=0.1, fire=0.0),  # raw = 0.1 × 1.0 = 0.10
    ]
    # 기존 L100-107 로직 재현 (회귀 동등성 비교용)
    expected_raw = {
        "brain": 0.60 * 1.5,
        "ai":    0.30 * 1.2,
        "infra": 0.10 * 1.0,
    }
    expected_total = sum(expected_raw.values())
    expected_norm  = {k: v / expected_total for k, v in expected_raw.items()}

    router = TagRouter()
    scores = router.score(signal_text=None, signal_vec=None, regions=signals)

    assert len(scores) == 3
    for rs in scores:
        assert rs.score == pytest.approx(expected_norm[rs.region_id], rel=1e-10)

    # 합 = 1.0 (L1 정규화 의미)
    assert sum(rs.score for rs in scores) == pytest.approx(1.0, rel=1e-10)


def test_tag_router_empty_regions_safe():
    """regions=[] 시 빈 list 반환 (기존 gate() L96-97 안전성 보존)."""
    router = TagRouter()
    assert router.score(None, None, []) == []
    # signal_text / signal_vec 가 있어도 결과는 동일
    assert router.score("뇌과학", np.zeros(64), []) == []


# ══════════════════════════════════════════════════════════
# M5 — RegionSignal.region_signature 필드  (+1)
# ══════════════════════════════════════════════════════════

def test_region_signal_signature_field_default_none():
    """region_signature 필드가 default None 으로 추가됨 — backward-compat 보호."""
    # 기존 코드처럼 signature 인자 없이 생성 가능해야 함
    sig = RegionSignal(
        region_id    = "r1",
        hub_strength = 0.5,
        fire_rate    = 0.3,
        top_hubs     = [],
        overload     = False,
        output_vec   = torch.zeros(8),
    )
    assert sig.region_signature is None

    # signature 제공 시 보존
    rsig = RegionSignature(dim=4)
    rsig.update(np.array([1.0, 0.0, 0.0, 0.0]))
    sig2 = _make_signal("r2", 0.5, 0.3, signature=rsig)
    assert sig2.region_signature is rsig
    assert sig2.region_signature.count == 1
