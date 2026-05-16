"""
Stage 2 Hybrid Routing — HybridRouter 검증 (M7).

Design Ref: docs/02-design/features/htp-thalamus-car_sub-2_design v1.md §2.5
Plan SC: FR-10 (α × vector + (1-α) × tag), FR-11 (α 변화 연속성)

stage-2-hybrid 범위 (130 → 133, 신규 +3):
- test_hybrid_alpha_continuity         — α∈{0.1, 0.5, 0.9} 연속성
- test_hybrid_breakdown_records        — RoutingScore.breakdown 에 tag/vector/alpha
- test_hybrid_extremes_match_pure      — α=0/α=1 시 순수 Tag/Vector 일치
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from htp.thalamus.signature     import RegionSignature
from htp.thalamus.region_signal import RegionSignal
from htp.thalamus.router        import (
    TagRouter, VectorRouter, HybridRouter,
)


def _make_region_with_centroid(
    rid: str, hub: float, fire: float, centroid: np.ndarray,
) -> RegionSignal:
    """centroid + hub_strength + fire_rate 가 모두 있는 fixture."""
    sig = RegionSignature(dim=centroid.shape[0])
    sig.update(centroid)
    return RegionSignal(
        region_id    = rid,
        hub_strength = hub,
        fire_rate    = fire,
        top_hubs     = [],
        overload     = False,
        output_vec   = torch.zeros(8),
        region_signature = sig,
    )


# 공통 fixture: brain/ai/infra 3 Region — Tag 와 Vector 의 결과가 다르도록 구성
def _three_regions() -> list[RegionSignal]:
    """- brain: 높은 hub (TagRouter 에서 우위) + query 와 cosine 0.7
       - ai:    중간 hub + query 와 cosine 1.0   (VectorRouter 에서 우위)
       - infra: 낮은 hub + query 와 cosine 0.0
    """
    return [
        _make_region_with_centroid(
            "brain", hub=0.8, fire=0.5,
            centroid=np.array([0.7, 0.7, 0.0, 0.0])),
        _make_region_with_centroid(
            "ai", hub=0.3, fire=0.2,
            centroid=np.array([1.0, 0.0, 0.0, 0.0])),
        _make_region_with_centroid(
            "infra", hub=0.1, fire=0.0,
            centroid=np.array([0.0, 0.0, 1.0, 0.0])),
    ]


# ══════════════════════════════════════════════════════════
# Test 1: α=0 / α=1 extremes — 순수 Tag/Vector 와 일치
# ══════════════════════════════════════════════════════════

def test_hybrid_extremes_match_pure():
    """α=0 → 순수 TagRouter 와 동일, α=1 → 순수 VectorRouter 와 동일.

    FR-10 의 수학적 경계조건 보장:
      mixed = α × v + (1-α) × t
      α=0 → mixed = t   (TagRouter 결과)
      α=1 → mixed = v   (VectorRouter 결과)
    """
    regions = _three_regions()
    query   = np.array([1.0, 0.0, 0.0, 0.0])

    tag = TagRouter()
    vec = VectorRouter(beta=0.0)

    pure_tag = {rs.region_id: rs.score
                for rs in tag.score(None, query, regions)}
    pure_vec = {rs.region_id: rs.score
                for rs in vec.score(None, query, regions)}

    # α=0 → TagRouter 동등
    h0 = HybridRouter(tag=tag, vec=vec, alpha=0.0)
    h0_scores = {rs.region_id: rs.score
                 for rs in h0.score(None, query, regions)}
    for rid in pure_tag:
        assert h0_scores[rid] == pytest.approx(pure_tag[rid], rel=1e-10)

    # α=1 → VectorRouter 동등
    h1 = HybridRouter(tag=tag, vec=vec, alpha=1.0)
    h1_scores = {rs.region_id: rs.score
                 for rs in h1.score(None, query, regions)}
    for rid in pure_vec:
        assert h1_scores[rid] == pytest.approx(pure_vec[rid], rel=1e-10)


# ══════════════════════════════════════════════════════════
# Test 2: α 연속성 — α 변화 시 score 분포가 부드럽게 이동
# ══════════════════════════════════════════════════════════

def test_hybrid_alpha_continuity():
    """α∈{0.1, 0.5, 0.9} 변화 시 결과 분포가 연속적 (코사인 유사도 > 0.5).

    Plan FR-11: "α 0.1→0.9 변화 시 선택 Region 집합 변화 연속적".
    이산적 jump 없이 score 가 점진 이동해야 함.
    """
    regions = _three_regions()
    query   = np.array([1.0, 0.0, 0.0, 0.0])

    def _scores_at(alpha: float) -> np.ndarray:
        h = HybridRouter(alpha=alpha)
        return np.array([rs.score
                         for rs in h.score(None, query, regions)])

    s01 = _scores_at(0.1)
    s05 = _scores_at(0.5)
    s09 = _scores_at(0.9)

    def _cos(a: np.ndarray, b: np.ndarray) -> float:
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na < 1e-8 or nb < 1e-8:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    # 인접 α 분포는 매우 유사 (연속성)
    assert _cos(s01, s05) > 0.5, f"α 0.1↔0.5 cosine={_cos(s01, s05):.3f}"
    assert _cos(s05, s09) > 0.5, f"α 0.5↔0.9 cosine={_cos(s05, s09):.3f}"

    # 양 끝 분포는 어느 정도 차이 있어야 함 (선형 보간이 의미 있음)
    # 같은 분포라면 hybrid 의 가치 없음 — 합리적 일관성 검증
    assert _cos(s01, s09) < 0.999, "α 0.1 와 0.9 가 사실상 동일 → hybrid 가치 없음"


# ══════════════════════════════════════════════════════════
# Test 3: breakdown 진단 정보 기록
# ══════════════════════════════════════════════════════════

def test_hybrid_breakdown_records():
    """RoutingScore.breakdown 에 tag / vector / alpha 모두 기록 — 진단 가능성.

    이 정보는 향후 β/α 튜닝 시 어느 Router 가 어느 정도 기여했는지 추적용.
    """
    regions = _three_regions()
    query   = np.array([1.0, 0.0, 0.0, 0.0])

    h = HybridRouter(alpha=0.5)
    scores = h.score(None, query, regions)

    required_keys = {"tag", "vector", "alpha"}
    for rs in scores:
        missing = required_keys - set(rs.breakdown.keys())
        assert not missing, (
            f"breakdown 누락 키: {missing} on region {rs.region_id}"
        )
        # alpha 는 일관된 값
        assert rs.breakdown["alpha"] == 0.5
        # mixed score = α × vec + (1-α) × tag 수학 일치
        recomputed = 0.5 * rs.breakdown["vector"] + 0.5 * rs.breakdown["tag"]
        assert rs.score == pytest.approx(recomputed, rel=1e-10)
