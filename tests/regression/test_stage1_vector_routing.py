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
from htp.thalamus.router           import (
    RouterStrategy, RoutingScore, TagRouter, VectorRouter,
)
from htp.thalamus.core_cells       import CoreCells


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


# ══════════════════════════════════════════════════════════
# M4 — VectorRouter  (+5: empty_vec, threshold, cold_start, clamp, β sweep)
# ══════════════════════════════════════════════════════════

def _make_signal_with_centroid(rid: str, centroid: np.ndarray) -> RegionSignal:
    """centroid 가 있는 RegionSignal — VectorRouter 테스트용 fixture."""
    sig = RegionSignature(dim=centroid.shape[0])
    sig.update(centroid)
    return RegionSignal(
        region_id    = rid,
        hub_strength = 0.5,
        fire_rate    = 0.3,
        top_hubs     = [],
        overload     = False,
        output_vec   = torch.zeros(8),
        region_signature = sig,
    )


def test_vector_router_empty_vec():
    """signal_vec=None 시 모든 score 0 + last_metrics.empty_vec=True 기록."""
    router = VectorRouter()
    regions = [
        _make_signal_with_centroid("r1", np.array([1.0, 0.0, 0.0, 0.0])),
        _make_signal_with_centroid("r2", np.array([0.0, 1.0, 0.0, 0.0])),
    ]
    scores = router.score(None, None, regions)
    assert all(rs.score == 0.0 for rs in scores)
    assert router.last_metrics["empty_vec"] is True
    assert router.last_metrics["active_count"] == 0


def test_vector_router_cold_start_uniform():
    """모든 Region count=0 시 균등 score 반환 — Review #3 회귀 보호.

    empty route 0건 보장 — vector mode 가 절대 죽지 않음.
    """
    router = VectorRouter()
    # signature=None 으로 cold start 시뮬레이션
    regions = [
        RegionSignal(region_id="r1", hub_strength=0.5, fire_rate=0.3,
                     top_hubs=[], overload=False, output_vec=torch.zeros(8)),
        RegionSignal(region_id="r2", hub_strength=0.5, fire_rate=0.3,
                     top_hubs=[], overload=False, output_vec=torch.zeros(8)),
        RegionSignal(region_id="r3", hub_strength=0.5, fire_rate=0.3,
                     top_hubs=[], overload=False, output_vec=torch.zeros(8)),
    ]
    query = np.array([1.0, 0.0, 0.0, 0.0])
    scores = router.score(None, query, regions)
    # 균등 분포 — 모두 1/3
    assert all(rs.score == pytest.approx(1.0 / 3.0) for rs in scores)
    # 모든 RoutingScore.breakdown 에 cold_start 마커
    assert all(rs.breakdown.get("cold_start") is True for rs in scores)
    # last_metrics 에도 기록
    assert router.last_metrics["cold_start"] is True
    assert router.last_metrics["active_count"] == 3


def test_vector_router_dynamic_threshold():
    """thr = μ + β×σ 정규화 — β=0.5 기본 동작 + similarity 차이 반영."""
    # 4-dim 공간에서 서로 다른 centroid
    regions = [
        _make_signal_with_centroid("close",  np.array([1.0, 0.0, 0.0, 0.0])),
        _make_signal_with_centroid("medium", np.array([0.7, 0.7, 0.0, 0.0])),
        _make_signal_with_centroid("far",    np.array([0.0, 0.0, 1.0, 0.0])),
    ]
    query = np.array([1.0, 0.0, 0.0, 0.0])
    router = VectorRouter(beta=0.5)
    scores = router.score(None, query, regions)

    by_rid = {rs.region_id: rs for rs in scores}
    # close 가 최고 점수 (similarity=1.0)
    assert by_rid["close"].score >= by_rid["medium"].score
    # far 는 thr 미만 → 0
    assert by_rid["far"].score == pytest.approx(0.0, abs=1e-6)
    # last_metrics 기록 검증
    m = router.last_metrics
    assert m["mu"] > 0 and m["sigma"] > 0
    assert m["thr"] == pytest.approx(m["mu"] + 0.5 * m["sigma"])


def test_vector_router_high_uniform_similarity():
    """모든 Region 의 similarity ≥ 0.95 시 thr 클램프 동작 — Review #1.

    부호 반전 없이 안전. thr ≤ 0.99 보장.
    """
    # 4-dim 공간 — 모든 centroid 가 query 와 매우 유사 (정규화 후 거의 동일)
    regions = [
        _make_signal_with_centroid("r1", np.array([1.0,    0.05, 0.0, 0.0])),
        _make_signal_with_centroid("r2", np.array([1.0,    0.10, 0.0, 0.0])),
        _make_signal_with_centroid("r3", np.array([0.99,   0.0,  0.0, 0.0])),
    ]
    query = np.array([1.0, 0.0, 0.0, 0.0])
    router = VectorRouter(beta=10.0)  # 큰 β → μ+β×σ 가 1.0 초과할 위험
    scores = router.score(None, query, regions)

    # thr 클램프로 0.99 이하 보장
    assert router.last_metrics["thr"] <= 0.99 + 1e-10
    # 모든 score 는 [0, 1] 범위 (부호 반전 없음)
    for rs in scores:
        assert 0.0 <= rs.score <= 1.0 + 1e-8


def test_vector_router_beta_sweep_metrics():
    """β∈{0.0, 0.5, 1.0} sweep 시 메트릭 단조성 — Review #6 핵심.

    예상 동작:
      β↑ → thr↑ → active_count↓ → entropy↓ → top1_score↑
    (precision/recall trade-off 의 정량 관찰)
    """
    # 4-dim 공간, 서로 다른 similarity 분포
    regions = [
        _make_signal_with_centroid("r1", np.array([1.0, 0.0, 0.0, 0.0])),
        _make_signal_with_centroid("r2", np.array([0.8, 0.6, 0.0, 0.0])),
        _make_signal_with_centroid("r3", np.array([0.5, 0.5, 0.7, 0.0])),
        _make_signal_with_centroid("r4", np.array([0.0, 0.0, 0.0, 1.0])),
    ]
    query = np.array([1.0, 0.0, 0.0, 0.0])

    metrics_by_beta: dict[float, dict] = {}
    for beta in (0.0, 0.5, 1.0):
        router = VectorRouter(beta=beta)
        router.score(None, query, regions)
        m = dict(router.last_metrics)
        metrics_by_beta[beta] = m

    # last_metrics 가 모든 필수 키 노출 (회귀 보호)
    required = {"beta", "mu", "sigma", "thr", "active_count",
                "entropy", "top1_score"}
    for beta, m in metrics_by_beta.items():
        missing = required - set(m.keys())
        assert not missing, f"β={beta} 누락 메트릭: {missing}"

    # 단조성: β↑ → thr↑ (μ+β×σ 단조 증가)
    assert (metrics_by_beta[0.0]["thr"]
            <= metrics_by_beta[0.5]["thr"]
            <= metrics_by_beta[1.0]["thr"])

    # 단조성: β↑ → active_count 감소 또는 동일
    assert (metrics_by_beta[0.0]["active_count"]
            >= metrics_by_beta[0.5]["active_count"]
            >= metrics_by_beta[1.0]["active_count"])

    # 단조성: β↑ → entropy 감소 또는 동일 (집중도↑)
    assert (metrics_by_beta[0.0]["entropy"]
            >= metrics_by_beta[1.0]["entropy"] - 1e-10)


# ══════════════════════════════════════════════════════════
# M6 — CoreCells router DI  (+3)
# ══════════════════════════════════════════════════════════

def test_core_cells_router_di_default_tag():
    """CoreCells() 기본 router 가 TagRouter — 회귀 보호 핵심."""
    cc = CoreCells()
    assert isinstance(cc.router, TagRouter)
    assert cc.router.mode == "tag"


def test_core_cells_router_swap_at_runtime():
    """런타임 router 교체 후 다음 gate() 가 새 router 사용 — DI 핵심 가치.

    Review #4: M6 의 자기방어적 가치 — VectorRouter 주입 시 다음 gate 호출이
    실제로 vector mode 로 동작함을 보장.
    """
    cc = CoreCells()
    signals = [
        _make_signal("brain", hub=0.6, fire=0.5),
        _make_signal("ai",    hub=0.3, fire=0.2),
    ]

    # 기본 (TagRouter) 결과
    mask_tag = cc.gate(signals)

    # router 교체
    cc.router = VectorRouter(beta=0.0)
    assert isinstance(cc.router, VectorRouter)

    # 같은 signals + 새 router 로 gate — 다른 동작 (cold start 균등)
    mask_vec = cc.gate(signals, signal_vec=np.array([1.0, 0.0, 0.0, 0.0]
                                                    + [0.0] * 60))
    # 두 결과 다름 — 단순히 router 가 바뀌어 새 logic 적용됨을 확인
    # cold start 균등 score 진입 (signature=None) → VectorRouter.last_metrics 기록
    assert cc.router.last_metrics["cold_start"] is True


def test_core_cells_vector_mode_with_signature():
    """signature 가 있는 Region 들에 대해 vector mode 가 정상 라우팅."""
    cc = CoreCells(router=VectorRouter(beta=0.0))
    # 4-dim 공간 fixture
    regions = [
        _make_signal_with_centroid("brain", np.array([1.0, 0.0, 0.0, 0.0])),
        _make_signal_with_centroid("ai",    np.array([0.9, 0.1, 0.0, 0.0])),
        _make_signal_with_centroid("infra", np.array([0.0, 0.0, 1.0, 0.0])),
    ]
    query = np.array([1.0, 0.0, 0.0, 0.0])

    mask = cc.gate(regions, signal_vec=query)
    # 모든 Region 에 대해 gating score 산출 (empty route 0건)
    assert set(mask.scores.keys()) == {"brain", "ai", "infra"}
    # brain (similarity=1.0) 의 gate 가 infra (similarity=0.0) 보다 크거나 같음
    # (sigmoid + theta_bias 영향으로 동률 가능성 있음, 단조 ≥ 검증)
    assert mask.scores["brain"] >= mask.scores["infra"]
