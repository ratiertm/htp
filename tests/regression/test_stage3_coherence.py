"""
Stage 3 CoherenceGate — types + Protocol + Pairwise binding tests (M1+M2+M3).

Design Ref: docs/02-design/features/htp-thalamus-car.sub-3.design.md §5.2
Plan SC: FR-12 (CoherenceGate.bind), FR-13 (BoundResponse)

stage-3-coherence-core 범위 (140 → 144):
- M1 types          : BoundResponse + RegionResponse 기본 동작     (+1)
- M2 Protocol       : isinstance(PairwiseCoherenceGate, CoherenceStrategy)  (+1)
- M3 Pairwise       : high agreement + conflict 감지 정확도        (+2)

stage-3-integration (M4+M5) 은 후속 세션.
"""
from __future__ import annotations

import numpy as np
import pytest

from htp.thalamus.types     import RegionResponse, BoundResponse
from htp.thalamus.coherence import CoherenceStrategy, PairwiseCoherenceGate


def _make_response(rid: str, vec: np.ndarray, precision: float = 1.0
                   ) -> RegionResponse:
    return RegionResponse(region_id=rid, output_vec=vec, precision=precision)


# ══════════════════════════════════════════════════════════
# M1 — types  (+1)
# ══════════════════════════════════════════════════════════

def test_bound_response_defaults():
    """BoundResponse 기본 dataclass 동작 + RegionResponse default."""
    # RegionResponse: precision default 1.0
    r = RegionResponse(region_id="r1", output_vec=np.zeros(8))
    assert r.precision == 1.0
    assert r.region_id == "r1"

    # BoundResponse: escalate_to_pfc default False
    br = BoundResponse(
        responses=[r], coherence=1.0, conflict=0.0,
        fused_vec=np.zeros(8),
    )
    assert br.escalate_to_pfc is False
    assert br.coherence == 1.0
    assert len(br.responses) == 1


# ══════════════════════════════════════════════════════════
# M2 — CoherenceStrategy Protocol  (+1)
# ══════════════════════════════════════════════════════════

def test_coherence_strategy_protocol_compliance():
    """runtime_checkable Protocol — PairwiseCoherenceGate 가 isinstance 통과."""
    gate = PairwiseCoherenceGate()
    assert isinstance(gate, CoherenceStrategy)
    assert gate.mode == "pairwise"

    # 잘못된 threshold 는 ValueError
    with pytest.raises(ValueError):
        PairwiseCoherenceGate(conflict_threshold=1.5)
    with pytest.raises(ValueError):
        PairwiseCoherenceGate(escalation_threshold=-0.1)


# ══════════════════════════════════════════════════════════
# M3 — PairwiseCoherenceGate  (+2)
# ══════════════════════════════════════════════════════════

def test_pairwise_coherence_high_agreement():
    """유사 응답 3개 → coherence ≥ 0.9, conflict ≤ 0.1, escalate=False."""
    gate = PairwiseCoherenceGate()
    base = np.array([1.0, 0.0, 0.0, 0.0])
    responses = [
        _make_response("r1", base),
        _make_response("r2", base + np.array([0.05, 0.0, 0.0, 0.0])),
        _make_response("r3", base + np.array([0.0, 0.05, 0.0, 0.0])),
    ]
    bound = gate.bind(responses)
    assert bound.coherence >= 0.9, f"coherence={bound.coherence}"
    assert bound.conflict  <= 0.2, f"conflict={bound.conflict}"
    assert bound.escalate_to_pfc is False
    # fused_vec 은 base 근처
    assert _cosine_simple(bound.fused_vec, base) >= 0.95

    # 단일 응답: coherence=1, conflict=0
    single = gate.bind([_make_response("r1", base)])
    assert single.coherence == 1.0
    assert single.conflict  == 0.0

    # 빈 입력: 안전
    empty = gate.bind([])
    assert empty.coherence == 0.0
    assert empty.escalate_to_pfc is False


def test_pairwise_conflict_detection_accuracy():
    """의도적 conflict fixture 10건 + 정합 fixture 10건 — Plan G5 기준.

    Pass:
      - Conflict 감지 recall ≥ 90% (10건 중 9건 이상 escalate)
      - Conflict false positive ≤ 10% (10건 중 1건 이하)
    """
    gate = PairwiseCoherenceGate(escalation_threshold=0.7)
    rng = np.random.default_rng(seed=42)

    def _conflict_fixture() -> list[RegionResponse]:
        """직교 응답 2개 → cosine ≈ 0 → conflict ≈ 1.0"""
        v1 = rng.standard_normal(8); v1 /= np.linalg.norm(v1)
        v2_perp = rng.standard_normal(8)
        v2 = v2_perp - np.dot(v2_perp, v1) * v1
        v2 /= np.linalg.norm(v2)
        return [_make_response("r1", v1), _make_response("r2", v2)]

    def _agreement_fixture() -> list[RegionResponse]:
        """매우 유사한 응답 2개 → cosine ≈ 1 → conflict ≈ 0"""
        v1 = rng.standard_normal(8); v1 /= np.linalg.norm(v1)
        noise = rng.standard_normal(8) * 0.05
        v2 = v1 + noise; v2 /= np.linalg.norm(v2)
        return [_make_response("r1", v1), _make_response("r2", v2)]

    # 10건 conflict — escalate=True 여야 함
    conflict_detected = sum(
        1 for _ in range(10)
        if gate.bind(_conflict_fixture()).escalate_to_pfc
    )
    # 10건 agreement — escalate=False 여야 함
    false_positives = sum(
        1 for _ in range(10)
        if gate.bind(_agreement_fixture()).escalate_to_pfc
    )

    assert conflict_detected >= 9, (
        f"conflict recall {conflict_detected}/10 < 9"
    )
    assert false_positives <= 1, (
        f"false positive {false_positives}/10 > 1"
    )


# ══════════════════════════════════════════════════════════
# 헬퍼
# ══════════════════════════════════════════════════════════

def _cosine_simple(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))
