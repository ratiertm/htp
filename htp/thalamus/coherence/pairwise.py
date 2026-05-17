"""
PairwiseCoherenceGate — O(N²) pairwise cosine similarity 기반 binding (sub-3 M3).

Design Ref: docs/02-design/features/htp-thalamus-car.sub-3.design.md §2.3
Plan SC: FR-12 (pairwise coherence + conflict + precision-weighted fusion)

수학:
  coherence = mean(cosine(r_i, r_j) for i<j)
  conflict  = max(1 - cosine(r_i, r_j) for i<j)
  fused_vec = Σ (p_i × r_i) / Σ p_i      (precision-weighted average)
  escalate_to_pfc = (conflict > escalation_threshold)

Plan §R2: N≥16 시 O(N²) 성능 저하 → LSH 전환 별도 사이클.
"""
from __future__ import annotations

import numpy as np

from ..types import RegionResponse, BoundResponse


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity. 영벡터 안전 처리."""
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class PairwiseCoherenceGate:
    """Pairwise cosine similarity 기반 binding (O(N²)).

    Plan FR-12. 향후 LSH 전환 시 동일 Protocol 의 새 구현체로 교체.
    """

    def __init__(
        self,
        conflict_threshold:   float = 0.3,
        escalation_threshold: float = 0.7,
    ):
        """
        conflict_threshold:    1 - cosine > threshold 면 pair conflict 로 카운트
                                (현 구현은 max conflict 만 반환하므로 진단용)
        escalation_threshold:  최대 conflict 가 threshold 초과 시 PFC 에스컬레이션
        """
        if not (0.0 <= conflict_threshold <= 1.0):
            raise ValueError(
                f"conflict_threshold ∈ [0,1], got {conflict_threshold}"
            )
        if not (0.0 <= escalation_threshold <= 1.0):
            raise ValueError(
                f"escalation_threshold ∈ [0,1], got {escalation_threshold}"
            )
        self.conflict_threshold   = conflict_threshold
        self.escalation_threshold = escalation_threshold

    @property
    def mode(self) -> str:
        return "pairwise"

    def bind(self, responses: list[RegionResponse]) -> BoundResponse:
        # 단일 응답 / 빈 입력 처리
        if len(responses) == 0:
            return BoundResponse(
                responses=[],
                coherence=0.0,
                conflict=0.0,
                fused_vec=np.zeros(64, dtype=np.float64),
                escalate_to_pfc=False,
            )
        if len(responses) == 1:
            r = responses[0]
            return BoundResponse(
                responses=[r],
                coherence=1.0,
                conflict=0.0,
                fused_vec=r.output_vec.copy(),
                escalate_to_pfc=False,
            )

        # 1) Pairwise cosine — O(N²)
        N = len(responses)
        sims: list[float] = []
        for i in range(N):
            for j in range(i + 1, N):
                sims.append(_cosine(responses[i].output_vec,
                                    responses[j].output_vec))

        coherence = float(np.mean(sims))
        # conflict = max disagreement = max(1 - s)
        conflict = float(max(0.0, max(1.0 - s for s in sims)))

        # 2) Precision-weighted 평균 fusion
        weights = np.array([r.precision for r in responses], dtype=np.float64)
        wsum    = float(weights.sum())
        if wsum < 1e-8:
            # 모든 precision=0 → 균등 평균으로 fallback (zero division 방지)
            weights = np.ones_like(weights)
            wsum    = float(weights.sum())

        stacked = np.stack([r.output_vec for r in responses])
        fused   = (weights[:, None] * stacked).sum(axis=0) / wsum

        return BoundResponse(
            responses       = list(responses),
            coherence       = coherence,
            conflict        = conflict,
            fused_vec       = fused,
            escalate_to_pfc = (conflict > self.escalation_threshold),
        )


__all__ = ["PairwiseCoherenceGate"]
