"""
HybridRouter — α × VectorRouter + (1-α) × TagRouter (Stage 2 M7).

Design Ref: docs/02-design/features/htp-thalamus-car_sub-2_design v1.md §2.5
Plan SC: FR-10 (혼합 score), FR-11 (α 변화 시 연속성)

핵심 동작:
  mixed[i] = α × vec.score(i) + (1-α) × tag.score(i)

Review #2: signal_text/signal_vec 양쪽 모두 sub-Router 에 전달.
            각 Router 내부 무시 책임은 Router 자체.
Review #7: 현재 sub-2 는 동기 순차 호출. 향후 async pipeline 검토는
            sub-5 EmbeddingBridge 진입 시 별도 사이클 (htp-thalamus-async-pipeline).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from .base          import RoutingScore
from .tag_router    import TagRouter
from .vector_router import VectorRouter

if TYPE_CHECKING:
    import numpy as np
    from ..region_signal import RegionSignal


class HybridRouter:
    """Tag + Vector 가중 결합 라우터.

    Plan FR-10, FR-11.
    """

    def __init__(
        self,
        tag:   "TagRouter | None"    = None,
        vec:   "VectorRouter | None" = None,
        alpha: float = 0.5,
    ):
        """
        tag   : TagRouter 인스턴스. None 이면 기본 TagRouter()
        vec   : VectorRouter 인스턴스. None 이면 기본 VectorRouter()
        alpha : [0.0, 1.0] — α × vector + (1-α) × tag
                  α=0 → 순수 Tag, α=1 → 순수 Vector
        """
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self.tag   = tag or TagRouter()
        self.vec   = vec or VectorRouter()
        self.alpha = alpha

    @property
    def mode(self) -> str:
        return "hybrid"

    def score(
        self,
        signal_text: "str | None",
        signal_vec:  "np.ndarray | None",
        regions:     "list[RegionSignal]",
    ) -> list[RoutingScore]:
        if not regions:
            return []

        # Review #2: 양쪽 인자를 모두 전달 — Protocol 계약 준수
        # Review #7 (sub-2 sync): VectorRouter 가 무거워지면 (sub-5 EmbeddingBridge
        # sLLM forward 등) 여기서 동기 병목 발생 가능. async 도입은 별도 사이클.
        tag_scores = {
            s.region_id: s.score
            for s in self.tag.score(signal_text, signal_vec, regions)
        }
        vec_scores = {
            s.region_id: s.score
            for s in self.vec.score(signal_text, signal_vec, regions)
        }

        out: list[RoutingScore] = []
        for r in regions:
            t = tag_scores.get(r.region_id, 0.0)
            v = vec_scores.get(r.region_id, 0.0)
            mixed = self.alpha * v + (1.0 - self.alpha) * t
            out.append(RoutingScore(
                region_id = r.region_id,
                score     = mixed,
                breakdown = {
                    "tag":   t,
                    "vector": v,
                    "alpha": self.alpha,
                },
            ))
        return out


__all__ = ["HybridRouter"]
