"""
TagRouter — 기본 라우팅 (회귀 보호 핵심).

Design Ref: docs/02-design/features/htp-thalamus-car_sub-2_design v1.md §2.3
Plan SC: FR-09 (routing_mode="tag" 기본값 유지, 회귀 57+8 = 65 보호)

기존 `CoreCells.gate()` 의 score 계산 로직 (raw = hub_strength × (1 + fire_rate))
을 RouterStrategy 인터페이스로 1:1 이관. 회귀 0건 보장.

신호 인자 무시 (Review #2):
    signal_text / signal_vec 은 받지만 사용하지 않음.
    Router 책임 — Protocol 계약을 위해 양쪽 모두 받지만, tag mode 는
    regions 의 PageRank/fire_rate 기반 통계만 사용.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from .base import RoutingScore

if TYPE_CHECKING:
    import numpy as np
    from ..region_signal import RegionSignal


class TagRouter:
    """Hub strength × (1 + fire rate) 기반 routing.

    이 Router 가 기본값 (CoreCells(router=TagRouter()))로 회귀 12/12 보호.
    """

    @property
    def mode(self) -> str:
        return "tag"

    def score(
        self,
        signal_text: "str | None",
        signal_vec:  "np.ndarray | None",
        regions:     "list[RegionSignal]",
    ) -> list[RoutingScore]:
        """기존 CoreCells.gate() 의 raw score + L1 정규화 단계 그대로 이관.

        CoreCells 내부의 sigmoid / theta_bias / precision 은 router 외부.
        Router 책임은 *어떤 Region 이 얼마나 관련 있는가* 의 normalized score 만.
        """
        if not regions:
            return []

        # raw score = hub_strength × (1 + fire_rate)  — 기존 core_cells.gate() L100-103
        raw: list[tuple[str, float]] = [
            (r.region_id, float(r.hub_strength) * (1.0 + float(r.fire_rate)))
            for r in regions
        ]

        # L1 정규화 — 기존 core_cells.gate() L106-107
        total = sum(v for _, v in raw) or 1.0

        return [
            RoutingScore(
                region_id=rid,
                score=v / total,
                breakdown={"tag": v / total,
                           "hub_strength": float(r.hub_strength),
                           "fire_rate":    float(r.fire_rate)},
            )
            for (rid, v), r in zip(raw, regions)
        ]


__all__ = ["TagRouter"]
