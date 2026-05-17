"""
CoherenceStrategy Protocol — 다중 Region 응답 binding 인터페이스.

Design Ref: docs/02-design/features/htp-thalamus-car.sub-3.design.md §2.1
Plan SC: FR-12

sub-2 RouterStrategy 와 동일 패턴 (runtime_checkable Protocol).

Stage 3: PairwiseCoherenceGate (O(N²))
Future:  LSHCoherenceGate — N≥16 시 별도 사이클 (htp-thalamus-coherence-lsh)
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable, TYPE_CHECKING

if TYPE_CHECKING:
    from ..types import RegionResponse, BoundResponse


@runtime_checkable
class CoherenceStrategy(Protocol):
    """다중 Region 응답을 단일 BoundResponse 로 binding.

    Plan §R2: N≥16 시 LSH 전환 임계. 그 시점에 동일 Protocol 의 새 구현체로 교체.
    """

    @property
    def mode(self) -> str:
        """식별자 — "pairwise" | "lsh" | ..."""
        ...

    def bind(self, responses: "list[RegionResponse]") -> "BoundResponse":
        """다중 응답 → 단일 BoundResponse.

        - coherence: 응답 간 평균 일치도 [0, 1]
        - conflict:  최대 pairwise 불일치 [0, 1]
        - fused_vec: precision-weighted 평균
        - escalate_to_pfc: conflict > threshold 시 True
        """
        ...


__all__ = ["CoherenceStrategy"]
