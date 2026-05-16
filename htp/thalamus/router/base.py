"""
RouterStrategy Protocol + RoutingScore dataclass.

Design Ref: docs/02-design/features/htp-thalamus-car_sub-2_design v1.md §2.1
Plan SC: FR-08 (RouterStrategy 다형성 토대)

OCP 원칙 — 새 라우팅 정책 추가 시 기존 코드 무변경.
sub-2: Tag/Vector/Hybrid
sub-5 (Stage 6): EmbeddingBridge 가 동일 Protocol 의 추가 구현체로 끼움.

DAG: 외부 의존 없음 (numpy 는 시그니처 type-hint 만, 런타임 미사용).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing      import Protocol, runtime_checkable, TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from ..region_signal import RegionSignal


@dataclass
class RoutingScore:
    """라우팅 결과 — Region 별 점수 + 진단 정보."""

    region_id: str
    score:     float            # [0.0, 1.0] 권장 (Router 마다 의미 다름)
    breakdown: dict = field(default_factory=dict)
    """진단용 — {"tag": 0.6, "vector": 0.4, "alpha": 0.5, ...}"""


@runtime_checkable
class RouterStrategy(Protocol):
    """Thalamus 라우팅 정책 인터페이스.

    Stage 1: TagRouter (기본 회귀 보호), VectorRouter
    Stage 2: HybridRouter
    Stage 6 (sub-5): EmbeddingBridgeRouter
    """

    @property
    def mode(self) -> str:
        """식별자 — "tag" | "vector" | "hybrid" | "embedding"."""
        ...

    def score(
        self,
        signal_text: "str | None",
        signal_vec:  "np.ndarray | None",
        regions:     "list[RegionSignal]",
    ) -> list[RoutingScore]:
        """signal → Region 별 score 리스트.

        - tag mode:    regions 의 hub_strength/fire_rate 기반 (signal_* 무시)
        - vector mode: signal_vec + regions[i].region_signature.similarity 기반
        - hybrid mode: 둘 다 가중 결합

        Review #2: 모든 호출자 (특히 HybridRouter) 는 양쪽 인자를 모두 전달.
                   인자 무시는 Router 의 책임.
        """
        ...


__all__ = ["RouterStrategy", "RoutingScore"]
