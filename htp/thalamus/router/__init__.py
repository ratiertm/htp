"""
htp.thalamus.router — Thalamus 라우팅 정책 (Strategy Pattern).

Design Ref: docs/02-design/features/htp-thalamus-car_sub-2_design v1.md §2.1
Plan SC: FR-08~FR-11 (Stage 1 + 2 Vector + Hybrid routing)

DAG 단방향:
    router/* → signature.py / region_signal.py / numpy (외부)
    금지: router/* → htp.runtime / htp.memory / htp.knowledge

Stage 1 (sub-2): TagRouter, VectorRouter
Stage 2 (sub-2): HybridRouter
Stage 6 (sub-5): EmbeddingBridgeRouter (실험 브랜치, 동일 Protocol)
"""
from __future__ import annotations

from .base          import RouterStrategy, RoutingScore
from .tag_router    import TagRouter
from .vector_router import VectorRouter
from .hybrid_router import HybridRouter

__all__ = [
    "RouterStrategy",
    "RoutingScore",
    "TagRouter",
    "VectorRouter",
    "HybridRouter",
]
