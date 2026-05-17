"""
htp.thalamus.coherence — Temporal binding 패키지 (sub-3 신설).

Design Ref: docs/02-design/features/htp-thalamus-car.sub-3.design.md §2.1-2.3
Plan SC: FR-12 (CoherenceGate.bind)

sub-2 router/ 패턴과 일관:
  - CoherenceStrategy Protocol (다형성)
  - PairwiseCoherenceGate (O(N²)) — sub-3 기본
  - 향후 LSHCoherenceGate (N≥16 trigger) — 별도 사이클

DAG 단방향:
  coherence/* → htp/thalamus/types.py + numpy
  금지: coherence/* → htp.runtime / htp.memory / htp.knowledge / htp.thalamus.router
"""
from __future__ import annotations

from .base     import CoherenceStrategy
from .pairwise import PairwiseCoherenceGate

__all__ = [
    "CoherenceStrategy",
    "PairwiseCoherenceGate",
]
