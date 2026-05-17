"""
htp.thalamus.types — 공통 dataclass (sub-3 신설).

Design Ref: docs/02-design/features/htp-thalamus-car.sub-3.design.md §2.2
Plan SC: FR-13 (BoundResponse dataclass)

Stage 3 (CoherenceGate) 의 입출력 단위.
DAG: numpy 만 의존. htp.thalamus 형제 모듈에서 import 가능, 역방향 금지.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RegionResponse:
    """Region 의 응답 단위 — CoherenceGate input.

    sub-3 OUT-OF-SCOPE 인 Region 응답 수집 메커니즘은 BrainRuntime 책임.
    이 dataclass 는 CoherenceGate.bind() 가 받는 표준 인터페이스.
    """
    region_id:  str
    output_vec: np.ndarray            # Region 의 의미 출력 (64-dim 권장)
    precision:  float = 1.0           # Friston precision (Stage 3 B1)


@dataclass
class BoundResponse:
    """Plan FR-13 — 다중 응답 binding 결과.

    coherence: 응답 간 평균 일치도 [0, 1]. 1=완전 일치, 0=완전 불일치.
    conflict:  최대 pairwise 불일치 [0, 1]. swr_priority 증폭에 사용.
    fused_vec: precision-weighted 평균 출력. PFC 입력.
    escalate_to_pfc: conflict > escalation_threshold 시 True. PFC top-down 트리거.
    """
    responses:       list             # list[RegionResponse]
    coherence:       float
    conflict:        float
    fused_vec:       np.ndarray
    escalate_to_pfc: bool = False


__all__ = ["RegionResponse", "BoundResponse"]
