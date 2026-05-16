"""
RegionSignature — Region 의 의미 중심점 (centroid + EMA + cosine similarity).

Design Ref: docs/02-design/features/htp-thalamus-car_sub-2_design v1.md §2.2
Plan SC: FR-06 (RegionSignature centroid + count + update + similarity)

Stage 1 (sub-2) 신설. content-addressable routing 의 토대.

DAG: htp/thalamus/ — htp/runtime/* 미참조. numpy 만 의존.
이 모듈은 router/* 가 참조 (router → signature 방향만, 역방향 금지).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RegionSignature:
    """Region 의 의미 중심점 — Online EMA centroid + cosine similarity.

    수학:
      cold start (count=0):
        centroid ← input_vec
      steady state (count >> 0):
        centroid ← centroid + (input_vec - centroid) / (count+1)
        lr = 1 / (count+1)  → 점진 평균 수렴

    냉시작 보호:
      centroid 가 영벡터 (count=0) 일 때 similarity() 는 0.0 반환.
      VectorRouter 는 이를 감지해 균등 score fallback (Review #3).
    """

    centroid: np.ndarray = field(default=None)  # type: ignore[assignment]
    count: int = 0
    dim: int = 64

    def __post_init__(self) -> None:
        if self.centroid is None:
            # 0-vector 초기화 (냉시작 마커)
            self.centroid = np.zeros(self.dim, dtype=np.float64)
        elif self.centroid.shape != (self.dim,):
            raise ValueError(
                f"RegionSignature centroid shape {self.centroid.shape} "
                f"!= expected ({self.dim},)"
            )

    def update(self, input_vec: np.ndarray) -> None:
        """EMA centroid 업데이트 — lr = 1 / (count + 1).

        Plan FR-06.
        """
        if input_vec.shape != (self.dim,):
            raise ValueError(
                f"input_vec shape {input_vec.shape} != ({self.dim},)"
            )
        lr = 1.0 / (self.count + 1)
        self.centroid = (1 - lr) * self.centroid + lr * input_vec
        self.count += 1

    def similarity(self, query_vec: np.ndarray) -> float:
        """cosine similarity. centroid 가 영벡터면 0.0 반환 (냉시작 보호)."""
        if query_vec.shape != (self.dim,):
            raise ValueError(
                f"query_vec shape {query_vec.shape} != ({self.dim},)"
            )
        nc = float(np.linalg.norm(self.centroid))
        nq = float(np.linalg.norm(query_vec))
        if nc < 1e-8 or nq < 1e-8:
            return 0.0
        return float(np.dot(self.centroid, query_vec) / (nc * nq))


__all__ = ["RegionSignature"]
