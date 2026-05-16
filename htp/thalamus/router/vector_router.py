"""
VectorRouter — Content-Addressable Routing (Stage 1 M4).

Design Ref: docs/02-design/features/htp-thalamus-car_sub-2_design v1.md §2.4
Plan SC: FR-08 (dynamic threshold μ+β×σ) + Review #1, #3, #6

핵심 동작:
  1. signal_vec → 모든 Region 의 RegionSignature.similarity 계산
  2. Dynamic threshold thr = min(μ + β·σ, 0.99)   [Review #1: 부호반전 방지]
  3. normalized_i = max(0, (s_i - thr) / (1 - thr + ε))
  4. cold start: 모든 Region count=0 → 균등 score      [Review #3]
  5. β sweep 메트릭 last_metrics 노출                    [Review #6]

last_metrics 사용:
  β=0.0 → thr=μ. 평균 이상 모두 활성 → entropy 높음, active_count 큼 (recall 우선)
  β=0.5 → thr=μ+0.5σ. 균형
  β=1.0 → thr=μ+σ. 1σ 초과만 → entropy 낮음, top1 confident (precision 우선)
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .base import RoutingScore

if TYPE_CHECKING:
    from ..region_signal import RegionSignal


def _shannon_entropy(scores: "list[float]") -> float:
    """Shannon entropy (nats). 분포 균일 시 ln(N), 한쪽 집중 시 0.

    Review #6: 정규화된 score 분포의 다양성 지표.
    β sweep 시 "라우팅 다양성 vs 정확도" trade-off 정량 관찰.
    """
    arr = np.array([s for s in scores if s > 1e-8], dtype=np.float64)
    if arr.size == 0:
        return 0.0
    total = arr.sum()
    if total <= 0:
        return 0.0
    p = arr / total
    return float(-(p * np.log(p + 1e-12)).sum())


class VectorRouter:
    """RegionSignature.similarity 기반 content-addressable routing.

    Review #6: last_metrics 딕셔너리에 β sweep 진단 정보 보존.
       keys: cold_start, empty_vec, beta, mu, sigma, thr, active_count,
             entropy, top1_score
    """

    def __init__(self, beta: float = 0.5):
        self.beta: float = beta
        self.last_metrics: dict = {}   # Review #6: 마지막 호출 진단

    @property
    def mode(self) -> str:
        return "vector"

    def score(
        self,
        signal_text: "str | None",
        signal_vec:  "np.ndarray | None",
        regions:     "list[RegionSignal]",
    ) -> list[RoutingScore]:
        if not regions:
            self.last_metrics = {
                "cold_start": False, "empty_vec": False,
                "active_count": 0, "entropy": 0.0,
                "top1_score": 0.0, "thr": None,
            }
            return []

        # 1) signal_vec 부재 — 빈 라우팅 + 메트릭 기록
        if signal_vec is None:
            self.last_metrics = {
                "cold_start": False, "empty_vec": True,
                "active_count": 0, "entropy": 0.0,
                "top1_score": 0.0, "thr": None,
            }
            return [RoutingScore(r.region_id, 0.0, {"vector": 0.0})
                    for r in regions]

        # 2) 냉시작 보호 (Review #3) — 모든 Region 의 signature 가 미초기화/count=0
        all_cold = all(
            r.region_signature is None or r.region_signature.count == 0
            for r in regions
        )
        if all_cold:
            uniform = 1.0 / max(len(regions), 1)
            self.last_metrics = {
                "cold_start": True, "empty_vec": False,
                "active_count": len(regions),
                "entropy": _shannon_entropy([uniform] * len(regions)),
                "top1_score": uniform, "thr": None,
            }
            return [RoutingScore(r.region_id, uniform,
                                 {"vector": 0.0, "cold_start": True})
                    for r in regions]

        # 3) similarity 계산
        sims = []
        for r in regions:
            sig = r.region_signature
            s = sig.similarity(signal_vec) if sig is not None else 0.0
            sims.append((r.region_id, s))

        scores = np.array([s for _, s in sims], dtype=np.float64)
        mu     = float(scores.mean())
        sigma  = float(scores.std())

        # 4) Dynamic threshold + Review #1 상한 클램프 (부호반전 방지)
        thr = min(mu + self.beta * sigma, 0.99)
        normalized = [
            max(0.0, (s - thr) / (1.0 - thr + 1e-8))
            for s in scores
        ]

        # 5) Review #6: β sweep 메트릭
        active_count = sum(1 for n in normalized if n > 1e-8)
        self.last_metrics = {
            "cold_start":   False,
            "empty_vec":    False,
            "beta":         self.beta,
            "mu":           mu,
            "sigma":        sigma,
            "thr":          thr,
            "active_count": active_count,
            "entropy":      _shannon_entropy(normalized),
            "top1_score":   float(max(normalized)) if normalized else 0.0,
        }

        return [
            RoutingScore(rid, float(ns), {"vector": float(s)})
            for (rid, s), ns in zip(sims, normalized)
        ]


__all__ = ["VectorRouter"]
