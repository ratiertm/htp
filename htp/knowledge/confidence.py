"""Query 결과의 confidence 측정 (sub-5 merge plan 작업 3 — I5).

Design Ref: docs/01-plan/features/htp-sub5-merge-plan.md §3

핵심 아이디어: top-1 과 top-2 의 cosine gap 이 좁으면 확신이 없다.
  - gap 큼 → top-1 이 명확히 우세 → 진짜 매칭
  - gap 좁음 → 비슷한 후보가 여럿 → 매칭 없거나 모호

발견 데이터 (Vault 99 entries, merge plan §2 작업 후 측정):
  - Hopfield (vault 에 없음): gap = 0.0051  → low confidence (목표 동작)
  - V-JEPA (vault 에 있음):    gap = 0.0042  → low confidence (false negative)
  - HTP thalamus (있음):       gap = 0.0168  → high confidence
  - 트레이딩 (있음):           gap = 0.0192  → high confidence

이 분포 기반으로 기본 threshold = 0.005 (Hopfield 와 V-JEPA 사이).
완벽하지 않으나 false negative 회피 우선. 사용자가 generator 인자로 조정 가능.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# Vault 실측 분포 기반 기본값 (merge plan §6 R3 의 조정 사항 반영)
# 측정값: Hopfield(없음) gap=0.0051, V-JEPA(있음) gap=0.0042, HTP(있음) gap=0.0168
# threshold = 0.01: Hopfield 잡힘 ✓, V-JEPA 는 false negative (top-1 cosine 자체는 노출).
# false negative 회피 우선 시 0.005 권장. 사용자가 query_v2(gap_threshold=) 로 조정 가능.
DEFAULT_GAP_THRESHOLD: float = 0.01


@dataclass
class ScoredResult:
    """query 결과 단일 항목 + 메타데이터."""
    entry_id:   str
    text:       str
    source:     str
    similarity: float
    rank:       int


@dataclass
class QueryResultV2:
    """confidence 포함 query 반환값 (sub-5 merge 작업 3 신규).

    기존 QueryResult (loop.py) 는 backward-compat 위해 유지.
    KnowledgeLoop.query_v2() 또는 future query() 가 이를 반환.
    """
    question:   str
    results:    list                  # list[ScoredResult]
    confidence: float                 # top-1 vs top-2 cosine gap
    has_match:  bool                  # gap > threshold

    @staticmethod
    def compute_confidence(
        similarities: "list[float]",
        gap_threshold: float = DEFAULT_GAP_THRESHOLD,
    ) -> "tuple[float, bool]":
        """Top-1 vs Top-2 cosine gap 으로 confidence 측정.

        Parameters
        ----------
        similarities : list[float]
            검색 결과의 cosine similarity 리스트 (정렬 불필요).
        gap_threshold : float
            이 값 이하면 "확신 없음".

        Returns
        -------
        (gap, has_match) : tuple[float, bool]
            gap : top-1 과 top-2 의 cosine 차이.
            has_match : gap > threshold.
        """
        if len(similarities) < 2:
            return 0.0, False
        sorted_sims = sorted(similarities, reverse=True)
        gap = float(sorted_sims[0] - sorted_sims[1])
        return gap, gap > gap_threshold


__all__ = [
    "DEFAULT_GAP_THRESHOLD",
    "ScoredResult",
    "QueryResultV2",
]
