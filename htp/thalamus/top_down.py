"""
Top-Down Signal  —  PFC → Thalamus 역방향 투영
================================================

생물학: 전전두엽(PFC)에서 시상으로 내려가는 역방향 연결
  - 전체 시상 연결의 약 40%가 피질→시상 방향
  - 목표 관련 Region을 미리 활성화 (예측적 게이팅)
  - Biased Competition Model (Desimone & Duncan 1995):
    top-down attention bias + bottom-up sensory signal

수학:
  [Stage 3-B4 / Friston review] Softmax prior:
    overlap_count[r] = |region_tags[r] ∩ goal_tags|
    biases[r]        = softmax(overlap_count / T)[r]

    Jaccard 대비 장점:
      - 합 = 1 (적법한 확률 분포) → VFE 계산 시 직접 prior 로 사용 가능
      - overlap=0 인 Region 도 최소 확률 할당 (Jaccard 는 0 절단)
      - temperature T 로 sharpness 조절

  CoreCells gate():
    biased_score = precision × score + td_weight × td_bias × strength

용도:
  BrainRuntime.run()에서 매 스텝 생성:
    _, td = pfc.decide(thal_out, regions)
  다음 스텝 Thalamus.step()에 전달:
    thalamus.step(data, top_down=td)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..runtime.region_runtime import RegionRuntime


@dataclass
class TopDownSignal:
    """PFC → Thalamus 역방향 바이어스 신호."""
    biases:   dict   # {region_id: bias, 합=1 (Stage 3-B4 이후)}
    strength: float  # 전체 top-down 강도 스케일 (0~1)
    step:     int


class TopDownBias:
    """
    PFC long_term_goals → Region softmax 확률 prior 계산기.

    goal 키워드와 Region specialty + @tag 의 교집합 크기를
    softmax 정규화하여 합=1 의 확률 분포 반환.
    """

    def __init__(self, temperature: float = 1.0):
        """
        temperature: softmax 온도
          - 낮을수록 (T→0) argmax 에 집중 (hard)
          - 높을수록 (T→∞) 균등 분포 (soft)
          - 기본 1.0 — 적당한 확률적 prior
        """
        self.temperature = temperature

    def compute(self,
                goals:    list[str],
                regions:  dict,
                step:     int,
                strength: float = 0.3) -> TopDownSignal:
        """
        goals:    long_term_goals 문자열 리스트
        regions:  {region_name: RegionRuntime}
        strength: 0~1, top-down 신호 전체 강도
        """
        if not goals or not regions:
            return TopDownSignal(biases={}, strength=0.0, step=step)

        # goal 단어 집합
        goal_set: set[str] = set()
        for g in goals:
            goal_set.update(w.lower() for w in g.replace("_", " ").split())

        # 각 Region 의 overlap count 수집
        overlap_counts: dict[str, int] = {}
        for rid, region in regions.items():
            spec = set(region.specialty.lower().replace("_", " ").split())
            node_tags: set[str] = set()
            for n in getattr(region, "_nodes", []):
                node_tags |= getattr(n.fn, "_htp_tags", set())
            region_tags = spec | node_tags
            overlap_counts[rid] = len(goal_set & region_tags)

        # Softmax 정규화 (torch 의존 제거 — 순수 Python)
        T = max(self.temperature, 1e-6)
        exps = {rid: math.exp(c / T) for rid, c in overlap_counts.items()}
        Z    = sum(exps.values()) or 1.0
        biases = {rid: e / Z for rid, e in exps.items()}

        return TopDownSignal(biases=biases, strength=strength, step=step)
