"""
Top-Down Signal  —  PFC → Thalamus 역방향 투영
================================================

생물학: 전전두엽(PFC)에서 시상으로 내려가는 역방향 연결
  - 전체 시상 연결의 약 40%가 피질→시상 방향
  - 목표 관련 Region을 미리 활성화 (예측적 게이팅)
  - Biased Competition Model (Desimone & Duncan 1995):
    top-down attention bias + bottom-up sensory signal

수학:
  td_bias[region] = |region_tags ∩ goal_tags| / |goal_tags|
                    (Jaccard-like 교집합 비율)

  CoreCells gate():
    biased_score = score + td_weight × td_bias × strength

용도:
  BrainRuntime.run()에서 매 스텝 생성:
    _, td = pfc.decide(thal_out, regions)
  다음 스텝 Thalamus.step()에 전달:
    thalamus.step(data, top_down=td)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..runtime.region_runtime import RegionRuntime


@dataclass
class TopDownSignal:
    """PFC → Thalamus 역방향 바이어스 신호."""
    biases:   dict   # {region_id: bias 0~1}
    strength: float  # 전체 top-down 강도 스케일 (0~1)
    step:     int


class TopDownBias:
    """
    PFC long_term_goals → Region 바이어스 계산기.

    goal 키워드와 Region의 specialty + @tag 교집합으로
    각 Region의 사전 활성화 바이어스를 계산.
    """

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

        biases: dict[str, float] = {}
        for rid, region in regions.items():
            # Region specialty 단어
            spec = set(region.specialty.lower().replace("_", " ").split())

            # 등록된 노드의 @tag 합집합
            node_tags: set[str] = set()
            for n in getattr(region, "_nodes", []):
                node_tags |= getattr(n.fn, "_htp_tags", set())

            region_tags = spec | node_tags
            overlap = goal_set & region_tags
            biases[rid] = len(overlap) / max(len(goal_set), 1)

        return TopDownSignal(biases=biases, strength=strength, step=step)
