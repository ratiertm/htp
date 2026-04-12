"""
NGE Trigger  —  시상 신경발생 트리거
=======================================

생물학: NRXN1 신호 전달 경로 (bioRxiv 2025)
  시상 → NRXN1 → outer radial glia → 피질 신경발생

메커니즘:
  1. CUSUM이 누적된 과부하 Region 식별
  2. 해당 Region의 NGE split 임계값을 일시적으로 낮춤
  3. check_split() 강제 호출 → 즉각 분열 유도
  4. 분열 성공 시 CUSUM 리셋 (과부하 해소)

수학:
  boost = min(overload_strength × 2.0, 4.0)
  new_threshold = max(original - boost, 0.5)   ← 하한선 보호
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..runtime.region_runtime import RegionRuntime


class NGETrigger:
    """
    시상 → 피질 신경발생 신호 전달.

    과부하 Region의 NGE split 임계값을 동적으로 낮춰
    즉각적인 노드 분열을 유도.
    분열 성공 시 CUSUM을 리셋해 과부하 상태를 해소.
    """

    def __init__(self, regions: dict):
        """
        regions: {region_name: RegionRuntime} — RegionRuntime 참조 딕셔너리
                 add_region() 시 Thalamus가 갱신.
        """
        self.regions      = regions
        self._trigger_log : list[dict] = []

    def fire(self, region_id: str, overload_strength: float):
        """
        과부하 Region의 NGE를 트리거.

        overload_strength: RegionSignal.hub_strength (PageRank 점수)
        """
        region = self.regions.get(region_id)
        if region is None or region.nge is None:
            return

        gcfg     = region.nge.gcfg
        original = gcfg.split_strength_threshold

        # CUSUM 누적에 비례한 임계값 낮춤
        boost    = min(overload_strength * 2.0, 4.0)
        gcfg.split_strength_threshold = max(original - boost, 0.5)

        new_nodes = region.nge.check_split(region._step)

        # 복원
        gcfg.split_strength_threshold = original

        # 분열 성공 시 CUSUM 리셋 (과부하 해소)
        if new_nodes:
            region._cusum_S = 0.0

        self._trigger_log.append({
            "region":   region_id,
            "strength": overload_strength,
            "boost":    boost,
            "spawned":  len(new_nodes) if new_nodes else 0,
        })

    def report(self) -> str:
        if not self._trigger_log:
            return "  [NGETrigger] no fires yet"
        lines = ["  [ NGETrigger Log ]"]
        for e in self._trigger_log[-5:]:
            lines.append(
                f"  region={e['region']}  boost={e['boost']:.2f}"
                f"  spawned={e['spawned']}"
            )
        return "\n".join(lines)
