"""
Region Runtime  —  HTPRuntime 확장
====================================

단일 피질 영역을 표현하는 런타임.
HTPRuntime을 상속하고 다음을 추가:

  - NodeGenerationEngine 통합 (동적 노드 생성)
  - PageRank 기반 허브 강도 측정
  - Shannon Entropy + CUSUM 과부하 감지
  - Thalamus 억제 신호 수신 (apply_suppression)

생물학:
  - 각 RegionRuntime은 독립 피질 영역 (언어, 기억, 감정 등)
  - NRXN1 메커니즘: 시상 과부하 신호 → 피질 신경발생 (NGE 트리거)
"""

from __future__ import annotations

import math
from typing import Optional, Any

import torch

from .htp_runtime import HTPRuntime, HTPConfig, Node, RunResult
from ..thalamus.region_signal import RegionSignal


class RegionRuntime(HTPRuntime):
    """
    단일 피질 영역 런타임.

    사용법:
      region = RegionRuntime("language", "text_processing")

      @region.node
      def parse(data): ...

      region.connect(parse, classify)
      result = region.run(data, entry=parse)
      sig    = region.collect_signal()  # Thalamus에 전송
    """

    def __init__(self,
                 region_name: str,
                 specialty:   str,
                 config:      Optional[HTPConfig] = None,
                 gen_config   = None):               # GenConfig | None
        super().__init__(config)
        self.region_name = region_name
        self.specialty   = specialty
        self._gen_config = gen_config
        self._step       = 0

        # NodeGenerationEngine — _ensure_built() 이후 초기화
        self.nge = None

        # CUSUM 상태 (Shannon Entropy 과부하 누적)
        self._cusum_S   : float = 0.0
        self._cusum_k   : float = 0.25   # 허용 농도 수준 (0=균일 ~ 1=완전집중)
        self._cusum_h   : float = 2.0    # 알람 임계값

    # ── 빌드 (NGE 통합) ────────────────────────────────

    def _ensure_built(self):
        super()._ensure_built()
        if self.nge is None:
            from ..core.node_generation_engine import NodeGenerationEngine, GenConfig
            gc = self._gen_config or GenConfig()
            self.nge = NodeGenerationEngine(self.wm, self.hfe, self.cfg, gc)
            self.nge.register(self._nodes)

    # ── 실행 (NGE check 포함) ──────────────────────────

    def run(self, data: Any,
            entry=None,
            max_depth: int = 8) -> RunResult:
        self._last_data = data
        result = super().run(data, entry, max_depth)
        self._step += 1

        # NGE: 매 스텝 split/sprout 체크
        if self.nge and self._nodes:
            self.nge.check_split(self._step)

            # sprout용 신호 생성 (현재 발화 경로 기반)
            fired_ids = [n.node_id for n in result.route_path]
            N = len(self._nodes)
            sig = torch.zeros(N, device=self.cfg.device)
            for i in fired_ids:
                if i < N:
                    sig[i] = 1.0
            self.nge.check_sprout(sig, fired_ids, self._step)

        return result

    # ── 신호 수집 (Thalamus 전송용) ───────────────────

    def collect_signal(self) -> RegionSignal:
        """
        현재 영역 상태를 RegionSignal로 요약.

        허브 강도: PageRank 점수 (hfe.top_hubs가 PageRank 반환)
        과부하: Shannon Entropy → 발화 집중도 → CUSUM 누적
        """
        self._ensure_built()

        # 상위 허브 (PageRank)
        top = self.hfe.top_hubs(3)
        hub_strength = top[0][1] if top else 0.0
        top_hubs = [
            (self._nodes[i].name if i < len(self._nodes) else str(i), s)
            for i, s in top
        ]

        # 발화율 (전체 노드 평균)
        fire_rate = (
            sum(self.wm.recent_fire_rate(n.node_id, 20) for n in self._nodes)
            / max(len(self._nodes), 1)
        )

        # Shannon Entropy 기반 발화 집중도
        concentration = self._entropy_concentration()

        # CUSUM 누적 과부하 감지
        self._cusum_S = max(0.0, self._cusum_S + concentration - self._cusum_k)
        overload = self._cusum_S > self._cusum_h

        return RegionSignal(
            region_id    = self.region_name,
            hub_strength = hub_strength,
            fire_rate    = fire_rate,
            top_hubs     = top_hubs,
            overload     = overload,
            output_vec   = self.wm.W.sum(dim=1).detach().clone(),
        )

    def apply_suppression(self, strength: float):
        """
        MatrixCells WTA 결과로 Thalamus에서 전송한 억제 신호 수신.
        strength = Δp (winner_prob - self_prob)
        W 전체를 비례 감소.
        """
        self._ensure_built()
        factor = 1.0 - min(strength * 0.3, 0.6)   # 최대 60% 억제
        self.wm.W.mul_(factor).clamp_(0.0, 1.0)

    # ── 내부 ──────────────────────────────────────────

    def _entropy_concentration(self, window: int = 20) -> float:
        """
        Shannon Entropy 기반 발화 집중도.
          H = -Σ p_i log(p_i)
          concentration = 1 - H / log(N)
        0 = 균일 분포 (건강), 1 = 완전 집중 (과부하)
        """
        if len(self.wm.fire_history) < 3:
            return 0.0

        recent = torch.stack(self.wm.fire_history[-window:]).mean(dim=0)
        total  = recent.sum()
        if total < 1e-8:
            return 0.0

        p = recent / total
        H = -(p * (p + 1e-8).log()).sum().item()
        H_max = math.log(max(len(recent), 2))
        return max(0.0, 1.0 - H / H_max)
