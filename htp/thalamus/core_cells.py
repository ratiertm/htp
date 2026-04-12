"""
Core Cells  —  시상 내용 게이팅 (Adaptive)
===========================================

생물학: 시상의 Core 세포층 (VPM, dLGN 등)
  - 특정 피질 영역에 정밀하고 강한 투영
  - Sigmoidal Gate로 부드러운 ON/OFF 전환

수학:
  [Phase 2] Sigmoidal Gate:
    g_i = σ(β * (score_i - θ))

  [Phase 3] 추가:
    (1) Top-down Bias:
        biased_score = score + td_weight × td_bias_i × td_strength

    (2) Hebbian Adaptive θ:
        win_history[i] = 0.1 × win_t + 0.9 × win_history[i]  (EMA)
        theta_bias[i] -= eta × win_history[i]
        clamp: theta_bias ∈ [-0.2, 0.2]

    최종:
        g_i = σ(β × (biased_score - (θ + theta_bias[i])))

생물학: 시상-피질 Hebbian 가소성 (반복 승자 → 게이트 강화)
"""

from __future__ import annotations

import math
from typing import List, Optional

from .region_signal import RegionSignal, GatingMask
from .top_down      import TopDownSignal


class CoreCells:
    """
    Sigmoidal Gate + Top-down Bias + Hebbian Adaptive Learning.

    인터페이스:
      gate(signals, top_down=None) → GatingMask
      update(winner_id, all_ids)   → None  (Hebbian 학습)
    """

    def __init__(self,
                 beta:      float = 5.0,
                 theta:     float = 0.3,
                 eta:       float = 0.05,
                 td_weight: float = 0.3):
        """
        beta      : Sigmoid 날카로움
        theta     : 기본 게이팅 임계값
        eta       : Hebbian 학습률
        td_weight : top-down 바이어스 가중치
        """
        self.beta      = beta
        self.theta     = theta
        self._eta      = eta
        self._td_weight = td_weight

        # Hebbian 적응 상태
        self._win_history : dict[str, float] = {}
        self._theta_bias  : dict[str, float] = {}

    # ── 게이팅 ────────────────────────────────────────

    def gate(self,
             signals:   List[RegionSignal],
             top_down:  Optional[TopDownSignal] = None) -> GatingMask:
        """
        signals + top-down → GatingMask.

        1. raw score = hub_strength × (1 + fire_rate)
        2. L1 정규화
        3. top-down 바이어스 가산 (있을 때만)
        4. Adaptive θ 반영
        5. Sigmoid sharpening
        """
        if not signals:
            return GatingMask(scores={})

        # 1. Raw score
        raw: dict[str, float] = {
            sig.region_id: sig.hub_strength * (1.0 + sig.fire_rate)
            for sig in signals
        }

        # 2. L1 정규화
        total = sum(raw.values()) or 1.0
        normalized = {rid: v / total for rid, v in raw.items()}

        # 3. Top-down 바이어스 (Biased Competition)
        td_biases: dict[str, float] = {}
        if top_down and top_down.strength > 0:
            for rid in normalized:
                td_biases[rid] = (
                    self._td_weight
                    * top_down.biases.get(rid, 0.0)
                    * top_down.strength
                )

        # 4 + 5. Adaptive θ + Sigmoid
        gated: dict[str, float] = {}
        for rid, score in normalized.items():
            biased_score = score + td_biases.get(rid, 0.0)
            eff_theta    = self.theta + self._theta_bias.get(rid, 0.0)
            gated[rid]   = 1.0 / (1.0 + math.exp(
                -self.beta * (biased_score - eff_theta)
            ))

        return GatingMask(scores=gated)

    # ── Hebbian 학습 ──────────────────────────────────

    def update(self, winner_id: str, all_ids: list[str]):
        """
        승자 Region 기반 Hebbian gate 파라미터 업데이트.

        win_history: EMA 승리율 (자주 이길수록 높아짐)
        theta_bias:  승리율에 비례해 낮아짐
                     → 자주 이기는 Region이 더 쉽게 게이팅됨
        """
        for rid in all_ids:
            win  = 1.0 if rid == winner_id else 0.0
            prev = self._win_history.get(rid, 0.0)
            self._win_history[rid] = 0.1 * win + 0.9 * prev

        for rid in all_ids:
            bias  = self._theta_bias.get(rid, 0.0)
            bias -= self._eta * self._win_history.get(rid, 0.0)
            self._theta_bias[rid] = max(-0.2, min(0.2, bias))

    def report(self) -> str:
        if not self._win_history:
            return "  [CoreCells] no history"
        lines = ["  [ CoreCells Adaptive State ]"]
        for rid in sorted(self._win_history):
            wh = self._win_history.get(rid, 0)
            tb = self._theta_bias.get(rid, 0)
            lines.append(f"  {rid:<14}  win_ema={wh:.3f}  theta_bias={tb:+.3f}")
        return "\n".join(lines)
