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

    (2) Hebbian Adaptive θ (anti-homeostatic, 반복 승자 강화):
        win_history[i] = 0.1 × win_t + 0.9 × win_history[i]  (EMA)

  [Stage 2-A3 / LeCun review] Homeostatic Plasticity 추가:
    (3) Homeostatic θ (과흥분 안정화):
        homeo_term[i] = η_hom × (fire_rate[i] - target_rate)

    두 term 은 polarity 가 상반되어 공존:
        theta_bias[i] += -η_heb × win_history[i]   (승자일수록 θ↓, 이기기 쉬움)
                        + η_hom × (r_i - r_target) (과흥분일수록 θ↑, 억제)
        clamp: theta_bias ∈ [-0.2, 0.2]

    최종 gate:
        g_i = σ(β × (precision_i · biased_score_i - (θ + theta_bias[i])))

생물학:
  - 시상-피질 Hebbian 가소성 (반복 승자 → 게이트 강화)
  - Turrigiano (2008) synaptic scaling — 활동 과흥분 → 흥분성 감소
"""

from __future__ import annotations

import math
from typing import List, Optional, TYPE_CHECKING

from .region_signal import RegionSignal, GatingMask
from .top_down      import TopDownSignal
from .router        import RouterStrategy, TagRouter

if TYPE_CHECKING:
    import numpy as np


class CoreCells:
    """
    Sigmoidal Gate + Top-down Bias + Hebbian Adaptive Learning.

    인터페이스:
      gate(signals, top_down=None, signal_text=None, signal_vec=None) → GatingMask
      update(winner_id, all_ids)   → None  (Hebbian 학습)

    [sub-2 M6] router DI:
      - router 기본값 = TagRouter() — 회귀 보호 (기존 raw score 와 동등)
      - VectorRouter / HybridRouter 주입 시 content-addressable / hybrid 모드
      - 런타임 router 교체 가능 (self.router = ...)
    """

    def __init__(self,
                 beta:        float = 5.0,
                 theta:       float = 0.3,
                 eta:         float = 0.05,   # Hebbian 학습률 (호환 유지)
                 eta_heb:     float | None = None,  # None 이면 eta 사용
                 eta_hom:     float = 0.02,   # [A3] Homeostatic 학습률
                 target_rate: float = 0.1,    # [A3] 목표 발화율
                 td_weight:   float = 0.3,
                 router:      "RouterStrategy | None" = None):
        """
        beta        : Sigmoid 날카로움
        theta       : 기본 게이팅 임계값
        eta / eta_heb: Hebbian 학습률 (승자 theta ↓)
        eta_hom     : Homeostatic 학습률 (과흥분 theta ↑)
        target_rate : 목표 발화율 (r_i > target → 억제, r_i < target → 강화)
        td_weight   : top-down 바이어스 가중치
        router      : [sub-2 M6] 라우팅 정책. None → TagRouter() (회귀 보호)
        """
        self.beta        = beta
        self.theta       = theta
        self._eta_heb    = eta_heb if eta_heb is not None else eta
        self._eta_hom    = eta_hom
        self._target_rate = target_rate
        self._td_weight  = td_weight

        # [sub-2 M6] Router DI — 기본값은 회귀 동등 (TagRouter)
        self.router: RouterStrategy = router or TagRouter()

        # Hebbian 적응 상태
        self._win_history : dict[str, float] = {}
        self._theta_bias  : dict[str, float] = {}

    # ── 게이팅 ────────────────────────────────────────

    def gate(self,
             signals:   List[RegionSignal],
             top_down:  Optional[TopDownSignal] = None,
             *,
             signal_text: "str | None" = None,
             signal_vec:  "np.ndarray | None" = None) -> GatingMask:
        """
        signals + top-down → GatingMask.

        1. router.score(signal_text, signal_vec, signals) → normalized scores
        2. top-down 바이어스 가산 (있을 때만)
        3. Adaptive θ 반영
        4. Sigmoid sharpening

        [sub-2 M6] 1단계가 self.router (기본 TagRouter) 로 위임.
        signal_text / signal_vec 은 keyword-only — 기존 호출자 (positional)
        는 None 으로 들어가 TagRouter 가 무시 → 회귀 동등.
        """
        if not signals:
            return GatingMask(scores={})

        # 1. [sub-2 M6] Router 위임 — Tag/Vector/Hybrid 다형성
        routing_scores = self.router.score(signal_text, signal_vec, signals)
        normalized: dict[str, float] = {
            rs.region_id: rs.score for rs in routing_scores
        }

        # 3. Top-down 바이어스 (Biased Competition)
        td_biases: dict[str, float] = {}
        if top_down and top_down.strength > 0:
            for rid in normalized:
                td_biases[rid] = (
                    self._td_weight
                    * top_down.biases.get(rid, 0.0)
                    * top_down.strength
                )

        # 4 + 5. Adaptive θ + Precision-weighted + Sigmoid
        # precision (Friston B3): 각 Region 의 신뢰도 — score amplification
        precision_map = {s.region_id: getattr(s, "precision", 1.0) for s in signals}

        gated: dict[str, float] = {}
        for rid, score in normalized.items():
            precision    = precision_map.get(rid, 1.0)
            biased_score = precision * score + td_biases.get(rid, 0.0)
            eff_theta    = self.theta + self._theta_bias.get(rid, 0.0)
            gated[rid]   = 1.0 / (1.0 + math.exp(
                -self.beta * (biased_score - eff_theta)
            ))

        return GatingMask(scores=gated)

    # ── Hebbian + Homeostatic 학습 ────────────────────

    def update(self, winner_id: str, all_ids: list[str],
               fire_rates: dict[str, float] | None = None):
        """
        승자 Region 기반 게이트 파라미터 업데이트 — 이중 메커니즘.

        (1) Hebbian (anti-homeostatic):
            win_history: EMA 승리율
            theta_bias -= eta_heb × win_history      (승자 Region θ ↓)

        (2) Homeostatic (Turrigiano synaptic scaling, Stage 2-A3):
            theta_bias += eta_hom × (fire_rate - target_rate)
            fire_rate > target → θ ↑ (과흥분 억제)
            fire_rate < target → θ ↓ (저활성 활성화)

        fire_rates 가 None 이면 homeostatic 생략 (하위 호환).
        """
        for rid in all_ids:
            win  = 1.0 if rid == winner_id else 0.0
            prev = self._win_history.get(rid, 0.0)
            self._win_history[rid] = 0.1 * win + 0.9 * prev

        for rid in all_ids:
            bias         = self._theta_bias.get(rid, 0.0)
            hebbian_term = -self._eta_heb * self._win_history.get(rid, 0.0)
            if fire_rates is not None and rid in fire_rates:
                homeo_term = self._eta_hom * (fire_rates[rid] - self._target_rate)
            else:
                homeo_term = 0.0
            bias += hebbian_term + homeo_term
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
