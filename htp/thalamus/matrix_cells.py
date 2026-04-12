"""
Matrix Cells  —  시상 상태 게이팅 (Winner-take-all)
=====================================================

생물학: 시상의 Matrix 세포층 (중심매체핵, 내측배측핵)
  - 피질 전반에 광범위한 조절 신호 투사
  - Winner-take-all 경쟁으로 주의 집중 결정
  - 측억제(Lateral Inhibition)로 패자 억제

수학: Lateral Inhibition + Softmax

  [1단계] Lateral Inhibition (생물학적 측억제):
    s_i(t+1) = ReLU(s_i(t) - w_lat * Σ_{j≠i} s_j(t))
    반복 iter회 → 강한 신호는 강해지고 약한 신호는 약해짐

  [2단계] Softmax (temperature T):
    p_i = exp(s_i / T) / Σ exp(s_j / T)
    T → 0: Hard WTA (winner만 1, 나머지 0)
    T = 1: Soft WTA (확률 분포)

  억제 강도: Δp_i = p_winner - p_i  (패자 Region에 전달)
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F

from .region_signal import RegionSignal, GatingMask, CompetitionResult


class MatrixCells:
    """
    Lateral Inhibition + Softmax Winner-take-all.

    생물학적 WTA 재현:
    - Lateral inhibition으로 강한 Region이 약한 Region 억제
    - Softmax로 최종 확률 분포 계산
    - 억제 강도(Δp)를 패자 Region에 피드백
    """

    def __init__(self,
                 temperature:   float = 1.0,
                 lateral_w:     float = 0.15,
                 lateral_iter:  int   = 3):
        """
        temperature  : Softmax 온도 (낮을수록 더 날카로운 WTA)
        lateral_w    : 측억제 강도 (0~1)
        lateral_iter : 측억제 반복 횟수
        """
        self.temperature  = temperature
        self.lateral_w    = lateral_w
        self.lateral_iter = lateral_iter

    def compete(self,
                signals: List[RegionSignal],
                gating:  GatingMask) -> CompetitionResult:
        """
        signals + gating → CompetitionResult.

        1. 원시 점수 = gating_score + 과부하 보너스
        2. Lateral Inhibition (lateral_iter회)
        3. Softmax (temperature)
        4. 억제 강도 계산 (Δp)
        """
        if not signals:
            raise ValueError("signals 리스트가 비어 있습니다")

        region_ids = [sig.region_id for sig in signals]

        # 1. 원시 점수: gating + 과부하 보너스 (과부하 Region 우선 처리)
        raw = torch.tensor([
            gating.scores.get(sig.region_id, 0.0) + (0.2 if sig.overload else 0.0)
            for sig in signals
        ], dtype=torch.float32)

        # 2. Lateral Inhibition
        s = raw.clone()
        for _ in range(self.lateral_iter):
            total      = s.sum()
            inhibition = self.lateral_w * (total - s)   # 자신 제외 합
            s          = F.relu(s - inhibition)

        # 3. Softmax (분모 폭발 방지: max 빼기)
        probs = torch.softmax(s / max(self.temperature, 1e-6), dim=0)

        # 4. 결과 조립
        winner_idx   = int(probs.argmax().item())
        winner_id    = region_ids[winner_idx]
        winner_score = float(probs[winner_idx])

        suppression = {
            region_ids[i]: max(0.0, winner_score - float(probs[i]))
            for i in range(len(region_ids))
            if i != winner_idx
        }

        return CompetitionResult(
            winner_id       = winner_id,
            winner_score    = winner_score,
            suppression_map = suppression,
            all_scores      = {rid: float(p)
                               for rid, p in zip(region_ids, probs.tolist())},
        )
