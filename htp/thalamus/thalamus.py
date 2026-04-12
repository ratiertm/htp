"""
Thalamus  —  시상 통합 게이팅 허브
=====================================

생물학: 시상 (Thalamus)
  - Neuron 2024: 단순 중계가 아닌 압축·게이팅·재구성 수행
  - CoreCells: 특정 피질 영역 정밀 투영 (VPM, dLGN 등)
  - MatrixCells: 광범위 배경 조절 + WTA 경쟁
  - NGETrigger: NRXN1 신호 → 과부하 영역 신경발생 유도

처리 순서 (매 step):
  1. 각 Region 신호 수집 (RegionSignal)
  2. CoreCells Sigmoidal Gate → GatingMask
  3. MatrixCells Lateral Inhibition + Softmax → CompetitionResult
  4. 과부하 Region → NGETrigger.fire()
  5. 승자 Region output_vec → JL Random Projection 압축 → 8-dim state_vec
  6. ThalamusOutput 반환 → PFCRuntime 전달

수학 요약:
  압축: z = Φ @ x,  Φ ~ N(0, 1/k)  (Johnson-Lindenstrauss Lemma)
  JL 보장: ||Φx - Φy||² ≈ ||x - y||²  (거리 보존, 학습 불필요)
"""

from __future__ import annotations

import math
from typing import Optional

import torch

from .region_signal  import RegionSignal, ThalamusOutput
from .core_cells     import CoreCells
from .matrix_cells   import MatrixCells
from .nge_trigger    import NGETrigger
from .top_down       import TopDownSignal


class Thalamus:
    """
    CoreCells + MatrixCells + NGETrigger 통합 오케스트레이터.

    사용법:
      thalamus = Thalamus(regions)      # regions: {name: RegionRuntime}
      out = thalamus.step(data)          # ThalamusOutput
    """

    def __init__(self,
                 regions:            dict,
                 temperature:        float = 1.0,
                 core_beta:          float = 5.0,
                 core_theta:         float = 0.3,
                 compress_dim:       int   = 8):
        """
        regions       : {region_name: RegionRuntime}
        temperature   : MatrixCells Softmax 온도
        core_beta     : CoreCells Sigmoid 날카로움
        core_theta    : CoreCells Sigmoid 임계값
        compress_dim  : JL Projection 출력 차원
        """
        self.regions      = regions
        self.core         = CoreCells(beta=core_beta, theta=core_theta)
        self.matrix       = MatrixCells(temperature=temperature)
        self.nge_trigger  = NGETrigger(regions)
        self.compress_dim = compress_dim
        self._step        = 0

        # JL Projection matrices: {input_dim → Φ [k × n]}
        self._proj: dict[int, torch.Tensor] = {}

    def step(self, data=None,
             top_down: Optional[TopDownSignal] = None) -> ThalamusOutput:
        """
        매 스텝 실행.

        top_down: PFC가 이전 스텝에서 생성한 TopDownSignal.
                  None이면 순수 bottom-up 처리.
        """
        self._step += 1

        if not self.regions:
            raise RuntimeError("Thalamus에 등록된 Region이 없습니다 — add_region() 먼저 호출하세요")

        # ── 1. 각 Region 신호 수집 ───────────────────
        signals = [r.collect_signal() for r in self.regions.values()]

        # ── 2. CoreCells: Sigmoidal Gate + Top-down Bias ─
        gating = self.core.gate(signals, top_down=top_down)

        # ── 3. MatrixCells: Lateral Inhibition + WTA ─────
        competition = self.matrix.compete(signals, gating)

        # ── 4. CoreCells Hebbian 학습 (승자 Region 강화) ──
        self.core.update(
            winner_id = competition.winner_id,
            all_ids   = [s.region_id for s in signals],
        )

        # ── 5. 과부하 → NGETrigger ────────────────────────
        for sig in signals:
            if sig.overload:
                self.nge_trigger.fire(sig.region_id, sig.hub_strength)

        # ── 5. JL Random Projection 압축 ─────────────
        winner_sig = next(
            (s for s in signals if s.region_id == competition.winner_id),
            signals[0]
        )
        state_vec = self._jl_compress(winner_sig.output_vec)

        return ThalamusOutput(
            winner    = competition.winner_id,
            state_vec = state_vec,
            gating    = gating,
            suppressed= competition.suppression_map,
            step      = self._step,
        )

    def _jl_compress(self, vec: torch.Tensor) -> torch.Tensor:
        """
        Johnson-Lindenstrauss Random Projection.

        Φ ∈ R^{k×n}, 각 원소 ~ N(0, 1/k)
        z = Φ @ x   (n-dim → k-dim)

        JL Lemma 보장:
          ||Φx - Φy||² ≈ ||x - y||²  (거리 보존)
          k = O(log(1/δ) / ε²) 이면 (1±ε) 오차 내 거리 보존
        """
        k = self.compress_dim
        n = len(vec)

        if n == 0:
            return torch.zeros(k, device=vec.device if hasattr(vec, 'device') else 'cpu')

        if n not in self._proj:
            # Gaussian random projection matrix (한 번만 생성, 고정)
            Phi = torch.randn(k, n)
            Phi = Phi / math.sqrt(k)   # JL 정규화
            if vec.is_cuda:
                Phi = Phi.cuda()
            self._proj[n] = Phi

        return self._proj[n] @ vec

    def status(self) -> str:
        """현재 Thalamus 상태 요약."""
        lines = [
            f"\n{'='*55}",
            f"  Thalamus  step={self._step}  regions={len(self.regions)}",
            f"  CoreCells   beta={self.core.beta}  theta={self.core.theta}",
            f"  MatrixCells T={self.matrix.temperature}  "
            f"lat_w={self.matrix.lateral_w}",
            self.nge_trigger.report(),
            f"{'='*55}",
        ]
        return "\n".join(lines)
