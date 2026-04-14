"""
Pruning Engine
HTP 런타임의 시냅스 가지치기 전담 엔진

4가지 전략:
  magnitude — 강도 < threshold 인 연결 제거 (기본)
  activity  — 양 끝 노드 발화 횟수 합이 낮은 연결 제거
  age       — 마지막 강화 이후 오래된 연결 제거
  combined  — decay + magnitude + activity 동시 적용
"""

from __future__ import annotations

import torch
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .hub_formation_engine import HubFormationEngine


# ─────────────────────────────────────────────
# 설정 / 결과
# ─────────────────────────────────────────────

@dataclass
class PruneConfig:
    strategy:            str   = "magnitude"
    decay_rate:          float = 0.005   # magnitude / combined 에서 매 호출마다 감쇠
    magnitude_threshold: float = 0.02   # 이 이하 연결 제거
    activity_window:     float = 30.0   # fire_count 합 최솟값 (activity 전략)
    age_threshold:       int   = 100    # 이 스텝 이상 강화 없으면 제거 (age 전략)
    hub_protect:         bool  = True   # 허브 노드 관여 연결은 제거 안 함


@dataclass
class PruneResult:
    pruned:        int
    strategy:      str
    decay_applied: bool
    protected:     int = 0

    def __repr__(self) -> str:
        d = "decay+" if self.decay_applied else ""
        return (
            f"PruneResult({d}{self.strategy}"
            f" | pruned={self.pruned}, protected={self.protected})"
        )


# ─────────────────────────────────────────────
# Pruning Engine
# ─────────────────────────────────────────────

class PruningEngine:
    """
    HubFormationEngine.W 를 공유 참조로 받아 가지치기만 전담.

    모든 W 수정은 in-place 연산 → HubFormationEngine 쪽 참조 항상 유효.

    age 전략:
      prune() 호출 때마다 W 스냅샷 비교 → 강화된 연결 나이 0 리셋.
      HubFormationEngine 내부 step() 을 직접 후킹하지 않아도 동작.
    """

    STRATEGIES = frozenset({"magnitude", "activity", "age", "combined"})

    def __init__(
        self,
        hub_engine: "HubFormationEngine",
        config: Optional[PruneConfig] = None,
    ):
        self._engine = hub_engine
        self.cfg     = config or PruneConfig()

        # age 전략 내부 상태
        self._age_matrix: Optional[torch.Tensor] = None
        self._W_prev:     Optional[torch.Tensor] = None

        self._total_pruned = 0
        self._call_count   = 0

    # ── 공개 API ────────────────────────────────────

    def prune(self, strategy: Optional[str] = None) -> PruneResult:
        """
        가지치기 실행.
        strategy 미지정 → config.strategy 사용.
        """
        s = (strategy or self.cfg.strategy).lower()
        if s not in self.STRATEGIES:
            raise ValueError(f"알 수 없는 전략 '{s}'. 선택: {sorted(self.STRATEGIES)}")

        if   s == "magnitude": result = self._magnitude()
        elif s == "activity":  result = self._activity()
        elif s == "age":       result = self._age()
        else:                  result = self._combined()     # "combined"

        self._total_pruned += result.pruned
        self._call_count   += 1
        return result

    @property
    def total_pruned(self) -> int:
        return self._total_pruned

    @property
    def call_count(self) -> int:
        return self._call_count

    # ── 내부: 허브 보호 마스크 ──────────────────────

    def _hub_mask(self) -> torch.Tensor:
        """허브 노드가 src 또는 dst 인 모든 연결 마스크 [n, n]"""
        h = self._engine.is_hub                     # [n] bool
        return h.unsqueeze(0) | h.unsqueeze(1)      # [n, n] bool

    def _apply_protection(
        self, mask: torch.Tensor
    ) -> tuple:
        """hub_protect 적용 후 (최종 마스크, protected 수) 반환"""
        if self.cfg.hub_protect:
            hm        = self._hub_mask()
            protected = int((mask & hm).sum().item())
            mask      = mask & ~hm
        else:
            protected = 0
        return mask, protected

    # ── 전략 1: magnitude ───────────────────────────

    def _magnitude(self) -> PruneResult:
        W = self._engine.W
        W.mul_(1.0 - self.cfg.decay_rate)           # in-place 감쇠

        weak = W < self.cfg.magnitude_threshold
        weak, protected = self._apply_protection(weak)

        pruned = int(weak.sum().item())
        W[weak] = 0.0
        return PruneResult(
            pruned=pruned, strategy="magnitude",
            decay_applied=True, protected=protected,
        )

    # ── 전략 2: activity ────────────────────────────

    def _activity(self) -> PruneResult:
        W  = self._engine.W
        fc = self._engine.fire_count                # [n] 누적 발화

        # 연결 양 끝 노드의 발화 합이 window 이하
        pair_act = fc.unsqueeze(0) + fc.unsqueeze(1)   # [n, n]
        low      = (W > 0) & (pair_act < self.cfg.activity_window)
        low, protected = self._apply_protection(low)

        pruned = int(low.sum().item())
        W[low]  = 0.0
        return PruneResult(
            pruned=pruned, strategy="activity",
            decay_applied=False, protected=protected,
        )

    # ── 전략 3: age ─────────────────────────────────

    def _age(self) -> PruneResult:
        W = self._engine.W
        n = W.shape[0]
        dev = W.device

        # 첫 호출 시 상태 초기화
        if self._age_matrix is None:
            self._age_matrix = torch.zeros(n, n, device=dev)
            self._W_prev     = W.clone()

        # 마지막 prune() 호출 이후 강화된 연결 → 나이 0 리셋
        strengthened = W > self._W_prev
        self._age_matrix[strengthened] = 0

        # 살아있는 연결 +1, 소멸된 연결 0
        self._age_matrix[W > 0]  += 1
        self._age_matrix[W == 0]  = 0

        # W 스냅샷 갱신
        self._W_prev.copy_(W)

        old = self._age_matrix > self.cfg.age_threshold
        old, protected = self._apply_protection(old)

        pruned = int(old.sum().item())
        W[old] = 0.0
        self._age_matrix[old] = 0
        return PruneResult(
            pruned=pruned, strategy="age",
            decay_applied=False, protected=protected,
        )

    # ── 전략 4: combined ────────────────────────────

    def _combined(self) -> PruneResult:
        """decay 감쇠 + magnitude + activity 동시 적용"""
        W = self._engine.W
        W.mul_(1.0 - self.cfg.decay_rate)

        fc   = self._engine.fire_count
        weak = W < self.cfg.magnitude_threshold
        low  = (fc.unsqueeze(0) + fc.unsqueeze(1)) < self.cfg.activity_window
        mask = weak | ((W > 0) & low)
        mask, protected = self._apply_protection(mask)

        pruned = int(mask.sum().item())
        W[mask] = 0.0
        return PruneResult(
            pruned=pruned, strategy="combined",
            decay_applied=True, protected=protected,
        )
