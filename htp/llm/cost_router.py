"""
CostRouter  —  EMA Cost Pressure + 모델 다운그레이드
=====================================================

EMA Cost Pressure:
  ema_cost = alpha * cost + (1-alpha) * ema_cost   (alpha=0.3)
  pressure = ema_cost / budget_per_step

  < 0.6   normal
  0.6~1.0 warn   (prompt 단축 권장)
  > 1.0   overload -> 모델 다운그레이드 또는 차단

Cost-aware Routing Score:
  route_score = quality * (1 - pressure) + cache_hit * pressure
  고압박: 캐시/저비용 우선
  저압박: 품질 우선
"""

from __future__ import annotations


class CostRouter:
    PRESSURE_WARN     = 0.6
    PRESSURE_OVERLOAD = 1.0
    PRESSURE_BLOCK    = 2.0   # 예산 2배 초과 시 LLM 호출 차단

    DOWNGRADE_MAP: dict[str, str] = {
        "claude-sonnet-4-6":         "claude-haiku-4-5-20251001",
        "claude-opus-4-6":           "claude-sonnet-4-6",
        "claude-haiku-4-5-20251001": "claude-haiku-4-5-20251001",
    }

    def __init__(self, budget_per_step: float = 0.01, ema_alpha: float = 0.3):
        self.budget      = budget_per_step
        self._alpha      = ema_alpha
        self._ema_cost   = 0.0
        self._ema_lat    = 0.0

    # ── 업데이트 ──────────────────────────────────────

    def update(self, cost: float, latency_ms: float):
        """호출 후 EMA 갱신."""
        self._ema_cost = self._alpha * cost       + (1 - self._alpha) * self._ema_cost
        self._ema_lat  = self._alpha * latency_ms + (1 - self._alpha) * self._ema_lat

    # ── 속성 ─────────────────────────────────────────

    @property
    def pressure(self) -> float:
        return self._ema_cost / max(self.budget, 1e-9)

    @property
    def status(self) -> str:
        p = self.pressure
        if p < self.PRESSURE_WARN:     return "normal"
        if p < self.PRESSURE_OVERLOAD: return "warn"
        return "overload"

    # ── 결정 ─────────────────────────────────────────

    def suggest_model(self, preferred: str) -> str:
        """overload 상태면 더 저렴한 모델 반환."""
        if self.status == "overload":
            return self.DOWNGRADE_MAP.get(preferred, preferred)
        return preferred

    def routing_score(self, quality: float, cache_hit: bool) -> float:
        """
        Cost-aware routing score.
        고압박: cache_hit 중요, 저압박: quality 중요.
        """
        p = min(self.pressure, 1.0)
        return quality * (1 - p) + float(cache_hit) * p

    def should_block(self) -> bool:
        """예산 2배 초과 시 LLM 호출 차단."""
        return self.pressure > self.PRESSURE_BLOCK

    def report(self) -> str:
        return (
            f"CostRouter  status={self.status}  "
            f"pressure={self.pressure:.3f}  "
            f"ema_cost=${self._ema_cost:.6f}  "
            f"ema_lat={self._ema_lat:.1f}ms  "
            f"budget=${self.budget:.4f}"
        )

    # ── sub-4 Stage 4 FR-18: 4-Level 의사결정 ─────────
    # Design Ref: docs/02-design/features/htp-thalamus-car.sub-4.design.md §3-3
    # 기존 7-method (update/pressure/status/suggest_model/routing_score/should_block/report)
    # 모두 보존. select_level 만 추가 (C-3).

    LEVEL_LOCAL    = 1   # TF-IDF / keyword 매칭 — 비용 0
    LEVEL_SLLM     = 2   # sLLM (e5-small 등) — 로컬 384-dim 임베딩
    LEVEL_API_SMALL = 3  # API 소형 — Haiku
    LEVEL_API_LARGE = 4  # API 대형 — Sonnet / Opus

    def select_level(self, query_complexity: float = 0.5) -> int:
        """4-Level 의사결정 — 비용 압박 × 쿼리 복잡도.

        Plan §SUCCESS Level 1-2 비율 70% 목표 → 압박 시 하향, 단순 쿼리는 하향.

        Returns
        -------
        int : LEVEL_LOCAL (1) ~ LEVEL_API_LARGE (4)
        """
        if not (0.0 <= query_complexity <= 1.0):
            raise ValueError(
                f"query_complexity ∈ [0,1], got {query_complexity}",
            )
        p = self.pressure
        # 비용 극압박 — 로컬 폴백
        if p > 0.8:
            return self.LEVEL_LOCAL
        # 압박 + 단순 쿼리 → sLLM
        if p > 0.5 and query_complexity < 0.5:
            return self.LEVEL_SLLM
        # 복잡 쿼리 → 대형 모델 (압박 없을 때만 의미 있음)
        if query_complexity > 0.8:
            return self.LEVEL_API_LARGE
        # 중간 쿼리 → API 소형
        if query_complexity > 0.5:
            return self.LEVEL_API_SMALL
        # default — sLLM (대부분 쿼리)
        return self.LEVEL_SLLM
