"""
PruningEngine  —  4가지 가지치기 전략 + 허브 보호.

Design Ref: docs/02-design/features/htp-review-improvements.design.md §3 Step 5
Plan SC: FR-07 (PruningEngine 파일 분리), FR-03 (Constructor DI — PruneConfig)

이 파일은 htp/core/ 트리에 속하므로 htp/runtime/* 를 import 하지 않는다 (DAG 강제).
의존: torch, htp.core.{config, weight_matrix, hub_formation}.

알고리즘 (4 전략):
  [1] Decay      — 매 스텝 감쇠 + 임계값 이하 제거
  [2] Usage      — N 스텝마다 발화율 낮은 엣지 제거
  [3] Redundancy — N 스텝마다 코사인 유사도 높은 중복 엣지 제거
  [4] Age        — N 스텝마다 오래 방치된 엣지 제거

생물학: 미세아교세포(Microglia) 시냅스 가지치기.
"""
from __future__ import annotations

from enum   import Enum
from typing import Optional

import torch

from .config        import PruneConfig
from .hub_formation import HubFormationEngine
from .weight_matrix import WeightMatrix


class PruneStrategy(Enum):
    DECAY    = "decay"      # 시간 감쇠
    USAGE    = "usage"      # 사용 빈도
    REDUND   = "redundancy" # 중복 경로
    AGE      = "age"        # 연결 나이 (마지막 강화 이후 스텝)


class PruningEngine:
    """
    4가지 가지치기 전략으로 네트워크를 정제.

    [1] Decay Pruning    - 매 스텝 감쇠, 임계값 이하 제거
    [2] Usage Pruning    - 최근 N 스텝 동안 거의 안 쓰인 엣지 제거
    [3] Redundancy Pruning - 두 노드의 연결 패턴이 너무 비슷하면
                             약한 쪽 제거 (중복 경로 정리)
    [4] Age Pruning      - 마지막 강화 이후 오래된 연결 제거

    hub_protect=True 이면 허브 노드 관여 연결은 제거하지 않음.
    """

    def __init__(self, wm: WeightMatrix, hfe: HubFormationEngine, cfg: PruneConfig):
        # Design Ref: htp-review-improvements §3 Step 5 — Constructor DI 전환
        # cfg 는 PruneConfig (was HTPConfig). hub_protect 등 모든 필드 동일.
        self.wm  = wm
        self.hfe = hfe
        self.cfg = cfg

        self.stats = {
            PruneStrategy.DECAY:  {"runs": 0, "pruned": 0},
            PruneStrategy.USAGE:  {"runs": 0, "pruned": 0},
            PruneStrategy.REDUND: {"runs": 0, "pruned": 0},
            PruneStrategy.AGE:    {"runs": 0, "pruned": 0},
        }
        self.prune_log: list[dict] = []

        # age 전략 내부 상태
        self._age_matrix: Optional[torch.Tensor] = None
        self._W_prev:     Optional[torch.Tensor] = None

    # ── 허브 보호 헬퍼 ──────────────────────────────

    def _hub_mask(self) -> torch.Tensor:
        """허브 노드가 src 또는 dst인 모든 연결 마스크 [n, n]."""
        h = self.hfe.is_hub                      # [n] bool
        return h.unsqueeze(0) | h.unsqueeze(1)   # [n, n] bool

    def _apply_protection(self, mask: torch.Tensor) -> tuple[torch.Tensor, int]:
        """hub_protect 적용 후 (최종 마스크, protected 수) 반환."""
        if self.cfg.hub_protect:
            hm        = self._hub_mask()
            protected = int((mask & hm).sum().item())
            mask      = mask & ~hm
        else:
            protected = 0
        return mask, protected

    # ── [1] Decay ────────────────────────────────────

    def decay_prune(self) -> int:
        """매 스텝 감쇠 후 임계값 이하 제거. 가장 기본적인 전략 - 항상 실행."""
        self.wm.W *= (1 - self.cfg.decay_rate)
        weak = self.wm.W < self.cfg.prune_threshold
        weak.fill_diagonal_(False)

        weak, _ = self._apply_protection(weak)

        count = int(weak.sum().item())
        self.wm.W[weak] = 0.0

        self.stats[PruneStrategy.DECAY]["runs"]   += 1
        self.stats[PruneStrategy.DECAY]["pruned"] += count
        return count

    # ── [2] Usage ────────────────────────────────────

    def usage_prune(self, step: int, interval: int = 10) -> int:
        """N 스텝마다 실행. 최근 window 내에서 사용 빈도가 낮은 엣지 제거."""
        if step % interval != 0:
            return 0

        W   = self.wm.W
        N   = self.wm.n
        win = self.cfg.usage_window

        for u in range(N):
            # 허브 노드의 출력 엣지는 약화하지 않음
            if self.cfg.hub_protect and self.hfe.is_hub[u]:
                continue
            ru = self.wm.recent_fire_rate(u, win)
            if ru < self.cfg.usage_min:
                mask = W[u] > 0
                mask[u] = False
                # 허브 목적지 엣지 보호
                if self.cfg.hub_protect:
                    mask = mask & ~self.hfe.is_hub
                W[u][mask] *= 0.85

        # threshold 이하 제거
        weak = (W > 0) & (W < self.cfg.prune_threshold)
        weak.fill_diagonal_(False)
        weak, _ = self._apply_protection(weak)

        count = int(weak.sum().item())
        W[weak] = 0.0

        self.stats[PruneStrategy.USAGE]["runs"]   += 1
        self.stats[PruneStrategy.USAGE]["pruned"] += count

        if count:
            self.prune_log.append({
                "step": step, "strategy": "usage", "pruned": count
            })
        return count

    # ── [3] Redundancy ───────────────────────────────

    def redundancy_prune(self, step: int, interval: int = 50) -> int:
        """
        N 스텝마다 실행.
        두 노드의 입력 패턴이 코사인 유사도 > threshold 이면
        연결 강도가 약한 쪽의 엣지를 제거.
        """
        if step % interval != 0:
            return 0

        W = self.wm.W
        N = self.wm.n
        pruned = 0

        # 입력 벡터 기준 유사도 계산
        col_norms = W.norm(dim=0, keepdim=True).clamp(min=1e-8)
        W_norm    = W / col_norms
        cos_sim   = torch.matmul(W_norm.T, W_norm)  # [N x N]
        cos_sim.fill_diagonal_(0)

        visited = set()
        for v in range(N):
            if v in visited:
                continue
            similar = (cos_sim[v] > self.cfg.redundancy_cos).nonzero(
                as_tuple=True)[0].tolist()
            similar = [u for u in similar if u != v and u not in visited]
            if not similar:
                continue
            for u in similar:
                str_v = float(W[:, v].sum())
                str_u = float(W[:, u].sum())
                weaker = u if str_u <= str_v else v

                # 허브 노드 관여 엣지는 제거하지 않음
                if self.cfg.hub_protect and self.hfe.is_hub[weaker]:
                    visited.add(u)
                    continue

                W[:, weaker] *= 0.7
                # 허브 소스에서 오는 엣지는 0으로 만들지 않음
                if self.cfg.hub_protect:
                    hub_src = self.hfe.is_hub
                    below   = (W[:, weaker] < self.cfg.prune_threshold) & ~hub_src
                else:
                    below   = W[:, weaker] < self.cfg.prune_threshold
                pruned_here = int(below.sum().item())
                W[:, weaker][below] = 0.0
                pruned += pruned_here
                visited.add(u)

        self.stats[PruneStrategy.REDUND]["runs"]   += 1
        self.stats[PruneStrategy.REDUND]["pruned"] += pruned

        if pruned:
            self.prune_log.append({
                "step": step, "strategy": "redundancy", "pruned": pruned
            })
        return pruned

    # ── [4] Age ──────────────────────────────────────

    def age_prune(self, step: int, interval: int = 20) -> int:
        """
        N 스텝마다 실행.
        마지막 강화 이후 age_threshold 스텝 이상 방치된 연결 제거.

        age matrix: 각 연결의 "마지막 강화 이후 경과 스텝"
          - 연결이 강화되면 -> 0 리셋
          - 연결이 살아있으면 -> +1
          - 연결이 소멸하면  -> 0
        """
        if step % interval != 0:
            return 0

        W   = self.wm.W
        dev = W.device

        # 첫 호출 시 상태 초기화
        if self._age_matrix is None:
            self._age_matrix = torch.zeros(W.shape[0], W.shape[0], device=dev)
            self._W_prev     = W.clone()

        # 마지막 prune 호출 이후 강화된 연결 -> 나이 0 리셋
        strengthened = W > self._W_prev
        self._age_matrix[strengthened] = 0

        # 살아있는 연결 +1, 소멸된 연결 0
        self._age_matrix[W > 0]  += 1
        self._age_matrix[W == 0]  = 0

        # W 스냅샷 갱신
        self._W_prev.copy_(W)

        old = self._age_matrix > self.cfg.age_threshold
        old, _ = self._apply_protection(old)

        count = int(old.sum().item())
        W[old] = 0.0
        self._age_matrix[old] = 0

        self.stats[PruneStrategy.AGE]["runs"]   += 1
        self.stats[PruneStrategy.AGE]["pruned"] += count

        if count:
            self.prune_log.append({
                "step": step, "strategy": "age", "pruned": count
            })
        return count

    # ── 통합 실행 ────────────────────────────────────

    def run_all(self, step: int) -> dict[str, int]:
        """네 전략 모두 실행. HTPRuntime이 매 스텝 호출."""
        return {
            "decay":  self.decay_prune(),
            "usage":  self.usage_prune(step),
            "redund": self.redundancy_prune(step),
            "age":    self.age_prune(step),
        }

    def report(self) -> str:
        lines = ["  [ Pruning Stats ]"]
        for s, d in self.stats.items():
            lines.append(
                f"  {s.value:<12}  runs={d['runs']:<5}  total_pruned={d['pruned']}"
            )
        return "\n".join(lines)


__all__ = ["PruningEngine", "PruneStrategy"]
