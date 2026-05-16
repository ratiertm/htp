"""
WeightMatrix  —  W 행렬 단일 소유, Phase 1 세 엔진이 참조.

Design Ref: docs/02-design/features/htp-review-improvements.design.md §3 Step 3
Plan SC: FR-05 (WeightMatrix 파일 분리)

이 파일은 htp/core/ 트리에 속하므로 htp/runtime/* 를 import 하지 않는다.
의존: torch 만 (DAG 강제, Design §2.2).
"""
from __future__ import annotations

import torch


class WeightMatrix:
    """
    연결 가중치 행렬 W[u][v] (u→v 엣지 강도).

    Phase 1 의 단일 소유 원칙:
      - 세 엔진(HubFormationEngine / PruningEngine / ActivationEngine)이
        이 객체의 *참조* 를 공유.
      - 쓰기는 HubFormationEngine(헤비안) + PruningEngine(제거).
      - 읽기는 ActivationEngine(전파 계산).

    Attributes
    ----------
    n : int
        현재 노드 수 (동적 확장 가능).
    dev : str
        torch device (e.g. "cpu", "cuda").
    W : torch.Tensor
        [n × n] 가중치 행렬. W[u][v] = u→v 엣지 강도.
    fire_history : list[torch.Tensor]
        usage pruning 용 발화 이력 (max 200 step).
    """

    def __init__(self, n: int, device: str):
        self.n   = n
        self.dev = device
        self.W   = torch.zeros(n, n, device=device)
        self._step = 0

        # 히스토리: 각 노드의 스텝별 발화 기록 (usage pruning용)
        self.fire_history: list[torch.Tensor] = []

    def set(self, u: int, v: int, w: float):
        self.W[u][v] = w

    def get(self, u: int, v: int) -> float:
        return float(self.W[u][v])

    def row(self, u: int) -> torch.Tensor:
        return self.W[u]

    def col(self, v: int) -> torch.Tensor:
        return self.W[:, v]

    def in_strength(self, v: int) -> float:
        return float(self.W[:, v].sum())

    def edge_count(self) -> int:
        return int((self.W > 0).sum().item())

    def record_fire(self, fired: torch.Tensor):
        """발화 기록 저장 (usage pruning용)."""
        self._step += 1
        self.fire_history.append(fired.clone())
        if len(self.fire_history) > 200:
            self.fire_history.pop(0)

    def recent_fire_rate(self, node_id: int, window: int) -> float:
        """최근 window 스텝 내 발화 비율."""
        if not self.fire_history:
            return 0.0
        recent = self.fire_history[-window:]
        fires  = sum(float(f[node_id]) for f in recent if node_id < len(f))
        return fires / len(recent)


__all__ = ["WeightMatrix"]
