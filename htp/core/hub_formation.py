"""
HubFormationEngine  —  Hebbian + Oja + PageRank 기반 허브 형성.

Design Ref: docs/02-design/features/htp-review-improvements.design.md §3 Step 4
Plan SC: FR-06 (HubFormationEngine 파일 분리)

이 파일은 htp/core/ 트리에 속하므로 htp/runtime/* 를 import 하지 않는다 (DAG 강제).
의존: torch, htp.core.weight_matrix, htp.core.config (모두 DAG 안전).

알고리즘:
  1. Graph Laplacian Diffusion 신호 전파 (W.T direction, D_in/D_out 분리 정규화)
  2. Oja's Rule 가중치 업데이트  (W[u][v] = W[pre][post] convention)
  3. PageRank 기반 허브 감지     (pr × N > hub_pr_threshold)
"""
from __future__ import annotations

import torch

from .config        import HubConfig
from .weight_matrix import WeightMatrix


class HubFormationEngine:
    """
    헤비안 학습 + 허브 승격.

    매 스텝:
      1. 입력 신호 전파  ->  발화 노드 결정
      2. 함께 발화한 쌍의 연결 강화 (Hebbian + Oja)
      3. PageRank 점수 > hub_pr_threshold 이면 허브 승격
    """

    def __init__(self, wm: WeightMatrix, cfg: HubConfig):
        # Design Ref: htp-review-improvements §3 Step 2 — Constructor DI
        # n_nodes / device 는 wm 에서 파생 (shared field 는 sub-config 에 복제 안 함)
        self.wm  = wm
        self.cfg = cfg                        # HubConfig (Phase 1 step-2)
        self.dev = wm.dev

        self.is_hub    = torch.zeros(wm.n, dtype=torch.bool, device=self.dev)
        self.fire_count= torch.zeros(wm.n, device=self.dev)
        self.step_count= 0

        # 허브 승격/강등 이벤트 로그
        self.hub_events: list[dict] = []

    def step(self, signal: torch.Tensor) -> torch.Tensor:
        """
        신호 전파 -> 발화 -> 헤비안 업데이트 -> 허브 감지.
        반환: fired [N] bool tensor
        """
        self.step_count += 1
        W = self.wm.W

        # 1. Graph Laplacian Diffusion 전파 (directed edges: W[u][v] = u→v)
        #    신호가 엣지 방향을 따라 전파되려면 W.T 를 써야 한다:
        #    (W.T @ s)[v] = Σ_u W[u][v]·s[u]  = v가 in-neighbors 로부터 받는 합
        #    D_out(u) = Σ_v W[u][v]  를 u 측에서 정규화, D_in(v) = Σ_u W[u][v] 를 v 측에서 정규화.
        D_out_inv_sqrt = W.sum(dim=1).clamp(min=1e-8).pow(-0.5)   # [N] out-degree
        D_in_inv_sqrt  = W.sum(dim=0).clamp(min=1e-8).pow(-0.5)   # [N] in-degree
        propagated     = D_in_inv_sqrt * (W.T @ (D_out_inv_sqrt * signal))
        dt         = 0.5
        energy     = (1 - dt) * signal + dt * propagated
        fired      = (energy > self.cfg.threshold).float()

        self.fire_count += fired
        self.wm.record_fire(fired)

        # 2. Oja's Rule: Δw_ij = η * y_i * (x_j - y_i * w_ij)  (textbook: W[post][pre])
        #    코드 컨벤션 W[u][v] = u→v 엣지 (u=pre, v=post) 이므로 textbook 대비 transpose.
        #    강화 텀 : W[u][v] += η · signal[u] · fired[v]      = pre·post 공동발화
        #    정규화  : W[u][v] -= η · fired[v]² · W[u][v]        = post² normalization
        #    효과    : 허브 폭발 방지, PCA 1st PC 방향으로 수렴
        pre  = signal                                        # [N] pre-synaptic
        post = fired                                         # [N] post-synaptic
        oja  = torch.outer(pre, post) - (post * post).unsqueeze(0) * W  # [N,N]
        oja.fill_diagonal_(0)
        self.wm.W += self.cfg.hebbian_lr * oja
        self.wm.W.clamp_(0, 1)

        # 3. 허브 감지 — PageRank 기반 (LeCun review A2)
        #    pr 는 확률 분포 (합=1) 이므로 노드 수에 독립적인 threshold 를 위해
        #    pr * N 을 비교 — "1/N 대비 몇 배" 중심성인지 판단.
        prev_hub = self.is_hub.clone()
        pr       = self.pagerank()
        pr_rel   = pr * self.wm.n                          # [N] 평균 대비 배수
        self.is_hub = pr_rel > self.cfg.hub_pr_threshold

        # 이벤트 로깅
        promoted = (~prev_hub & self.is_hub).nonzero(as_tuple=True)[0].tolist()
        demoted  = (prev_hub & ~self.is_hub).nonzero(as_tuple=True)[0].tolist()
        if promoted:
            self.hub_events.append({"step": self.step_count, "type": "promote", "nodes": promoted})
        if demoted:
            self.hub_events.append({"step": self.step_count, "type": "demote",  "nodes": demoted})

        return fired

    def hub_indices(self) -> list[int]:
        return self.is_hub.nonzero(as_tuple=True)[0].tolist()

    def pagerank(self, alpha: float = 0.85,
                 tol: float = 1e-5, max_iter: int = 30) -> torch.Tensor:
        """
        PageRank (Power Iteration) — W[u][v] = u→v 엣지 convention.

        표준 수식:
            PR(v) = (1-α)/N + α · Σ_{u: u→v} PR(u) / out_deg(u)

        out-degree 정규화를 쓴다 (발신자 u 의 rank 가 out-link 수만큼 분배).
        행렬 형태: M[v][u] = W[u][v] / out_deg(u)  →  r = α·M·r + tp
                  M.T[u][v] = W[u][v] / out_deg(u)  →  M = (W / out_deg).T

        반환: [N] 노드 중요도 벡터 (합=1), 허브-of-hub 감지용.
        """
        W       = self.wm.W
        N       = W.shape[0]
        out_deg = W.sum(dim=1)                                 # [N] 발신자 out-degree
        dangling_mask = (out_deg < 1e-8)                       # terminal / 끊긴 노드
        out_safe = out_deg.clamp(min=1e-8).unsqueeze(1)        # [N,1]
        Wr      = W / out_safe                                 # row-normalized
        r       = torch.ones(N, device=W.device) / N
        tp      = (1 - alpha) / N
        for _ in range(max_iter):
            # dangling 노드의 rank 는 균등 재분배 (표준 PR 처리, 누설 방지)
            dangling_r = float(r[dangling_mask].sum()) / N if bool(dangling_mask.any()) else 0.0
            r_new = alpha * (Wr.T @ r + dangling_r) + tp
            if (r_new - r).abs().max() < tol:
                break
            r = r_new
        return r

    def top_hubs(self, k: int = 5) -> list[tuple[int, float]]:
        """PageRank 기반 상위 허브 반환 (토폴로지 반영)."""
        pr   = self.pagerank()
        vals, idx = torch.topk(pr, k=min(k, len(pr)))
        return [(int(i), float(v)) for i, v in zip(idx.tolist(), vals.tolist())]


__all__ = ["HubFormationEngine"]
