"""
HTP Runtime  -  Hub Topology Programming
=========================================

세 엔진이 하나의 WeightMatrix(W)를 공유하며 협력.

  WeightMatrix        W[u][v] = u->v 연결 강도 (단일 소유)
  HubFormationEngine  헤비안 학습 + 허브 승격
  PruningEngine       4가지 가지치기 전략 + 허브 보호
  ActivationEngine    캐스케이드 전파 + 시맨틱 라우팅
  HTPRuntime          세 엔진 통합 오케스트레이터

사용법:
  rt = HTPRuntime()

  @rt.node
  def parse(data): ...

  @rt.node
  @tag("success")
  def to_cache(data): ...

  @rt.node
  @terminal
  def log_result(data): ...

  rt.connect(parse, to_cache)
  result = rt.run(data, entry=parse)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Any, Optional, Set, List, Dict, Tuple
from enum import Enum
import torch

from htp.core.config        import HubConfig, PruneConfig, ActivationConfig
from htp.core.weight_matrix import WeightMatrix  # Re-exported below (Step 3)


# ======================================================
# 설정 (Facade — Design Ref: htp-review-improvements §2.3, Step 1)
# ======================================================
#
# HTPConfig 는 sub-config(HubConfig/PruneConfig/ActivationConfig) 들을 묶는 facade.
# Backward compatibility 를 위해 다음을 보존한다:
#   - flat 키워드 생성자:  HTPConfig(hub_pr_threshold=3.0)  → self.hub.hub_pr_threshold = 3.0
#   - flat 속성 접근:      cfg.hub_pr_threshold              → self.hub.hub_pr_threshold (via __getattr__)
#   - top-level 필드:      n_nodes, device 는 모든 엔진이 공유하므로 facade 본체에 유지
#
# 새 권장 사용 방식 (옵션):
#   HTPConfig(hub=HubConfig(hub_pr_threshold=3.0))
#
class HTPConfig:
    """
    Facade over Phase 1 sub-configs.

    Top-level fields (shared across engines):
      - n_nodes: int      네트워크 노드 수
      - device:  str      torch device ('cpu' or 'cuda')

    Sub-configs (engine-specific):
      - hub:        HubConfig         HubFormationEngine 파라미터
      - prune:      PruneConfig       PruningEngine 파라미터
      - activation: ActivationConfig  ActivationEngine 파라미터 (현재 비어있음)

    Backward-compat constructors (모두 동등):
        HTPConfig()
        HTPConfig(n_nodes=128)
        HTPConfig(hub_pr_threshold=3.0)             # flat 키워드 → self.hub 로 위임
        HTPConfig(hub=HubConfig(hub_pr_threshold=3.0))   # 새 권장 방식
    """

    __slots__ = ("n_nodes", "device", "hub", "prune", "activation")

    def __init__(self,
                 n_nodes:    int = 64,
                 device:     Optional[str] = None,
                 hub:        Optional[HubConfig]        = None,
                 prune:      Optional[PruneConfig]      = None,
                 activation: Optional[ActivationConfig] = None,
                 **kwargs: Any):
        # Top-level shared
        object.__setattr__(self, "n_nodes", n_nodes)
        object.__setattr__(self, "device",
                           device if device is not None
                           else ("cuda" if torch.cuda.is_available() else "cpu"))
        # Sub-configs (factory defaults)
        object.__setattr__(self, "hub",        hub        if hub        is not None else HubConfig())
        object.__setattr__(self, "prune",      prune      if prune      is not None else PruneConfig())
        object.__setattr__(self, "activation", activation if activation is not None else ActivationConfig())

        # Flat-keyword backward compat:
        # dispatch each unknown kwarg to the first sub-config that owns it.
        for k, v in kwargs.items():
            assigned = False
            for sub in (self.hub, self.prune, self.activation):
                if hasattr(sub, k):
                    setattr(sub, k, v)
                    assigned = True
                    break
            if not assigned:
                raise TypeError(
                    f"HTPConfig got unexpected keyword argument: {k!r}. "
                    f"Known top-level: n_nodes/device/hub/prune/activation. "
                    f"Known sub-config fields: "
                    f"{sorted(set(HubConfig.__dataclass_fields__) | set(PruneConfig.__dataclass_fields__) | set(ActivationConfig.__dataclass_fields__))}"
                )

    # ── Backward-compat attribute access ────────────────────
    # __getattr__ is only called when normal lookup fails (i.e., not in __slots__).
    # Forward unknown attribute reads to the appropriate sub-config.
    def __getattr__(self, name: str) -> Any:
        # NOTE: __slots__ fields go through normal lookup, not here.
        sub_dict = (
            ("hub",        HubConfig.__dataclass_fields__),
            ("prune",      PruneConfig.__dataclass_fields__),
            ("activation", ActivationConfig.__dataclass_fields__),
        )
        for sub_name, fields in sub_dict:
            if name in fields:
                return getattr(getattr(self, sub_name), name)
        raise AttributeError(f"'HTPConfig' object has no attribute {name!r}")

    def __setattr__(self, name: str, value: Any) -> None:
        # Allow direct set of __slots__ fields
        if name in ("n_nodes", "device", "hub", "prune", "activation"):
            object.__setattr__(self, name, value)
            return
        # Flat field assignment: route to the sub-config that owns it
        sub_dict = (
            ("hub",        HubConfig.__dataclass_fields__),
            ("prune",      PruneConfig.__dataclass_fields__),
            ("activation", ActivationConfig.__dataclass_fields__),
        )
        for sub_name, fields in sub_dict:
            if name in fields:
                setattr(object.__getattribute__(self, sub_name), name, value)
                return
        raise AttributeError(
            f"Cannot set unknown HTPConfig attribute: {name!r}. "
            f"To add new fields, edit the appropriate sub-config in htp/core/config.py"
        )

    def __repr__(self) -> str:
        return (f"HTPConfig(n_nodes={self.n_nodes}, device={self.device!r}, "
                f"hub={self.hub}, prune={self.prune}, activation={self.activation})")


# ======================================================
# WeightMatrix  -  htp/core/weight_matrix.py 로 이전됨 (Step 3)
# 위 ``from htp.core.weight_matrix import WeightMatrix`` 가 re-export 역할.
# 사용자 코드의 `from htp.runtime.htp_runtime import WeightMatrix` 는 그대로 동작.
# ======================================================


# ======================================================
# Hub Formation Engine
# ======================================================

class HubFormationEngine:
    """
    헤비안 학습 + 허브 승격.

    매 스텝:
      1. 입력 신호 전파  ->  발화 노드 결정
      2. 함께 발화한 쌍의 연결 강화 (Hebbian)
      3. PageRank 점수 > hub_pr_threshold 이면 허브 승격
    """

    def __init__(self, wm: WeightMatrix, cfg: HubConfig):
        # Design Ref: htp-review-improvements §3 Step 2 — Constructor DI 전환
        # n_nodes / device 는 wm 에서 파생 (shared field 는 sub-config 에 복제하지 않음)
        self.wm  = wm
        self.cfg = cfg                        # HubConfig (was HTPConfig)
        self.dev = wm.dev                     # was cfg.device

        self.is_hub    = torch.zeros(wm.n, dtype=torch.bool, device=self.dev)   # was cfg.n_nodes
        self.fire_count= torch.zeros(wm.n, device=self.dev)                     # was cfg.n_nodes
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


# ======================================================
# Pruning Engine  -  4가지 독립 전략 + 허브 보호
# ======================================================

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

    def __init__(self, wm: WeightMatrix, hfe: HubFormationEngine, cfg: HTPConfig):
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
        """허브 노드가 src 또는 dst인 모든 연결 마스크 [n, n]"""
        h = self.hfe.is_hub                      # [n] bool
        return h.unsqueeze(0) | h.unsqueeze(1)   # [n, n] bool

    def _apply_protection(self, mask: torch.Tensor) -> tuple[torch.Tensor, int]:
        """hub_protect 적용 후 (최종 마스크, protected 수) 반환"""
        if self.cfg.hub_protect:
            hm        = self._hub_mask()
            protected = int((mask & hm).sum().item())
            mask      = mask & ~hm
        else:
            protected = 0
        return mask, protected

    # ── [1] Decay ────────────────────────────────────

    def decay_prune(self) -> int:
        """
        매 스텝 감쇠 후 임계값 이하 제거.
        가장 기본적인 전략 - 항상 실행.
        """
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
        """
        N 스텝마다 실행.
        최근 window 내에서 사용 빈도가 낮은 엣지 제거.
        """
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


# ======================================================
# 노드 / 결과 / 데코레이터
# ======================================================

@dataclass
class Node:
    fn:         Callable
    node_id:    int
    name:       str   = ""
    call_count: int   = 0
    total_ms:   float = 0.0

    def __post_init__(self):
        self.name = self.fn.__name__

    def run(self, data: Any) -> Any:
        t0  = time.perf_counter()
        out = self.fn(data)
        self.total_ms += (time.perf_counter() - t0) * 1000
        self.call_count += 1
        return out

    @property
    def avg_ms(self) -> float:
        return self.total_ms / self.call_count if self.call_count else 0.0


@dataclass
class RunResult:
    input_data:  Any
    route_path:  list
    outputs:     dict
    hub_ids:     list
    pruned:      dict
    total_ms:    float = 0.0

    def summary(self) -> str:
        path = " -> ".join(n.name for n in self.route_path)
        hubs = [self.route_path[i].name
                for i, n in enumerate(self.route_path)
                if n.node_id in self.hub_ids]
        return f"{path}  hubs={hubs}  ({self.total_ms:.1f}ms)"


def tag(*tags):
    """시맨틱 라우팅 태그 데코레이터."""
    def decorator(fn):
        fn._htp_tags = set(t.lower() for t in tags)
        return fn
    return decorator


def terminal(fn):
    """캐스케이드 종착점 데코레이터."""
    fn._htp_terminal = True
    return fn


# ======================================================
# Activation Engine
# ======================================================

FIRE_FLOOR = 0.08

class ActivationEngine:
    """
    캐스케이드 전파 + 시맨틱 배제 라우팅.
    HubFormationEngine 의 step() 을 호출해 학습도 함께 진행.
    """

    def __init__(self, wm: WeightMatrix,
                 hfe: HubFormationEngine,
                 cfg: HTPConfig):
        self.wm  = wm
        self.hfe = hfe
        self.cfg = cfg
        self._nodes: list[Node] = []

    def register(self, nodes: list[Node]):
        self._nodes = nodes

    def run(self, data: Any,
            entry: Optional[Node] = None,
            max_depth: int = 8) -> tuple[list, dict, list]:
        """
        반환: (route_path, outputs, hub_ids)
        """
        fired_n = []
        outputs = {}
        path    = []
        visited : Set[int] = set()
        hub_ids : list     = []
        current = data

        # entry 강제 실행
        if entry:
            current = entry.run(current)
            fired_n.append(entry)
            outputs[entry.name] = current
            path.append(entry)
            visited.add(entry.node_id)
            prev = {entry.node_id}
        else:
            prev = set()

        # 캐스케이드
        for _ in range(max_depth):
            sig = self._make_signal(current, prev, visited)
            if sig.max().item() < FIRE_FLOOR:
                break

            fired_mask = self.hfe.step(sig)
            hub_ids    = self.hfe.hub_indices()

            raw_ids = [
                int(i)
                for i in fired_mask.nonzero(as_tuple=True)[0].tolist()
                if int(i) not in visited and int(i) < len(self._nodes)
            ]
            if not raw_ids:
                break

            final_ids = self._semantic_filter(raw_ids, prev, current, visited)
            if not final_ids:
                break

            prev = set()
            terminal_fired = False
            for idx in final_ids:
                n = self._nodes[idx]
                try:
                    out = n.run(current)
                    fired_n.append(n)
                    outputs[n.name] = out
                    path.append(n)
                    visited.add(idx)
                    prev.add(idx)
                    if out is not None:
                        current = out
                    if getattr(n.fn, "_htp_terminal", False):
                        terminal_fired = True
                except Exception as e:
                    print(f"  [warn] {n.name}: {e}")

            if terminal_fired:
                break

        return path, outputs, hub_ids

    def _make_signal(self, data, prev_ids, visited):
        N   = len(self._nodes)
        dev = self.cfg.device
        sig = torch.zeros(N, device=dev)

        for uid in prev_ids:
            sig[uid] = 1.0

        label, kws = self._extract(data)
        for n in self._nodes:
            ntags = getattr(n.fn, "_htp_tags", set())
            if not ntags:
                continue
            overlap = ntags & (kws | {label})
            if overlap:
                boost = min(0.4 + 0.20 * len(overlap), 0.90)
                if sig[n.node_id].item() < boost:
                    sig[n.node_id] = boost

        return sig.clamp(0.0, 1.0)

    def _semantic_filter(self, candidate_ids, prev_ids, data, visited):
        label, kws = self._extract(data)
        if not label and not kws:
            return candidate_ids

        result = list(candidate_ids)
        for uid in prev_ids:
            rivals = [v for v in candidate_ids
                      if float(self.wm.W[uid][v]) > 0.01]
            if len(rivals) < 2:
                continue
            matched     = [v for v in rivals
                           if getattr(self._nodes[v].fn, "_htp_tags", set())
                           & (kws | {label})]
            not_matched = [v for v in rivals if v not in matched]
            if matched:
                for v in not_matched:
                    if v in result:
                        result.remove(v)
        return result

    def _extract(self, data):
        label = ""
        kws: set[str] = set()
        if isinstance(data, dict):
            label = str(data.get("label", "")).lower()
            # 각 문자열 값을 공백 split 하여 개별 키워드로 편입
            # (dict value 자체를 한 덩어리로 쓰면 tag 매칭 확률이 급감)
            for v in data.values():
                if isinstance(v, str):
                    kws.update(v.lower().split())
        elif isinstance(data, str):
            kws = set(data.lower().split())
        return label, kws


# ======================================================
# HTPRuntime  -  세 엔진 통합 오케스트레이터
# ======================================================

class HTPRuntime:
    """
    Hub Topology Programming 통합 런타임.

    사용법:
      rt = HTPRuntime()

      @rt.node
      def parse(data): ...

      @rt.node
      @tag("success")
      @terminal
      def log_result(data): ...

      rt.connect(parse, log_result)
      result = rt.run(my_data, entry=parse)
    """

    def __init__(self, config: Optional[HTPConfig] = None):
        self.cfg          = config or HTPConfig()
        self._nodes       : list[Node] = []
        self._name_map    : dict       = {}
        self._node_count              = 0
        self._built                   = False
        self._run_log     : list[RunResult] = []

        # 엔진 (빌드 전까지 None)
        self.wm  : Optional[WeightMatrix]       = None
        self.hfe : Optional[HubFormationEngine]  = None
        self.pe  : Optional[PruningEngine]       = None
        self.ae  : Optional[ActivationEngine]    = None

    # ── 노드 등록 ─────────────────────────────────

    def node(self, fn: Callable) -> Callable:
        n = Node(fn=fn, node_id=self._node_count)
        self._nodes.append(n)
        self._name_map[fn.__name__] = n
        self._node_count += 1
        fn._htp_node = n
        return fn

    # ── 연결 ──────────────────────────────────────

    def connect(self, src: Callable, dst: Callable,
                weight: float = 0.3) -> "HTPRuntime":
        self._ensure_built()
        u = src._htp_node.node_id
        v = dst._htp_node.node_id
        self.wm.set(u, v, weight)
        print(f"  {src.__name__} -> {dst.__name__}  w={weight:.2f}")
        return self   # 체이닝 가능

    # ── 실행 ──────────────────────────────────────

    def run(self, data: Any,
            entry: Optional[Callable] = None,
            max_depth: int = 8) -> RunResult:
        """
        단일 데이터 실행.
        캐스케이드 전파 -> 가지치기 -> 결과 반환.
        """
        self._ensure_built()
        t0 = time.perf_counter()

        entry_node = entry._htp_node if entry else None
        path, outputs, hub_ids = self.ae.run(data, entry_node, max_depth)

        # 매 실행 후 가지치기
        pruned = self.pe.run_all(self.hfe.step_count)

        result = RunResult(
            input_data = data,
            route_path = path,
            outputs    = outputs,
            hub_ids    = hub_ids,
            pruned     = pruned,
            total_ms   = (time.perf_counter() - t0) * 1000,
        )
        self._run_log.append(result)
        return result

    def batch(self, dataset: list, entry: Optional[Callable] = None) -> list:
        """배치 실행."""
        return [self.run(d, entry=entry) for d in dataset]

    # ── 상태 조회 ─────────────────────────────────

    def status(self):
        if not self._built:
            print("[HTP] Not built yet")
            return

        SEP = "=" * 66
        print(f"\n{SEP}")
        print("  HTPRuntime Status")
        print(SEP)
        print(f"  nodes      : {self._node_count}")
        print(f"  steps      : {self.hfe.step_count}")
        print(f"  edges      : {self.wm.edge_count()}")
        print(f"  hubs       : {int(self.hfe.is_hub.sum().item())}")
        print(f"  hub_protect: {self.cfg.hub_protect}")
        print()

        print(f"  {'node':<16} {'calls':>6}  {'avg_ms':>7}  {'strength':>9}  hub")
        print(f"  {'-'*52}")
        for n in self._nodes:
            hub   = "* HUB" if self.hfe.is_hub[n.node_id] else "     "
            instr = self.wm.in_strength(n.node_id)
            print(f"  {n.name:<16} {n.call_count:>6}  "
                  f"{n.avg_ms:>7.2f}  {instr:>9.3f}  {hub}")

        print()
        print(self.pe.report())

        # 상위 허브
        print()
        print("  [ Top Hubs ]")
        for nid, strength in self.hfe.top_hubs(5):
            if nid < len(self._nodes):
                name = self._nodes[nid].name
                bar  = "|" * int(strength * 6)
                print(f"  {name:<16}  str={strength:.3f}  {bar}")

        # 강화된 엣지
        print()
        print("  [ Reinforced Edges (w > 0.5) ]")
        for u in range(self._node_count):
            for v in range(self._node_count):
                w = self.wm.get(u, v)
                if w > 0.5:
                    print(f"  {self._nodes[u].name:<14} -> "
                          f"{self._nodes[v].name:<14}  w={w:.3f}")
        print(SEP)

    # ── 내부 ──────────────────────────────────────

    def _ensure_built(self):
        if self._built:
            return
        N = max(self._node_count, 1)
        self.cfg.n_nodes = N

        self.wm  = WeightMatrix(N, self.cfg.device)
        # Design Ref: htp-review-improvements §3 Step 2 — HFE 가 HubConfig 만 받음
        self.hfe = HubFormationEngine(self.wm, self.cfg.hub)
        self.pe  = PruningEngine(self.wm, self.hfe, self.cfg)     # Step 5에서 PruneConfig로 전환 예정
        self.ae  = ActivationEngine(self.wm, self.hfe, self.cfg)  # Step 6에서 ActivationConfig로 전환 예정
        self.ae.register(self._nodes)

        self._built = True
        print(f"[HTP] Runtime built  -  {N} nodes  "
              f"| hfe + pe(hub_protect={self.cfg.hub_protect}) + ae  on {self.cfg.device}")

    @property
    def log(self) -> list:
        return self._run_log


# ======================================================
# 데모
# ======================================================

def demo():
    SEP = "=" * 66
    print(SEP)
    print("  HTPRuntime  -  Hub Topology Programming")
    print("  HubFormationEngine + PruningEngine(4 strategies) + ActivationEngine")
    print(SEP)

    rt = HTPRuntime(HTPConfig(
        hebbian_lr      = 0.13,
        decay_rate      = 0.005,
        prune_threshold = 0.02,
        usage_window    = 15,
        usage_min       = 0.05,
        redundancy_cos  = 0.95,
        threshold       = 0.35,
        hub_protect     = True,
        age_threshold   = 50,
    ))

    @rt.node
    def parse(data):
        text = str(data).lower().strip()
        return {"text": text, "len": len(text)}

    @rt.node
    def classify(data):
        text = data.get("text", "") if isinstance(data, dict) else str(data)
        err = ["error", "fail", "bug", "timeout", "fatal", "oom"]
        ok  = ["success", "ok", "done", "deployed", "completed"]
        if any(w in text for w in err):
            return {**data, "label": "error"}
        if any(w in text for w in ok):
            return {**data, "label": "success"}
        return {**data, "label": "neutral"}

    @rt.node
    @tag("success", "ok", "done", "deployed", "completed", "cache")
    def to_cache(data):
        label = data.get("label", "") if isinstance(data, dict) else ""
        print(f"        [CACHE]  {label}")
        return {**data, "cached": True}

    @rt.node
    @tag("error", "fail", "bug", "timeout", "fatal", "oom")
    def to_alert(data):
        label = data.get("label", "") if isinstance(data, dict) else ""
        print(f"        [ALERT]  {label}  <- error!")
        return {**data, "alerted": True}

    @rt.node
    @terminal
    @tag("success", "error", "neutral", "cached", "alerted")
    def log_result(data):
        label   = data.get("label",   "") if isinstance(data, dict) else ""
        cached  = data.get("cached",  False)
        alerted = data.get("alerted", False)
        st = "cached" if cached else ("alerted" if alerted else "passed")
        print(f"        [LOG]    {label}  status={st}")
        return data

    print("\n[ connections ]")
    (rt.connect(parse,    classify,   weight=0.55)
       .connect(classify, to_cache,   weight=0.30)
       .connect(classify, to_alert,   weight=0.30)
       .connect(to_cache, log_result, weight=0.35)
       .connect(to_alert, log_result, weight=0.35))

    dataset = [
        ("success: task completed",    "success"),
        ("ok everything is fine",      "success"),
        ("error: connection failed",   "error"),
        ("done processing batch",      "success"),
        ("success: cache hit",         "success"),
        ("bug found in module",        "error"),
        ("ok all systems go",          "success"),
        ("error: timeout on db",       "error"),
        ("success: deployed v2",       "success"),
        ("fail: disk quota exceeded",  "error"),
        ("done all tasks complete",    "success"),
        ("fatal error: oom",           "error"),
    ]

    print("\n[ batch run ]\n")
    correct = 0
    for i, (text, expected) in enumerate(dataset, 1):
        result = rt.run(text, entry=parse)
        path   = " -> ".join(n.name for n in result.route_path)
        routed = ("to_cache" if "to_cache" in result.outputs
                  else "to_alert" if "to_alert" in result.outputs
                  else "?")
        ok = ((expected == "success" and routed == "to_cache") or
              (expected == "error"   and routed == "to_alert"))
        if ok:
            correct += 1
        mark = "O" if ok else "X"
        print(f"  [{i:02d}] {mark}  '{text}'")
        print(f"        {path}")

    total = len(dataset)
    print(f"\n  accuracy: {correct}/{total}  ({correct*100//total}%)")

    rt.status()


if __name__ == "__main__":
    demo()
