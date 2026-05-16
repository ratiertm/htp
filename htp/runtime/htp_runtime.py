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

from htp.core.config        import (                             # Re-exported (Step 1+7)
    HubConfig,
    PruneConfig,
    ActivationConfig,
    HTPConfig,
)
from htp.core.weight_matrix import WeightMatrix                  # Re-exported (Step 3)
from htp.core.hub_formation import HubFormationEngine            # Re-exported (Step 4)
from htp.core.pruning       import PruningEngine, PruneStrategy  # Re-exported (Step 5)
from htp.core.activation    import (                             # Re-exported (Step 6)
    ActivationEngine,
    Node,
    RunResult,
    tag,
    terminal,
    FIRE_FLOOR,
)


# ──────────────────────────────────────────────────────
# Phase 1 components have all been moved to htp/core/* (Steps 1-7):
#   HTPConfig + sub-configs   → htp/core/config.py        (Step 1+7)
#   WeightMatrix              → htp/core/weight_matrix.py (Step 3)
#   HubFormationEngine        → htp/core/hub_formation.py (Step 4)
#   PruningEngine + Strategy  → htp/core/pruning.py       (Step 5)
#   ActivationEngine + Node + tag/terminal + FIRE_FLOOR
#                             → htp/core/activation.py    (Step 6)
#
# This file (htp_runtime.py) now contains only:
#   - HTPRuntime orchestrator
#   - demo() function
#   - Re-exports for backward compatibility
# ──────────────────────────────────────────────────────


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
        # Design Ref: htp-review-improvements §3 Step 2/5/6 — Constructor DI 완료
        self.hfe = HubFormationEngine(self.wm, self.cfg.hub)
        self.pe  = PruningEngine(self.wm, self.hfe, self.cfg.prune)
        self.ae  = ActivationEngine(self.wm, self.hfe, self.cfg.activation)
        self.ae.register(self._nodes)

        self._built = True
        print(f"[HTP] Runtime built  -  {N} nodes  "
              f"| hfe + pe(hub_protect={self.cfg.hub_protect}) + ae  on {self.cfg.device}")

    @property
    def log(self) -> list:
        return self._run_log


# ======================================================
# 데모 — htp/runtime/_demo.py 로 분리됨 (Step 7, ≤250줄 SUCCESS 기준)
# ======================================================
# 사용자 코드 호환을 위해 re-export:
from ._demo import demo  # noqa: E402


if __name__ == "__main__":
    demo()
