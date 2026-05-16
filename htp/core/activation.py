"""
ActivationEngine  —  캐스케이드 전파 + 시맨틱 배제 라우팅.

Design Ref: docs/02-design/features/htp-review-improvements.design.md §3 Step 6
Plan SC: FR-08 (ActivationEngine 파일 분리), FR-03 (Constructor DI — ActivationConfig)

이 파일은 htp/core/ 트리에 속하므로 htp/runtime/* 를 import 하지 않는다 (DAG 강제).
의존: torch, htp.core.{config, weight_matrix, hub_formation}.

내용:
  - Node / RunResult dataclass (사용자 함수의 노드 wrapping)
  - tag() / terminal() 데코레이터 (시맨틱 라우팅 메타데이터)
  - FIRE_FLOOR 상수 (캐스케이드 종료 신호 임계값)
  - ActivationEngine (HFE.step 호출 + 시맨틱 필터링)

⚠️ 12/12 라우팅 회귀의 핵심 — _extract() 의 dict-value split (Stage 1 bug #3)
   과 _semantic_filter() 의 prev-tag 매칭 로직이 모두 여기에 있음.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing      import Callable, Any, Optional, Set

import torch

from .config        import ActivationConfig
from .hub_formation import HubFormationEngine
from .weight_matrix import WeightMatrix


# ══════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════

FIRE_FLOOR = 0.08    # 캐스케이드 진행 가능한 최소 신호 강도


# ══════════════════════════════════════════════════════════
# Node / RunResult dataclass
# ══════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════
# Decorators — tag / terminal
# ══════════════════════════════════════════════════════════

def tag(*tags):
    """시맨틱 라우팅 태그 데코레이터.

    예: @tag("success", "done") → 입력 데이터의 라벨/키워드와 매칭되면 boost.
    """
    def decorator(fn):
        fn._htp_tags = set(t.lower() for t in tags)
        return fn
    return decorator


def terminal(fn):
    """캐스케이드 종착점 데코레이터.

    이 노드가 발화하면 그 스텝에서 캐스케이드 종료.
    """
    fn._htp_terminal = True
    return fn


# ══════════════════════════════════════════════════════════
# ActivationEngine
# ══════════════════════════════════════════════════════════

class ActivationEngine:
    """
    캐스케이드 전파 + 시맨틱 배제 라우팅.
    HubFormationEngine 의 step() 을 호출해 학습도 함께 진행.
    """

    def __init__(self,
                 wm:  WeightMatrix,
                 hfe: HubFormationEngine,
                 cfg: ActivationConfig):
        # Design Ref: htp-review-improvements §3 Step 6 — Constructor DI
        # cfg 는 ActivationConfig (was HTPConfig). device 는 wm 에서 파생.
        self.wm  = wm
        self.hfe = hfe
        self.cfg = cfg
        self._nodes: list[Node] = []

    def register(self, nodes: list[Node]):
        self._nodes = nodes

    def run(self,
            data: Any,
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
        dev = self.wm.dev                       # was self.cfg.device — Step 6 DI
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
        # ⚠️ Stage 1 bug #3 수정 — dict value 공백 split.
        # dict-value 자체를 한 덩어리로 쓰면 tag 매칭 확률이 급감.
        label = ""
        kws: set[str] = set()
        if isinstance(data, dict):
            label = str(data.get("label", "")).lower()
            for v in data.values():
                if isinstance(v, str):
                    kws.update(v.lower().split())
        elif isinstance(data, str):
            kws = set(data.lower().split())
        return label, kws


__all__ = [
    "ActivationEngine",
    "Node",
    "RunResult",
    "tag",
    "terminal",
    "FIRE_FLOOR",
]
