"""
Node Generation Engine  -  동적 노드 생성
==========================================

뇌의 신경발생(Neurogenesis)에 대응.
가지치기(Pruning)의 반대 방향 - 네트워크가 성장.

3가지 생성 전략:

  [1] hub_split()
      허브 노드가 과부하 -> 둘로 분열
      A(과부하) -> A' + A''  각각 절반의 연결을 상속
      뇌의 세포 분열 / 피질 컬럼 형성과 유사

  [2] sprout()
      입력 신호가 기존 노드와 잘 매칭 안 될 때
      새 탐색 노드를 약한 연결로 생성
      뇌의 새 수상돌기 발아(Dendritic Sprouting)와 유사

  [3] interpolate()
      두 노드 사이 라우팅이 자주 실패할 때
      중간에 새 중계 노드 삽입
      뇌의 인터뉴런(Interneuron) 생성과 유사
"""

from __future__ import annotations

import time
import math
from dataclasses import dataclass, field
from typing import Callable, Any, Optional, Set
from enum import Enum
import torch

# htp_runtime 에서 공통 클래스 임포트
from ..runtime.htp_runtime import (
    HTPConfig, WeightMatrix, HubFormationEngine,
    PruningEngine, ActivationEngine,
    Node, RunResult, tag, terminal,
    FIRE_FLOOR
)


# ══════════════════════════════════════════════════════════
# 생성 설정 (HTPConfig 확장)
# ══════════════════════════════════════════════════════════

@dataclass
class GenConfig:
    # hub_split 트리거
    split_strength_threshold: float = 5.0   # 허브 강도 임계값 (높여서 성급한 분열 방지)
    split_call_threshold:     int   = 15    # 최소 호출 횟수 (충분히 검증된 노드만 분열)
    split_cooldown:           int   = 30    # 재분열 최소 간격 (스텝) - 연쇄 분열 방지
    maturity_calls:           int   = 5     # 자식 노드가 분열 가능하려면 최소 이 횟수 호출되어야 함
    global_cooldown:          int   = 15    # 마지막 분열 후 시스템 전체 쿨다운 (스텝)

    # sprout 트리거
    sprout_miss_threshold:    int   = 3     # N번 연속 매칭 실패 시 발아
    sprout_init_weight:       float = 0.15  # 새 노드 초기 연결 강도

    # interpolate 트리거
    interp_fail_threshold:    int   = 4     # N번 연속 라우팅 실패 시 중계 삽입
    interp_init_weight:       float = 0.25  # 중계 노드 초기 연결 강도

    # 전체 제한
    max_nodes:                int   = 128   # 최대 노드 수
    max_gen_per_run:          int   = 1     # 한 번 run() 에서 최대 1개 생성 (연쇄 방지)


# ══════════════════════════════════════════════════════════
# 생성 이벤트 로그
# ══════════════════════════════════════════════════════════

class GenStrategy(Enum):
    SPLIT       = "split"
    SPROUT      = "sprout"
    INTERPOLATE = "interpolate"


@dataclass
class GenEvent:
    step:       int
    strategy:   GenStrategy
    trigger:    str          # 트리거 설명
    new_node_id:int
    new_node_name: str
    parent_ids: list         # 분열/삽입 시 부모 노드


# ══════════════════════════════════════════════════════════
# Node Generation Engine
# ══════════════════════════════════════════════════════════

class NodeGenerationEngine:
    """
    세 가지 전략으로 새 노드를 동적 생성.
    생성된 노드는 WeightMatrix와 노드 목록에 즉시 반영.
    """

    def __init__(self, wm: WeightMatrix,
                 hfe: HubFormationEngine,
                 cfg: HTPConfig,
                 gen_cfg: GenConfig):
        self.wm      = wm
        self.hfe     = hfe
        self.cfg     = cfg
        self.gcfg    = gen_cfg
        self._nodes  : list[Node] = []   # 런타임과 공유 참조
        self._stats  = {s: 0 for s in GenStrategy}
        self.events  : list[GenEvent] = []

        # 상태 추적
        self._split_last       : dict[int, int] = {}  # {node_id: last_split_step}
        self._global_last_split: int = -9999           # 마지막 분열 스텝 (전역)
        self._miss_count       : int = 0               # 연속 매칭 실패 횟수
        self._fail_pairs       : dict[tuple, int] = {} # {(u,v): fail_count}
        self._immature         : set = set()           # 아직 성숙 안 된 자식 노드 ID

    def register(self, nodes: list[Node]):
        """HTPRuntime 의 노드 목록과 참조 공유"""
        self._nodes = nodes

    # ── [1] Hub Split ─────────────────────────────────

    def check_split(self, step: int) -> list[Node]:
        """
        과부하 허브 감지 -> 분열.

        분열 조건:
          - 허브 강도 > split_strength_threshold
          - 호출 횟수 > split_call_threshold  (충분히 검증된 노드만)
          - 자식 노드가 아닌 것 (immature 아닌 것)
          - 마지막 분열 후 split_cooldown 스텝 경과  (노드별)
          - 전체 마지막 분열 후 global_cooldown 경과  (연쇄 분열 방지)
        """
        if len(self._nodes) >= self.gcfg.max_nodes:
            return []

        # 전역 쿨다운: 최근 분열 후 시스템 전체가 안정화될 시간 확보
        if step - self._global_last_split < self.gcfg.global_cooldown:
            return []

        new_nodes = []
        gcfg = self.gcfg

        for n in list(self._nodes):
            nid = n.node_id

            # 성숙 조건: 직접 생성된 자식은 충분히 호출돼야 분열 가능
            if nid in self._immature:
                if n.call_count < gcfg.maturity_calls:
                    continue
                else:
                    self._immature.discard(nid)  # 성숙 완료

            # 허브 조건 - calls=0 이면 좀비 허브이므로 제외
            if not self.hfe.is_hub[nid] or n.call_count == 0:
                continue

            instr = self.wm.in_strength(nid)
            if instr < gcfg.split_strength_threshold:
                continue
            if n.call_count < gcfg.split_call_threshold:
                continue
            last = self._split_last.get(nid, -9999)
            if step - last < gcfg.split_cooldown:
                continue

            # 분열 실행 (한 번에 하나만)
            child_a, child_b = self._do_split(n, step)
            new_nodes.extend([child_a, child_b])
            self._split_last[nid]   = step
            self._global_last_split = step
            self._stats[GenStrategy.SPLIT] += 2
            break  # 이번 스텝은 하나만

        return new_nodes

    def _do_split(self, parent: Node, step: int) -> tuple:
        """
        부모 노드를 두 자식으로 분열.
        연결을 강도 기준으로 절반씩 상속.
        """
        N   = self.wm.n
        W   = self.wm.W
        pid = parent.node_id

        # 들어오는 연결 강도 순으로 정렬
        in_weights  = [(u, float(W[u][pid])) for u in range(N) if float(W[u][pid]) > 0]
        out_weights = [(v, float(W[pid][v])) for v in range(N) if float(W[pid][v]) > 0]
        in_weights.sort(key=lambda x: -x[1])
        out_weights.sort(key=lambda x: -x[1])

        mid_in  = max(1, len(in_weights) // 2)
        mid_out = max(1, len(out_weights) // 2)

        # 자식 A: 상위 절반 연결
        child_a = self._create_node(
            f"{parent.name}_a",
            parent.fn,
            in_edges  = in_weights[:mid_in],
            out_edges = out_weights[:mid_out],
        )
        # 자식 B: 하위 절반 연결
        child_b = self._create_node(
            f"{parent.name}_b",
            parent.fn,
            in_edges  = in_weights[mid_in:],
            out_edges = out_weights[mid_out:],
        )

        # 두 자식 간 약한 단방향 연결 (순환 경로 방지)
        self.wm.set(child_a.node_id, child_b.node_id, 0.08)

        # 부모 연결 점진적 약화 (0.5로 유지 - 급격한 소멸 방지)
        self.wm.W[pid] *= 0.5
        self.wm.W[:, pid] *= 0.5

        # 자식을 immature(미성숙) 목록에 등록
        self._immature.add(child_a.node_id)
        self._immature.add(child_b.node_id)

        ev = GenEvent(
            step         = step,
            strategy     = GenStrategy.SPLIT,
            trigger      = f"{parent.name} overloaded (str={self.wm.in_strength(pid):.2f}, calls={parent.call_count})",
            new_node_id  = child_a.node_id,
            new_node_name= f"{child_a.name} + {child_b.name}",
            parent_ids   = [pid],
        )
        self.events.append(ev)
        print(f"  [GEN] SPLIT  {parent.name} -> {child_a.name} + {child_b.name}  (immature until {self.gcfg.maturity_calls} calls)")
        return child_a, child_b

    # ── [2] Sprout ────────────────────────────────────

    def check_sprout(self, signal: torch.Tensor,
                     fired_ids: list[int], step: int) -> Optional[Node]:
        """
        신호가 강한데 아무 노드도 발화하지 않으면 -> 새 탐색 노드 발아.
        """
        if len(self._nodes) >= self.gcfg.max_nodes:
            return None

        if signal.max().item() < 0.6:
            self._miss_count = 0
            return None
        if fired_ids:
            self._miss_count = 0
            return None

        self._miss_count += 1
        if self._miss_count < self.gcfg.sprout_miss_threshold:
            return None

        self._miss_count = 0

        top_signal_nodes = signal.topk(min(3, len(self._nodes))).indices.tolist()

        new_node = self._create_node(
            f"sprout_{len(self._nodes)}",
            self._make_passthrough_fn(),
            in_edges  = [(nid, self.gcfg.sprout_init_weight)
                         for nid in top_signal_nodes],
            out_edges = [(nid, self.gcfg.sprout_init_weight)
                         for nid in top_signal_nodes],
        )

        ev = GenEvent(
            step         = step,
            strategy     = GenStrategy.SPROUT,
            trigger      = f"signal={signal.max().item():.2f} but no node fired",
            new_node_id  = new_node.node_id,
            new_node_name= new_node.name,
            parent_ids   = top_signal_nodes,
        )
        self.events.append(ev)
        self._stats[GenStrategy.SPROUT] += 1
        print(f"  [GEN] SPROUT  {new_node.name}  (signal miss x{self.gcfg.sprout_miss_threshold})")
        return new_node

    # ── [3] Interpolate ───────────────────────────────

    def record_routing_fail(self, src_id: int, dst_id: int):
        """라우팅 실패 기록 (ActivationEngine 이 호출)"""
        key = (src_id, dst_id)
        self._fail_pairs[key] = self._fail_pairs.get(key, 0) + 1

    def check_interpolate(self, step: int) -> list[Node]:
        """
        특정 노드 쌍 사이 라우팅이 반복 실패 -> 중계 노드 삽입.
        """
        if len(self._nodes) >= self.gcfg.max_nodes:
            return []

        new_nodes = []
        threshold = self.gcfg.interp_fail_threshold

        for (uid, vid), count in list(self._fail_pairs.items()):
            if count < threshold:
                continue
            if uid >= len(self._nodes) or vid >= len(self._nodes):
                continue

            src = self._nodes[uid]
            dst = self._nodes[vid]
            w   = self.gcfg.interp_init_weight

            mid = self._create_node(
                f"relay_{src.name}_{dst.name}",
                self._make_passthrough_fn(),
                in_edges  = [(uid, w)],
                out_edges = [(vid, w)],
            )
            new_nodes.append(mid)
            self._fail_pairs[(uid, vid)] = 0   # 카운터 리셋

            ev = GenEvent(
                step         = step,
                strategy     = GenStrategy.INTERPOLATE,
                trigger      = f"{src.name}->{dst.name} failed x{count}",
                new_node_id  = mid.node_id,
                new_node_name= mid.name,
                parent_ids   = [uid, vid],
            )
            self.events.append(ev)
            self._stats[GenStrategy.INTERPOLATE] += 1
            print(f"  [GEN] INTERP  relay: {src.name} -> {mid.name} -> {dst.name}")

        return new_nodes

    # ── 공통 유틸 ─────────────────────────────────────

    def _create_node(self, name: str, fn: Callable,
                     in_edges:  list[tuple] = None,
                     out_edges: list[tuple] = None) -> Node:
        """
        WeightMatrix 크기 확장 + 노드 생성 + 엣지 연결.
        """
        new_id = len(self._nodes)

        # W 행렬 한 행/열 확장
        old_n = self.wm.n
        new_n = old_n + 1
        new_W = torch.zeros(new_n, new_n, device=self.wm.dev)
        new_W[:old_n, :old_n] = self.wm.W
        self.wm.W = new_W
        self.wm.n = new_n

        # 허브 마스크 확장
        self.hfe.is_hub    = torch.cat([
            self.hfe.is_hub,
            torch.zeros(1, dtype=torch.bool, device=self.wm.dev)
        ])
        self.hfe.fire_count= torch.cat([
            self.hfe.fire_count,
            torch.zeros(1, device=self.wm.dev)
        ])

        # 엣지 설정 - 유효한 엣지만 (빈 자식 방지)
        for u, w in (in_edges or []):
            if u < old_n and w > 0.01:
                self.wm.W[u][new_id] = w
        for v, w in (out_edges or []):
            if v < old_n and w > 0.01:
                self.wm.W[new_id][v] = w

        # 노드 객체 생성
        def named_fn(data, _name=name, _fn=fn):
            return _fn(data)
        named_fn.__name__      = name
        named_fn._htp_tags     = getattr(fn, "_htp_tags",     set())
        named_fn._htp_terminal = getattr(fn, "_htp_terminal", False)

        n = Node(fn=named_fn, node_id=new_id)
        self._nodes.append(n)
        return n

    def _make_passthrough_fn(self) -> Callable:
        """데이터를 그대로 통과시키는 기본 함수"""
        def passthrough(data):
            return data
        return passthrough

    def report(self) -> str:
        lines = ["\n  [ Node Generation Stats ]"]
        for s, count in self._stats.items():
            lines.append(f"  {s.value:<12}  generated={count}")
        if self.events:
            lines.append(f"\n  [ Recent Gen Events (last 5) ]")
            for ev in self.events[-5:]:
                lines.append(
                    f"  step={ev.step:<4}  {ev.strategy.value:<12}"
                    f"  {ev.new_node_name:<30}  trigger: {ev.trigger}"
                )
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════
# HTPRuntime v2  -  4엔진 통합
# ══════════════════════════════════════════════════════════

class HTPRuntime:
    """
    Hub Topology Programming 완전체.

      WeightMatrix         - W 공유
      HubFormationEngine   - 헤비안 + 허브
      PruningEngine        - 4가지 가지치기 + 허브 보호
      NodeGenerationEngine - 3가지 노드 생성
      ActivationEngine     - 캐스케이드 전파

    균형:
      Pruning    - 불필요한 연결/노드 제거
      Generation - 새로운 패턴에 새 노드 생성
      -> 네트워크가 데이터에 맞게 살아서 진화
    """

    def __init__(self, config: Optional[HTPConfig] = None,
                 gen_config: Optional[GenConfig] = None):
        self.cfg      = config     or HTPConfig()
        self.gcfg     = gen_config or GenConfig()
        self._nodes   : list[Node] = []
        self._name_map: dict       = {}
        self._node_count           = 0
        self._built                = False
        self._run_log : list       = []

        self.wm  = None
        self.hfe = None
        self.pe  = None
        self.nge = None
        self.ae  = None

    # ── 노드 등록 ─────────────────────────────────────

    def node(self, fn: Callable) -> Callable:
        n = Node(fn=fn, node_id=self._node_count)
        self._nodes.append(n)
        self._name_map[fn.__name__] = n
        self._node_count += 1
        fn._htp_node = n
        return fn

    # ── 연결 ──────────────────────────────────────────

    def connect(self, src: Callable, dst: Callable,
                weight: float = 0.3) -> "HTPRuntime":
        self._ensure_built()
        u = src._htp_node.node_id
        v = dst._htp_node.node_id
        self.wm.set(u, v, weight)
        print(f"  {src.__name__} -> {dst.__name__}  w={weight:.2f}")
        return self

    # ── 실행 ──────────────────────────────────────────

    def run(self, data: Any,
            entry: Optional[Callable] = None,
            max_depth: int = 8) -> RunResult:
        self._ensure_built()
        t0 = time.perf_counter()

        entry_node = entry._htp_node if entry else None
        path, outputs, hub_ids = self.ae.run(data, entry_node, max_depth)

        step = self.hfe.step_count

        # 1. 가지치기
        pruned = self.pe.run_all(step)

        # 2. 노드 생성 체크
        gen_count = 0

        # 2-a. 허브 과부하 분열
        if gen_count < self.gcfg.max_gen_per_run:
            new = self.nge.check_split(step)
            gen_count += len(new)

        # 2-b. 중계 노드 삽입
        if gen_count < self.gcfg.max_gen_per_run:
            new = self.nge.check_interpolate(step)
            gen_count += len(new)

        res = RunResult(
            input_data = data,
            route_path = path,
            outputs    = outputs,
            hub_ids    = hub_ids,
            pruned     = pruned,
            total_ms   = (time.perf_counter() - t0) * 1000,
        )
        self._run_log.append(res)
        return res

    # ── 상태 ──────────────────────────────────────────

    def status(self):
        SEP = "=" * 66
        print(f"\n{SEP}")
        print("  HTPRuntime v2 Status")
        print(SEP)
        print(f"  nodes (current) : {len(self._nodes)}")
        print(f"  steps           : {self.hfe.step_count}")
        print(f"  edges           : {self.wm.edge_count()}")
        print(f"  hubs            : {int(self.hfe.is_hub.sum().item())}")
        print()
        print(f"  {'node':<22} {'calls':>6}  {'strength':>9}  hub")
        print(f"  {'-'*54}")
        for n in self._nodes:
            hub   = "★ HUB" if (n.node_id < len(self.hfe.is_hub)
                                 and self.hfe.is_hub[n.node_id]) else "     "
            instr = (self.wm.in_strength(n.node_id)
                     if n.node_id < self.wm.n else 0.0)
            print(f"  {n.name:<22} {n.call_count:>6}  {instr:>9.3f}  {hub}")

        print(self.pe.report())
        print(self.nge.report())
        print(SEP)

    # ── 내부 ──────────────────────────────────────────

    def _ensure_built(self):
        if self._built:
            return
        N = max(self._node_count, 1)
        self.cfg.n_nodes = N

        self.wm  = WeightMatrix(N, self.cfg.device)
        self.hfe = HubFormationEngine(self.wm, self.cfg)
        self.pe  = PruningEngine(self.wm, self.hfe, self.cfg)
        self.nge = NodeGenerationEngine(self.wm, self.hfe, self.cfg, self.gcfg)
        self.nge.register(self._nodes)
        self.ae  = ActivationEngine(self.wm, self.hfe, self.cfg)
        self.ae.register(self._nodes)

        self._built = True
        print(f"[HTP] Runtime v2 built  -  {N} nodes  "
              f"| hfe + pe + nge + ae  on {self.cfg.device}")

    @property
    def log(self):
        return self._run_log


# ══════════════════════════════════════════════════════════
# 데모  -  허브 분열 + 발아 시연
# ══════════════════════════════════════════════════════════

def demo():
    SEP = "=" * 66
    print(SEP)
    print("  HTPRuntime v2  -  Pruning + Generation")
    print("  노드가 줄기도 하고 늘기도 하는 살아있는 네트워크")
    print(SEP)

    rt = HTPRuntime(
        config = HTPConfig(
            hub_threshold   = 1.5,
            hebbian_lr      = 0.13,
            decay_rate      = 0.003,
            prune_threshold = 0.02,
            threshold       = 0.35,
        ),
        gen_config = GenConfig(
            split_strength_threshold = 3.8,
            split_call_threshold     = 12,
            split_cooldown           = 25,
            global_cooldown          = 12,
            maturity_calls           = 5,
            sprout_miss_threshold    = 3,
            max_gen_per_run          = 1,
            max_nodes                = 20,
        )
    )

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
        print(f"        [ALERT]  {label}")
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
        ("success: migration done",    "success"),
        ("error: out of memory",       "error"),
        ("ok: health check passed",    "success"),
        ("bug: null pointer",          "error"),
        ("completed: backup job",      "success"),
        ("timeout: api call",          "error"),
        ("deployed: frontend v3",      "success"),
        ("fail: auth token expired",   "error"),
        ("success: migration done",    "success"),
        ("error: out of memory",       "error"),
        ("ok: health check passed",    "success"),
        ("bug: null pointer",          "error"),
        ("completed: backup job",      "success"),
        ("timeout: api call",          "error"),
        ("deployed: frontend v3",      "success"),
        ("fail: disk full",            "error"),
        ("success: api v2 launched",   "success"),
        ("error: ssl expired",         "error"),
        ("done: cleanup finished",     "success"),
        ("fatal: segfault",            "error"),
        ("ok: rollback complete",      "success"),
        ("bug: race condition",        "error"),
        ("completed: export job",      "success"),
        ("fail: rate limit hit",       "error"),
    ]

    print(f"\n[ 총 {len(dataset)}개 데이터 실행 - 허브 분열 관찰 ]\n")
    init_node_count = len(rt._nodes)

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
        node_mark = f"  [nodes: {len(rt._nodes)}]" if len(rt._nodes) != init_node_count else ""
        print(f"  [{i:02d}] {mark}  '{text}'{node_mark}")
        print(f"        {path}")
        init_node_count = len(rt._nodes)

    total = len(dataset)
    print(f"\n  accuracy: {correct}/{total}  ({correct*100//total}%)")
    rt.status()


if __name__ == "__main__":
    demo()
