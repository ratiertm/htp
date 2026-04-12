"""
Activation Engine v2  —  Cascade + Semantic Exclusion Routing

핵심 수정:
  [A] sig[prev]=1.0  →  구조적 전파 (엣지 가중치로 다음 노드 에너지 전달)
  [B] semantic boost  →  label/키워드 매칭 노드 신호 강화
  [C] semantic exclusion  →  같은 prev에서 나온 노드 중
                              시맨틱 매칭 노드가 있으면 비매칭 노드는 fired 목록에서 제외
"""

import time
from dataclasses import dataclass, field
from typing import Callable, Any, Optional, Set
from ..core.hub_formation_engine import HubFormationEngine, HTPConfig
import torch

FIRE_FLOOR = 0.08


# ── 노드 ──────────────────────────────────────────────

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


# ── 결과 ──────────────────────────────────────────────

@dataclass
class ActivationResult:
    input_data:   Any
    fired_nodes:  list
    outputs:      dict
    route_path:   list
    total_ms:     float = 0.0
    hub_snapshot: list  = field(default_factory=list)

    def summary(self) -> str:
        return " -> ".join(n.name for n in self.route_path)


# ── tag 데코레이터 ─────────────────────────────────────

def tag(*tags):
    """
    노드에 의미 태그를 달아 시맨틱 라우팅 활성화.

      @rt.node
      @tag("success", "done")
      def to_cache(data): ...

    data["label"] == "success" 이면:
      - to_cache 신호 강화
      - 같은 소스에서 나온 to_alert 등 비매칭 노드는 발화 억제
    """
    def decorator(fn):
        fn._htp_tags = set(t.lower() for t in tags)
        return fn
    return decorator



def terminal(fn):
    """
    노드를 캐스케이드 종착점으로 표시.
    이 노드가 발화하면 cascade 루프 종료.

      @rt.node
      @terminal
      def log_result(data): ...
    """
    fn._htp_terminal = True
    return fn

# ── Runtime ───────────────────────────────────────────

class Runtime:

    def __init__(self, config: Optional[HTPConfig] = None):
        self._nodes      : list = []
        self._name_map   : dict = {}
        self._node_count        = 0
        self._engine     : Optional[HubFormationEngine] = None
        self._config            = config or HTPConfig()
        self._built             = False
        self._log        : list = []

    # ── 노드 등록 ────────────────────────────────────

    def node(self, fn: Callable) -> Callable:
        n = Node(fn=fn, node_id=self._node_count)
        self._nodes.append(n)
        self._name_map[fn.__name__] = n
        self._node_count += 1
        fn._htp_node = n
        return fn

    # ── 연결 ─────────────────────────────────────────

    def connect(self, src: Callable, dst: Callable, weight: float = 0.3):
        self._ensure_built()
        u = src._htp_node.node_id
        v = dst._htp_node.node_id
        self._engine.W[u][v] = weight
        print(f"  {src.__name__} -> {dst.__name__}  w={weight:.2f}")

    # ── 캐스케이드 활성화 ─────────────────────────────

    def activate(self, data: Any,
                 entry: Optional[Callable] = None,
                 max_depth: int = 8) -> ActivationResult:
        self._ensure_built()
        t0      = time.perf_counter()
        fired_n = []
        outputs = {}
        path    = []
        visited : Set[int] = set()
        hub_seen: list     = []
        current = data

        # entry 강제 실행
        if entry:
            en      = entry._htp_node
            current = en.run(current)
            fired_n.append(en)
            outputs[en.name] = current
            path.append(en)
            visited.add(en.node_id)
            prev = {en.node_id}
        else:
            prev = set()

        # 캐스케이드 루프
        for _ in range(max_depth):
            sig = self._make_signal(current, prev, visited)
            if sig.max().item() < FIRE_FLOOR:
                break

            sr       = self._engine.step(sig)
            hub_seen = sr.hub_indices.tolist()

            # 엔진이 발화시킨 노드 (visited 제외)
            raw_ids = [
                int(i)
                for i in sr.fired.nonzero(as_tuple=True)[0].tolist()
                if int(i) not in visited and int(i) < len(self._nodes)
            ]
            if not raw_ids:
                break

            # [C] 시맨틱 배제 필터 적용
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

        hubs = [self._nodes[i] for i in hub_seen if i < len(self._nodes)]
        res  = ActivationResult(
            input_data   = data,
            fired_nodes  = fired_n,
            outputs      = outputs,
            route_path   = path,
            total_ms     = (time.perf_counter() - t0) * 1000,
            hub_snapshot = hubs,
        )
        self._log.append(res)
        return res

    # ── 신호 생성 ─────────────────────────────────────

    def _make_signal(self, data: Any,
                     prev_ids: Set[int],
                     visited:  Set[int]) -> torch.Tensor:
        """
        [A] prev 노드를 sig=1.0 으로  →  W 전파로 다음 노드 에너지 전달
        [B] 시맨틱 부스트  →  label 매칭 노드 sig 강화
        """
        N   = self._node_count
        dev = self._engine.cfg.device
        sig = torch.zeros(N, device=dev)

        # [A] 구조 신호: prev = 1.0
        for uid in prev_ids:
            sig[uid] = 1.0

        # [B] 시맨틱 신호
        label, kws = self._extract_semantics(data)
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

    # ── 시맨틱 배제 필터 ──────────────────────────────

    def _semantic_filter(self,
                         candidate_ids: list,
                         prev_ids: Set[int],
                         data: Any,
                         visited: Set[int]) -> list:
        """
        prev 노드에서 나오는 경쟁 노드들 중
        시맨틱 매칭 노드가 있으면 → 비매칭 노드는 제거.

        예: classify(prev) → [to_cache, to_alert] 중
            label=error 이면 to_alert 만 통과, to_cache 제거
        """
        label, kws = self._extract_semantics(data)
        if not label and not kws:
            return candidate_ids

        # prev 노드의 직접 출력 엣지 대상 노드들을 그룹화
        # 같은 prev에서 나오는 경쟁 그룹에서만 배제 적용
        result = list(candidate_ids)

        for uid in prev_ids:
            # uid 에서 나가는 엣지의 후보 노드들
            rivals = [
                v for v in candidate_ids
                if float(self._engine.W[uid][v]) > 0.01
            ]
            if len(rivals) < 2:
                continue  # 경쟁자 없으면 배제 불필요

            # 시맨틱 매칭 여부 판단
            matched     = []
            not_matched = []
            for v in rivals:
                ntags = getattr(self._nodes[v].fn, "_htp_tags", set())
                if ntags and ntags & (kws | {label}):
                    matched.append(v)
                else:
                    not_matched.append(v)

            # 매칭 노드가 있으면 비매칭 노드 제거
            if matched:
                for v in not_matched:
                    if v in result:
                        result.remove(v)

        return result

    # ── 공통 유틸 ─────────────────────────────────────

    def _extract_semantics(self, data):
        label = ""
        kws   = set()
        if isinstance(data, dict):
            label = str(data.get("label", "")).lower()
            kws   = {str(v).lower() for v in data.values() if isinstance(v, str)}
        elif isinstance(data, str):
            kws = set(data.lower().split())
        return label, kws

    def inject_engine(self, engine: HubFormationEngine) -> None:
        """외부에서 생성된 엔진 주입 — HTPRuntime 전용."""
        self._engine = engine
        self._built  = True

    def _ensure_built(self):
        if self._built:
            return
        N   = max(self._node_count, 1)
        cfg = HTPConfig(
            n_nodes         = N,
            threshold       = self._config.threshold,
            hub_threshold   = self._config.hub_threshold,
            hebbian_lr      = self._config.hebbian_lr,
            decay_rate      = self._config.decay_rate,
            prune_threshold = self._config.prune_threshold,
        )
        self._engine = HubFormationEngine(cfg)
        self._built  = True
        print(f"[HTP] Runtime built  —  {N} nodes")

    @property
    def log(self):
        return self._log


# ── 데모 ──────────────────────────────────────────────

def demo():
    SEP = "=" * 64
    print(SEP)
    print("  Activation Engine v2  —  Cascade + Semantic Exclusion")
    print(SEP)

    cfg = HTPConfig(
        hub_threshold   = 1.5,
        hebbian_lr      = 0.13,
        decay_rate      = 0.005,
        prune_threshold = 0.02,
        threshold       = 0.35,
    )
    rt = Runtime(config=cfg)

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
        print(f"        [CACHE]  label={label}")
        return {**data, "cached": True}

    @rt.node
    @tag("error", "fail", "bug", "timeout", "fatal", "oom")
    def to_alert(data):
        label = data.get("label", "") if isinstance(data, dict) else ""
        print(f"        [ALERT]  label={label}  <- error!")
        return {**data, "alerted": True}

    @rt.node
    @terminal
    @tag("success", "error", "neutral", "cached", "alerted")
    def log_result(data):
        label   = data.get("label",   "") if isinstance(data, dict) else ""
        cached  = data.get("cached",  False)
        alerted = data.get("alerted", False)
        st = "cached" if cached else ("alerted" if alerted else "passed")
        print(f"        [LOG]    label={label}  status={st}")
        return data

    print("\n[ connections ]")
    rt.connect(parse,    classify,   weight=0.55)
    rt.connect(classify, to_cache,   weight=0.30)
    rt.connect(classify, to_alert,   weight=0.30)
    rt.connect(to_cache, log_result, weight=0.35)
    rt.connect(to_alert, log_result, weight=0.35)

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

    print("\n[ cascade activation ]\n")
    correct = 0
    for i, (text, expected) in enumerate(dataset, 1):
        result = rt.activate(text, entry=parse)
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
        print(f"        path: {path}")

    total = len(dataset)
    pct   = correct * 100 // total
    print(f"\n  routing accuracy: {correct}/{total}  ({pct}%)")

    W     = rt._engine.W
    nodes = rt._nodes
    print(f"\n{SEP}")
    print("  node stats")
    print(SEP)
    print(f"  {'node':<16} {'calls':>6}  {'strength':>9}  hub")
    print(f"  {'-'*48}")
    for n in nodes:
        hub   = "* HUB" if rt._engine.is_hub[n.node_id] else "     "
        instr = float(W[:, n.node_id].sum())
        print(f"  {n.name:<16} {n.call_count:>6}  {instr:>9.3f}  {hub}")

    print("\n  [ reinforced edges (w > 0.40) ]")
    for u in range(len(nodes)):
        for v in range(len(nodes)):
            w = float(W[u][v])
            if w > 0.40:
                star = "  ** HUB EDGE **" if w > 0.70 else ""
                print(f"  {nodes[u].name:<14} -> {nodes[v].name:<14}  w={w:.3f}{star}")
    print(SEP)


if __name__ == "__main__":
    demo()