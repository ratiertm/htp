"""
HTPRuntime demo  -  12/12 routing demonstration.

Design Ref: docs/02-design/features/htp-review-improvements.design.md §3 Step 7
이 데모는 step-7 에서 htp_runtime.py 에서 분리됨 (≤250줄 SUCCESS 기준 달성용).

실행:
    python -m htp.runtime._demo
    또는
    python -c "from htp.runtime._demo import demo; demo()"
"""
from __future__ import annotations

from .htp_runtime import HTPRuntime, HTPConfig
from ..core.activation import tag, terminal


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
