"""
LLMRegion 사용 데모 — sub-4 C-1 보완.

Design Ref: docs/02-design/features/htp-thalamus-car.sub-4.design.md §3 C-1

MockLLMNode 사용 — Anthropic API 키 불필요. 실제 사용 시 use_mock=False 로 교체.

실행:
  source .venv/bin/activate
  python examples/llm_region_demo.py
"""
from __future__ import annotations

from htp.llm.llm_region          import LLMRegion
from htp.llm.cost_router         import CostRouter
from htp.runtime.brain_runtime   import BrainRuntime, PFCRuntime


def demo_single_region():
    """LLMRegion 단독 사용 — run / collect_signal / cost report."""
    print("\n── Demo 1: LLMRegion 단독 ──")
    region = LLMRegion(
        region_name = "language",
        specialty   = "language",
        use_mock    = True,
    )

    for text in ["사과를 자르면 무엇이 나오나요?",
                 "에러 로그를 분석해 주세요",
                 "메모리 누수 가능성 진단"]:
        result = region.run(text)
        print(f"  in : {text}")
        print(f"  out: {result}")

    sig = region.collect_signal()
    print(f"\n  signal: {sig.region_id} "
          f"precision={sig.precision:.2f}, fire_rate={sig.fire_rate:.2f}, "
          f"overload={sig.overload}")
    print(region.cost_report())


def demo_cost_router_levels():
    """CostRouter.select_level 4-Level 시연."""
    print("\n── Demo 2: CostRouter 4-Level 의사결정 ──")
    r = CostRouter(budget_per_step=0.01)

    cases = [
        (0.1, "단순 (lookup)"),
        (0.4, "약간 복잡"),
        (0.6, "중간"),
        (0.9, "복잡 (reasoning)"),
    ]
    print("  initial pressure=0:")
    for c, desc in cases:
        lv = r.select_level(c)
        names = {1: "LOCAL", 2: "sLLM", 3: "API_SMALL", 4: "API_LARGE"}
        print(f"    complexity={c:.1f} ({desc:18s}) → level={lv} {names[lv]}")

    # 압박 시뮬레이션
    r._ema_cost = r.budget * 0.9   # pressure = 0.9
    print(f"\n  pressure {r.pressure:.2f} (압박):")
    for c, desc in cases:
        lv = r.select_level(c)
        names = {1: "LOCAL", 2: "sLLM", 3: "API_SMALL", 4: "API_LARGE"}
        print(f"    complexity={c:.1f} ({desc:18s}) → level={lv} {names[lv]}")


def demo_brain_runtime_with_llm_region():
    """BrainRuntime 에 LLMRegion 통합 — RegionRuntime 과 동시 사용."""
    print("\n── Demo 3: BrainRuntime + LLMRegion 통합 ──")

    # PFCRuntime 기본값
    pfc = PFCRuntime()
    brain = BrainRuntime(pfc_config=pfc)

    # LLMRegion 3개 (mock) — language / code / memory
    for name, specialty in [("language", "language"),
                            ("code",     "code"),
                            ("memory",   "memory")]:
        region = LLMRegion(region_name=name, specialty=specialty, use_mock=True)
        brain.add_region(name, region)

    # 1 step 실행
    action = brain.run("사과의 의미를 분석해 주세요")
    print(f"  action: type={action.type}, winner={action.winner}")
    print(f"  reason: {action.reason}")

    # 각 region 의 비용 보고
    for name, region in brain.regions.items():
        if hasattr(region, "cost_report"):
            print(region.cost_report())


if __name__ == "__main__":
    demo_single_region()
    demo_cost_router_levels()
    demo_brain_runtime_with_llm_region()
    print("\n✓ Demo 완료. 실제 API 사용 시 use_mock=False 로 변경.")
