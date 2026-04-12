"""
AsyncBrainRuntime  —  asyncio.gather + SLA Timeout
====================================================

BrainRuntime 확장:
  - 모든 Region을 asyncio.gather로 병렬 실행
  - wait_for(sla_sec) 로 SLA 시간 초과 Region 무시
  - 동기 run() 인터페이스 유지 (executor 래핑)

수학:
  tasks = [asyncio.wait_for(r.arun(data), timeout=sla) for r in regions]
  results = await asyncio.gather(*tasks, return_exceptions=True)
  # TimeoutError -> 해당 Region 결과 무시, 캐시 반환
"""

from __future__ import annotations

import asyncio
from typing import Any

from .brain_runtime import BrainRuntime
from ..thalamus.region_signal import Action


class AsyncBrainRuntime(BrainRuntime):
    """
    비동기 병렬 Brain.

    Parameters
    ----------
    pfc_config : PFCRuntime 설정 (None이면 기본값)
    sla_sec    : 각 Region arun() 최대 대기 시간 (초)

    사용법
    ------
    # 비동기
    brain = AsyncBrainRuntime(sla_sec=8.0)
    brain.add_region("language", LLMRegionRuntime(...))
    action = await brain.arun("입력 데이터")

    # 동기 (내부적으로 asyncio.run 사용)
    action = brain.run("입력 데이터")
    """

    def __init__(self, pfc_config=None, sla_sec: float = 5.0):
        super().__init__(pfc_config)
        self.sla_sec = sla_sec

    # ── 비동기 실행 ───────────────────────────────────

    async def arun(self, data: Any) -> Action:
        """
        모든 Region 병렬 실행 후 Thalamus → PFC → Action.

        arun() 없는 Region은 ThreadPoolExecutor로 동기 실행.
        """
        self._ensure_thalamus()
        self._step += 1

        async def safe_run(region):
            try:
                if hasattr(region, "arun"):
                    return await asyncio.wait_for(
                        region.arun(data), timeout=self.sla_sec
                    )
                # 동기 Region: executor에서 실행
                loop = asyncio.get_event_loop()
                return await asyncio.wait_for(
                    loop.run_in_executor(None, region.run, data),
                    timeout=self.sla_sec,
                )
            except asyncio.TimeoutError:
                print(f"  [timeout] {getattr(region, 'region_name', '?')}")
            except Exception as e:
                print(f"  [error]   {getattr(region, 'region_name', '?')}: {e}")
            return None

        await asyncio.gather(*[safe_run(r) for r in self.regions.values()])

        # Thalamus → PFC (동기, 연산량 적음)
        thal_out      = self.thalamus.step(data, top_down=self._last_td)
        action, td    = self.pfc.decide(thal_out, regions=self.regions)
        self._last_td = td

        for rid, strength in thal_out.suppressed.items():
            if rid in self.regions and strength > 0:
                self.regions[rid].apply_suppression(strength)

        if self._cc is not None:
            self._cc.apply(thal_out)

        # winner Region의 LLM 응답을 action.result에 채움
        winner_region = self.regions.get(action.winner)
        if winner_region is not None:
            last = getattr(winner_region, "_last_result", None)
            if last is not None:
                action.result = last.outputs.get(action.winner)

        return action

    # ── 동기 호환 ─────────────────────────────────────

    def run(self, data: Any) -> Action:
        """동기 호환 인터페이스 — arun() 래핑."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Jupyter 등 이미 루프가 실행 중인 환경
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    return pool.submit(asyncio.run, self.arun(data)).result()
            return loop.run_until_complete(self.arun(data))
        except RuntimeError:
            return asyncio.run(self.arun(data))

    # ── 비용 보고 ─────────────────────────────────────

    def cost_report(self) -> str:
        SEP = "=" * 62
        lines = [f"\n{SEP}", f"  AsyncBrainRuntime cost report  step={self._step}", SEP]
        for name, region in self.regions.items():
            if hasattr(region, "cost_report"):
                lines.append(region.cost_report())
            else:
                lines.append(f"  [{name}] (no cost tracking)")
        lines.append(SEP)
        return "\n".join(lines)
