"""
PipelinedBrainRuntime — 다중 입력의 3-stage pipeline 병렬 실행.

Design Ref: docs/02-design/features/htp-thalamus-car.sub-4.design.md §3-5
Plan SC: FR-22 (PipelinedBrainRuntime, throughput ≥ 1.5× AsyncBrainRuntime)

AsyncBrainRuntime 차이:
  AsyncBrainRuntime    : 한 step 의 Region 들을 asyncio.gather 로 병렬.
                          step N → step N+1 은 순차 (각 step 후 PFC binding 대기).
  PipelinedBrainRuntime: N 개 입력을 3-stage pipeline 으로 실행.
                          step N 의 PFC binding 과 step N+1 의 Region collect 가 겹침.

Stage 분할:
  S1 Region.run/arun  : Region 별 입력 처리 (가장 무거움 — LLM API 등)
  S2 Thalamus.step    : RegionSignal 수집 + WTA
  S3 PFC.decide       : action 결정 + memory + suppression

Pipeline throughput: 이론치 ≈ N / max(t_S1, t_S2, t_S3).
순차 baseline    : Σ (t_S1 + t_S2 + t_S3) per input.
LLM 시나리오에서 S1 이 dominant 이므로 S2+S3 가 다음 입력의 S1 과 겹쳐
throughput 가 ≈ 1.5-2× 향상 기대.

호환성: BrainRuntime 의 모든 메서드 보존. pipelined_arun(inputs) 신규.
       기존 arun(single_input) 은 super().arun 그대로 사용 가능 (AsyncBrainRuntime 상속 시).
"""
from __future__ import annotations

import asyncio
from typing import Any

from .async_brain_runtime import AsyncBrainRuntime
from ..thalamus.region_signal import Action


class PipelinedBrainRuntime(AsyncBrainRuntime):
    """다중 입력의 3-stage pipeline 병렬 실행.

    Parameters
    ----------
    pfc_config : PFCRuntime 설정
    sla_sec    : Region arun timeout
    buffer_size: pipeline buffer 깊이 (default=3, S1/S2/S3 동시 진행 가능 수)
    """

    def __init__(
        self,
        pfc_config = None,
        sla_sec:     float = 5.0,
        buffer_size: int   = 3,
    ):
        super().__init__(pfc_config=pfc_config, sla_sec=sla_sec)
        if buffer_size < 1:
            raise ValueError(f"buffer_size ≥ 1 required, got {buffer_size}")
        self.buffer_size = buffer_size

    async def pipelined_arun(self, inputs: "list[Any]") -> "list[Action]":
        """N 개 입력을 pipeline 으로 실행. 결과는 입력 순서 보존.

        구현 — semaphore 로 S1 (Region.arun) 동시성 buffer_size 로 제한.
        S2/S3 (Thalamus + PFC) 은 input-순서 serialize 하여 state 일관성 보존.

        Plan Stage 5 throughput 목표 1.5× 는 N ≥ buffer_size × 2 시 의미 있음.

        **제약 — 입력 독립성 가정 (sub-4 Report §6 외부 리뷰 합의)**:
          pipelined_arun 은 입력들이 서로 *기억 의존이 없는* 시나리오 가정.
          S1 stage 가 buffer_size 만큼 동시 실행되므로 입력 N+1 의 Region.arun
          이 입력 N 의 PFC 결과 / Memory 상태를 보기 전에 시작될 수 있다.

          순차 기억 의존이 필요한 시나리오 (예: 입력 N+1 이 입력 N 의 결과를 회상해
          사용) 는 super().arun() (AsyncBrainRuntime) 을 N 번 호출하는 순차 모드 사용.

          batch 분석, 독립 query 다발 처리 같은 read-only 또는 stateless 시나리오에서
          사용. 입력 간 인과 관계가 필요하면 sequential.
        """
        if not inputs:
            return []

        self._ensure_thalamus()

        sem = asyncio.Semaphore(self.buffer_size)
        # input_idx → (S1 결과 region 응답 dict).
        # Region.arun 결과는 region 객체 내부 (_last_result) 에 저장되므로
        # S1 의 "완료 신호" 만 모으면 됨.
        s1_done: list[asyncio.Event] = [
            asyncio.Event() for _ in inputs
        ]

        async def s1_worker(idx: int, data: Any) -> None:
            """S1: 모든 Region 의 arun 을 병렬로 실행."""
            async with sem:
                await self._run_all_regions(data)
                s1_done[idx].set()

        # S1 task 모두 즉시 schedule
        s1_tasks = [
            asyncio.create_task(s1_worker(i, d))
            for i, d in enumerate(inputs)
        ]

        results: list[Action] = []
        for i, data in enumerate(inputs):
            # S1 idx i 완료까지 대기 (pipeline 정밀화: 이전 S2+S3 중 다음 S1 진행)
            await s1_done[i].wait()

            # S2: Thalamus
            self._step += 1
            thal_out = self.thalamus.step(data, top_down=self._last_td)

            # S3: PFC + suppression + cortical
            action, td = self.pfc.decide(thal_out, regions=self.regions)
            self._last_td = td

            for rid, strength in thal_out.suppressed.items():
                if rid in self.regions and strength > 0:
                    self.regions[rid].apply_suppression(strength)

            if self._cc is not None:
                self._cc.apply(thal_out)

            # winner Region 의 최근 결과 결합
            winner_region = self.regions.get(action.winner)
            if winner_region is not None:
                last = getattr(winner_region, "_last_result", None)
                if last is not None:
                    if hasattr(last, "outputs"):
                        action.result = last.outputs.get(action.winner)
                    else:
                        action.result = last

            results.append(action)

        # S1 task 마무리
        await asyncio.gather(*s1_tasks, return_exceptions=True)
        return results

    # ── 내부: 한 입력에 대해 모든 Region.arun 병렬 ─────
    async def _run_all_regions(self, data: Any) -> None:
        """super().arun 의 첫 부분과 동일 — Region 만 병렬 실행."""
        async def safe_run(region):
            try:
                if hasattr(region, "arun"):
                    return await asyncio.wait_for(
                        region.arun(data), timeout=self.sla_sec,
                    )
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

    # ── 동기 호환 ─────────────────────────────────────
    def pipelined_run(self, inputs: "list[Any]") -> "list[Action]":
        """동기 호환 — pipelined_arun 래핑."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    return pool.submit(
                        asyncio.run, self.pipelined_arun(inputs),
                    ).result()
            return loop.run_until_complete(self.pipelined_arun(inputs))
        except RuntimeError:
            return asyncio.run(self.pipelined_arun(inputs))


__all__ = ["PipelinedBrainRuntime"]
