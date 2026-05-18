"""
ExternalRegion — RegionRuntime 비상속 추상.

Design Ref: docs/02-design/features/htp-thalamus-car.sub-4.design.md §3-1
Plan SC: FR-16 (ExternalRegion 추상)

PageRank / Hebbian / NGE 의존성을 끌어오지 않고 Region 의 핵심 interface 만 만족:
  - run / arun        : 외부 호출
  - collect_signal    : Thalamus 가 보는 외부 region 의 신호 (가짜 hub/fire/precision)
  - apply_suppression : default no-op

graphify 영향: LLM / 외부 API 노드를 brain-like hub 그래프에서 분리 → isolated 감소.

DAG: htp/runtime/external_region.py 는 htp.thalamus.region_signal 만 import.
     상속자 (예: LLMRegion) 는 htp/llm 등 외부 layer 에서 정의.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from htp.thalamus.region_signal import RegionSignal


class ExternalRegion(ABC):
    """외부 호출 (LLM API, search, RAG, ...) 을 Region 추상으로 노출.

    하위 클래스가 반드시 구현:
      - run(data) -> Any
      - collect_signal() -> RegionSignal

    선택적 override:
      - arun(data)              : default = sync run 을 await
      - apply_suppression(s)    : default = no-op

    BrainRuntime 호환성:
      BrainRuntime 일부 코드가 RegionRuntime 의 내부 속성 (`_nodes`,
      `_cusum_S`, `_cusum_h`) 을 직접 참조함. ExternalRegion 에 dummy
      값을 두어 AttributeError 회피. 외부 region 의 의미상 PageRank /
      CUSUM 모두 trivial 이므로 빈 list / 0 / 무한대 로 표현.
    """

    # 식별자 — 하위 클래스가 __init__ 에서 설정
    region_name: str
    specialty:   str
    step:        int

    # BrainRuntime 호환 dummy 속성 — 외부 region 은 hub 그래프 / CUSUM 미보유
    _nodes:   list = []     # type: ignore[assignment]  # 빈 리스트 (BrainRuntime 의 `if region._nodes:` 가짜)
    _cusum_S: float = 0.0   # 누적 entropy — 외부 region 은 0 고정
    _cusum_h: float = 1e9   # 임계값 — 절대 발화 안 함 (수면 트리거 회피)

    @abstractmethod
    def run(self, data: Any) -> Any:
        """동기 외부 호출. 반환: 도메인별 결과 (dict 권장)."""
        ...

    async def arun(self, data: Any) -> Any:
        """비동기 외부 호출. default 는 sync run 호출.

        하위 클래스가 진짜 async API 를 쓰면 override.
        """
        return self.run(data)

    @abstractmethod
    def collect_signal(self) -> RegionSignal:
        """Thalamus 가 보는 외부 region 의 신호.

        - hub_strength, fire_rate : 외부 호출 빈도 등으로 환산 (의미적 hub 없음)
        - top_hubs                : 일반적으로 빈 리스트
        - overload                : 비용 압박 / API rate-limit 등
        - precision               : 외부 호출 신뢰도 (예: CostRouter.pressure 역수)
        - output_vec              : 의미 vec 없으면 placeholder (e.g. torch.zeros(1))
        """
        ...

    def apply_suppression(self, strength: float) -> None:
        """default = no-op.

        외부 호출은 일반적으로 suppression 영향 없음. 하위 클래스가 필요하면
        호출 빈도 감소 등으로 변환.
        """
        return None


__all__ = ["ExternalRegion"]
