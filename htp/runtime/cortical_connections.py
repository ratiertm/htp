"""
Cortical Connections  —  Region 간 약한 직접 연결
===================================================

생물학: cortico-cortical long-range connections
  - 피질 영역 간 직접 연결 (시상 경유 없음)
  - 전체 피질 연결의 60%가 cortico-cortical
  - 시상 억제를 받은 영역이 측면으로 정보 전달

메커니즘:
  Thalamus가 Region_i를 억제했을 때,
  i → j 직접 연결이 있으면 Region_j에 약한 신호 전달:

    signal_j += W_cc[(i,j)] × (1 - suppression_i) × 0.2

  이를 통해 억제된 영역의 정보가 완전히 차단되지 않고
  약하게 다른 영역으로 전달됨.

사용법:
  cc = brain.enable_cortical_connections()
  cc.add_connection("language", "memory", weight=0.15)
  cc.add_connection("memory",   "emotion", weight=0.10)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..thalamus.region_signal import ThalamusOutput


class CorticalConnections:
    """
    Region 간 약한 직접 연결.

    - `add_connection(src, dst, weight)`: 연결 추가 (weight 0~1)
    - `apply(thal_out)`: 억제된 Region에서 연결된 Region으로 신호 전달
    """

    def __init__(self, regions: dict):
        self._regions = regions
        self._W: dict[tuple[str, str], float] = {}

    def add_connection(self, src: str, dst: str, weight: float = 0.1):
        """
        src Region → dst Region 직접 연결 추가.
        weight: 연결 강도 (0~1, 권장 0.05~0.2)
        """
        if src not in self._regions:
            raise ValueError(f"Region '{src}' not found")
        if dst not in self._regions:
            raise ValueError(f"Region '{dst}' not found")
        self._W[(src, dst)] = max(0.0, min(1.0, weight))
        print(f"  [CorticalConn] {src} -> {dst}  w={weight:.3f}")

    def apply(self, thal_out: "ThalamusOutput"):
        """
        억제된 Region에서 연결된 Region으로 약한 신호 전달.
        Thalamus.step() 이후 BrainRuntime.run()에서 호출.
        """
        for (src, dst), w in self._W.items():
            if src not in thal_out.suppressed:
                continue

            suppression = thal_out.suppressed[src]
            # 억제 강도가 강할수록 신호가 약해짐
            signal = w * (1.0 - suppression) * 0.2

            if signal < 0.005:
                continue

            dst_region = self._regions.get(dst)
            if dst_region is None:
                continue

            try:
                dst_region._ensure_built()
                # W 행렬에 약한 신호 가산 (허브 연결 강화 효과)
                dst_region.wm.W.mul_(1.0 + signal)
                dst_region.wm.W.clamp_(0.0, 1.0)
            except Exception:
                pass

    def report(self) -> str:
        if not self._W:
            return "  [CorticalConn] no connections"
        lines = ["  [ Cortical Connections ]"]
        for (src, dst), w in self._W.items():
            lines.append(f"  {src:<14} -> {dst:<14}  w={w:.3f}")
        return "\n".join(lines)
