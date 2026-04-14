# deprecated_phase1

Phase 1 초기 구현에서 사용되었으나 이후 `htp/runtime/htp_runtime.py`로
통합·이관된 모듈들. Review Feedback Integration Stage 2-A1에서 정리.

| 파일 | 대체 |
|------|------|
| `hub_formation_engine.py` (BCM-like Hebbian) | `htp/runtime/htp_runtime.py` (Oja's Rule 기반) |
| `pruning_engine.py` (단순 3전략) | `htp/runtime/htp_runtime.py` (4전략 + 허브 보호) |
| `activation_engine.py` (초기 캐스케이드) | `htp/runtime/htp_runtime.py` (시맨틱 배제 라우팅 포함) |

보존 이유: 수학 수식·주석이 다른 formulation 참고용으로 가치 있음.
런타임에는 import되지 않으며 `htp/core/__init__.py`는 `node_generation_engine`만 export한다.
