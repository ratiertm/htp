"""htp.core — 동적 노드 생성 엔진 (NGE)

Phase 1 초기 구현 중 `hub_formation_engine.py`, `pruning_engine.py`는
`htp/runtime/htp_runtime.py`로 통합되며 이관되었고, 원본은
`archive/deprecated_phase1/`로 이동되었다.

Live 모듈:
    node_generation_engine (NGE) — split / sprout / interpolate
"""
from .node_generation_engine import (
    NodeGenerationEngine,
    GenConfig,
    GenEvent,
    GenStrategy,
)

__all__ = [
    "NodeGenerationEngine",
    "GenConfig",
    "GenEvent",
    "GenStrategy",
]
