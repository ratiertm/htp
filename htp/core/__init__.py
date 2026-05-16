"""htp.core — Phase 1 엔진 + Phase 2 NGE.

Design Ref: docs/02-design/features/htp-review-improvements.design.md §2.2 — DAG 강제.
이 패키지는 htp/runtime/* 를 import 하지 않는다. 단, 과도기적으로
node_generation_engine 이 htp_runtime 의 dataclass 를 import 하므로
초기화 시 순환 충돌을 피하기 위해 PEP 562 lazy attribute loading 을 사용한다.

Live 모듈:
    config                       — HubConfig / PruneConfig / ActivationConfig (Step 1)
    node_generation_engine (NGE) — split / sprout / interpolate

Phase 1 초기 구현 중 ``hub_formation_engine.py``, ``pruning_engine.py`` 는
``htp/runtime/htp_runtime.py`` 로 통합되며 이관되었고, 원본은
``archive/deprecated_phase1/`` 로 이동되었다.
"""
from __future__ import annotations

# Step 1: sub-config 는 외부 의존이 없어 eager import 안전
from .config import HubConfig, PruneConfig, ActivationConfig
# Step 3: WeightMatrix 는 torch 만 의존 (DAG 안전)
from .weight_matrix import WeightMatrix


# NGE 관련 심볼은 lazy 로드 — htp_runtime → htp.core → NGE → htp_runtime 순환을 회피.
# PEP 562 (Python 3.7+) — `from htp.core import NodeGenerationEngine` 호출 시점에 비로소 로드.
_LAZY_NGE_NAMES = {"NodeGenerationEngine", "GenConfig", "GenEvent", "GenStrategy"}


def __getattr__(name: str):
    if name in _LAZY_NGE_NAMES:
        from .node_generation_engine import (
            NodeGenerationEngine,
            GenConfig,
            GenEvent,
            GenStrategy,
        )
        return locals()[name]
    raise AttributeError(f"module 'htp.core' has no attribute {name!r}")


__all__ = [
    # Step 1 — sub-configs
    "HubConfig",
    "PruneConfig",
    "ActivationConfig",
    # Step 3 — WeightMatrix
    "WeightMatrix",
    # NGE (lazy)
    "NodeGenerationEngine",
    "GenConfig",
    "GenEvent",
    "GenStrategy",
]
