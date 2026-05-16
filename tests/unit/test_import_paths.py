"""
Public + legacy import path verification.

Design Ref: docs/02-design/features/htp-review-improvements.design.md §4
Plan SC: FR-10 (re-export 호환성)

3개의 import 경로가 동일 객체를 반환하는지 영구 검증:
  - htp                      (공개 API, htp/__init__.py)
  - htp.core                 (새 권장 경로)
  - htp.runtime.htp_runtime  (옛 호환 경로 — server.py, tests 가 사용)

이 테스트가 깨지면 외부 사용자 코드가 깨졌다는 신호.
"""
from __future__ import annotations


# ══════════════════════════════════════════════════════════
# WeightMatrix
# ══════════════════════════════════════════════════════════

def test_weight_matrix_4_import_paths_identical():
    from htp                          import WeightMatrix as P_top
    from htp.core                     import WeightMatrix as P_core
    from htp.core.weight_matrix       import WeightMatrix as P_new
    from htp.runtime.htp_runtime      import WeightMatrix as P_legacy
    assert P_top is P_core is P_new is P_legacy


# ══════════════════════════════════════════════════════════
# HubFormationEngine
# ══════════════════════════════════════════════════════════

def test_hub_formation_engine_paths_identical():
    from htp                          import HubFormationEngine as P_top
    from htp.core                     import HubFormationEngine as P_core
    from htp.core.hub_formation       import HubFormationEngine as P_new
    from htp.runtime.htp_runtime      import HubFormationEngine as P_legacy
    assert P_top is P_core is P_new is P_legacy


# ══════════════════════════════════════════════════════════
# PruningEngine + PruneStrategy
# ══════════════════════════════════════════════════════════

def test_pruning_engine_paths_identical():
    from htp                          import PruningEngine as P_top
    from htp.core                     import PruningEngine as P_core
    from htp.core.pruning             import PruningEngine as P_new
    from htp.runtime.htp_runtime      import PruningEngine as P_legacy
    assert P_top is P_core is P_new is P_legacy


def test_prune_strategy_paths_identical():
    """tests/regression/test_phase1_pruning.py:18 이 사용하는 옛 경로 호환."""
    from htp.core                     import PruneStrategy as P_core
    from htp.core.pruning             import PruneStrategy as P_new
    from htp.runtime.htp_runtime      import PruneStrategy as P_legacy
    assert P_core is P_new is P_legacy


# ══════════════════════════════════════════════════════════
# ActivationEngine + Node + RunResult + decorators + FIRE_FLOOR
# ══════════════════════════════════════════════════════════

def test_activation_engine_paths_identical():
    from htp                          import ActivationEngine as P_top
    from htp.core                     import ActivationEngine as P_core
    from htp.core.activation          import ActivationEngine as P_new
    from htp.runtime.htp_runtime      import ActivationEngine as P_legacy
    assert P_top is P_core is P_new is P_legacy


def test_node_dataclass_paths_identical():
    """Node 는 htp/core/node_generation_engine.py 가 import 함 — 호환 필수."""
    from htp                          import Node as P_top
    from htp.core                     import Node as P_core
    from htp.core.activation          import Node as P_new
    from htp.runtime.htp_runtime      import Node as P_legacy
    assert P_top is P_core is P_new is P_legacy


def test_decorators_paths_identical():
    from htp                          import tag, terminal
    from htp.core                     import tag as tag_core, terminal as terminal_core
    from htp.core.activation          import tag as tag_new, terminal as terminal_new
    from htp.runtime.htp_runtime      import tag as tag_legacy, terminal as terminal_legacy
    assert tag is tag_core is tag_new is tag_legacy
    assert terminal is terminal_core is terminal_new is terminal_legacy


def test_fire_floor_constant_paths_identical():
    from htp                          import FIRE_FLOOR as F_top
    from htp.core                     import FIRE_FLOOR as F_core
    from htp.core.activation          import FIRE_FLOOR as F_new
    from htp.runtime.htp_runtime      import FIRE_FLOOR as F_legacy
    assert F_top == F_core == F_new == F_legacy == 0.08


# ══════════════════════════════════════════════════════════
# Sub-configs (Step 1 신설)
# ══════════════════════════════════════════════════════════

def test_sub_config_paths_identical():
    from htp.core                     import HubConfig, PruneConfig, ActivationConfig
    from htp.core.config              import HubConfig as H2, PruneConfig as P2, ActivationConfig as A2
    assert HubConfig is H2
    assert PruneConfig is P2
    assert ActivationConfig is A2


# ══════════════════════════════════════════════════════════
# Star import — htp/__init__.py 전체 표면
# ══════════════════════════════════════════════════════════

def test_star_import_succeeds():
    """from htp import * 가 에러 없이 동작."""
    namespace = {}
    exec("from htp import *", namespace)
    # 주요 심볼 존재 확인
    for sym in ["HTPRuntime", "HTPConfig", "WeightMatrix",
                "HubFormationEngine", "PruningEngine", "ActivationEngine",
                "Node", "RunResult", "tag", "terminal", "FIRE_FLOOR",
                "BrainRuntime", "MemorySystem"]:
        assert sym in namespace, f"public API missing: {sym}"
