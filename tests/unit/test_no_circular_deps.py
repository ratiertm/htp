"""
DAG enforcement — htp/core/* 는 htp/runtime/* 를 import 하지 않는다.

Design Ref: docs/02-design/features/htp-review-improvements.design.md §2.2
Plan SC: FR-11 (구조적 결합도 감소 영구 보호)

이 테스트가 깨지면 누군가 핵심 DAG 규칙을 어긴 것:
  - htp/core/*.py 는 torch + dataclasses + 같은 htp/core 형제만 import 가능
  - htp/runtime/*.py 가 htp/core/* 를 import (단방향)
  - htp/__init__.py 만 모든 곳에 접근

이 규칙을 어기면 step-1 에서 발견한 순환 import 가 재발할 위험.
"""
from __future__ import annotations

import ast
import pathlib

import pytest


# 검사 대상 디렉토리
_PROJECT_ROOT  = pathlib.Path(__file__).parent.parent.parent
_CORE_DIR      = _PROJECT_ROOT / "htp" / "core"
_KNOWLEDGE_DIR = _PROJECT_ROOT / "htp" / "knowledge"           # sub-1
_ROUTER_DIR    = _PROJECT_ROOT / "htp" / "thalamus" / "router"  # sub-2 M8
_COHERENCE_DIR = _PROJECT_ROOT / "htp" / "thalamus" / "coherence"  # sub-3 M6


def _from_modules(py_file: pathlib.Path) -> list[str]:
    """py_file 의 모든 `from X import ...` 의 X 목록."""
    tree = ast.parse(py_file.read_text())
    return [
        node.module for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    ]


def _direct_imports(py_file: pathlib.Path) -> list[str]:
    """py_file 의 모든 `import X` 의 X 목록."""
    tree = ast.parse(py_file.read_text())
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
    return imports


# ══════════════════════════════════════════════════════════
# DAG 규칙 1: htp/core/*.py 는 htp.runtime 을 import 하지 않는다
# ══════════════════════════════════════════════════════════

# 예외: node_generation_engine.py 는 Phase 1 레거시 호환 — 향후 분리 대상이나
# 현재는 그대로 둠 (Plan §2.2 "위치 유지").
_DAG_EXEMPT = {"node_generation_engine.py"}


@pytest.mark.parametrize("py_file", [
    p for p in _CORE_DIR.glob("*.py")
    if p.name != "__init__.py" and p.name not in _DAG_EXEMPT
])
def test_core_file_does_not_import_runtime(py_file: pathlib.Path):
    """htp/core/<file>.py 가 htp.runtime 을 import 하지 않아야 함."""
    from_mods = _from_modules(py_file)
    direct    = _direct_imports(py_file)

    violations = [m for m in from_mods + direct
                  if m and ("htp.runtime" in m or m.startswith("..runtime"))]
    assert not violations, (
        f"DAG violation in {py_file.name}: {violations}\n"
        f"htp/core/* 는 htp/runtime/* 를 import 할 수 없음 (단방향 DAG)"
    )


# ══════════════════════════════════════════════════════════
# DAG 규칙 (htp-thalamus-car sub-1):
# htp/knowledge/*.py 는 htp.runtime / htp.thalamus / htp.memory 모두 미참조
# Design Ref: htp-thalamus-car.design.md §2.2
# ══════════════════════════════════════════════════════════

@pytest.mark.parametrize("py_file", [
    p for p in _KNOWLEDGE_DIR.glob("*.py")
    if p.name != "__init__.py"
] if _KNOWLEDGE_DIR.exists() else [])
def test_knowledge_file_dag_isolation(py_file: pathlib.Path):
    """htp/knowledge/<file>.py 가 htp.runtime/thalamus/memory 를 import 하지 않아야 함."""
    from_mods = _from_modules(py_file)
    direct    = _direct_imports(py_file)

    forbidden = ("htp.runtime", "htp.thalamus", "htp.memory")
    violations = [
        m for m in from_mods + direct
        if m and any(f in m for f in forbidden)
    ]
    assert not violations, (
        f"DAG violation in htp/knowledge/{py_file.name}: {violations}\n"
        f"htp/knowledge/* 는 htp.runtime/thalamus/memory 를 import 할 수 없음"
    )


# ══════════════════════════════════════════════════════════
# DAG 규칙 (htp-thalamus-car sub-2 M8):
# htp/thalamus/router/*.py 는 htp.runtime / htp.memory / htp.knowledge 미참조
# 허용: htp/thalamus 형제 (signature, region_signal) + numpy + typing
# Design Ref: htp-thalamus-car_sub-2_design v1.md §3
# ══════════════════════════════════════════════════════════

@pytest.mark.parametrize("py_file", [
    p for p in _ROUTER_DIR.glob("*.py")
    if p.name != "__init__.py"
] if _ROUTER_DIR.exists() else [])
def test_router_file_dag_isolation(py_file: pathlib.Path):
    """htp/thalamus/router/<file>.py 의 DAG 단방향성 보장.

    sub-5 (Stage 6 EmbeddingBridge) 시 동일 Protocol 추가 구현체가 끼워질
    때도 이 규칙 유지 — runtime / memory / knowledge 로의 역방향 금지.
    """
    from_mods = _from_modules(py_file)
    direct    = _direct_imports(py_file)

    forbidden = ("htp.runtime", "htp.memory", "htp.knowledge")
    violations = [
        m for m in from_mods + direct
        if m and any(f in m for f in forbidden)
    ]
    assert not violations, (
        f"DAG violation in htp/thalamus/router/{py_file.name}: {violations}\n"
        f"htp/thalamus/router/* 는 htp.runtime/memory/knowledge 미참조 유지"
    )


# ══════════════════════════════════════════════════════════
# DAG 규칙 (htp-thalamus-car sub-3 M6):
# htp/thalamus/coherence/*.py 는 htp.runtime/memory/knowledge 미참조
# 또한 htp.thalamus.router 도 미참조 (coherence ↔ router 독립)
# Design Ref: htp-thalamus-car.sub-3.design.md §3
# ══════════════════════════════════════════════════════════

@pytest.mark.parametrize("py_file", [
    p for p in _COHERENCE_DIR.glob("*.py")
    if p.name != "__init__.py"
] if _COHERENCE_DIR.exists() else [])
def test_coherence_file_dag_isolation(py_file: pathlib.Path):
    """htp/thalamus/coherence/<file>.py 의 DAG 단방향성 보장."""
    from_mods = _from_modules(py_file)
    direct    = _direct_imports(py_file)

    forbidden = ("htp.runtime", "htp.memory",
                 "htp.knowledge", "htp.thalamus.router")
    violations = [
        m for m in from_mods + direct
        if m and any(f in m for f in forbidden)
    ]
    assert not violations, (
        f"DAG violation in htp/thalamus/coherence/{py_file.name}: {violations}\n"
        f"htp/thalamus/coherence/* 는 runtime/memory/knowledge/router 미참조 유지"
    )


# ══════════════════════════════════════════════════════════
# DAG 규칙 2: htp 패키지 전체가 순환 없이 import 가능
# ══════════════════════════════════════════════════════════

def test_top_level_htp_import_no_circular():
    """import htp 가 ImportError 없이 깨끗하게 완료되어야 함."""
    # 실행만 되면 통과 (이 자체로 순환 부재 증명)
    import htp
    assert htp.__name__ == "htp"


def test_core_init_lazy_loading_works():
    """htp/core/__init__.py 가 PEP 562 lazy __getattr__ 로 NGE 를 늦게 로드."""
    # NGE 심볼은 처음 접근 시점에 로드되어야 한다 (순환 회피 패턴)
    from htp import core
    # 명시 export 된 sub-config 들은 즉시 접근 가능
    assert hasattr(core, "HubConfig")
    assert hasattr(core, "PruneConfig")
    assert hasattr(core, "ActivationConfig")
    assert hasattr(core, "WeightMatrix")
    # NGE 는 lazy — getattr 호출 시점에 로드
    NodeGenerationEngine = core.NodeGenerationEngine
    assert NodeGenerationEngine is not None


# ══════════════════════════════════════════════════════════
# DAG 규칙 3: htp/core/__init__.py 의 eager import 는 안전한 모듈만
# ══════════════════════════════════════════════════════════

def test_core_init_eager_imports_are_safe():
    """htp/core/__init__.py 가 eager 로 import 하는 모듈은 모두 DAG 안전 해야 함."""
    init_file = _CORE_DIR / "__init__.py"
    from_mods = _from_modules(init_file)

    # eager 로 가져오는 sub-module: .config, .weight_matrix, .hub_formation, .pruning, .activation
    eager_targets = {m for m in from_mods if m and m.startswith(".") and "node_generation" not in m}

    # 각 eager target 파일이 DAG 안전한지 (runtime 미참조)
    for target in eager_targets:
        # ".config" → "config.py"
        module_name = target.lstrip(".")
        target_file = _CORE_DIR / f"{module_name}.py"
        if not target_file.exists():
            continue  # 옵션 모듈 (예: 향후 추가될 수 있는 파일)

        target_from = _from_modules(target_file)
        violations = [m for m in target_from if "htp.runtime" in m or "..runtime" in m]
        assert not violations, (
            f"Eager import {target} → {target_file.name} pulls in runtime: {violations}\n"
            f"이는 step-1 에서 발견한 순환 import 재발 위험."
        )
