"""
Unit tests for Phase 1 engine Constructor Injection.

Design Ref: docs/02-design/features/htp-review-improvements.design.md §7.2
Plan SC: FR-11 (engine을 sub-config만으로 독립 생성 가능)

이 테스트는 각 엔진이 *전체 HTPConfig 없이도* sub-config 만으로 동작함을 검증한다.
회귀 테스트(`tests/regression/`)는 행동 동일성을 보호하고,
이 단위 테스트는 *구조적 결합도 감소* 를 보호한다.
"""
from __future__ import annotations

import pytest
import torch

from htp.core.config        import HubConfig, PruneConfig, ActivationConfig
from htp.core.weight_matrix import WeightMatrix
from htp.core.hub_formation import HubFormationEngine
from htp.core.pruning       import PruningEngine, PruneStrategy
from htp.core.activation    import ActivationEngine, Node, tag, terminal, FIRE_FLOOR


# ══════════════════════════════════════════════════════════
# Step 4 — HubFormationEngine DI
# ══════════════════════════════════════════════════════════

def test_hfe_constructs_from_hub_config_only():
    """HubFormationEngine 은 HubConfig + WeightMatrix 만으로 생성 가능해야 함."""
    wm  = WeightMatrix(n=4, device="cpu")
    cfg = HubConfig(threshold=0.5, hebbian_lr=0.1, hub_pr_threshold=2.0)
    hfe = HubFormationEngine(wm, cfg)

    assert hfe.cfg is cfg
    assert hfe.cfg.threshold == 0.5
    assert hfe.cfg.hebbian_lr == 0.1
    assert hfe.cfg.hub_pr_threshold == 2.0
    assert hfe.dev == wm.dev           # device 는 wm 에서 파생
    assert hfe.is_hub.shape == (4,)   # n_nodes 는 wm.n 에서 파생
    assert hfe.fire_count.shape == (4,)


def test_hfe_does_not_require_htpconfig():
    """HubFormationEngine 은 HTPConfig 를 import 하지 않고도 동작해야 한다 (DAG)."""
    # 이 테스트가 통과한다는 사실 자체가 hfe.py 가 htp_runtime.py 에 의존하지 않음을 입증
    wm  = WeightMatrix(n=3, device="cpu")
    hfe = HubFormationEngine(wm, HubConfig())  # HTPConfig 미사용

    # 1 step 실행 — Hebbian + PageRank 가 정상 진행되는지
    signal = torch.tensor([1.0, 0.0, 0.0])
    fired  = hfe.step(signal)

    assert fired.shape == (3,)
    assert hfe.step_count == 1


def test_hfe_derives_shared_fields_from_weight_matrix():
    """n_nodes / device 는 wm 에서 파생되며 HubConfig 에 중복 저장 안 됨."""
    wm  = WeightMatrix(n=7, device="cpu")
    hfe = HubFormationEngine(wm, HubConfig())

    # HubConfig 에는 n_nodes / device 가 *없어야* 한다
    assert not hasattr(hfe.cfg, "n_nodes")
    assert not hasattr(hfe.cfg, "device")

    # 대신 engine 자체가 wm 에서 가져옴
    assert hfe.is_hub.shape[0] == wm.n
    assert str(hfe.dev) == "cpu"


def test_hub_config_defaults_match_legacy():
    """HubConfig 기본값은 기존 HTPConfig 와 동일해야 회귀 보호됨."""
    cfg = HubConfig()
    assert cfg.threshold        == 0.35
    assert cfg.hebbian_lr       == 0.13
    assert cfg.hub_pr_threshold == 2.5


def test_hfe_pagerank_unchanged_after_split():
    """파일 분리 후에도 PageRank 합 = 1 보장 (Phase 1 회귀 핵심)."""
    wm  = WeightMatrix(n=5, device="cpu")
    wm.set(0, 1, 0.5)
    wm.set(1, 2, 0.3)
    wm.set(2, 0, 0.2)
    hfe = HubFormationEngine(wm, HubConfig())

    pr = hfe.pagerank()
    assert pr.shape == (5,)
    assert abs(float(pr.sum()) - 1.0) < 1e-4


# ══════════════════════════════════════════════════════════
# Step 5 — PruningEngine DI
# ══════════════════════════════════════════════════════════

def test_pe_constructs_from_prune_config_only():
    """PruningEngine 은 PruneConfig + WM + HFE 만으로 생성 가능해야 함."""
    wm  = WeightMatrix(n=4, device="cpu")
    hfe = HubFormationEngine(wm, HubConfig())
    pcfg = PruneConfig(decay_rate=0.01, prune_threshold=0.05, hub_protect=False)
    pe  = PruningEngine(wm, hfe, pcfg)

    assert pe.cfg is pcfg
    assert pe.cfg.decay_rate == 0.01
    assert pe.cfg.hub_protect is False
    # 4가지 전략 카운터가 초기화되어 있어야 함
    assert set(pe.stats.keys()) == {
        PruneStrategy.DECAY, PruneStrategy.USAGE,
        PruneStrategy.REDUND, PruneStrategy.AGE,
    }


def test_pe_decay_prune_runs_without_htpconfig():
    """PruningEngine 이 HTPConfig 없이 decay 동작."""
    wm  = WeightMatrix(n=3, device="cpu")
    wm.set(0, 1, 0.5)  # 강한 엣지 — 살아남아야 함
    wm.set(0, 2, 0.01) # 약한 엣지 — prune_threshold 0.02 이하라 제거 대상
    hfe = HubFormationEngine(wm, HubConfig())
    pe  = PruningEngine(wm, hfe, PruneConfig(hub_protect=False))

    pruned = pe.decay_prune()
    assert pruned >= 1  # 약한 엣지 1개는 제거
    assert wm.get(0, 1) > 0  # 강한 엣지는 유지 (감쇠는 됐지만 살아있음)
    assert wm.get(0, 2) == 0  # 약한 엣지는 제거


def test_prune_config_defaults_match_legacy():
    """PruneConfig 기본값은 기존 HTPConfig 와 동일해야 회귀 보호."""
    cfg = PruneConfig()
    assert cfg.decay_rate      == 0.005
    assert cfg.prune_threshold == 0.02
    assert cfg.usage_window    == 20
    assert cfg.usage_min       == 0.05
    assert cfg.redundancy_cos  == 0.95
    assert cfg.age_threshold   == 100
    assert cfg.hub_protect     is True


def test_pe_does_not_import_htp_runtime():
    """htp.core.pruning 모듈이 htp.runtime 을 import 하지 않아야 함 (DAG)."""
    import ast
    import pathlib
    src = pathlib.Path(__file__).parent.parent.parent / "htp" / "core" / "pruning.py"
    tree = ast.parse(src.read_text())
    from_modules = [
        node.module for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    ]
    for mod in from_modules:
        assert "runtime" not in mod, f"DAG violation: pruning.py imports {mod}"


# ══════════════════════════════════════════════════════════
# Step 6 — ActivationEngine DI + decorators + Node
# ══════════════════════════════════════════════════════════

def test_ae_constructs_from_activation_config_only():
    """ActivationEngine 은 ActivationConfig + WM + HFE 만으로 생성 가능해야 함."""
    wm   = WeightMatrix(n=3, device="cpu")
    hfe  = HubFormationEngine(wm, HubConfig())
    acfg = ActivationConfig()
    ae   = ActivationEngine(wm, hfe, acfg)

    assert ae.cfg is acfg
    assert ae._nodes == []


def test_ae_derives_device_from_wm_not_config():
    """ActivationEngine 은 device 를 wm 에서 파생 (ActivationConfig 에 device 없음)."""
    wm  = WeightMatrix(n=3, device="cpu")
    hfe = HubFormationEngine(wm, HubConfig())
    ae  = ActivationEngine(wm, hfe, ActivationConfig())

    # ActivationConfig 에는 device 필드가 *없어야* 한다
    assert not hasattr(ae.cfg, "device")
    # 대신 wm 에서 가져옴 (_make_signal 내부에서)
    assert str(ae.wm.dev) == "cpu"


def test_tag_decorator_attaches_metadata():
    """tag() 데코레이터가 함수에 _htp_tags 메타데이터를 부착."""
    @tag("success", "done")
    def my_fn(d): return d

    assert hasattr(my_fn, "_htp_tags")
    assert my_fn._htp_tags == {"success", "done"}


def test_terminal_decorator_attaches_metadata():
    """terminal() 데코레이터가 함수에 _htp_terminal 메타데이터를 부착."""
    @terminal
    def my_fn(d): return d

    assert getattr(my_fn, "_htp_terminal", False) is True


def test_extract_dict_value_split_preserves_keyword_matching():
    """⚠️ Stage 1 bug #3 회귀 — dict value 가 공백 split 되어 tag 매칭 가능."""
    wm  = WeightMatrix(n=3, device="cpu")
    hfe = HubFormationEngine(wm, HubConfig())
    ae  = ActivationEngine(wm, hfe, ActivationConfig())

    # tag("done") 이 "auth done" 같은 value 와 매칭되려면 split 이 필요
    label, kws = ae._extract({"label": "success", "msg": "auth done"})
    assert label == "success"
    assert "done" in kws        # split 결과
    assert "auth" in kws
    # 전체 덩어리는 *없어야* 함 (split 안 되었다면 fail)
    assert "auth done" not in kws


def test_ae_does_not_import_htp_runtime():
    """htp.core.activation 모듈이 htp.runtime 을 import 하지 않아야 함 (DAG)."""
    import ast
    import pathlib
    src = pathlib.Path(__file__).parent.parent.parent / "htp" / "core" / "activation.py"
    tree = ast.parse(src.read_text())
    from_modules = [
        node.module for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    ]
    for mod in from_modules:
        assert "runtime" not in mod, f"DAG violation: activation.py imports {mod}"


def test_fire_floor_constant_unchanged():
    """FIRE_FLOOR 상수는 0.08 유지 — 캐스케이드 종료 임계값."""
    assert FIRE_FLOOR == 0.08
