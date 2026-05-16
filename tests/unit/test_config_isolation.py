"""
HTPConfig facade backward compatibility unit tests.

Design Ref: docs/02-design/features/htp-review-improvements.design.md §2.3 + §7.2
Plan SC: FR-02 (HTPConfig facade), FR-11 (구조 검증)

이 테스트는 HTPConfig facade 가 다음 4가지 호출 패턴 모두 지원함을 영구 보호한다:
  1. 기본 생성자                 HTPConfig()
  2. flat 키워드 (legacy)        HTPConfig(hub_pr_threshold=3.0)
  3. sub-config 직접 주입 (new)  HTPConfig(hub=HubConfig(...))
  4. flat 속성 접근 (legacy)     cfg.hub_pr_threshold

Flat 호환 레이어가 사라지면 `server.py`/기존 사용자 코드가 모두 깨지므로
이 테스트가 그 회귀를 막아준다.
"""
from __future__ import annotations

import pytest

from htp                import HTPConfig
from htp.core.config    import (
    HubConfig, PruneConfig, ActivationConfig,
    # htp-thalamus-car sub-1 (Stage 0)
    RoutingConfig, CoherenceConfig, LLMBridgeConfig, PipelineConfig,
)


# ══════════════════════════════════════════════════════════
# 기본 동작
# ══════════════════════════════════════════════════════════

def test_default_constructor_creates_all_sub_configs():
    cfg = HTPConfig()
    assert isinstance(cfg.hub,        HubConfig)
    assert isinstance(cfg.prune,      PruneConfig)
    assert isinstance(cfg.activation, ActivationConfig)
    assert cfg.n_nodes == 64
    assert cfg.device in ("cpu", "cuda")


def test_default_field_values_match_legacy_htp_config():
    """기본값이 step-1 이전 HTPConfig 와 동일해야 회귀 보호."""
    cfg = HTPConfig()
    # HubConfig
    assert cfg.hub.threshold        == 0.35
    assert cfg.hub.hebbian_lr       == 0.13
    assert cfg.hub.hub_pr_threshold == 2.5
    # PruneConfig
    assert cfg.prune.decay_rate      == 0.005
    assert cfg.prune.prune_threshold == 0.02
    assert cfg.prune.hub_protect     is True
    assert cfg.prune.age_threshold   == 100


# ══════════════════════════════════════════════════════════
# Backward compat — flat 키워드 생성자
# ══════════════════════════════════════════════════════════

def test_flat_keyword_constructor_dispatches_to_hub():
    """server.py 같은 옛 코드가 HTPConfig(hub_pr_threshold=3.0) 호출하면
    HubConfig.hub_pr_threshold 로 위임."""
    cfg = HTPConfig(hub_pr_threshold=3.0)
    assert cfg.hub.hub_pr_threshold == 3.0
    # 다른 sub-config 는 기본값 유지
    assert cfg.prune.decay_rate == 0.005


def test_flat_keyword_constructor_dispatches_to_prune():
    cfg = HTPConfig(decay_rate=0.01, hub_protect=False)
    assert cfg.prune.decay_rate  == 0.01
    assert cfg.prune.hub_protect is False
    # HubConfig 는 무영향
    assert cfg.hub.hub_pr_threshold == 2.5


def test_flat_keyword_with_multiple_sub_config_fields():
    """여러 sub-config 의 필드를 동시에 flat 으로 전달."""
    cfg = HTPConfig(
        hub_pr_threshold = 3.0,    # HubConfig
        decay_rate       = 0.01,   # PruneConfig
        n_nodes          = 128,    # top-level
    )
    assert cfg.hub.hub_pr_threshold == 3.0
    assert cfg.prune.decay_rate     == 0.01
    assert cfg.n_nodes              == 128


def test_unknown_kwarg_raises_type_error():
    """오타·존재하지 않는 필드는 명시적 TypeError."""
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        HTPConfig(nonsense_field=1)


# ══════════════════════════════════════════════════════════
# Backward compat — flat 속성 read/write
# ══════════════════════════════════════════════════════════

def test_flat_attribute_read_delegates_to_sub_config():
    """cfg.hub_pr_threshold → cfg.hub.hub_pr_threshold (__getattr__ 위임)."""
    cfg = HTPConfig(hub=HubConfig(hub_pr_threshold=5.0))
    assert cfg.hub_pr_threshold == 5.0   # flat read
    assert cfg.hub.hub_pr_threshold == 5.0  # nested read
    # 두 경로가 같은 값을 반환해야 함
    assert cfg.hub_pr_threshold == cfg.hub.hub_pr_threshold


def test_flat_attribute_write_delegates_to_sub_config():
    """cfg.hub_pr_threshold = 4.0 → cfg.hub.hub_pr_threshold = 4.0."""
    cfg = HTPConfig()
    cfg.hub_pr_threshold = 4.0
    assert cfg.hub.hub_pr_threshold == 4.0


def test_top_level_field_assignment_works():
    """HTPRuntime._ensure_built 가 self.cfg.n_nodes = N 를 호출함."""
    cfg = HTPConfig()
    cfg.n_nodes = 256
    assert cfg.n_nodes == 256


def test_setattr_to_unknown_field_raises():
    """알 수 없는 속성 set 은 거부 — 오타 방지."""
    cfg = HTPConfig()
    with pytest.raises(AttributeError, match="Cannot set unknown"):
        cfg.totally_made_up_field = 42


# ══════════════════════════════════════════════════════════
# Sub-config 직접 주입 (새 권장 방식)
# ══════════════════════════════════════════════════════════

def test_direct_sub_config_injection():
    """사용자가 HubConfig 를 직접 만들어 주입할 수 있어야 함."""
    hub_cfg = HubConfig(hebbian_lr=0.2, threshold=0.5)
    cfg     = HTPConfig(hub=hub_cfg)
    assert cfg.hub is hub_cfg


def test_mixed_direct_and_flat_kwargs():
    """sub-config 직접 주입 + flat 키워드 혼합 — flat 이 사후 override."""
    cfg = HTPConfig(
        hub        = HubConfig(hub_pr_threshold=2.0),
        decay_rate = 0.01,   # PruneConfig 의 필드를 flat 으로
    )
    assert cfg.hub.hub_pr_threshold == 2.0
    assert cfg.prune.decay_rate     == 0.01


# ══════════════════════════════════════════════════════════
# htp-thalamus-car sub-1 (Stage 0) — 4 신규 sub-config
# Design Ref: htp-thalamus-car.design.md §3 + §5.2
# ══════════════════════════════════════════════════════════

def test_routing_config_isolation():
    """RoutingConfig 독립 생성 + 기본값. Design §3.1."""
    cfg = RoutingConfig()
    assert cfg.mode           == "tag"     # 회귀 보호: 기본값 tag mode
    assert cfg.alpha          == 0.5
    assert cfg.threshold_beta == 0.5
    assert cfg.warmup_steps   == 10

    # 사용자 정의
    cfg2 = RoutingConfig(mode="vector", alpha=0.7)
    assert cfg2.mode  == "vector"
    assert cfg2.alpha == 0.7


def test_coherence_config_isolation():
    """CoherenceConfig 독립 생성 + LSH transition threshold 16. Design §3.1."""
    cfg = CoherenceConfig()
    assert cfg.conflict_threshold  == 0.3
    assert cfg.agreement_threshold == 0.7
    assert cfg.novelty_boost       == 1.0
    assert cfg.lsh_transition_n    == 16   # ModalEncoder 통합 시점과 동일 임계


def test_subconfig_flat_kwarg_dispatch():
    """HTPConfig(threshold_beta=0.7, lsh_transition_n=32) 가 올바른 sub-config 로 위임됨."""
    cfg = HTPConfig(
        threshold_beta   = 0.7,    # RoutingConfig 의 필드
        lsh_transition_n = 32,     # CoherenceConfig 의 필드
        embed_dim        = 768,    # LLMBridgeConfig 의 필드
        buffer_size      = 5,      # PipelineConfig 의 필드
    )
    assert cfg.routing.threshold_beta     == 0.7
    assert cfg.coherence.lsh_transition_n == 32
    assert cfg.llm_bridge.embed_dim       == 768
    assert cfg.pipeline.buffer_size       == 5

    # flat attribute read 도 동일 동작
    assert cfg.threshold_beta     == 0.7
    assert cfg.lsh_transition_n   == 32
    assert cfg.embed_dim          == 768
    assert cfg.buffer_size        == 5

    # 기본값 보호: 다른 sub-config 는 영향 없음
    assert cfg.hub.hub_pr_threshold == 2.5  # Phase 1 기본값 유지
