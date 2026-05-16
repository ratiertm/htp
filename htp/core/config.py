"""
Phase 1 sub-config dataclasses + HTPConfig facade.

Design Ref: docs/02-design/features/htp-review-improvements.design.md §2.3 — Sub-config split
Plan SC: FR-01 (4-sub-config dataclass), FR-02 (HTPConfig facade)

이 파일은 htp/core/ 트리에 속하므로 htp/runtime/* 를 import 하지 않는다 (DAG 강제).
HTPConfig facade 는 step-7 에서 htp/runtime/htp_runtime.py 로부터 여기로 이전됨.

각 sub-config 는 해당 엔진이 *현재* 실제로 사용하는 필드만 포함한다.
n_nodes / device 는 모든 엔진이 공유하므로 HTPConfig facade top-level 에 유지된다.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing      import Any, Optional

import torch


# ══════════════════════════════════════════════════════════
# HubConfig  -  HubFormationEngine 전용 파라미터
# ══════════════════════════════════════════════════════════
@dataclass
class HubConfig:
    """HubFormationEngine 의 발화·헤비안·허브 승격 파라미터."""
    threshold:        float = 0.35  # 노드 발화 임계값 (Laplacian 전파 후 energy 비교)
    hebbian_lr:       float = 0.13  # Oja's Rule 학습률
    hub_pr_threshold: float = 2.5   # PageRank 기준 허브 승격 (1/N 대비 배수)


# ══════════════════════════════════════════════════════════
# PruneConfig  -  PruningEngine 전용 파라미터
# ══════════════════════════════════════════════════════════
@dataclass
class PruneConfig:
    """PruningEngine 의 4가지 전략 + 허브 보호 파라미터."""
    decay_rate:      float = 0.005  # 시간 감쇠율 (매 스텝)
    prune_threshold: float = 0.02   # 이 이하면 엣지 제거
    usage_window:    int   = 20     # 사용 빈도 측정 윈도우 (스텝)
    usage_min:       float = 0.05   # 윈도우 내 최소 발화 비율
    redundancy_cos:  float = 0.95   # 코사인 유사도 임계값 (중복 노드 감지)
    age_threshold:   int   = 100    # age 전략: 이 스텝 이상 강화 없으면 제거
    hub_protect:     bool  = True   # 허브 노드 관여 연결 보호


# ══════════════════════════════════════════════════════════
# ActivationConfig  -  ActivationEngine 전용 (현재 비어있음, 향후 확장용)
# ══════════════════════════════════════════════════════════
@dataclass
class ActivationConfig:
    """
    ActivationEngine 파라미터 placeholder.

    현재 ActivationEngine 은 HTPConfig 필드를 직접 읽지 않는다
    (max_depth 는 run() 의 함수 인자). 향후 시맨틱 라우팅 임계값 등이
    추가될 때 이 dataclass 에 모음.
    """
    pass


# ══════════════════════════════════════════════════════════
# RoutingConfig  -  Content-Addressable Routing (Stage 0, htp-thalamus-car sub-1)
# ══════════════════════════════════════════════════════════
#
# Design Ref: docs/02-design/features/htp-thalamus-car.design.md §3.1
# Plan SC: FR-01 (htp-thalamus-car)

@dataclass
class RoutingConfig:
    """Content-Addressable Routing 설정 (CoreCells._gate_*).

    sub-1 에서는 mode 만 정의되고 vector/hybrid 동작은 sub-2 에서 활성화.
    """
    mode:           str   = "tag"   # "tag" | "vector" | "hybrid"
    alpha:          float = 0.5     # hybrid 모드 vector 비중
    threshold_beta: float = 0.5     # θ = μ + β×σ (dynamic threshold)
    warmup_steps:   int   = 10      # signature 냉시작 보호


# ══════════════════════════════════════════════════════════
# CoherenceConfig  -  CoherenceGate (sub-3 에서 활성화)
# ══════════════════════════════════════════════════════════
@dataclass
class CoherenceConfig:
    """CoherenceGate 의 temporal binding 파라미터."""
    conflict_threshold:  float = 0.3
    agreement_threshold: float = 0.7
    novelty_boost:       float = 1.0   # conflict → SWR priority 증폭 계수
    lsh_transition_n:    int   = 16    # Region ≥ N 이면 LSH 근사 전환


# ══════════════════════════════════════════════════════════
# LLMBridgeConfig  -  EmbeddingBridge + CostRouter (sub-4, sub-5)
# ══════════════════════════════════════════════════════════
@dataclass
class LLMBridgeConfig:
    """sLLM 임베딩 프로젝션 + 4-Level CostRouter 파라미터."""
    embedding_model:           str   = "BAAI/bge-small-ko-v1.5"
    embed_dim:                 int   = 384
    cost_level_thresholds:     tuple = (0.2, 0.5, 0.8)
    budget_pressure_threshold: float = 0.8


# ══════════════════════════════════════════════════════════
# PipelineConfig  -  PipelinedBrainRuntime (sub-4)
# ══════════════════════════════════════════════════════════
@dataclass
class PipelineConfig:
    """L3 파이프라인 병렬성 버퍼 크기."""
    buffer_size: int = 3


# ══════════════════════════════════════════════════════════
# HTPConfig  -  Facade over sub-configs (Step 7 에서 이전됨)
# ══════════════════════════════════════════════════════════
#
# HTPConfig 는 sub-config(HubConfig/PruneConfig/ActivationConfig) 들을 묶는 facade.
# Backward compatibility 를 위해 다음을 보존한다:
#   - flat 키워드 생성자:  HTPConfig(hub_pr_threshold=3.0)  → self.hub.hub_pr_threshold = 3.0
#   - flat 속성 접근:      cfg.hub_pr_threshold              → self.hub.hub_pr_threshold (via __getattr__)
#   - top-level 필드:      n_nodes, device 는 모든 엔진이 공유하므로 facade 본체에 유지
#
# 새 권장 사용 방식 (옵션):
#   HTPConfig(hub=HubConfig(hub_pr_threshold=3.0))

# Design Ref: htp-thalamus-car sub-1 §3.2 — facade 확장으로 신규 4 sub-config 흡수.
# 기존 (hub/prune/activation) + 신규 (routing/coherence/llm_bridge/pipeline) = 7 sub-config.
_SUBCONFIG_NAMES: tuple = (
    "hub", "prune", "activation",        # htp-review-improvements (이전 사이클)
    "routing", "coherence", "llm_bridge", "pipeline",  # htp-thalamus-car sub-1 (Stage 0)
)


def _all_subconfig_fields() -> set:
    """모든 sub-config dataclass field 이름 합집합 (flat kwarg dispatch 용)."""
    return (
        set(HubConfig.__dataclass_fields__)
        | set(PruneConfig.__dataclass_fields__)
        | set(ActivationConfig.__dataclass_fields__)
        | set(RoutingConfig.__dataclass_fields__)
        | set(CoherenceConfig.__dataclass_fields__)
        | set(LLMBridgeConfig.__dataclass_fields__)
        | set(PipelineConfig.__dataclass_fields__)
    )


class HTPConfig:
    """
    Facade over Phase 1-5 + Thalamus CAR sub-configs.

    Top-level fields (shared across engines):
      - n_nodes: int      네트워크 노드 수
      - device:  str      torch device ('cpu' or 'cuda')

    Sub-configs:
      - hub:        HubConfig         HubFormationEngine
      - prune:      PruneConfig       PruningEngine
      - activation: ActivationConfig  ActivationEngine
      - routing:    RoutingConfig     CoreCells routing mode (htp-thalamus-car sub-1)
      - coherence:  CoherenceConfig   CoherenceGate (sub-3)
      - llm_bridge: LLMBridgeConfig   EmbeddingBridge + CostRouter (sub-4/5)
      - pipeline:   PipelineConfig    PipelinedBrainRuntime (sub-4)
    """

    __slots__ = ("n_nodes", "device",
                 "hub", "prune", "activation",
                 "routing", "coherence", "llm_bridge", "pipeline")

    def __init__(self,
                 n_nodes:    int = 64,
                 device:     Optional[str] = None,
                 hub:        Optional[HubConfig]        = None,
                 prune:      Optional[PruneConfig]      = None,
                 activation: Optional[ActivationConfig] = None,
                 routing:    Optional[RoutingConfig]    = None,
                 coherence:  Optional[CoherenceConfig]  = None,
                 llm_bridge: Optional[LLMBridgeConfig]  = None,
                 pipeline:   Optional[PipelineConfig]   = None,
                 **kwargs: Any):
        object.__setattr__(self, "n_nodes", n_nodes)
        object.__setattr__(self, "device",
                           device if device is not None
                           else ("cuda" if torch.cuda.is_available() else "cpu"))
        # Phase 1 sub-configs (이전 사이클)
        object.__setattr__(self, "hub",        hub        if hub        is not None else HubConfig())
        object.__setattr__(self, "prune",      prune      if prune      is not None else PruneConfig())
        object.__setattr__(self, "activation", activation if activation is not None else ActivationConfig())
        # CAR sub-configs (Stage 0)
        object.__setattr__(self, "routing",    routing    if routing    is not None else RoutingConfig())
        object.__setattr__(self, "coherence",  coherence  if coherence  is not None else CoherenceConfig())
        object.__setattr__(self, "llm_bridge", llm_bridge if llm_bridge is not None else LLMBridgeConfig())
        object.__setattr__(self, "pipeline",   pipeline   if pipeline   is not None else PipelineConfig())

        for k, v in kwargs.items():
            assigned = False
            for name in _SUBCONFIG_NAMES:
                sub = getattr(self, name)
                if hasattr(sub, k):
                    setattr(sub, k, v)
                    assigned = True
                    break
            if not assigned:
                raise TypeError(
                    f"HTPConfig got unexpected keyword argument: {k!r}. "
                    f"Known top-level: n_nodes/device/{'/'.join(_SUBCONFIG_NAMES)}. "
                    f"Known sub-config fields: {sorted(_all_subconfig_fields())}"
                )

    # ── Backward-compat attribute access ──────────────────
    def __getattr__(self, name: str) -> Any:
        # __slots__ fields 는 정상 lookup, 여기 안 옴.
        # flat 속성을 sub-config 로 위임.
        for sub_name in _SUBCONFIG_NAMES:
            try:
                sub = object.__getattribute__(self, sub_name)
            except AttributeError:
                continue
            cls = type(sub)
            if hasattr(cls, "__dataclass_fields__") and name in cls.__dataclass_fields__:
                return getattr(sub, name)
        raise AttributeError(f"'HTPConfig' object has no attribute {name!r}")

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self.__slots__:
            object.__setattr__(self, name, value)
            return
        # flat 속성 set 위임
        for sub_name in _SUBCONFIG_NAMES:
            try:
                sub = object.__getattribute__(self, sub_name)
            except AttributeError:
                continue
            cls = type(sub)
            if hasattr(cls, "__dataclass_fields__") and name in cls.__dataclass_fields__:
                setattr(sub, name, value)
                return
        raise AttributeError(
            f"Cannot set unknown HTPConfig attribute: {name!r}. "
            f"To add new fields, edit the appropriate sub-config in htp/core/config.py"
        )

    def __repr__(self) -> str:
        subs = ", ".join(f"{n}={getattr(self, n)}" for n in _SUBCONFIG_NAMES)
        return f"HTPConfig(n_nodes={self.n_nodes}, device={self.device!r}, {subs})"


__all__ = [
    # Phase 1 (이전 사이클)
    "HubConfig", "PruneConfig", "ActivationConfig",
    # Thalamus CAR sub-1 (Stage 0)
    "RoutingConfig", "CoherenceConfig", "LLMBridgeConfig", "PipelineConfig",
    # Facade
    "HTPConfig",
]
