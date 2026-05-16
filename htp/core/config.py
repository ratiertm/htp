"""
Sub-config dataclasses for Phase 1 engines.

Design Ref: docs/02-design/features/htp-review-improvements.design.md §2.3 — Sub-config split
Plan SC: FR-01 (4-sub-config dataclass introduction)

이 파일은 htp/core/ 트리에 속하므로 htp/runtime/* 를 import 하지 않는다 (DAG 강제).
HTPConfig facade 는 htp/runtime/htp_runtime.py 에 위치하며 이 모듈을 임포트한다.

각 sub-config 는 해당 엔진이 *현재* 실제로 사용하는 필드만 포함한다.
n_nodes / device 는 모든 엔진이 공유하므로 HTPConfig top-level 에 유지된다.
"""
from __future__ import annotations

from dataclasses import dataclass


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


__all__ = ["HubConfig", "PruneConfig", "ActivationConfig"]
