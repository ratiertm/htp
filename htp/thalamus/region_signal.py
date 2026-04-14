"""
Region Signal  —  Region → Thalamus 통신 단위
==============================================

생물학: 피질 → 시상 피드포워드 신호
  - 허브 강도: 해당 영역에서 가장 활성화된 핵심 노드의 PageRank 점수
  - 발화율: 영역 전체 노드의 평균 최근 발화 빈도
  - overload: Shannon Entropy + CUSUM 기반 과부하 감지

ThalamusOutput: 시상 → PFC 압축 신호
  - state_vec: JL Random Projection으로 압축된 64차원 벡터 (Stage 4 이후; 이전 8-dim)
  - suppressed: Lateral Inhibition WTA 억제 강도
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class RegionSignal:
    """피질 Region → Thalamus 통신 단위."""
    region_id:    str
    hub_strength: float       # PageRank 기반 최고 허브 점수 (0~1/N 범위)
    fire_rate:    float       # 전체 노드 평균 최근 발화율 (0~1)
    top_hubs:     list        # [(node_name, pagerank_score), ...]
    overload:     bool        # Shannon Entropy + CUSUM 과부하 여부
    output_vec:   torch.Tensor  # W.sum(dim=1) — 출력 강도 벡터 (압축 전)
    precision:    float = 1.0   # [Stage 3-B1] Friston precision. 높을수록 신뢰도 ↑,
                                # CoreCells gate 에서 score 를 amplification 하는 scaler.
                                # 1.0 = 중립. Stage 3-B2 에서 Region 이 동적 계산.


@dataclass
class GatingMask:
    """CoreCells Sigmoidal Gating 결과."""
    scores: dict   # {region_id: gate_strength 0~1}


@dataclass
class CompetitionResult:
    """MatrixCells Lateral Inhibition + Softmax WTA 결과."""
    winner_id:       str
    winner_score:    float
    suppression_map: dict   # {region_id: Δp 억제 강도 0~1}
    all_scores:      dict   # {region_id: softmax_prob}


@dataclass
class ThalamusOutput:
    """Thalamus → PFCRuntime 출력 (JL 압축 벡터)."""
    winner:     str
    state_vec:  torch.Tensor   # JL Random Projection 64-dim (Stage 4 이후)
    gating:     GatingMask
    suppressed: dict           # {region_id: suppression_strength}
    step:       int


@dataclass
class Action:
    """PFCRuntime 최종 결정."""
    type:      str    # "execute" | "inhibit"
    winner:    str
    result:    Any    = None
    reason:    str    = ""
    redirect:  str    = ""
