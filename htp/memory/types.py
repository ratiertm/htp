"""
Memory System Types — L2 Episode, L3 Pattern, MemoryContext
============================================================

설계 문서: design/htp_memory_design_final.md §2
LeCun 검토 반영:
  - 64-dim state_vec (해마 place-cell sparse 표현)
  - SWR priority = novelty × reward
  - CA3-CA1 양방향 recall
"""
from __future__ import annotations

import struct
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

import torch


# ───────────────────────────────────────────────────────
# Numpy-free tensor ↔ bytes 변환 헬퍼
# ───────────────────────────────────────────────────────

def tensor_to_bytes(t: torch.Tensor) -> bytes:
    """float32 tensor → raw bytes (numpy 의존 없음)."""
    flat = t.detach().cpu().contiguous().to(torch.float32).flatten().tolist()
    return struct.pack(f"{len(flat)}f", *flat)


def bytes_to_tensor(b: bytes) -> torch.Tensor:
    """raw bytes → float32 1D tensor."""
    if not b:
        return torch.zeros(0, dtype=torch.float32)
    n = len(b) // 4  # float32 = 4 bytes
    values = struct.unpack(f"{n}f", b)
    return torch.tensor(values, dtype=torch.float32)


@dataclass
class Episode:
    """L2 에피소드 메모리 단위 — 해마 단기 에피소드 인코딩."""
    episode_id:   str = ""                      # UUID (save 시 자동 생성)
    step:         int = 0                        # BrainRuntime._step
    winner:       str = ""                       # 이긴 Region 이름
    action_type:  str = ""                       # "execute" | "inhibit"
    score:        float = 0.0                    # PFC combined score (reward)
    state_vec:    bytes = b""                    # 64-dim float32 blob
    context:      str = ""                       # 입력 요약 50자
    outcome:      Optional[str] = None           # 사후 "success"/"fail"
    recall_count: int = 0                        # CA1 재활성화 횟수
    novelty:      float = 1.0                    # SWR 태깅용 — 1 - L3 매칭 신뢰도
    swr_tagged:   bool = False                   # consolidation 대상 여부
    session_id:   str = ""
    timestamp:    float = field(default_factory=time.time)


@dataclass
class Pattern:
    """L3 패턴 메모리 — 신피질 일반화 표현 (Go-CLS 통과)."""
    pattern_id:    str
    centroid_vec:  bytes                          # 64-dim EMA 중심
    best_winner:   str                            # 다수 성공 Region
    success_rate:  float
    episode_count: int
    winner_dist:   dict                           # {region: success_count}
    snr:           float                          # μ(scores) / σ(scores)
    generalize_ok: bool = True
    updated_at:    float = field(default_factory=time.time)


@dataclass
class MemoryContext:
    """recall() 반환 — CA3-CA1 양방향 처리 결과."""
    # CA3: pattern completion
    completed_vec:   torch.Tensor                 # 복원된 완성 벡터
    mismatch:        float                        # CA1 불일치 L2 거리

    # CA1: 가치 기반 선택
    candidates:      list                         # list[Episode]
    recommendation:  Optional[str]                # best_winner
    confidence:      float                        # 예측 신뢰도

    # 메타
    pattern:         Optional[Pattern]            # 매칭된 L3 패턴
    is_novel:        bool                         # mismatch >= threshold
