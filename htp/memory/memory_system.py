"""
MemorySystem — L1+L2+L3 통합 (CA3-CA1 양방향 recall + CUSUM 트리거)
====================================================================

설계 문서: design/htp_memory_design_final.md §5-7

  L1: PFCRuntime.working_memory (deque[7])     — 세션 내 (이미 구현)
  L2: EpisodeStore (SQLite)                    — 세션 간 에피소드
  L3: PatternStore (JSON + Online Hebbian EMA) — 장기 패턴 일반화

CA3: pattern completion   (부분 입력 → centroid 복원)
CA1: mismatch detection   (완성본과 원본 비교)
     + 가치 기반 선택      (score × log(recall_count + 2))
"""
from __future__ import annotations

import math
import uuid
from pathlib import Path
from typing import Optional

import torch

from .episode_store import EpisodeStore
from .pattern_store import PatternStore
from .types import Episode, MemoryContext, bytes_to_tensor, tensor_to_bytes


class MemorySystem:
    CA1_MISMATCH_THRESHOLD = 0.3      # 이 이상이면 "새로운 상황"
    SWR_PRIORITY_THRESHOLD = 0.5      # novelty × score >= 이면 SWR 태깅
    CONSOLIDATE_TOP_K      = 200      # on_overload 시 조회할 에피소드 수

    def __init__(self,
                 memory_dir: str | Path = ".htp",
                 session_id: Optional[str] = None):
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.l2 = EpisodeStore(self.memory_dir / "memory.db")
        self.l3 = PatternStore(self.memory_dir / "patterns.json")
        self.session_id = session_id or str(uuid.uuid4())
        self._last_episode_id: Optional[str] = None

    # ── 저장 ─────────────────────────────────────

    def save(self,
             state_vec:   torch.Tensor,
             step:        int,
             winner:      str,
             action_type: str,
             score:       float,
             context:     str) -> str:
        """
        에피소드 저장.
        novelty = 1 - L3 매칭 신뢰도 → L3 에 패턴 없을수록 1.0 에 가까움.
        """
        novelty = 1.0 - self.l3.match_confidence(state_vec)
        ep = Episode(
            step        = step,
            winner      = winner,
            action_type = action_type,
            score       = score,
            state_vec   = tensor_to_bytes(state_vec),
            context     = context[:50],
            novelty     = novelty,
            session_id  = self.session_id,
        )
        ep_id = self.l2.save(ep)
        self._last_episode_id = ep_id
        return ep_id

    # ── CA3-CA1 recall ──────────────────────────

    def recall(self, state_vec: torch.Tensor) -> MemoryContext:
        """
        CA3: pattern completion
        CA1: mismatch detection + 가치 기반 선택
        """
        completed_vec, pattern = self.l3.complete(state_vec)
        mismatch = (state_vec - completed_vec).norm().item()
        is_novel = mismatch >= self.CA1_MISMATCH_THRESHOLD

        if not is_novel:
            winner_filter = pattern.best_winner if pattern else None
            candidates = self.l2.search_similar(
                state_vec, top_k=10, winner_filter=winner_filter,
            )
        else:
            candidates = self.l2.search_similar(state_vec, top_k=5)

        recommendation: Optional[str] = None
        confidence: float = 0.0
        if candidates:
            for ep in candidates:
                self.l2.increment_recall(ep.episode_id)
            best = max(
                candidates,
                key=lambda e: (e.score or 0.0) * math.log((e.recall_count or 0) + 2),
            )
            recommendation = best.winner
            confidence     = best.score or 0.0

        return MemoryContext(
            completed_vec  = completed_vec,
            mismatch       = mismatch,
            candidates     = candidates,
            recommendation = recommendation,
            confidence     = confidence,
            pattern        = pattern,
            is_novel       = is_novel,
        )

    # ── CUSUM overload → consolidation ──────────

    def on_overload(self, region_id: str):
        """
        CUSUM overload = 피질 과부하 = 수면 신호.
        SWR 태깅 → Online Hebbian consolidation 실행.
        """
        self.l2.tag_swr(priority_threshold=self.SWR_PRIORITY_THRESHOLD)

        # 전체 SWR-tagged 에피소드 가져오기 — zero 벡터로 무관 검색
        sample = self.l2.recent(1)
        if sample and sample[0].state_vec:
            probe = bytes_to_tensor(sample[0].state_vec)
            zero = torch.zeros_like(probe)
            tagged = self.l2.search_similar(
                zero, top_k=self.CONSOLIDATE_TOP_K, swr_only=True,
            )
            # 위 search_similar 는 유사도 정렬이지만 swr_only=True 필터가 먼저 적용됨.
            # 만약 zero 벡터와 유사도가 0 인 항목이 다수이면 그 중 임의 순.
        else:
            tagged = []

        self.l3.consolidate(tagged)

    # ── 사후 평가 ────────────────────────────────

    def feedback(self, outcome: str, episode_id: Optional[str] = None):
        ep_id = episode_id or self._last_episode_id
        if ep_id:
            self.l2.feedback(ep_id, outcome)

    # ── 조회 ─────────────────────────────────────

    def episode_count(self) -> int:
        return self.l2.count()

    def pattern_count(self) -> int:
        return self.l3.pattern_count()
