"""
Stage 3 Integration — BrainRuntime coherence DI + Memory swr_priority 확장.

Design Ref: docs/02-design/features/htp-thalamus-car.sub-3.design.md §2.4-2.5
Plan SC: FR-14, FR-15

stage-3-integration 범위 (144 → 146):
- M4 BrainRuntime DI : coherence=None 기본 (회귀 동등) + 옵션 활성 (+1)
- M5 swr_priority    : conflict_magnitude 단조 증가 (+1)
- M6 DAG             : tests/unit/test_no_circular_deps.py parametrize 자동 +2 (coherence/ 2 파일)
"""
from __future__ import annotations

import tempfile

import numpy as np
import pytest
import torch

from htp.runtime.brain_runtime import BrainRuntime
from htp.memory.memory_system  import MemorySystem
from htp.thalamus.coherence    import PairwiseCoherenceGate


# ══════════════════════════════════════════════════════════
# M4 — BrainRuntime coherence DI  (+1)
# ══════════════════════════════════════════════════════════

def test_brain_runtime_coherence_optional():
    """coherence=None (기본) 시 기존 동작 동등 + 옵션 활성화 가능.

    회귀 보호 핵심 — Stage 5 통합 테스트 7건이 깨지지 않음.
    """
    with tempfile.TemporaryDirectory() as td:
        # 기본: coherence=None
        rt1 = BrainRuntime(memory_dir=td, enable_memory=False)
        assert rt1.coherence is None
        assert rt1._last_bound_response is None

        # 옵션: PairwiseCoherenceGate 주입
        gate = PairwiseCoherenceGate()
        rt2  = BrainRuntime(memory_dir=td, enable_memory=False,
                            coherence=gate)
        assert rt2.coherence is gate
        # 외부 inspection 진입점이 존재
        assert hasattr(rt2, "_last_bound_response")


# ══════════════════════════════════════════════════════════
# M5 — MemorySystem swr_priority 확장  (+1)
# ══════════════════════════════════════════════════════════

def test_swr_priority_conflict_amplification():
    """priority = novelty × reward × (1 + conflict_magnitude) — Plan FR-15.

    검증 포인트:
      1. conflict_magnitude=0 → 기존 식 (novelty × reward) 동등 — 회귀 보호
      2. conflict_magnitude > 0 → 단조 증가
      3. save() 가 in-memory dict 에 보존, tag_swr 에서 사용
    """
    with tempfile.TemporaryDirectory() as td:
        mem = MemorySystem(memory_dir=td)

        # 1. 회귀 동등 — conflict=0 시 기존 식
        p0 = mem.swr_priority(novelty=0.8, reward=0.6, conflict_magnitude=0.0)
        assert p0 == pytest.approx(0.8 * 0.6, rel=1e-10)

        # 2. 단조 증가
        p1 = mem.swr_priority(novelty=0.8, reward=0.6, conflict_magnitude=0.5)
        p2 = mem.swr_priority(novelty=0.8, reward=0.6, conflict_magnitude=1.0)
        assert p1 > p0
        assert p2 > p1
        # conflict=1.0 → 정확히 2배 증폭
        assert p2 == pytest.approx(p0 * 2.0, rel=1e-10)

        # 3. save() → in-memory dict 보존
        state = torch.zeros(64)
        ep_id_with_conflict = mem.save(
            state_vec=state, step=1, winner="r1", action_type="execute",
            score=0.7, context="test", conflict_magnitude=0.5,
        )
        ep_id_no_conflict = mem.save(
            state_vec=state, step=2, winner="r2", action_type="execute",
            score=0.7, context="test",
        )
        # conflict 있는 ep 만 dict 에
        assert ep_id_with_conflict in mem._conflict_by_episode
        assert mem._conflict_by_episode[ep_id_with_conflict] == 0.5
        assert ep_id_no_conflict not in mem._conflict_by_episode
