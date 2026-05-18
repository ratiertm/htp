"""Bridge Q2 retune — encoder 별 CoherenceGate threshold default 검증.

Design Ref: docs/02-design/features/htp-bridge-integration-design.md §Q2 retune (2026-05-18)

측정 (15 entries × 3 source 의 intra/inter conflict 분포):
  TF-IDF       : conflict ≈ 1.0 포화 → escalation_threshold=1.0 (비활성).
  EmbeddingBridge : intra max=0.124, inter max=0.141 → (0.105, 0.135).

검증:
  1. TfidfJLEncoder → (0.5, 1.0)
  2. EmbeddingBridge → (0.105, 0.135)
  3. 미지 encoder → 보수적 default (0.3, 0.7)
  4. override (coherence_thresholds=(c, e)) 가 default 보다 우선
  5. TFIDF 단계에서 escalate=False 항상 유지 (sparse 노이즈 차단)
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from htp.knowledge import KnowledgeLoop, KnowledgeStore
from htp.knowledge.encoder import TfidfJLEncoder


_SKIP_HF = os.environ.get("HTP_SKIP_HF_DOWNLOAD", "").lower() in ("1", "true")


def _make_loop_tfidf(tmp: Path, **kwargs) -> KnowledgeLoop:
    store = KnowledgeStore(tmp / "log.jsonl")
    return KnowledgeLoop(encoder=TfidfJLEncoder(dim=32), store=store, **kwargs)


def test_tfidf_default_thresholds():
    """TfidfJLEncoder → (0.5, 1.0). escalation_threshold=1.0 으로 비활성."""
    with tempfile.TemporaryDirectory() as td:
        loop = _make_loop_tfidf(Path(td))
        assert loop.coherence_conflict_threshold == 0.5
        assert loop.coherence_escalation_threshold == 1.0


@pytest.mark.skipif(_SKIP_HF, reason="HF download skipped")
def test_embedding_default_thresholds():
    """EmbeddingBridge → (0.105, 0.135). e5 분포 측정 기반."""
    from htp.knowledge.embedding import EmbeddingBridge
    with tempfile.TemporaryDirectory() as td:
        store = KnowledgeStore(Path(td) / "log.jsonl")
        loop = KnowledgeLoop(encoder=EmbeddingBridge(), store=store)
        assert abs(loop.coherence_conflict_threshold - 0.105) < 1e-9
        assert abs(loop.coherence_escalation_threshold - 0.135) < 1e-9


def test_unknown_encoder_conservative_default():
    """알려지지 않은 encoder → 보수적 default (0.3, 0.7)."""
    class _FakeEncoder:
        dim = 8
        _fitted = True
        def encode(self, text):
            return np.ones(8, dtype=np.float32)
        def fit(self, corpus):
            pass

    with tempfile.TemporaryDirectory() as td:
        store = KnowledgeStore(Path(td) / "log.jsonl")
        loop = KnowledgeLoop(encoder=_FakeEncoder(), store=store)
        assert loop.coherence_conflict_threshold == 0.3
        assert loop.coherence_escalation_threshold == 0.7


def test_explicit_override_wins():
    """coherence_thresholds=(c, e) 가 encoder default 보다 우선."""
    with tempfile.TemporaryDirectory() as td:
        loop = _make_loop_tfidf(
            Path(td),
            coherence_thresholds=(0.2, 0.4),
        )
        assert loop.coherence_conflict_threshold == 0.2
        assert loop.coherence_escalation_threshold == 0.4


def test_tfidf_escalate_never_triggered():
    """Q2 retune: TFIDF 단계에서 escalate=False 항상.

    Bridge Q2 발견: TF-IDF + JL 의 cosine 이 대부분 0 가까움 → conflict ≈ 1.0 포화.
    escalation_threshold=1.0 으로 두면 `conflict > 1.0` 절대 False.
    """
    with tempfile.TemporaryDirectory() as td:
        loop = _make_loop_tfidf(Path(td))
        # 의도적으로 이질 코퍼스 누적
        loop.ingest("Transformer self attention encoder layer", source="AI")
        loop.ingest("Redis LRU cache eviction policy", source="인프라")
        loop.ingest("로드밸런서 round robin scheduling", source="인프라")
        loop.ingest("CDN edge caching TTL fastly", source="인프라")

        result = loop.ingest(
            "감마 진동 의식 통합 신피질 PFC 시상",
            source="뇌과학",
        )
        assert result.coherence_info is not None
        # escalation_threshold=1.0 으로 절대 escalate 되지 않음
        assert result.coherence_info["escalate"] is False, (
            f"TFIDF 에서 escalate 가 트리거됨 (기대: 항상 False): "
            f"{result.coherence_info}"
        )
