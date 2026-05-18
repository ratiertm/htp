"""I5 Confidence Score 검증 (sub-5 merge plan 작업 3)."""
from __future__ import annotations

import os
import pytest

from htp.knowledge.confidence import QueryResultV2, DEFAULT_GAP_THRESHOLD


_SKIP = os.environ.get("HTP_SKIP_HF_DOWNLOAD", "").lower() in ("1", "true")


def test_confidence_clear_match():
    """명확한 매칭: top-1 >> top-2 → confidence 높음."""
    sims = [0.92, 0.78, 0.71, 0.65]
    gap, has_match = QueryResultV2.compute_confidence(sims)
    assert gap > 0.1
    assert has_match is True


def test_confidence_no_match():
    """매칭 없음: 모든 similarity 비슷 → confidence 낮음.

    Vault Hopfield 실측 분포 (top1=0.861, top2=0.856) 재현.
    """
    sims = [0.861, 0.856, 0.853, 0.849]
    gap, has_match = QueryResultV2.compute_confidence(sims)
    # gap ≈ 0.005, threshold = 0.01 → has_match False
    assert gap < DEFAULT_GAP_THRESHOLD
    assert has_match is False


def test_confidence_single_entry():
    """entry 1개: gap 0 → has_match=False."""
    gap, has_match = QueryResultV2.compute_confidence([0.9])
    assert gap == 0.0
    assert has_match is False


@pytest.mark.skipif(_SKIP, reason="HF download skipped")
def test_query_v2_real_data_no_match():
    """KnowledgeLoop.query_v2() 실데이터 (vault 에 없는 주제) → has_match False.

    Vault Hopfield 시나리오 재현.
    """
    import tempfile
    from pathlib import Path
    from htp.knowledge import KnowledgeLoop, KnowledgeStore
    from htp.knowledge.embedding import EmbeddingBridge

    with tempfile.TemporaryDirectory() as td:
        store = KnowledgeStore(Path(td) / "knowledge_log.jsonl")
        loop = KnowledgeLoop(encoder=EmbeddingBridge(), store=store)

        # vault 와 무관한 일반 entries — 어떤 query 도 명확 매칭 안 됨
        for text in ["일지: 오늘 회의가 있었다",
                     "프로젝트 진행 상황 점검",
                     "기술 도구 비교 정리"]:
            loop.ingest(text, source="general")

        # Hopfield 같이 entries 와 무관한 query
        result = loop.query_v2("Hopfield network attractor dynamics", top_k=3)
        assert len(result.results) >= 2
        # 모든 cosine 비슷 → gap 작음 → no match
        assert result.has_match is False
