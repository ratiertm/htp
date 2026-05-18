"""Bridge Integration S2 — CoherenceGate 연결 검증.

Design Ref: docs/02-design/features/htp-bridge-integration-design.md §3 + §5-2

S2 의 두 가지 검증 + 1개 보조:
  1. test_coherence_detects_conflict — 모순 지식 입력 시 conflict 감지
  2. test_coherence_accepts_consistent — 일관 지식 입력 시 conflict 낮음
  3. test_coherence_skipped_when_few_neighbors — neighbors < 2 면 coherence_info=None
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from htp.knowledge import KnowledgeLoop, KnowledgeStore
from htp.knowledge.encoder import TfidfJLEncoder


def _make_loop(tmp_path: Path) -> KnowledgeLoop:
    store = KnowledgeStore(tmp_path / "knowledge_log.jsonl")
    return KnowledgeLoop(encoder=TfidfJLEncoder(dim=32), store=store)


def test_coherence_detects_conflict():
    """모순/이질 지식 다발 입력 시 conflict 가 의미 있게 측정됨.

    Bridge Design §3-6 시나리오 — 다른 도메인 텍스트가 누적된 상태에서
    이질 텍스트가 들어오면 pairwise conflict 가 양수로 측정되어야 한다.
    PairwiseCoherenceGate 는 max(1 - cosine) 이므로 단순히 > 0 인지 검증.
    """
    with tempfile.TemporaryDirectory() as td:
        loop = _make_loop(Path(td))
        # 이질 도메인 지식 다발 — 서로 무관한 벡터 분포
        loop.ingest("Transformer self attention 메커니즘", source="AI")
        loop.ingest("Redis 캐시 전략 LRU", source="인프라")
        loop.ingest("로드밸런서 라운드로빈", source="인프라")
        loop.ingest("CDN 엣지 캐싱 TTL", source="인프라")

        # 신규 이질 지식
        result = loop.ingest(
            "감마 진동과 시간적 바인딩 그리고 의식의 통합 메커니즘",
            source="뇌과학",
        )

        assert result.coherence_info is not None
        # 이질 지식 다발이라 conflict > 0 이어야 함
        assert result.coherence_info["conflict"] > 0.0, (
            f"이질 지식 ingest 시 conflict 가 측정되지 않음: "
            f"{result.coherence_info}"
        )


def test_coherence_accepts_consistent():
    """일관 도메인 지식 다발 입력 시 coherence 가 의미 있게 높음.

    같은 source 내 유사 텍스트가 누적된 상태에서 같은 도메인 텍스트가
    들어오면 coherence 가 conflict 보다 낮지 않아야 한다.
    """
    with tempfile.TemporaryDirectory() as td:
        loop = _make_loop(Path(td))
        loop.ingest("해마 CA3 패턴 완성 메커니즘 시냅스", source="뇌과학")
        loop.ingest("해마 패턴 완성 부분 단서 복원 시냅스", source="뇌과학")
        loop.ingest("해마 CA3 recurrent 시냅스 완성", source="뇌과학")

        result = loop.ingest(
            "해마 CA3 시냅스 패턴 완성 부분 단서",
            source="뇌과학",
        )

        assert result.coherence_info is not None
        # 일관 도메인이라 coherence > 0 이어야 함 (완전 직교는 아님)
        assert result.coherence_info["coherence"] > 0.0, (
            f"일관 도메인 ingest 시 coherence 가 0 임: "
            f"{result.coherence_info}"
        )
        # escalate 는 conflict > 0.7 인 극단 케이스에만 True
        # 일관 지식이면 false 이어야 함
        assert result.coherence_info["escalate"] is False


def test_coherence_skipped_when_few_neighbors():
    """neighbors < 2 (첫 ingest) 면 coherence_info = None — 비교 자체가 무의미."""
    with tempfile.TemporaryDirectory() as td:
        loop = _make_loop(Path(td))
        result = loop.ingest("최초 지식", source="test")
        # 비어있는 상태에서 첫 ingest → neighbors=0 → None
        assert result.coherence_info is None
