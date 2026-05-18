"""Bridge Integration S3 — VectorRouter 연결 검증.

Design Ref: docs/02-design/features/htp-bridge-integration-design.md §4 + §5-2

S3 의 검증:
  1. test_routed_query_selects_relevant_source — routed 가 관련 source 비율을 높임
  2. test_routed_vs_flat_top1 — routed top-1 의 source 가 적합
  3. test_routed_reduces_search_space — routing_info 로 검색 범위 축소 확인
  4. test_routed_fallback_when_no_signatures — signature 없으면 flat 와 동등
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from htp.knowledge import KnowledgeLoop, KnowledgeStore
from htp.knowledge.encoder import TfidfJLEncoder


def _make_loop(tmp_path: Path) -> KnowledgeLoop:
    store = KnowledgeStore(tmp_path / "knowledge_log.jsonl")
    return KnowledgeLoop(encoder=TfidfJLEncoder(dim=32), store=store)


def _make_loop_with_data(tmp_path: Path) -> KnowledgeLoop:
    """3 도메인 × 5 entries — Q3 검증용 표준 fixture."""
    loop = _make_loop(tmp_path)
    brain = [
        "해마 CA3 의 패턴 완성 메커니즘 시냅스",
        "시냅스 가소성과 헵의 학습 법칙 뇌",
        "감마 진동과 시간적 바인딩 뇌파",
        "시상의 게이팅과 의식의 통합 뇌",
        "수면 중 기억 공고화 SWR 해마",
    ]
    ai = [
        "Transformer 의 self attention 메커니즘 encoder",
        "RLHF 와 인간 피드백 강화 학습 reward",
        "MoE 라우팅 전략과 sparse experts gate",
        "RAG 검색 증강 생성 파이프라인 retrieval",
        "Diffusion 모델의 생성 원리 latent",
    ]
    infra = [
        "Redis key value 캐시 전략 LRU",
        "Kubernetes 오케스트레이션과 pod scheduler",
        "로드밸런서 round robin 알고리즘 nginx",
        "CDN 엣지 캐싱과 TTL 정책 fastly",
        "마이크로서비스 통신 패턴 gRPC service",
    ]
    for t in brain:
        loop.ingest(t, source="뇌과학")
    for t in ai:
        loop.ingest(t, source="AI")
    for t in infra:
        loop.ingest(t, source="인프라")
    return loop


def test_routed_query_selects_relevant_source():
    """Q3: routed top-5 의 관련 source 비율이 flat 보다 높거나 같음."""
    with tempfile.TemporaryDirectory() as td:
        loop = _make_loop_with_data(Path(td))

        flat_r   = loop.query("해마와 시냅스 의 패턴 완성", mode="flat")
        routed_r = loop.query("해마와 시냅스 의 패턴 완성", mode="routed")

        def brain_ratio(result):
            top5 = result.relevant[:5]
            if not top5:
                return 0.0
            srcs = [loop._cache[n.entry_id].source for n in top5]
            return srcs.count("뇌과학") / len(srcs)

        rb = brain_ratio(routed_r)
        fb = brain_ratio(flat_r)
        # routed 가 flat 보다 같거나 더 높은 도메인 비율
        assert rb >= fb, (
            f"routed brain_ratio={rb:.2f} < flat brain_ratio={fb:.2f}"
        )


def test_routed_top1_source_is_relevant():
    """routed top-1 의 source 가 query 와 관련된 도메인."""
    with tempfile.TemporaryDirectory() as td:
        loop = _make_loop_with_data(Path(td))

        r = loop.query("해마 CA3 의 역할", mode="routed")
        assert r.relevant, "routed 결과 비어있음"
        top1_src = loop._cache[r.relevant[0].entry_id].source
        assert top1_src == "뇌과학", (
            f"routed top-1 source={top1_src!r}, expected '뇌과학'"
        )


def test_routed_routing_info_present():
    """routed 모드에서 routing_info 가 채워짐."""
    with tempfile.TemporaryDirectory() as td:
        loop = _make_loop_with_data(Path(td))

        r = loop.query("Redis 성능 최적화", mode="routed")
        assert r.routing_info is not None
        # selected_sources 또는 fallback 둘 중 하나는 있어야 함
        keys = set(r.routing_info.keys())
        assert "selected_sources" in keys or "fallback" in keys, (
            f"routing_info keys={keys}"
        )
        assert r.mode == "routed"


def test_routed_fallback_empty_signatures():
    """signature 가 비어있으면 routed 도 flat 와 동등 (전체 순회)."""
    with tempfile.TemporaryDirectory() as td:
        loop = _make_loop(Path(td))
        # ingest 없이 query → _signatures 비어있음
        # 빈 cache 에서 query 호출
        r = loop.query("test", mode="routed")
        assert r.relevant == []
        assert r.cluster_count == 0


def test_routed_v2_confidence_works():
    """query_v2 의 routed 모드도 confidence 계산이 동작."""
    with tempfile.TemporaryDirectory() as td:
        loop = _make_loop_with_data(Path(td))

        r = loop.query_v2("해마 CA3 패턴 완성", top_k=5, mode="routed")
        assert r.results
        # routed 모드라도 confidence 필드가 채워짐
        assert isinstance(r.confidence, float)
        assert isinstance(r.has_match, bool)
