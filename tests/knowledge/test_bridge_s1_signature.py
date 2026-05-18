"""Bridge Integration S1 — RegionSignature 연결 검증.

Design Ref: docs/02-design/features/htp-bridge-integration-design.md §2 + §5-2

S1 의 두 가지 검증:
  1. test_signature_learns_from_ingest — source 별 RegionSignature 가 ingest 데이터로 학습
  2. test_signature_domain_discrimination — Q1: 같은 도메인 query 가 해당 signature 와 더 높은 유사도
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
    """3 도메인 × 5 entries — Q1/Q3 검증용 표준 fixture."""
    loop = _make_loop(tmp_path)
    brain = [
        "해마 CA3 의 패턴 완성 메커니즘",
        "시냅스 가소성과 헵의 학습 법칙",
        "감마 진동과 시간적 바인딩",
        "시상의 게이팅과 의식의 통합",
        "수면 중 기억 공고화 SWR",
    ]
    ai = [
        "Transformer 의 self attention 메커니즘",
        "RLHF 와 인간 피드백 강화 학습",
        "MoE 라우팅 전략과 sparse experts",
        "RAG 검색 증강 생성 파이프라인",
        "Diffusion 모델의 생성 원리",
    ]
    infra = [
        "Redis key value 캐시 전략",
        "Kubernetes 오케스트레이션과 pod",
        "로드밸런서 round robin 알고리즘",
        "CDN 엣지 캐싱과 TTL 정책",
        "마이크로서비스 통신 패턴 gRPC",
    ]
    for t in brain:
        loop.ingest(t, source="뇌과학")
    for t in ai:
        loop.ingest(t, source="AI")
    for t in infra:
        loop.ingest(t, source="인프라")
    return loop


def test_signature_learns_from_ingest():
    """source 별 RegionSignature 가 ingest 데이터로 학습됨."""
    with tempfile.TemporaryDirectory() as td:
        loop = _make_loop(Path(td))
        loop.ingest("해마 CA3 패턴 완성", source="뇌과학")
        loop.ingest("시냅스 가소성과 학습", source="뇌과학")
        loop.ingest("Redis 캐시 전략", source="인프라")

        assert "뇌과학" in loop._signatures
        assert "인프라" in loop._signatures
        assert loop._signatures["뇌과학"].count == 2
        assert loop._signatures["인프라"].count == 1


def test_signature_rebuild_from_cache():
    """KnowledgeLoop 재생성 시 cache 에서 signature 가 자동 재구축."""
    with tempfile.TemporaryDirectory() as td:
        loop1 = _make_loop(Path(td))
        loop1.ingest("해마 CA3 패턴 완성", source="뇌과학")
        loop1.ingest("시냅스 가소성과 학습", source="뇌과학")
        loop1.ingest("Redis 캐시 전략", source="인프라")

        # 새 KnowledgeLoop 로 재로드
        loop2 = _make_loop(Path(td))
        assert "뇌과학" in loop2._signatures
        assert loop2._signatures["뇌과학"].count == 2
        assert loop2._signatures["인프라"].count == 1


def test_signature_domain_discrimination():
    """Q1: 같은 도메인 query 가 해당 signature 와 더 높은 유사도."""
    with tempfile.TemporaryDirectory() as td:
        loop = _make_loop_with_data(Path(td))

        q_vec = loop.encoder.encode("기억과 학습의 신경 메커니즘")
        sim_brain = loop._signatures["뇌과학"].similarity(q_vec)
        sim_infra = loop._signatures["인프라"].similarity(q_vec)

        assert sim_brain > sim_infra, (
            f"뇌과학 유사도 ({sim_brain:.3f}) 가 "
            f"인프라 유사도 ({sim_infra:.3f}) 보다 높아야 함"
        )
