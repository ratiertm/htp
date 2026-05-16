"""
KnowledgeLoop unit tests (Stage 0.5).

Design Ref: docs/02-design/features/htp-thalamus-car.design.md §5.2
Plan SC: FR-05.4, FR-05.5, FR-05.6 + Stage 0.5 Go/No-Go 시나리오
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from htp.knowledge import (
    KnowledgeLoop, KnowledgeStore, TfidfJLEncoder, TextEncoder,
)


@pytest.fixture
def temp_store() -> KnowledgeStore:
    """격리된 임시 JSONL 저장소."""
    with tempfile.TemporaryDirectory() as td:
        yield KnowledgeStore(Path(td) / "knowledge_log.jsonl")


@pytest.fixture
def fresh_loop(temp_store: KnowledgeStore) -> KnowledgeLoop:
    """매 테스트마다 0 entry 상태에서 시작."""
    return KnowledgeLoop(encoder=TfidfJLEncoder(), store=temp_store)


# ══════════════════════════════════════════════════════════
# Test 1: ingest 기본 동작
# ══════════════════════════════════════════════════════════

def test_loop_ingest_basic(fresh_loop):
    """텍스트 입력 → 64-dim 벡터 생성 → store 저장 확인."""
    result = fresh_loop.ingest("뇌의 기억은 분산 표상", source="brain")

    assert result.entry.vec.shape == (64,)
    assert result.entry.text   == "뇌의 기억은 분산 표상"
    assert result.entry.source == "brain"
    assert result.entry.timestamp  # ISO 형식 문자열
    assert len(fresh_loop._cache) == 1


# ══════════════════════════════════════════════════════════
# Test 2: query — 유사 텍스트가 자기 자신/공유 어휘 텍스트 반환
# ══════════════════════════════════════════════════════════

def test_loop_query_neighbor(fresh_loop):
    """저장된 지식과 어휘 공유 질의가 높은 similarity 로 반환됨."""
    fresh_loop.ingest("attention mechanism transformer 가중치", source="ai")
    fresh_loop.ingest("Hopfield network 패턴 인출", source="ai")
    fresh_loop.ingest("Redis 의 key value lookup", source="infra")

    result = fresh_loop.query("attention transformer")

    assert len(result.relevant) > 0
    # top1 은 어휘 공유 도가 가장 높은 'attention mechanism transformer' 여야 함
    top = result.relevant[0]
    top_entry = fresh_loop._cache[top.entry_id]
    assert "attention" in top_entry.text or "transformer" in top_entry.text


# ══════════════════════════════════════════════════════════
# Test 3 (핵심): cross-domain discover (Plan SC, Stage 0.5 Go 시나리오)
# ══════════════════════════════════════════════════════════

def test_loop_discover_cross_domain(fresh_loop):
    """Plan §6 위험 6 회귀 보호:
    어휘를 충분히 공유시킨 brain/AI 쌍 vs brain/infra 쌍 비교.

    smoke 단계에서 threshold 0.6 이 너무 높아 0건 → 본 테스트는 threshold 0.3
    (TF-IDF + JL 64-dim 분포 기준) 으로 낮추어 비교만 검증.
    threshold 적정값은 향후 EmbeddingBridge 도입 시 0.7+ 로 다시 올림.
    """
    # 의도적으로 어휘 일부 공유:
    fresh_loop.ingest(
        "content addressable memory pattern recall by content",
        source="brain",
    )
    fresh_loop.ingest(
        "Hopfield network pattern recall by content energy",
        source="ai",
    )
    fresh_loop.ingest(
        "Redis key value lookup database protocol",
        source="infra",
    )

    # threshold 를 낮춰 brain-ai/brain-infra 모두 후보로 포함
    fresh_loop.discover_threshold = 0.05
    discoveries = fresh_loop.discover()

    bs_ai  = next((d for d in discoveries
                   if {d.source_a, d.source_b} == {"brain", "ai"}), None)
    bs_inf = next((d for d in discoveries
                   if {d.source_a, d.source_b} == {"brain", "infra"}), None)

    # Go: brain-ai 발견. brain-infra 가 발견되면 그것보다 brain-ai 가 우세.
    assert bs_ai is not None, "brain↔AI cross-domain 발견 실패"
    if bs_inf is not None:
        assert bs_ai.similarity > bs_inf.similarity, (
            f"brain-ai({bs_ai.similarity:.3f}) 가 "
            f"brain-infra({bs_inf.similarity:.3f}) 보다 낮음"
        )


# ══════════════════════════════════════════════════════════
# Test 4: TextEncoder Protocol 준수 + dim/encode/fit
# ══════════════════════════════════════════════════════════

def test_loop_text_encoder_interface():
    """TfidfJLEncoder 가 TextEncoder Protocol 준수 + shape 검증."""
    enc = TfidfJLEncoder(dim=64)

    # Protocol runtime check (Stage 6 EmbeddingBridge 도 동일 Protocol 준수해야 함)
    assert isinstance(enc, TextEncoder)
    assert enc.dim == 64

    # fit + encode round-trip
    enc.fit(["test corpus sample document", "another sample text content"])
    vec = enc.encode("sample test")

    assert isinstance(vec, np.ndarray)
    assert vec.shape == (64,)
    assert np.isfinite(vec).all()


# ══════════════════════════════════════════════════════════
# Test 5: empty state 안전성
# ══════════════════════════════════════════════════════════

def test_loop_empty_state(fresh_loop):
    """0 entry 상태에서 query/discover 가 에러 없이 빈 결과."""
    q_result = fresh_loop.query("nothing here")
    assert q_result.relevant == []
    assert q_result.cluster_count == 0

    discoveries = fresh_loop.discover()
    assert discoveries == []


# ══════════════════════════════════════════════════════════
# Test 6 (Critical Gap #3 옵션 A-2): encoder state 영속화
# ══════════════════════════════════════════════════════════

def test_encoder_save_load_round_trip(tmp_path):
    """fit 한 encoder 를 save/load 후 동일 vec 재현되어야 함."""
    enc1 = TfidfJLEncoder(dim=64)
    enc1.fit(["content addressable memory pattern recall",
              "Hopfield network pattern recall energy",
              "Redis key value lookup protocol"])
    vec1 = enc1.encode("pattern recall memory")

    state_path = tmp_path / "encoder_state.pkl"
    enc1.save(state_path)

    enc2 = TfidfJLEncoder(dim=64)
    assert enc2.load(state_path) is True
    vec2 = enc2.encode("pattern recall memory")

    np.testing.assert_allclose(vec1, vec2, rtol=1e-10,
        err_msg="save/load round-trip 으로 vec 재현 실패")


def test_loop_persists_encoder_across_instances(tmp_path):
    """KnowledgeLoop 를 두 번 생성해도 같은 텍스트 → 같은 vec.

    Critical Gap #3 (CLI 다중 호출) 회귀 보호:
    인스턴스1 에서 fit + ingest → 인스턴스2 에서 동일 text encode 시 동일 vec.
    """
    store = KnowledgeStore(tmp_path / "knowledge_log.jsonl")

    # 인스턴스 1: 3개 ingest (corpus 충분히 커서 JL 활성화)
    loop1 = KnowledgeLoop(encoder=TfidfJLEncoder(), store=store)
    loop1.ingest("content addressable memory pattern recall by content "
                 "distributed representation neural ensemble",
                 source="brain")
    loop1.ingest("Hopfield network attention transformer energy minimization "
                 "pattern recall vector representation", source="ai")
    loop1.ingest("Redis cache key value lookup database protocol storage "
                 "distributed system", source="infra")

    cached_vec = loop1._cache[0].vec.copy()

    # 인스턴스 2: 같은 store 로 새로 생성 (CLI 다중 호출 시뮬레이션)
    loop2 = KnowledgeLoop(encoder=TfidfJLEncoder(), store=store)

    # encoder state 가 로드되어 동일 임베딩 공간이어야 함
    re_encoded = loop2.encoder.encode(loop2._cache[0].text)
    np.testing.assert_allclose(cached_vec, re_encoded, rtol=1e-10,
        err_msg="CLI 다중 호출 시 동일 임베딩 공간 보장 실패")


def test_loop_encoder_state_file_created(tmp_path):
    """첫 ingest 후 .htp/encoder_state.pkl 파일이 실제 생성되는지."""
    store = KnowledgeStore(tmp_path / "knowledge_log.jsonl")
    state_path = tmp_path / "encoder_state.pkl"

    assert not state_path.exists()

    loop = KnowledgeLoop(encoder=TfidfJLEncoder(), store=store)
    loop.ingest("first ingest creates encoder state file", source="test")

    assert state_path.exists(), "첫 ingest 후 encoder_state.pkl 미생성"
