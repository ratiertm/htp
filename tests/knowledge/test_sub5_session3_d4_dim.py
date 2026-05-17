"""
sub-5 session-3 — D4 (HTP 구조 학습은 위에서) + RegionSignature dim 동적.

Design Ref: docs/02-design/features/htp-thalamus-car.sub-5.design.md §6.2 (Session 3)
Plan SC: FR-10 (dim 동적), FR-11 (D4)

신규 (187 → 189, +2):
- region_signature_dim_dynamic            — RegionSignature(dim=384) 동작
- d4_htp_structure_learns_post_embedding  — HTP 학습은 RegionSignature 에서만
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from htp.thalamus.signature import RegionSignature


_SKIP_DOWNLOAD = os.environ.get("HTP_SKIP_HF_DOWNLOAD", "").lower() in ("1", "true")


# ══════════════════════════════════════════════════════════
# T1: dim 동적 호환 (Plan FR-10)
# ══════════════════════════════════════════════════════════

def test_region_signature_dim_dynamic():
    """RegionSignature(dim=384) — backward-compat 유지 + 새 dim 지원.

    sub-5 의 EmbeddingBridge default = 384-dim (multilingual-e5-small).
    기존 sub-2 의 dim=64 도 그대로 작동해야 함.
    """
    # 기존: dim=64 (sub-2 default)
    sig64 = RegionSignature()
    assert sig64.dim == 64
    assert sig64.centroid.shape == (64,)

    sig64.update(np.ones(64))
    assert sig64.count == 1
    assert sig64.similarity(np.ones(64)) > 0.99  # 자기 자신과 매칭

    # 신규: dim=384 (sub-5 embedding)
    sig384 = RegionSignature(dim=384)
    assert sig384.dim == 384
    assert sig384.centroid.shape == (384,)

    vec = np.random.RandomState(42).standard_normal(384)
    vec /= np.linalg.norm(vec)
    sig384.update(vec)
    assert sig384.count == 1
    assert sig384.similarity(vec) > 0.99

    # 자동 추론: centroid 가 명시 주어지면 dim 자동
    custom_centroid = np.zeros(768, dtype=np.float64)
    sig_auto = RegionSignature(centroid=custom_centroid)
    assert sig_auto.dim == 768   # 자동 추론
    assert sig_auto.centroid.shape == (768,)

    # 차원 불일치 query 는 ValueError
    with pytest.raises(ValueError):
        sig384.similarity(np.zeros(64))


# ══════════════════════════════════════════════════════════
# T2: D4 — HTP 학습은 위에서만 (EmbeddingBridge 도입 후에도)
# ══════════════════════════════════════════════════════════

@pytest.mark.skipif(_SKIP_DOWNLOAD, reason="HF download skipped")
def test_d4_htp_structure_learns_post_embedding():
    """D4 검증 — EmbeddingBridge 사용 시에도:
       1. RegionSignature.centroid 가 update 로 변화 (HTP 학습 유지)
       2. EmbeddingBridge weights 는 불변 (D1)
       3. 두 시스템이 독립적으로 학습 + freeze 공존
    """
    from htp.knowledge.embedding import EmbeddingBridge

    bridge = EmbeddingBridge()
    weights_hash_before = bridge.weights_hash()

    # RegionSignature 가 EmbeddingBridge 의 vec 로 학습 (위 layer)
    sig = RegionSignature(dim=bridge.dim)
    assert sig.count == 0
    assert float(np.linalg.norm(sig.centroid)) == 0.0   # 초기 0

    # 3 개의 의미 유사 텍스트로 centroid 형성
    texts = [
        "pattern completion in memory retrieval",
        "associative memory recall mechanism",
        "stored pattern retrieval from partial cue",
    ]
    for txt in texts:
        v = bridge.encode(txt)
        sig.update(v)

    # D4 검증 ①: HTP 구조 (RegionSignature) 가 학습됨
    assert sig.count == 3
    centroid_norm = float(np.linalg.norm(sig.centroid))
    assert centroid_norm > 0.5, "centroid 형성 실패"

    # D4 검증 ②: 같은 의미 query 가 높은 cosine
    query_vec = bridge.encode("retrieval of stored memory pattern")
    sim = sig.similarity(query_vec)
    assert sim > 0.6, f"학습된 centroid 의 의미 매칭 실패: sim={sim}"

    # D1 검증 (재확인): EmbeddingBridge weights 는 불변
    weights_hash_after = bridge.weights_hash()
    assert weights_hash_before == weights_hash_after, (
        "D4 위반 — EmbeddingBridge 가 fine-tune 됨 (HTP 학습이 아래 layer 로 전파)"
    )
