"""e5 prefix 적용 검증 (sub-5 merge plan 작업 2).

Design Ref: docs/01-plan/features/htp-sub5-merge-plan.md §2

intfloat/multilingual-e5 시리즈는 query 에 "query: ", 문서에 "passage: "
prefix 를 붙여야 최적 성능. 본 테스트는 두 mode 의 벡터가 *다름* 을 검증
(같으면 prefix 미적용).
"""
from __future__ import annotations

import os

import numpy as np
import pytest


_SKIP_DOWNLOAD = os.environ.get("HTP_SKIP_HF_DOWNLOAD", "").lower() in ("1", "true")
pytestmark = pytest.mark.skipif(
    _SKIP_DOWNLOAD,
    reason="HuggingFace model download skipped",
)


def test_e5_prefix_query_vs_passage():
    """query prefix 와 passage prefix 가 다른 벡터를 생성하는지 확인.

    같은 텍스트라도 prefix 가 다르면 벡터가 다름 (prefix 미적용이면 동일).
    단, 의미는 가까워야 하므로 cosine 은 0.8 이상이지만 1 미만.
    """
    from htp.knowledge.embedding import EmbeddingBridge

    bridge = EmbeddingBridge()
    text = "pattern completion in hippocampus"

    vec_passage = bridge.encode(text)
    vec_query   = bridge.encode_query(text)

    cosine = float(
        np.dot(vec_passage, vec_query)
        / (np.linalg.norm(vec_passage) * np.linalg.norm(vec_query) + 1e-12)
    )

    # prefix 미적용이면 두 벡터가 동일 → cosine 1.0
    assert cosine < 0.999, (
        f"query 와 passage 벡터가 사실상 동일 (cosine={cosine:.4f}) — "
        f"prefix 미적용 의심"
    )
    # 같은 텍스트이므로 의미적으로는 가까워야 함
    assert cosine > 0.8, (
        f"같은 텍스트의 query/passage cosine={cosine:.4f} 너무 낮음 — "
        f"의미 보존 실패"
    )
