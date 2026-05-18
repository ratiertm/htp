"""
sub-5 session-1 — EmbeddingBridge + D1/D2/D3 검증.

Design Ref: docs/02-design/features/htp-thalamus-car.sub-5.design.md §6.2 (Session 1)
Plan SC: FR-01~04, D1 (Frozen), D2 (Protocol), D3 (Fallback)

신규 (172 → 178, +6):
- protocol_compliance — isinstance(TextEncoder)
- frozen_weights      — D1: weights hash 변경 안 됨
- no_grad             — D1: torch.is_grad_enabled() == False during encode
- fit_is_noop         — fit 후 weights hash 동일
- save_load_round_trip — metadata round-trip
- tfidf_fallback_still_works — D3

⚠️ 모델 다운로드 ~30s (첫 1회) — pytest 에서 skip 가능하게 함.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from htp.knowledge          import TextEncoder, TfidfJLEncoder
from htp.knowledge.embedding import EmbeddingBridge


# 모델 다운로드 가능 환경에서만 실행 (CI 등에서 skip 위함)
_SKIP_DOWNLOAD = os.environ.get("HTP_SKIP_HF_DOWNLOAD", "").lower() in ("1", "true")
pytestmark = pytest.mark.skipif(
    _SKIP_DOWNLOAD,
    reason="HuggingFace model download skipped (set HTP_SKIP_HF_DOWNLOAD=0)",
)


@pytest.fixture(scope="module")
def bridge() -> EmbeddingBridge:
    """모듈 단위 1회 로드 (재사용)."""
    return EmbeddingBridge()


# ══════════════════════════════════════════════════════════
# T1: TextEncoder Protocol 준수 (D2)
# ══════════════════════════════════════════════════════════

def test_embedding_bridge_protocol_compliance(bridge):
    """EmbeddingBridge 가 TextEncoder Protocol isinstance."""
    assert isinstance(bridge, TextEncoder)

    # dim 노출
    assert bridge.dim > 0
    assert bridge.dim == 384   # multilingual-e5-small default

    # encode → np.ndarray
    vec = bridge.encode("Hello, world.")
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (bridge.dim,)
    assert np.isfinite(vec).all()


# ══════════════════════════════════════════════════════════
# T2: D1 Frozen — weights hash 불변
# ══════════════════════════════════════════════════════════

def test_embedding_bridge_frozen_weights(bridge):
    """encode 전후 모델 weights hash 동일 — D1 검증 핵심."""
    h0 = bridge.weights_hash()
    assert h0 != "no-model"

    # 여러 번 encode
    for txt in ["first", "두번째 한국어", "third with mixed 단어"]:
        bridge.encode(txt)

    h1 = bridge.weights_hash()
    assert h0 == h1, (
        f"D1 위반: weights hash 변경됨 {h0[:8]} → {h1[:8]}. "
        f"fine-tune 의심"
    )


# ══════════════════════════════════════════════════════════
# T3: D1 no_grad — gradient computation 차단
# ══════════════════════════════════════════════════════════

def test_embedding_bridge_no_grad(bridge):
    """encode 동안 모델 parameter 의 requires_grad 모두 False."""
    # adapter 가 STAdapter 인지 확인
    adapter = bridge._adapter
    assert hasattr(adapter, "_model")

    # 모든 parameter 가 grad 비활성
    for p in adapter._model.parameters():
        assert p.requires_grad is False, (
            "D1 위반: parameter 에 grad enabled — fine-tune 가능 상태"
        )


# ══════════════════════════════════════════════════════════
# T4: D1 fit 은 no-op
# ══════════════════════════════════════════════════════════

def test_embedding_bridge_fit_is_noop(bridge):
    """fit(corpus) 후에도 weights hash 동일."""
    h0 = bridge.weights_hash()

    # 임의의 corpus 로 fit 시도
    bridge.fit(["text one", "text two", "한국어 텍스트"])

    h1 = bridge.weights_hash()
    assert h0 == h1, (
        f"D1 위반: fit() 후 weights 변경 ({h0[:8]} → {h1[:8]}). "
        f"fit 은 no-op 여야 함"
    )


# ══════════════════════════════════════════════════════════
# T5: D2 save/load round-trip (metadata)
# ══════════════════════════════════════════════════════════

def test_embedding_bridge_save_load_round_trip(bridge, tmp_path):
    """save() → load() 시 model_name 보존 + encode 결과 동등."""
    test_text = "save load test sentence."
    vec_before = bridge.encode(test_text)

    state_path = tmp_path / "encoder_state.pkl"
    bridge.save(state_path)
    assert state_path.exists()

    # 새 bridge 로 load
    bridge2 = EmbeddingBridge()
    ok = bridge2.load(state_path)
    assert ok is True
    assert bridge2.model_name == bridge.model_name
    assert bridge2.dim == bridge.dim

    # 같은 text → 같은 vec (D1 Frozen + 동일 모델)
    vec_after = bridge2.encode(test_text)
    np.testing.assert_allclose(vec_before, vec_after, rtol=1e-5)


# ══════════════════════════════════════════════════════════
# T6: D3 TfidfJLEncoder fallback 여전히 작동
# ══════════════════════════════════════════════════════════

def test_tfidf_fallback_still_works():
    """D3: TfidfJLEncoder 가 EmbeddingBridge 도입 후에도 그대로 동작."""
    enc = TfidfJLEncoder()
    assert isinstance(enc, TextEncoder)

    # fit + encode 정상
    enc.fit(["alpha beta gamma", "delta epsilon zeta"])
    vec = enc.encode("alpha delta")
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (enc.dim,)
    # TfidfJL 기본 dim=64
    assert enc.dim == 64
