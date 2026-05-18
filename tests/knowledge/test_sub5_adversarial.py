"""D1 원칙 adversarial 검증 (sub-5 merge plan 작업 1).

Design Ref: docs/01-plan/features/htp-sub5-merge-plan.md §1

의도적으로 D1 (Frozen weights) 을 위반하려 시도했을 때 방어가 작동하는지 검증.
이 테스트가 존재하는 한, D1 위반 코드가 PR 에 들어와도 CI 에서 차단됨.

실제 위험 시나리오:
  - fine-tune 코드를 추가했지만 학습 데이터가 적어서 weights 변화가
    hash 정밀도 안에 숨는 경우 → adversarial test 가 잡아야 함.

Note: 실제 코드 경로는 `htp.knowledge.embedding` (bridge.py 안의 EmbeddingBridge).
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


def test_d1_adversarial_grad_enable_blocked():
    """D1 위반 시도: requires_grad 를 True 로 바꿔도 no_grad 컨텍스트가 방어.

    누군가 STAdapter 외부에서 의도적으로 grad 를 활성화해도, encode 내부의
    torch.no_grad() 컨텍스트가 gradient 계산을 차단하여 weights 가 불변.
    """
    from htp.knowledge.embedding import EmbeddingBridge

    bridge = EmbeddingBridge()
    weights_before = bridge.weights_hash()

    # 의도적 위반 시도: grad 활성화
    for p in bridge._adapter._model.parameters():
        p.requires_grad = True

    # encode 호출 — no_grad context 가 여전히 방어
    vec = bridge.encode("test adversarial input — should not change weights")

    # 결과: 벡터는 정상 생성되지만 weights 는 불변
    assert vec.shape == (bridge.dim,)
    assert np.isfinite(vec).all()
    assert bridge.weights_hash() == weights_before, (
        "D1 위반: requires_grad=True 상태에서 encode 후 weights 변경됨. "
        "torch.no_grad() 컨텍스트가 작동하지 않음."
    )

    # 정리: 원래 상태 복원 (다른 테스트 영향 방지 — module-scope fixture 가 아니더라도 안전)
    for p in bridge._adapter._model.parameters():
        p.requires_grad = False


def test_d1_adversarial_fit_with_data_noop():
    """D1 위반 시도: fit() 에 대량 데이터를 넣어도 weights 불변.

    누군가 EmbeddingBridge.fit() 을 실제 학습 시도로 오해/악용해도,
    fit 이 no-op 로 구현되어 있어 weights 불변.
    """
    from htp.knowledge.embedding import EmbeddingBridge

    bridge = EmbeddingBridge()
    weights_before = bridge.weights_hash()

    # 의도적 위반 시도: 실제 학습 데이터 대량 주입
    corpus = [
        "pattern completion in hippocampal CA3",
        "associative memory recall mechanism",
        "neural network learning rule",
        "synaptic plasticity Hebbian update",
        "sharp wave ripple consolidation",
    ] * 100   # 500 텍스트 — 실제 fine-tune 이라면 weights 가 변할 양

    bridge.fit(corpus)   # D1: no-op 이어야 함

    assert bridge.weights_hash() == weights_before, (
        "D1 위반: fit(500 corpus) 후 weights 변경됨. fit 이 no-op 가 아님."
    )
