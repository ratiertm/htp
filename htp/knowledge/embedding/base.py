"""
BaseEmbeddingModel — 사전학습 모델 어댑터 Protocol (sub-5).

Design Ref: docs/02-design/features/htp-thalamus-car.sub-5.design.md §3.1

EmbeddingBridge 내부에서 모델 종류별 어댑터 다형성.
예: STAdapter (sentence-transformers) / HFAdapter (transformers raw)
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable, TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


@runtime_checkable
class BaseEmbeddingModel(Protocol):
    """모델 어댑터 인터페이스 — encode 만 책임 (D1 frozen)."""

    @property
    def dim(self) -> int: ...

    @property
    def model_name(self) -> str: ...

    def encode_one(self, text: str) -> "np.ndarray": ...

    def encode_batch(self, texts: list[str]) -> "np.ndarray": ...


__all__ = ["BaseEmbeddingModel"]
