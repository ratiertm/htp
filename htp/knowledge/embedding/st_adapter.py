"""
STAdapter — sentence-transformers 어댑터 (sub-5 default).

Design Ref: docs/02-design/features/htp-thalamus-car.sub-5.design.md §3.3
Plan: D1 (Frozen)

D1 강제:
  1. model.eval()  명시
  2. p.requires_grad = False  for all parameters
  3. encode 내부 torch.no_grad() context
"""
from __future__ import annotations

import numpy as np


class STAdapter:
    """sentence-transformers SentenceTransformer 래퍼 (Frozen).

    Plan SC: FR-02 (sentence-transformers 기반)
    D1 검증: weights hash 가 encode 전후 동일해야 함.
    """

    def __init__(self, model_name: str):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            ) from e

        import torch

        self._model = SentenceTransformer(model_name)

        # D1: Frozen — eval mode + grad disabled
        self._model.eval()
        for p in self._model.parameters():
            p.requires_grad = False

        self._dim = int(self._model.get_sentence_embedding_dimension())
        self._model_name = model_name
        self._torch = torch

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return self._model_name

    def encode_one(self, text: str) -> np.ndarray:
        # D1: torch.no_grad() context — gradient computation 완전 차단
        with self._torch.no_grad():
            vec = self._model.encode(
                text, normalize_embeddings=True, show_progress_bar=False,
            )
        return np.asarray(vec, dtype=np.float64)

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        with self._torch.no_grad():
            vecs = self._model.encode(
                texts, normalize_embeddings=True,
                batch_size=8, show_progress_bar=False,
            )
        return np.asarray(vecs, dtype=np.float64)


__all__ = ["STAdapter"]
