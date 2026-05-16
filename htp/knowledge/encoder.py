"""
TextEncoder Protocol + MVP TfidfJLEncoder.

Design Ref: docs/02-design/features/htp-thalamus-car.design.md §4.1, §4.2
Plan SC: FR-05.2 (TextEncoder Protocol), FR-05.3 (MVP TF-IDF+JL)

이 파일은 htp/knowledge/ 트리 — DAG 강제: htp/runtime/* 미참조.
의존: sklearn, numpy 만.

TextEncoder Protocol 은 KnowledgeLoop / LLMRegion / RegionSignature 공유.
Stage 6 EmbeddingBridge 가 동일 Protocol 의 다른 구현이 됨.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.random_projection      import GaussianRandomProjection


@runtime_checkable
class TextEncoder(Protocol):
    """텍스트 → 64-dim 벡터 인코딩 프로토콜."""

    @property
    def dim(self) -> int: ...
    def encode(self, text: str) -> np.ndarray: ...
    def fit(self, corpus: list[str]) -> None: ...


class TfidfJLEncoder:
    """TF-IDF + Gaussian Random Projection MVP.

    Design Ref: §4.2 — 의도적 조잡. Stage 6 EmbeddingBridge 로 교체 가능.

    구현 세부:
      - ngram_range=(1,2): unigram + bigram (영문 술어 공유 + phrase 보강)
      - token_pattern: 한/영 모두 매칭
      - GaussianRandomProjection(n=64): JL Lemma 차원 보존
      - L2 정규화: cosine similarity 비교를 위해 단위 벡터화
    """

    def __init__(self, dim: int = 64, max_features: int = 5000,
                 random_state: int = 42):
        self._dim = dim
        self._tfidf = TfidfVectorizer(
            max_features = max_features,
            ngram_range  = (1, 2),
            lowercase    = True,
            token_pattern = r"(?u)\b\w+\b",
        )
        self._jl = GaussianRandomProjection(
            n_components = dim,
            random_state = random_state,
        )
        self._fitted = False

    @property
    def dim(self) -> int:
        return self._dim

    def fit(self, corpus: list[str]) -> None:
        """전체 코퍼스로 어휘 + JL 행렬 학습."""
        if not corpus:
            return
        X_sparse = self._tfidf.fit_transform(corpus)
        # GaussianRandomProjection 은 n_features ≥ n_components 필요
        # max_features 가 dim 보다 작으면 fallback
        if X_sparse.shape[1] < self._dim:
            self._jl = None
        else:
            self._jl.fit(X_sparse)
        self._fitted = True

    def encode(self, text: str) -> np.ndarray:
        """text → 64-dim 벡터 (L2 정규화)."""
        if not self._fitted:
            self.fit([text])

        X_sparse = self._tfidf.transform([text])

        if self._jl is None:
            # max_features 부족 fallback: dense 변환 + zero-pad 또는 truncate
            x = np.asarray(X_sparse.toarray()).flatten()
            if x.shape[0] < self._dim:
                x = np.pad(x, (0, self._dim - x.shape[0]))
            else:
                x = x[:self._dim]
        else:
            x = np.asarray(self._jl.transform(X_sparse)).flatten()

        norm = float(np.linalg.norm(x))
        return x / norm if norm > 1e-8 else x


__all__ = ["TextEncoder", "TfidfJLEncoder"]
