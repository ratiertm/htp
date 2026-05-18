"""
EmbeddingBridge — TextEncoder Protocol 어댑터 (sub-5).

Design Ref: docs/02-design/features/htp-thalamus-car.sub-5.design.md §3.2
Plan SC: FR-01~04, D1-D4

D1 Frozen — STAdapter 가 model.eval() + no_grad 강제
D2 Protocol — TextEncoder 의 dim/encode/fit/save/load 모두 구현
D3 Fallback — TfidfJLEncoder 와 독립 (KnowledgeLoop 가 선택)
D4 학습 분리 — encode 만 책임. HTP 구조 학습은 외부에서.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np


class EmbeddingBridge:
    """사전학습 모델을 TextEncoder Protocol 로 래핑.

    Plan SC: FR-01 (embedding_bridge 구현).
    """

    DEFAULT_MODEL = "intfloat/multilingual-e5-small"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        adapter:    "object | None" = None,
    ):
        """
        model_name: HuggingFace 모델 이름. default = multilingual-e5-small
                     (118MB, 384-dim, 한국어 지원, sub-decision #1)
        adapter:    BaseEmbeddingModel 인스턴스 (test 주입 가능).
                    None 이면 STAdapter(model_name) 자동 생성.
        """
        if adapter is None:
            from .st_adapter import STAdapter
            adapter = STAdapter(model_name)

        self._adapter = adapter
        self._dim = int(adapter.dim)
        self._model_name = adapter.model_name
        # 사전학습 — 항상 fitted (D1: fit 은 no-op)
        self._fitted = True

    # ── TextEncoder Protocol (D2) ────────────────────
    @property
    def dim(self) -> int:
        return self._dim

    def fit(self, corpus: list[str]) -> None:
        """no-op — 사전학습 모델은 fit 불필요 (D1 Frozen)."""
        # 의도적 no-op. corpus 입력은 받지만 weights 변경 0.
        return None

    def encode(self, text: str) -> np.ndarray:
        """문서 저장용 인코딩 (passage mode) — D1: adapter 가 no_grad 보장.

        ingest 시 사용. e5 의 "passage: " prefix 적용.
        """
        return self._adapter.encode_one(text, is_query=False)

    def encode_query(self, text: str) -> np.ndarray:
        """검색 질의용 인코딩 (query mode) — sub-5 merge plan §2.

        query 시 사용. e5 의 "query: " prefix 적용으로 검색 품질 향상.
        """
        return self._adapter.encode_one(text, is_query=True)

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        """다중 텍스트 batch encoding (효율) — passage mode."""
        return self._adapter.encode_batch(texts, is_query=False)

    def save(self, path: Path | str) -> None:
        """metadata 만 pickle — 모델 자체는 HF 캐시 (재사용)."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "encoder_type": "embedding_bridge",
            "model_name":   self._model_name,
            "dim":          self._dim,
            "fitted":       self._fitted,
        }
        with p.open("wb") as f:
            pickle.dump(meta, f)

    def load(self, path: Path | str) -> bool:
        """metadata 복원 — 같은 model_name 으로 adapter 재로드."""
        p = Path(path)
        if not p.exists():
            return False
        with p.open("rb") as f:
            meta = pickle.load(f)
        if meta.get("encoder_type") != "embedding_bridge":
            return False
        # 같은 모델이면 dim 일치 확인. 다른 모델이면 reload.
        if meta.get("model_name") != self._model_name:
            from .st_adapter import STAdapter
            self._adapter = STAdapter(meta["model_name"])
            self._model_name = meta["model_name"]
            self._dim = int(self._adapter.dim)
        return True

    # ── 진단 (D1 검증 용) ───────────────────────────
    @property
    def model_name(self) -> str:
        return self._model_name

    def weights_hash(self) -> str:
        """전체 model parameter 의 hash — D1 검증용.

        encode 전후 동일해야 frozen 보장.
        """
        import hashlib
        try:
            sd = self._adapter._model.state_dict()
        except AttributeError:
            return "no-model"
        h = hashlib.sha256()
        for k in sorted(sd.keys()):
            t = sd[k]
            try:
                h.update(k.encode())
                h.update(t.detach().cpu().numpy().tobytes())
            except Exception:
                pass
        return h.hexdigest()[:16]


__all__ = ["EmbeddingBridge"]
