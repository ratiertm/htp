"""
htp.knowledge.embedding — 사전학습 embedding 모델 어댑터 (sub-5 신설).

Design Ref: docs/02-design/features/htp-thalamus-car.sub-5.design.md §1, §3
Plan SC: FR-01~17, D1-D4 design 원칙

핵심: `TextEncoder` Protocol 의 추가 구현체 `EmbeddingBridge`.
TfidfJLEncoder ↔ EmbeddingBridge 1:1 교체 가능 (D2).

DAG 단방향:
  embedding/* → sentence-transformers / numpy / torch
  금지: embedding/* → htp.runtime / htp.thalamus / htp.memory
"""
from __future__ import annotations

from .base   import BaseEmbeddingModel
from .bridge import EmbeddingBridge

__all__ = [
    "BaseEmbeddingModel",
    "EmbeddingBridge",
]
