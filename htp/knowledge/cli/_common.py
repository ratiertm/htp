"""Shared helpers for CLI subcommands (sub-5 — encoder 선택 통합)."""
from __future__ import annotations

from ..encoder import TfidfJLEncoder
from ..loop    import KnowledgeLoop


def make_loop(encoder_type: str = "tfidf") -> KnowledgeLoop:
    """argparse 의 --encoder 값을 KnowledgeLoop 생성에 반영 (D3 fallback).

    encoder_type:
      - "tfidf":     TfidfJLEncoder (기본, 빠름)
      - "embedding": EmbeddingBridge (사전학습, 정확)
    """
    if encoder_type == "embedding":
        try:
            from ..embedding import EmbeddingBridge
            enc = EmbeddingBridge()
        except ImportError as e:
            # D3: 미설치 환경 안전 fallback
            print(f"[warn] embedding 사용 불가 — fallback to tfidf: {e}")
            enc = TfidfJLEncoder()
    else:
        enc = TfidfJLEncoder()
    return KnowledgeLoop(encoder=enc)


__all__ = ["make_loop"]
