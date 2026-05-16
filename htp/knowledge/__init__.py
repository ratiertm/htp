"""htp.knowledge — Knowledge Loop MVP (Stage 0.5).

Design Ref: htp_thalamus_car_design v4.md §0.5 (Rev 1.3)
            docs/02-design/features/htp-thalamus-car.design.md §4

DAG 규칙: htp/knowledge/ 는 htp/runtime/, htp/thalamus/, htp/memory/ 를 import 하지 않는다.
의존: sklearn, numpy, 표준 라이브러리만.
"""
from .encoder     import TextEncoder, TfidfJLEncoder
from .loop        import (
    KnowledgeLoop, KnowledgeEntry, Neighbor,
    IngestResult, QueryResult, Discovery,
)
from .persistence import KnowledgeStore

__all__ = [
    "TextEncoder", "TfidfJLEncoder",
    "KnowledgeLoop", "KnowledgeEntry", "Neighbor",
    "IngestResult", "QueryResult", "Discovery",
    "KnowledgeStore",
]
