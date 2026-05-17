"""htp.knowledge — Knowledge Loop MVP (Stage 0.5) + L2 sidequest CLI polish.

Design Ref:
  - sub-1: htp_thalamus_car_design v4.md §0.5 (Rev 1.3)
            docs/02-design/features/htp-thalamus-car.design.md §4
  - L2 sidequest: docs/02-design/features/htp-knowledge-cli-polish.design.md

DAG 규칙: htp/knowledge/ 는 htp/runtime/, htp/thalamus/, htp/memory/ 를 import 하지 않는다.
의존: sklearn, numpy, 표준 라이브러리만.
"""
from .encoder     import TextEncoder, TfidfJLEncoder
from .types       import KnowledgeEntry, Tombstone     # L2 sidequest session-1
from .loop        import (
    KnowledgeLoop, Neighbor,
    IngestResult, QueryResult, Discovery,
)
from .persistence import KnowledgeStore
from .migrate     import migrate_add_uuid              # L2 sidequest session-1

__all__ = [
    "TextEncoder", "TfidfJLEncoder",
    "KnowledgeLoop", "KnowledgeEntry", "Tombstone", "Neighbor",
    "IngestResult", "QueryResult", "Discovery",
    "KnowledgeStore",
    "migrate_add_uuid",
]
