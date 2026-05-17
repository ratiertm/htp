"""
KnowledgeEntry + Tombstone — knowledge 도메인 dataclass (L2 sidequest session-1 신설).

Design Ref: docs/02-design/features/htp-knowledge-cli-polish.design.md §2.1
Plan SC: FR-12 (Tags) + Plan §6 Decision Record (UUID 전면 도입)

기존 `loop.py` 의 KnowledgeEntry 는 여기로 이동. backward-compat 위해
`loop.py` 가 동일 클래스를 re-export.

DAG: numpy / dataclasses / uuid / datetime 만 의존. 형제 모듈 미참조.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field

import numpy as np


@dataclass
class KnowledgeEntry:
    """Knowledge Loop 의 기본 단위.

    L2 sidequest 확장:
      - id: UUID4 (sub-decision #3) — 새 entry 자동 부여, 기존 7 entry 는
        load_all 시 in-memory 자동 부여 (영속화는 migration 명령으로)
      - tags: list[str] (sub-decision #2) — 사후 태그 추가 가능
    """
    text: str
    vec: np.ndarray
    source: str
    timestamp: str
    neighbors: list = field(default_factory=list)
    conflict_count: int = 0
    # ── L2 sidequest 확장 ──────────────────────
    id:   str       = field(default_factory=lambda: str(uuid.uuid4()))
    tags: list[str] = field(default_factory=list)


@dataclass
class Tombstone:
    """삭제/수정 마커 — JSONL 에 별도 라인으로 append.

    kind:
      - "delete":  ref_id 매칭 entry 를 load_all 결과에서 제외
      - "edit":    ref_id 매칭 entry 를 replacement_id 의 entry 로 대체

    Append-only 보존: 기존 entry 라인은 절대 수정하지 않음.
    """
    kind:           str           # "delete" | "edit"
    ref_id:         str           # 타깃 entry UUID
    timestamp:      str
    replacement_id: str | None = None    # edit 시 새 entry 의 UUID


__all__ = ["KnowledgeEntry", "Tombstone"]
