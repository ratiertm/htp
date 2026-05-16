"""
KnowledgeStore — JSONL append-only 영속 저장소.

Design Ref: docs/02-design/features/htp-thalamus-car.design.md §4.4
Plan SC: 'JSON 파일 (.htp/knowledge_log.jsonl)' 선택

DAG: htp/knowledge/ — htp/runtime/* 미참조. 표준 라이브러리 + numpy 만 의존.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# 순환 참조 회피: KnowledgeEntry 는 loop.py 에 정의. type-hint 만 문자열 사용.


class KnowledgeStore:
    """JSONL append-only 영속 저장소.

    파일: .htp/knowledge_log.jsonl (이전 사이클 .htp/ 디렉토리 재사용)
    포맷: 1 line = 1 entry, JSON. vec 은 list 직렬화.
    """

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def default(cls) -> "KnowledgeStore":
        return cls(Path(".htp/knowledge_log.jsonl"))

    def append(self, entry) -> None:
        """KnowledgeEntry 1줄 append."""
        rec = {
            "text":           entry.text,
            "vec":            entry.vec.tolist(),
            "source":         entry.source,
            "timestamp":      entry.timestamp,
            "neighbors":      list(entry.neighbors),
            "conflict_count": entry.conflict_count,
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def load_all(self) -> list:
        """전체 entry 로드. 손상 line 은 skip."""
        # lazy import to avoid 순환
        from .loop import KnowledgeEntry

        if not self.path.exists():
            return []
        entries = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                entries.append(KnowledgeEntry(
                    text=rec["text"],
                    vec=np.array(rec["vec"], dtype=np.float64),
                    source=rec.get("source", ""),
                    timestamp=rec.get("timestamp", ""),
                    neighbors=rec.get("neighbors", []),
                    conflict_count=rec.get("conflict_count", 0),
                ))
        return entries


__all__ = ["KnowledgeStore"]
