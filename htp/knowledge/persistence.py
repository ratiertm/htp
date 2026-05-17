"""
KnowledgeStore — JSONL append-only 영속 저장소.

Design Ref:
  - sub-1: docs/02-design/features/htp-thalamus-car.design.md §4.4
  - L2 sidequest session-1: docs/02-design/features/htp-knowledge-cli-polish.design.md §2.2

L2 sidequest 확장:
  - append_tombstone(): delete/edit 마커를 별도 라인으로 append
  - load_all(): tombstone 을 적용하여 valid entry 만 반환
  - 기존 entry (UUID 없음) 는 in-memory 자동 부여 — backward-compat

DAG: htp/knowledge/ — htp/runtime/* 미참조. 표준 라이브러리 + numpy 만 의존.
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path

import numpy as np

# 순환 참조 회피: KnowledgeEntry 는 types.py 에 정의 (session-1 신규).
# loop.py 가 backward-compat 으로 re-export.


# ══════════════════════════════════════════════════════════
# JSONL 레코드 종류 식별
# ══════════════════════════════════════════════════════════

_REC_KEY_TOMBSTONE = "__tombstone__"   # tombstone 라인 마커


class KnowledgeStore:
    """JSONL append-only 영속 저장소.

    파일: .htp/knowledge_log.jsonl
    포맷:
      1 line = 1 entry  ({"text": ..., "vec": [...], "id": "...", "tags": [...], ...})
      또는    = 1 tombstone ({"__tombstone__": true, "kind": "delete", "ref_id": "...", ...})

    load_all() 가 tombstone 을 적용해 valid entry 만 반환.
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
            # L2 sidequest 신규 필드
            "id":             getattr(entry, "id", None),
            "tags":           getattr(entry, "tags", []),
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def append_tombstone(self, ts) -> None:
        """Tombstone (delete/edit 마커) 1줄 append.

        L2 sidequest session-1: append-only 보존을 위해 기존 entry 라인은
        수정하지 않고 별도 라인으로 marker 추가.
        """
        rec = {
            _REC_KEY_TOMBSTONE: True,
            "kind":           ts.kind,
            "ref_id":         ts.ref_id,
            "timestamp":      ts.timestamp,
            "replacement_id": ts.replacement_id,
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def load_all(self) -> list:
        """전체 entry 로드 (tombstone 적용 후).

        알고리즘:
          1. 모든 라인 순회 — entry / tombstone 분리
          2. delete tombstone: ref_id 매칭 entry 제외
          3. edit tombstone: ref_id 매칭 entry 를 replacement_id 의 entry 로 대체
          4. UUID 없는 legacy entry: in-memory UUID 자동 부여
        """
        # lazy import to avoid 순환 (types.py 가 KnowledgeEntry 소유)
        from .types import KnowledgeEntry

        if not self.path.exists():
            return []

        # ── 1차 패스: raw 레코드 분리 ──
        entries_by_id: dict[str, KnowledgeEntry] = {}
        order: list[str] = []                   # 삽입 순서 보존
        tombstones: list[dict] = []

        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Tombstone 라인
                if rec.get(_REC_KEY_TOMBSTONE):
                    tombstones.append(rec)
                    continue

                # Entry 라인 — UUID 자동 부여 (legacy)
                ent_id = rec.get("id") or str(uuid.uuid4())
                entry = KnowledgeEntry(
                    text           = rec["text"],
                    vec            = np.array(rec["vec"], dtype=np.float64),
                    source         = rec.get("source", ""),
                    timestamp      = rec.get("timestamp", ""),
                    neighbors      = rec.get("neighbors", []),
                    conflict_count = rec.get("conflict_count", 0),
                    id             = ent_id,
                    tags           = list(rec.get("tags", []) or []),
                )

                # 중복 id 방어 — 같은 id 가 두 번 나오면 후자가 우선 (edit replacement)
                if ent_id in entries_by_id:
                    # order 갱신: 기존 위치 유지하되 내용은 덮어쓰기
                    entries_by_id[ent_id] = entry
                else:
                    entries_by_id[ent_id] = entry
                    order.append(ent_id)

        # ── 2차 패스: tombstone 적용 ──
        for ts in tombstones:
            kind   = ts.get("kind")
            ref_id = ts.get("ref_id")
            if not ref_id:
                continue
            if kind == "delete":
                entries_by_id.pop(ref_id, None)
                if ref_id in order:
                    order.remove(ref_id)
            elif kind == "edit":
                # edit: ref_id 의 entry 제거 (replacement_id 의 entry 가 이미 entries 에 존재해야 함)
                entries_by_id.pop(ref_id, None)
                if ref_id in order:
                    order.remove(ref_id)
            # 기타 kind 는 무시 (forward-compat)

        return [entries_by_id[eid] for eid in order if eid in entries_by_id]


__all__ = ["KnowledgeStore"]
