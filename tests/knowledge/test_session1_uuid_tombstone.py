"""
L2 sidequest session-1 — UUID + Tombstone + Migration tests.

Design Ref: docs/02-design/features/htp-knowledge-cli-polish.design.md §2.1-2.2, §2.7
Plan SC: T1, T4 (session-1 부분) + migration

신규 테스트 (148 → 151):
- test_entry_uuid_default                  — UUID4 자동 부여 (sub-decision #3)
- test_tombstone_round_trip_at_store_level — append_tombstone + load_all
- test_migrate_add_uuid_idempotent         — migration 옵셔널 명령
"""
from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib  import Path

import numpy as np
import pytest

from htp.knowledge import (
    KnowledgeEntry, Tombstone, KnowledgeStore, migrate_add_uuid,
)


# ══════════════════════════════════════════════════════════
# T1: UUID 자동 부여 (sub-decision #3)
# ══════════════════════════════════════════════════════════

def test_entry_uuid_default():
    """KnowledgeEntry() 가 UUID4 를 default_factory 로 자동 부여.

    backward-compat: 기존 호출자가 id 인자 없이 생성 가능.
    """
    e1 = KnowledgeEntry(
        text="t", vec=np.zeros(8), source="s", timestamp="2026-05-17",
    )
    e2 = KnowledgeEntry(
        text="t", vec=np.zeros(8), source="s", timestamp="2026-05-17",
    )

    # UUID4 형식 — 36 chars (8-4-4-4-12)
    assert isinstance(e1.id, str)
    assert len(e1.id) == 36
    assert e1.id.count("-") == 4

    # 두 entry 의 UUID 는 서로 달라야 함
    assert e1.id != e2.id

    # tags 도 default 빈 list
    assert e1.tags == []

    # 명시 id 도 가능
    custom = KnowledgeEntry(
        text="t", vec=np.zeros(8), source="s", timestamp="2026-05-17",
        id="custom-id-123", tags=["foo", "bar"],
    )
    assert custom.id == "custom-id-123"
    assert custom.tags == ["foo", "bar"]


# ══════════════════════════════════════════════════════════
# T4: Tombstone round-trip at store level
# ══════════════════════════════════════════════════════════

def test_tombstone_round_trip_at_store_level():
    """append_tombstone(delete) → load_all 시 ref_id 매칭 entry 미반환.

    FR-18 핵심: JSONL append-only 보호.
    기존 entry 라인은 절대 수정되지 않음 — tombstone marker 만 별도 append.
    """
    with tempfile.TemporaryDirectory() as td:
        store = KnowledgeStore(Path(td) / "knowledge_log.jsonl")

        # 3 entry 저장
        e1 = KnowledgeEntry(
            text="first",  vec=np.array([1.0, 0.0, 0.0, 0.0]),
            source="brain", timestamp="2026-05-17T10:00:00+00:00",
        )
        e2 = KnowledgeEntry(
            text="second", vec=np.array([0.0, 1.0, 0.0, 0.0]),
            source="ai",    timestamp="2026-05-17T10:05:00+00:00",
        )
        e3 = KnowledgeEntry(
            text="third",  vec=np.array([0.0, 0.0, 1.0, 0.0]),
            source="infra", timestamp="2026-05-17T10:10:00+00:00",
        )
        store.append(e1)
        store.append(e2)
        store.append(e3)

        # 1차 load — 3 entry 모두
        loaded = store.load_all()
        assert len(loaded) == 3
        loaded_ids = {e.id for e in loaded}
        assert e1.id in loaded_ids
        assert e2.id in loaded_ids
        assert e3.id in loaded_ids

        # e2 삭제 (tombstone)
        store.append_tombstone(Tombstone(
            kind="delete",
            ref_id=e2.id,
            timestamp=datetime.now(timezone.utc).isoformat(),
        ))

        # 2차 load — e2 제외
        loaded_after = store.load_all()
        assert len(loaded_after) == 2
        loaded_after_ids = {e.id for e in loaded_after}
        assert e1.id in loaded_after_ids
        assert e2.id not in loaded_after_ids   # 삭제 확인
        assert e3.id in loaded_after_ids

        # JSONL 무손상 검증: raw line 수 = 3 entries + 1 tombstone
        with (Path(td) / "knowledge_log.jsonl").open() as f:
            raw_lines = [l for l in f if l.strip()]
        assert len(raw_lines) == 4

        # 기존 entry 라인은 수정되지 않음 (e1, e2, e3 의 text 가 그대로)
        texts_in_raw = []
        for line in raw_lines:
            rec = json.loads(line)
            if not rec.get("__tombstone__"):
                texts_in_raw.append(rec["text"])
        assert "first"  in texts_in_raw
        assert "second" in texts_in_raw   # e2 line 도 *raw* 에 보존
        assert "third"  in texts_in_raw


# ══════════════════════════════════════════════════════════
# Migration: --add-uuid 옵셔널 명령
# ══════════════════════════════════════════════════════════

def test_migrate_add_uuid_idempotent():
    """legacy jsonl (id 필드 없는 raw entry) → migration → 모든 entry UUID 보유.

    절차:
      1. id 필드 없는 raw entry 직접 작성 (legacy 시뮬레이션)
      2. migrate_add_uuid 호출
      3. 백업 파일 존재 확인
      4. 새 jsonl 의 모든 entry 에 id 필드 존재
      5. 재실행 idempotent (이미 UUID 있어도 안전)
    """
    with tempfile.TemporaryDirectory() as td:
        jsonl_path = Path(td) / "knowledge_log.jsonl"

        # 1. legacy entry 직접 작성 (id 필드 없음)
        with jsonl_path.open("w") as f:
            for i, text in enumerate(["alpha", "beta", "gamma"]):
                rec = {
                    "text":      text,
                    "vec":       [1.0, 0.0, 0.0, 0.0],
                    "source":    f"src{i}",
                    "timestamp": f"2026-05-17T10:0{i}:00+00:00",
                    "neighbors": [],
                    "conflict_count": 0,
                    # id, tags 필드 의도적 누락 (legacy)
                }
                f.write(json.dumps(rec) + "\n")

        # 2. migration
        result = migrate_add_uuid(jsonl_path)
        assert result["migrated"] == 3
        assert result["had_uuids"] is True

        # 3. 백업 파일 존재
        backup = Path(result["backup_path"])
        assert backup.exists()
        with backup.open() as f:
            backup_lines = [json.loads(l) for l in f if l.strip()]
        # 백업의 raw 는 id 없음 (legacy 상태)
        assert all("id" not in rec or rec["id"] is None for rec in backup_lines)

        # 4. 새 jsonl 의 모든 entry 에 id 필드
        store = KnowledgeStore(jsonl_path)
        loaded = store.load_all()
        assert len(loaded) == 3
        assert all(e.id is not None and len(e.id) == 36 for e in loaded)
        # 3개 모두 서로 다른 UUID
        assert len({e.id for e in loaded}) == 3

        # 5. 재실행 idempotent — 다시 migration 해도 안전 (UUID 보존)
        ids_before = [e.id for e in loaded]
        result2 = migrate_add_uuid(jsonl_path)
        assert result2["migrated"] == 3

        loaded2 = store.load_all()
        ids_after = [e.id for e in loaded2]
        # 재실행 시 동일 UUID 유지 (이미 jsonl 에 id 가 있으므로 load 시 그대로)
        assert sorted(ids_before) == sorted(ids_after)
