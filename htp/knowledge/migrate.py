"""
Migration helpers — legacy jsonl 에 UUID 영구 부여 (L2 sidequest session-1).

Design Ref: docs/02-design/features/htp-knowledge-cli-polish.design.md §2.7
Plan: 8 sub-decision #8 — 옵셔널 migration 명령

CLI: `python -m htp.knowledge migrate --add-uuid`

기본 동작:
  1. .htp/knowledge_log.jsonl 백업 → .htp/knowledge_log.pre-uuid.bak
  2. load_all 로 in-memory UUID 부여
  3. 새 jsonl 작성 (모든 entry 에 UUID 포함, tombstone 보존)
"""
from __future__ import annotations

import shutil
from pathlib import Path

from .persistence import KnowledgeStore


def migrate_add_uuid(jsonl_path: Path | str,
                     backup_suffix: str = ".pre-uuid.bak") -> dict:
    """기존 jsonl 의 entry 에 UUID 영구 부여.

    절차:
      1. 백업 생성 (`<path>.pre-uuid.bak`)
      2. KnowledgeStore.load_all 로 entry 복원 (legacy UUID 자동 부여)
      3. 새 jsonl 작성 (모든 entry 에 UUID 포함, tombstone 손실)

    반환: {"migrated": N, "backup_path": str, "had_uuids": bool}

    참고: tombstone 은 마이그레이션 시 제거됨 — load_all 이 이미 적용한
    후이므로 정합성 유지. 이후 새 jsonl 은 깨끗한 entry list.
    """
    p = Path(jsonl_path)
    if not p.exists():
        return {"migrated": 0, "backup_path": None, "had_uuids": False}

    # 1. 백업
    backup = p.with_suffix(p.suffix + backup_suffix)
    shutil.copy2(p, backup)

    # 2. load — UUID 자동 부여 (legacy 도)
    store = KnowledgeStore(p)
    entries = store.load_all()

    # 3. 백업 후 새로 작성 (truncate)
    p.unlink()  # 새 KnowledgeStore 가 부모 디렉토리는 보존
    new_store = KnowledgeStore(p)
    for entry in entries:
        new_store.append(entry)

    return {
        "migrated":    len(entries),
        "backup_path": str(backup),
        "had_uuids":   all(e.id for e in entries),
    }


def migrate_add_interpretation(jsonl_path: Path | str,
                               backup_suffix: str = ".pre-interpretation.bak"
                               ) -> dict:
    """기존 jsonl 의 entry 에 interpretation 필드 명시적 부여 (None default).

    backward-compat 는 KnowledgeStore.load_all 의 `.get("interpretation")` fallback
    으로 이미 보장되므로 *필수는 아님*. 다만 외부 도구 / 문서 호환성을 위해
    명시화하고 싶을 때 사용.

    절차:
      1. 백업 생성 (`<path>.pre-interpretation.bak`)
      2. load_all 로 entry 복원 (interpretation 없는 entry 는 None)
      3. 새 jsonl 작성 (모든 entry 에 interpretation 포함)

    반환: {"migrated": N, "backup_path": str, "had_interpretation": int}
    """
    p = Path(jsonl_path)
    if not p.exists():
        return {"migrated": 0, "backup_path": None,
                "had_interpretation": 0}

    backup = p.with_suffix(p.suffix + backup_suffix)
    shutil.copy2(p, backup)

    store   = KnowledgeStore(p)
    entries = store.load_all()
    had_inter = sum(1 for e in entries
                     if getattr(e, "interpretation", None))

    p.unlink()
    new_store = KnowledgeStore(p)
    for entry in entries:
        new_store.append(entry)

    return {
        "migrated":           len(entries),
        "backup_path":        str(backup),
        "had_interpretation": had_inter,
    }


__all__ = ["migrate_add_uuid", "migrate_add_interpretation"]
