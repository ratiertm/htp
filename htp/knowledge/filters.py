"""
filter_entries — source/since/tag 필터 헬퍼 (L2 sidequest session-2 신설).

Design Ref: docs/02-design/features/htp-knowledge-cli-polish.design.md §2.4
Plan SC: FR-07~FR-09

Sub-decision #4: `--since` 파싱은 stdlib datetime + 정규식 (의존성 0).
지원 형식:
  - "Nd"        — N days ago from now() UTC
  - "YYYY-MM"   — 월 첫날 UTC
  - "YYYY-MM-DD" — 해당 일 00:00 UTC
  - ISO datetime — datetime.fromisoformat() 호환

DAG: 형제 모듈 (`.types`) 만 참조.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone, timedelta
from typing   import TYPE_CHECKING

if TYPE_CHECKING:
    from .types import KnowledgeEntry


_SINCE_DAYS_PATTERN = re.compile(r"^(\d+)d$")
_SINCE_YEAR_MONTH   = re.compile(r"^(\d{4})-(\d{2})$")


def filter_entries(
    entries: "list[KnowledgeEntry]",
    source:  "str | None" = None,
    since:   "str | None" = None,
    tag:     "str | None" = None,
) -> "list[KnowledgeEntry]":
    """source/since/tag 필터 적용. None 인자는 무시 (필터 미적용).

    여러 필터 조합 시 AND 적용 (모두 만족하는 entry 만 반환).
    """
    out = list(entries)
    if source is not None:
        out = [e for e in out if e.source == source]
    if since is not None:
        cutoff = parse_since(since)
        out = [e for e in out if _parse_ts(e.timestamp) >= cutoff]
    if tag is not None:
        out = [e for e in out if tag in e.tags]
    return out


def parse_since(spec: str) -> datetime:
    """`--since` 인자를 datetime (UTC) 으로 파싱.

    지원:
      - "30d"        → now() - 30 days (UTC)
      - "2026-04"    → 2026-04-01 00:00 UTC
      - "2026-04-15" → 2026-04-15 00:00 UTC
      - ISO datetime → datetime.fromisoformat (UTC fallback)

    Raises ValueError on unsupported format.
    """
    spec = spec.strip()

    # 1) "Nd" 형식
    if m := _SINCE_DAYS_PATTERN.match(spec):
        return datetime.now(timezone.utc) - timedelta(days=int(m.group(1)))

    # 2) "YYYY-MM" 형식 (월 첫날)
    if m := _SINCE_YEAR_MONTH.match(spec):
        return datetime(int(m.group(1)), int(m.group(2)), 1,
                        tzinfo=timezone.utc)

    # 3) ISO datetime (date or datetime)
    try:
        dt = datetime.fromisoformat(spec)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        pass

    raise ValueError(
        f"invalid --since: {spec!r} "
        f"(supported: 'Nd', 'YYYY-MM', 'YYYY-MM-DD', ISO datetime)"
    )


def _parse_ts(ts: str) -> datetime:
    """Entry timestamp → datetime (UTC). tz-naive 는 UTC 로 간주."""
    if not ts:
        # 빈 timestamp → 매우 오래된 시점 (모든 since 통과)
        return datetime(1970, 1, 1, tzinfo=timezone.utc)
    try:
        dt = datetime.fromisoformat(ts)
    except ValueError:
        return datetime(1970, 1, 1, tzinfo=timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


__all__ = ["filter_entries", "parse_since"]
