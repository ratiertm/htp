"""list / delete / edit / tag subcommand (L2 sidequest session-3 F4)."""
from __future__ import annotations

import sys

from ._common import make_loop
from ..filters import filter_entries
from ..loop    import KnowledgeLoop


def _make_loop(args=None) -> KnowledgeLoop:
    enc_type = getattr(args, "encoder", "tfidf") if args else "tfidf"
    return make_loop(enc_type)


# ── list ──────────────────────────────────────────
def list_run(args) -> int:
    loop = _make_loop(args)
    try:
        filtered = filter_entries(
            loop._cache,
            source=args.source, since=args.since, tag=args.tag,
        )
    except ValueError as e:
        print(f"list: {e}")
        return 2

    limited = filtered[: args.limit]
    if not limited:
        print("no entries")
        return 0

    print(f"     id        source          timestamp                preview")
    print(f"     --------  --------------  -----------------------  -------")
    for e in limited:
        preview = e.text[:50] + ("..." if len(e.text) > 50 else "")
        ts = (e.timestamp or "")[:19]
        # htp-conflict-interpretation: 해석 보유 entry 는 💡 마크.
        marker = "💡 " if getattr(e, "interpretation", None) else "   "
        print(f"  {marker}{e.id[:8]}  {e.source:14}  {ts:23}  {preview}")
    print(f"\n({len(limited)} of {len(filtered)} after filter)")
    return 0


# ── delete ────────────────────────────────────────
def delete_run(args) -> int:
    loop = _make_loop(args)
    ok = loop.delete(args.entry_id)
    if not ok:
        # full UUID 가 아닐 수도 — prefix 매칭 시도
        target = next(
            (e for e in loop._cache if e.id.startswith(args.entry_id)),
            None,
        )
        if target is None:
            print(f"delete: entry not found: {args.entry_id}",
                  file=sys.stderr)
            return 1
        ok = loop.delete(target.id)
    if ok:
        print(f"✓ deleted (id={args.entry_id})")
        return 0
    return 1


# ── edit ──────────────────────────────────────────
def edit_run(args) -> int:
    loop = _make_loop(args)
    target = _resolve_id(loop, args.entry_id)
    if target is None:
        print(f"edit: entry not found: {args.entry_id}", file=sys.stderr)
        return 1
    edited = loop.edit(target, args.text)
    if edited is None:
        return 1
    print(f"✓ edited (id={edited.id[:8]})")
    return 0


# ── tag ───────────────────────────────────────────
def tag_run(args) -> int:
    loop = _make_loop(args)
    target = _resolve_id(loop, args.entry_id)
    if target is None:
        print(f"tag: entry not found: {args.entry_id}", file=sys.stderr)
        return 1
    new_tags = [t.strip() for t in args.add.split(",") if t.strip()]
    e = loop.add_tags(target, new_tags)
    if e is None:
        return 1
    print(f"✓ tags: {e.tags} (id={e.id[:8]})")
    return 0


def _resolve_id(loop, partial: str) -> str | None:
    """전체 UUID 또는 prefix 매칭."""
    full = next((e.id for e in loop._cache if e.id == partial), None)
    if full:
        return full
    matches = [e.id for e in loop._cache if e.id.startswith(partial)]
    if len(matches) == 1:
        return matches[0]
    return None
