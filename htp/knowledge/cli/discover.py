"""discover subcommand — filter 지원 (L2 sidequest session-3 F3)."""
from __future__ import annotations

from ..encoder import TfidfJLEncoder
from ..filters import filter_entries
from ..loop    import KnowledgeLoop


def run(args) -> int:
    loop = KnowledgeLoop(encoder=TfidfJLEncoder())

    if args.threshold is not None:
        loop.discover_threshold = args.threshold

    if args.source or args.since or args.tag:
        try:
            filtered = filter_entries(
                loop._cache,
                source=args.source, since=args.since, tag=args.tag,
            )
        except ValueError as e:
            print(f"discover: {e}")
            return 2
        loop._cache = filtered

    discoveries = loop.discover()
    if not discoveries:
        print(f"Cross-domain 발견 없음 (threshold={loop.discover_threshold:.2f}).")
        return 0
    print(f"⚡ Top 발견 {len(discoveries)}건:")
    for d in discoveries:
        a = loop._cache[d.entry_a_id]
        b = loop._cache[d.entry_b_id]
        print(f"  [{d.similarity:+.2f}] {d.source_a}:\"{a.text[:50]}...\"")
        print(f"             ↔ {d.source_b}:\"{b.text[:50]}...\"")
    return 0
