"""query subcommand — filter 지원 (L2 sidequest session-3 F3)."""
from __future__ import annotations

from ._common import make_loop
from ..filters import filter_entries
from ..loop    import KnowledgeLoop


def run(args) -> int:
    loop = make_loop(getattr(args, "encoder", "tfidf"))

    # filter 가 있으면 cache 를 사전 필터링 후 query
    if args.source or args.since or args.tag:
        try:
            filtered = filter_entries(
                loop._cache,
                source=args.source, since=args.since, tag=args.tag,
            )
        except ValueError as e:
            print(f"query: {e}")
            return 2
        # 임시로 cache 대체 (in-memory 만)
        loop._cache = filtered

    result = loop.query(args.question)
    if not result.relevant:
        print("저장된 지식이 없습니다 (필터 후 0건일 수 있음).")
        return 0
    print(f"◆ '{args.question}' 관련 {len(result.relevant)}건 "
          f"({result.cluster_count}개 클러스터):")
    for n in result.relevant:
        e = loop._cache[n.entry_id]
        preview = e.text[:80] + ("..." if len(e.text) > 80 else "")
        print(f"  [{n.similarity:+.2f}] {preview} ({e.source})")
    return 0
