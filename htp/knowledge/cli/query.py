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

    # sub-5 merge 작업 3: confidence 포함 query (I5)
    result = loop.query_v2(args.question, top_k=10)
    if not result.results:
        print("저장된 지식이 없습니다 (필터 후 0건일 수 있음).")
        return 0

    # confidence header
    if result.has_match:
        print(f"◆ '{args.question}' 매칭 {len(result.results)}건 "
              f"(confidence={result.confidence:+.4f}):")
    else:
        print(f"⚠ '{args.question}' Low confidence (gap={result.confidence:+.4f}) — "
              f"확실한 매칭 없음:")

    for r in result.results:
        preview = r.text[:80] + ("..." if len(r.text) > 80 else "")
        marker = "  " if result.has_match else "  ?"
        print(f"{marker}[{r.similarity:+.3f}] {preview} ({r.source})")
    return 0
