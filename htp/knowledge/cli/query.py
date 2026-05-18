"""query subcommand — filter + Bridge §4 routed/compare 모드 (S3)."""
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

    mode = getattr(args, "mode", "flat")
    if mode == "compare":
        return _run_compare(loop, args.question)

    # sub-5 merge 작업 3: confidence 포함 query (I5) + Bridge §4 mode
    result = loop.query_v2(args.question, top_k=10, mode=mode)
    if not result.results:
        print("저장된 지식이 없습니다 (필터 후 0건일 수 있음).")
        return 0

    _print_v2(args.question, result, mode=mode, loop=loop)
    return 0


def _print_v2(question: str, result, mode: str, loop) -> None:
    """공통 출력 — flat/routed 라벨 + confidence header + entries."""
    label = "flat" if mode == "flat" else "routed"
    if result.has_match:
        print(f"◆ [{label}] '{question}' 매칭 {len(result.results)}건 "
              f"(confidence={result.confidence:+.4f}):")
    else:
        print(f"⚠ [{label}] '{question}' Low confidence "
              f"(gap={result.confidence:+.4f}) — 확실한 매칭 없음:")

    marker = "  " if result.has_match else "  ?"
    for r in result.results:
        preview = r.text[:80] + ("..." if len(r.text) > 80 else "")
        print(f"{marker}[{r.similarity:+.3f}] {preview} ({r.source})")


def _run_compare(loop, question: str) -> int:
    """Bridge §4-4 A/B 비교: flat vs routed 결과 나란히 출력."""
    flat_r   = loop.query(question, mode="flat")
    routed_r = loop.query(question, mode="routed")

    total = len(loop._cache)
    print(f"── flat (전체 {total} entries) ──")
    _print_flat(flat_r, loop)

    print()
    if routed_r.routing_info is None:
        print("── routed (signature 없음 → flat fallback) ──")
    else:
        ri = routed_r.routing_info
        sel = ri.get("selected_sources", [])
        cnt = ri.get("candidate_count", "?")
        ent = ri.get("entropy", 0.0)
        print(f"── routed (active={sel}, candidates={cnt}, "
              f"entropy={ent:.2f}) ──")
    _print_flat(routed_r, loop)

    # top-1 비교
    if flat_r.relevant and routed_r.relevant:
        f1 = flat_r.relevant[0].entry_id
        r1 = routed_r.relevant[0].entry_id
        if f1 == r1:
            print("\n✓ top-1 동일")
        else:
            print("\n⚡ top-1 다름 — routed 가 다른 entry 선택")
    return 0


def _print_flat(result, loop) -> None:
    """legacy QueryResult (.relevant) 출력."""
    if not result.relevant:
        print("  (no result)")
        return
    for n in result.relevant[:5]:
        e = loop._cache[n.entry_id]
        preview = e.text[:70] + ("..." if len(e.text) > 70 else "")
        print(f"  [{n.similarity:+.3f}] ({e.source}) {preview}")
