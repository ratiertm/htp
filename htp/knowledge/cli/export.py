"""export subcommand — markdown / json / obsidian (L2 sidequest session-3 F5)."""
from __future__ import annotations

import sys

from ..encoder    import TfidfJLEncoder
from ..exporters  import export_markdown, export_json, export_obsidian
from ..filters    import filter_entries
from ..loop       import KnowledgeLoop


def run(args) -> int:
    loop = KnowledgeLoop(encoder=TfidfJLEncoder())

    try:
        entries = filter_entries(
            loop._cache,
            source=args.source, since=args.since, tag=args.tag,
        )
    except ValueError as e:
        print(f"export: {e}", file=sys.stderr)
        return 2

    if args.format == "markdown":
        sys.stdout.write(export_markdown(entries, group_by=args.group_by))
        return 0
    if args.format == "json":
        sys.stdout.write(export_json(entries))
        sys.stdout.write("\n")
        return 0
    if args.format == "obsidian":
        if not args.dir_path:
            print("export obsidian: --dir is required", file=sys.stderr)
            return 2
        n = export_obsidian(entries, args.dir_path)
        print(f"✓ {n} files written to {args.dir_path}")
        return 0
    return 1
