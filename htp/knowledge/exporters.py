"""
exporters — knowledge entries 를 외부 포맷으로 변환 (L2 sidequest session-3 신설).

Design Ref: docs/02-design/features/htp-knowledge-cli-polish.design.md §2.5
Plan SC: FR-14 (markdown) / FR-15 (json) / FR-16 (obsidian)

3 포맷:
  - markdown: source 별 섹션 + timestamp 정렬
  - json:     원본 vec 포함 array (round-trip 가능)
  - obsidian: 파일 단위 split + YAML frontmatter

DAG: 형제 모듈 (.types) 만 참조.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib     import Path
from typing      import TYPE_CHECKING

if TYPE_CHECKING:
    from .types import KnowledgeEntry


# ══════════════════════════════════════════════════════════
# Markdown
# ══════════════════════════════════════════════════════════

def export_markdown(entries: "list[KnowledgeEntry]",
                    group_by: str = "source",
                    title: str = "HTP Knowledge Export") -> str:
    """source 별 섹션 + timestamp 정렬 markdown.

    group_by:
      - "source": source 별 H2 섹션
      - "flat":   timestamp 순 단일 list
    """
    lines: list[str] = [f"# {title}", ""]

    if not entries:
        lines.append("_no entries_")
        return "\n".join(lines)

    if group_by == "source":
        groups: dict[str, list] = defaultdict(list)
        for e in entries:
            groups[e.source].append(e)
        for source in sorted(groups.keys()):
            lines.append(f"## {source}")
            lines.append("")
            sorted_entries = sorted(groups[source], key=lambda e: e.timestamp)
            for e in sorted_entries:
                lines.extend(_markdown_entry(e))
                lines.append("")
    else:   # flat
        sorted_entries = sorted(entries, key=lambda e: e.timestamp)
        for e in sorted_entries:
            lines.append(f"## {e.source} — {e.timestamp}")
            lines.append("")
            lines.extend(_markdown_entry(e, include_source=False))
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _markdown_entry(e: "KnowledgeEntry",
                    include_source: bool = True) -> list[str]:
    out = [f"- **{e.timestamp}**"]
    if include_source:
        out[0] += f"  (`{e.source}`)"
    out.append(f"  {e.text}")
    if e.tags:
        out.append(f"  - tags: " + ", ".join(f"`{t}`" for t in e.tags))
    out.append(f"  - id: `{e.id}`")
    return out


# ══════════════════════════════════════════════════════════
# JSON
# ══════════════════════════════════════════════════════════

def export_json(entries: "list[KnowledgeEntry]",
                include_vec: bool = True) -> str:
    """JSON array — vec 포함 round-trip 가능.

    include_vec=False 시 vec 제외 (slim export).
    """
    records = []
    for e in entries:
        rec = {
            "id":             e.id,
            "text":           e.text,
            "source":         e.source,
            "timestamp":      e.timestamp,
            "tags":           list(e.tags),
            "neighbors":      list(e.neighbors),
            "conflict_count": e.conflict_count,
        }
        if include_vec:
            rec["vec"] = e.vec.tolist()
        records.append(rec)
    return json.dumps(records, ensure_ascii=False, indent=2)


# ══════════════════════════════════════════════════════════
# Obsidian (파일 단위 split + frontmatter)
# ══════════════════════════════════════════════════════════

def export_obsidian(entries: "list[KnowledgeEntry]",
                    dir_path: Path | str) -> int:
    """파일 단위 markdown + YAML frontmatter.

    파일명: {timestamp_compact}-{source}-{id_short}.md
    frontmatter: id / source / tags / created

    반환: 작성된 파일 수
    """
    out_dir = Path(dir_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for e in entries:
        # 파일명: timestamp 의 일부 사용 (sortable)
        ts_compact = (e.timestamp or "unknown").replace(":", "")[:19]
        id_short   = (e.id or "")[:8]
        safe_source = (e.source or "unknown").replace("/", "_")
        filename = f"{ts_compact}-{safe_source}-{id_short}.md"

        # YAML frontmatter
        fm_lines = ["---"]
        fm_lines.append(f"id: {e.id}")
        fm_lines.append(f"source: {e.source}")
        if e.tags:
            fm_lines.append("tags:")
            for t in e.tags:
                fm_lines.append(f"  - {t}")
        else:
            fm_lines.append("tags: []")
        fm_lines.append(f"created: {e.timestamp}")
        fm_lines.append("---")
        fm_lines.append("")
        fm_lines.append(e.text)
        fm_lines.append("")

        (out_dir / filename).write_text(
            "\n".join(fm_lines), encoding="utf-8",
        )
        count += 1

    return count


__all__ = ["export_markdown", "export_json", "export_obsidian"]
