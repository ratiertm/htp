"""
L2 sidequest session-3 — exporters + cli/ dispatch tests.

Design Ref: docs/02-design/features/htp-knowledge-cli-polish.design.md §2.5-2.6
Plan SC: T7, T8, T9

신규 테스트:
- test_export_markdown_grouped      — source 별 섹션 + timestamp 정렬
- test_export_json_round_trip       — vec 포함, JSON round-trip
- test_export_obsidian_files        — 파일 단위 split + frontmatter
- test_cli_dispatch_smoke           — argparse 통합 smoke (4 subcommand)
"""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib  import Path

import numpy as np
import pytest

from htp.knowledge import (
    KnowledgeEntry, KnowledgeStore,
    export_markdown, export_json, export_obsidian,
)
from htp.knowledge.cli import main


def _make_entries() -> list[KnowledgeEntry]:
    return [
        KnowledgeEntry(
            text="brain 패턴 인출 분산 표상",
            vec=np.array([1.0, 0.0, 0.0, 0.0]),
            source="brain",
            timestamp="2026-05-10T10:00:00+00:00",
            tags=["memory", "distributed"],
        ),
        KnowledgeEntry(
            text="ai 어텐션 매커니즘",
            vec=np.array([0.0, 1.0, 0.0, 0.0]),
            source="ai",
            timestamp="2026-05-15T12:00:00+00:00",
            tags=["attention"],
        ),
        KnowledgeEntry(
            text="brain 시상 라우팅",
            vec=np.array([0.0, 0.0, 1.0, 0.0]),
            source="brain",
            timestamp="2026-05-12T15:00:00+00:00",
            tags=[],
        ),
    ]


# ══════════════════════════════════════════════════════════
# T7: Markdown — source 별 섹션 + timestamp 정렬
# ══════════════════════════════════════════════════════════

def test_export_markdown_grouped():
    md = export_markdown(_make_entries(), group_by="source")

    # source 별 H2 섹션 존재
    assert "## brain" in md
    assert "## ai" in md
    # 알파벳 정렬 — "ai" 가 "brain" 보다 앞
    assert md.index("## ai") < md.index("## brain")

    # brain 섹션 안에 timestamp 정렬 (10일 → 12일 순)
    brain_section = md[md.index("## brain"):]
    assert brain_section.index("2026-05-10") < brain_section.index("2026-05-12")

    # tags 표시
    assert "memory" in md
    assert "distributed" in md
    assert "attention" in md

    # flat 모드
    md_flat = export_markdown(_make_entries(), group_by="flat")
    # flat 은 source 별 섹션 없음 (전체 timestamp 정렬)
    assert "## brain — 2026-05-10" in md_flat or "brain — 2026-05-10" in md_flat


def test_export_markdown_empty():
    """빈 entries 안전 처리."""
    md = export_markdown([])
    assert "no entries" in md


# ══════════════════════════════════════════════════════════
# T8: JSON — round-trip
# ══════════════════════════════════════════════════════════

def test_export_json_round_trip():
    entries = _make_entries()
    js = export_json(entries, include_vec=True)
    parsed = json.loads(js)

    assert len(parsed) == 3
    by_id = {r["id"]: r for r in parsed}

    # id / text / source / tags / vec 모두 포함
    for e in entries:
        rec = by_id[e.id]
        assert rec["text"] == e.text
        assert rec["source"] == e.source
        assert rec["tags"] == e.tags
        assert "vec" in rec
        # vec round-trip
        recovered = np.array(rec["vec"])
        np.testing.assert_allclose(recovered, e.vec, rtol=1e-10)

    # include_vec=False
    js_slim = export_json(entries, include_vec=False)
    parsed_slim = json.loads(js_slim)
    assert all("vec" not in r for r in parsed_slim)


# ══════════════════════════════════════════════════════════
# Obsidian — 파일 단위 split + frontmatter
# ══════════════════════════════════════════════════════════

def test_export_obsidian_files(tmp_path):
    entries = _make_entries()
    n = export_obsidian(entries, tmp_path)
    assert n == 3

    files = sorted(tmp_path.glob("*.md"))
    assert len(files) == 3

    # 모든 파일에 frontmatter 존재
    for f in files:
        content = f.read_text(encoding="utf-8")
        assert content.startswith("---")
        assert "id:" in content
        assert "source:" in content
        assert "tags:" in content
        assert "created:" in content
        # frontmatter 끝 + 본문
        assert content.count("---") >= 2

    # tags 가 있는 entry (brain, 2026-05-10) 의 파일 찾기
    brain_files = [f for f in files
                   if "brain" in f.name and "2026-05-10" in f.name]
    assert len(brain_files) == 1, f"expected 1 brain file, got {brain_files}"
    brain_content = brain_files[0].read_text(encoding="utf-8")
    assert "- memory" in brain_content
    assert "- distributed" in brain_content


# ══════════════════════════════════════════════════════════
# T9: CLI dispatch smoke
# ══════════════════════════════════════════════════════════

def test_cli_dispatch_smoke(tmp_path, monkeypatch, capsys):
    """argparse + dispatch 통합 smoke — 4 subcommand 호출 가능."""
    # 격리된 .htp 디렉토리 (현재 작업 디렉토리 변경)
    monkeypatch.chdir(tmp_path)

    # 1) ingest
    rc = main(["ingest", "--source", "brain", "content addressable memory pattern recall"])
    assert rc == 0

    # 2) ingest 두 번째 — 같은 corpus 에 추가
    rc = main(["ingest", "--source", "ai", "Hopfield network pattern recall"])
    assert rc == 0

    # 3) list — 2 entries
    rc = main(["list"])
    assert rc == 0
    captured = capsys.readouterr()
    assert "brain" in captured.out
    assert "ai" in captured.out

    # 4) query — 검색
    rc = main(["query", "pattern recall"])
    assert rc == 0

    # 5) export markdown to stdout
    capsys.readouterr()  # clear buffer
    rc = main(["export", "--format", "markdown"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "## brain" in out or "## ai" in out

    # 6) export json
    capsys.readouterr()
    rc = main(["export", "--format", "json"])
    assert rc == 0
    out_json = capsys.readouterr().out.strip()
    parsed = json.loads(out_json)
    assert len(parsed) == 2

    # 7) Help 표시 (subcommand 없이 호출 시 SystemExit)
    with pytest.raises(SystemExit):
        main([])
