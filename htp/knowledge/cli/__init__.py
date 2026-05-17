"""
htp.knowledge.cli — argparse 기반 CLI dispatch (L2 sidequest session-3 신설).

Design Ref: docs/02-design/features/htp-knowledge-cli-polish.design.md §2.6
Plan SC: 8 sub-decision #7 — CLI dispatch 위치 = cli/__init__.py

기존 `python -m htp.knowledge {ingest,query,discover}` + 신규
`{list,delete,edit,tag,export,migrate}` 모두 통합 dispatch.

DAG: cli/* → loop/filters/exporters/migrate (htp.knowledge 형제만).
     금지: cli/* → htp.runtime/thalamus/memory.
"""
from __future__ import annotations

import argparse
import sys


def main(argv: "list[str] | None" = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return _dispatch(args)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="htp.knowledge",
        description="HTP Knowledge Loop CLI (L2 polish)",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # ── ingest ────────────────────────────────────────
    p_ing = sub.add_parser("ingest", help="텍스트 지식 입력")
    p_ing.add_argument("--source", required=True, help="지식 출처")
    p_ing.add_argument("text", nargs="?", default=None,
                       help="텍스트 (생략 시 stdin)")
    p_ing.add_argument("--file", help="단일 파일 경로 (전체 내용 = 1 entry)")
    p_ing.add_argument("--dir",  dest="dir_path", help="디렉토리 (각 파일 = 1 entry)")
    p_ing.add_argument("--pattern", default="*.md",
                       help="--dir 사용 시 glob 패턴 (기본 *.md)")
    p_ing.add_argument("--tag", action="append", default=[],
                       help="ingest 시 추가할 tag (반복 가능)")

    # ── query ─────────────────────────────────────────
    p_q = sub.add_parser("query", help="유사 지식 검색")
    p_q.add_argument("question", help="질의 텍스트")
    p_q.add_argument("--source", help="filter: source 일치만")
    p_q.add_argument("--since",  help="filter: 'Nd' / 'YYYY-MM' / ISO")
    p_q.add_argument("--tag",    help="filter: tag 포함")

    # ── discover ──────────────────────────────────────
    p_d = sub.add_parser("discover", help="cross-domain 발견")
    p_d.add_argument("--threshold", type=float, default=None,
                     help="유사도 임계값 (기본 0.6)")
    p_d.add_argument("--source", help="filter: source 일치 entry 만 후보")
    p_d.add_argument("--since",  help="filter: 'Nd' / 'YYYY-MM' / ISO")
    p_d.add_argument("--tag",    help="filter: tag 포함")

    # ── list ──────────────────────────────────────────
    p_l = sub.add_parser("list", help="entry 목록 (id + summary)")
    p_l.add_argument("--source", help="filter: source")
    p_l.add_argument("--since",  help="filter: 'Nd' / 'YYYY-MM' / ISO")
    p_l.add_argument("--tag",    help="filter: tag")
    p_l.add_argument("--limit",  type=int, default=20)

    # ── delete ────────────────────────────────────────
    p_del = sub.add_parser("delete", help="entry 삭제 (tombstone)")
    p_del.add_argument("--id", required=True, dest="entry_id",
                       help="삭제할 entry UUID")

    # ── edit ──────────────────────────────────────────
    p_e = sub.add_parser("edit", help="entry 본문 수정 (id 유지)")
    p_e.add_argument("--id", required=True, dest="entry_id")
    p_e.add_argument("--text", required=True, help="새 본문")

    # ── tag ───────────────────────────────────────────
    p_t = sub.add_parser("tag", help="entry 에 tag 추가")
    p_t.add_argument("--id", required=True, dest="entry_id")
    p_t.add_argument("--add", required=True,
                     help="추가할 tags (콤마 구분, 예: memory,distributed)")

    # ── export ────────────────────────────────────────
    p_x = sub.add_parser("export", help="외부 포맷 출력")
    p_x.add_argument("--format", choices=["markdown", "json", "obsidian"],
                     default="markdown")
    p_x.add_argument("--source", help="filter: source")
    p_x.add_argument("--since",  help="filter: 'Nd' / 'YYYY-MM' / ISO")
    p_x.add_argument("--tag",    help="filter: tag")
    p_x.add_argument("--dir", dest="dir_path",
                     help="obsidian 포맷 시 출력 디렉토리 (필수)")
    p_x.add_argument("--group-by", choices=["source", "flat"], default="source")

    # ── migrate ───────────────────────────────────────
    p_m = sub.add_parser("migrate", help="기존 jsonl maintenance")
    p_m.add_argument("--add-uuid", action="store_true",
                     help="기존 entry 에 UUID 영구 부여")

    return parser


def _dispatch(args) -> int:
    # lazy import — 시작 시간 단축 + DAG 검사 영향 최소
    if args.cmd == "ingest":
        from . import ingest as _ingest
        return _ingest.run(args)
    if args.cmd == "query":
        from . import query as _query
        return _query.run(args)
    if args.cmd == "discover":
        from . import discover as _disc
        return _disc.run(args)
    if args.cmd in {"list", "delete", "edit", "tag"}:
        from . import list_cmd
        return getattr(list_cmd, f"{args.cmd}_run")(args)
    if args.cmd == "export":
        from . import export as _export
        return _export.run(args)
    if args.cmd == "migrate":
        from .. import migrate as _mig
        from ..persistence import KnowledgeStore
        if args.add_uuid:
            result = _mig.migrate_add_uuid(KnowledgeStore.default().path)
            print(f"migrated {result['migrated']} entries")
            print(f"backup: {result['backup_path']}")
            return 0
        print("migrate: nothing to do (use --add-uuid)")
        return 1
    return 1


__all__ = ["main"]
