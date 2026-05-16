"""htp.knowledge CLI — argparse dispatcher.

Design Ref: docs/02-design/features/htp-thalamus-car.design.md §4.5
Plan SC: FR-05.6

사용:
    python -m htp.knowledge ingest --source <src> "<text>"
    python -m htp.knowledge query "<question>"
    python -m htp.knowledge discover
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from .encoder import TfidfJLEncoder
from .loop    import KnowledgeLoop


def _make_loop() -> KnowledgeLoop:
    """Default 인코더 + 저장소로 KnowledgeLoop 인스턴스."""
    return KnowledgeLoop(encoder=TfidfJLEncoder())


def _cmd_ingest(loop: KnowledgeLoop, source: str, text: str) -> int:
    result = loop.ingest(text, source=source)
    norm = float(np.linalg.norm(result.entry.vec))
    print(f"✓ 저장 완료  (vec norm: {norm:.2f})")
    if result.neighbors:
        print(f"◆ 유사 지식 {len(result.neighbors)}건:")
        for n in result.neighbors:
            entry = loop._cache[n.entry_id]
            marker = "←대조" if n.similarity < 0.3 else "✓"
            preview = entry.text[:60] + ("..." if len(entry.text) > 60 else "")
            print(f"  [{n.similarity:+.2f}] {preview} ({entry.source}) {marker}")
    if result.resonances:
        top = result.resonances[0]
        entry = loop._cache[top.entry_id]
        print(f"⚡ 공명: '{source}' ↔ '{entry.source}' 유사도 {top.similarity:.2f}")
    return 0


def _cmd_query(loop: KnowledgeLoop, question: str) -> int:
    result = loop.query(question)
    if not result.relevant:
        print("저장된 지식이 없습니다. 먼저 `ingest` 하세요.")
        return 0
    print(f"◆ '{question}' 관련 지식 {len(result.relevant)}건 "
          f"({result.cluster_count}개 클러스터):")
    for n in result.relevant:
        entry = loop._cache[n.entry_id]
        preview = entry.text[:80] + ("..." if len(entry.text) > 80 else "")
        print(f"  [{n.similarity:+.2f}] {preview} ({entry.source})")
    return 0


def _cmd_discover(loop: KnowledgeLoop) -> int:
    discoveries = loop.discover()
    if not discoveries:
        print("Cross-domain 발견 없음. 다른 source 의 지식을 더 추가하세요.")
        print(f"(현재 threshold={loop.discover_threshold:.2f}. "
              f"낮추려면 `--threshold` 옵션 또는 KnowledgeLoop 생성자에서 조정)")
        return 0
    print(f"⚡ Top 발견 {len(discoveries)}건:")
    for d in discoveries:
        a = loop._cache[d.entry_a_id]
        b = loop._cache[d.entry_b_id]
        print(f"  [{d.similarity:+.2f}] {d.source_a}:\"{a.text[:50]}...\"")
        print(f"             ↔ {d.source_b}:\"{b.text[:50]}...\"")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="htp.knowledge",
        description="HTP Knowledge Loop MVP (Stage 0.5)",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ingest = sub.add_parser("ingest", help="텍스트 지식 입력")
    p_ingest.add_argument("--source", required=True, help="지식 출처 (예: '뇌과학')")
    p_ingest.add_argument("text", help="저장할 텍스트")

    p_query = sub.add_parser("query", help="유사 지식 검색")
    p_query.add_argument("question", help="질의 텍스트")

    p_disc = sub.add_parser("discover", help="cross-domain 발견")
    p_disc.add_argument("--threshold", type=float, default=None,
                        help="유사도 임계값 (기본 0.6)")

    args = parser.parse_args(argv)

    loop = _make_loop()
    if args.cmd == "discover" and args.threshold is not None:
        loop.discover_threshold = args.threshold

    if args.cmd == "ingest":
        return _cmd_ingest(loop, args.source, args.text)
    if args.cmd == "query":
        return _cmd_query(loop, args.question)
    if args.cmd == "discover":
        return _cmd_discover(loop)
    return 1


if __name__ == "__main__":
    sys.exit(main())
