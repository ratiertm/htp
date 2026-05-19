"""ingest subcommand — 단일 / batch / stdin (L2 sidequest session-3)."""
from __future__ import annotations

import sys
from pathlib import Path

from ._common import make_loop
from ..loop    import KnowledgeLoop


def _make_loop(args=None) -> KnowledgeLoop:
    enc_type = getattr(args, "encoder", "tfidf") if args else "tfidf"
    return make_loop(enc_type)


def run(args) -> int:
    # 경로별 분기
    if args.file:
        return _ingest_file(args)
    if args.dir_path:
        return _ingest_dir(args)
    return _ingest_text_or_stdin(args)


# ── 단일 텍스트 / stdin (F1 + F2) ─────────────────
def _ingest_text_or_stdin(args) -> int:
    text = args.text
    if text is None:
        # stdin pipe (F2)
        if sys.stdin.isatty():
            print("ingest: no text and stdin is a TTY", file=sys.stderr)
            return 2
        text = sys.stdin.read().strip()
        if not text:
            print("ingest: stdin is empty", file=sys.stderr)
            return 2
    return _do_ingest_single(args, text)


def _do_ingest_single(args, text: str) -> int:
    loop = _make_loop(args)
    result = loop.ingest(text, source=args.source)
    if args.tag:
        loop.add_tags(result.entry.id, args.tag)
    print(f"✓ saved (id={result.entry.id[:8]}, source={args.source})")

    # Bridge §3 (S2): CoherenceGate 정합성 신호 표시.
    ci = result.coherence_info
    if ci is not None:
        if ci["escalate"]:
            print(f"  ⚠ 충돌 감지 (coherence={ci['coherence']:.2f}, "
                  f"conflict={ci['conflict']:.2f})")
            print(f"     → 기존 지식과 모순될 수 있음")
            # htp-conflict-memory: recall 먼저 노출 (이전 비슷한 충돌)
            rh = result.recall_hint
            if rh:
                print(f"  📚 이전 유사 충돌 "
                      f"(mismatch={rh['mismatch']:.2f}, quality={rh['quality']:.2f}):")
                print(f"     trigger: {rh['prev_trigger']}")
                print(f"     해석: {rh['prev_interpretation'][:180]}"
                      f"{'...' if len(rh['prev_interpretation']) > 180 else ''}")
            # htp-conflict-interpretation §1: 새 LLM 해석
            if result.entry.interpretation:
                preview = result.entry.interpretation[:180]
                suffix  = "..." if len(result.entry.interpretation) > 180 else ""
                print(f"  💡 새 해석: {preview}{suffix}")
        else:
            print(f"  ✓ 정합성 양호 (coherence={ci['coherence']:.2f})")
    return 0


# ── 단일 파일 (F1) ────────────────────────────────
def _ingest_file(args) -> int:
    p = Path(args.file)
    if not p.exists():
        print(f"ingest: file not found: {p}", file=sys.stderr)
        return 2
    text = p.read_text(encoding="utf-8").strip()
    if not text:
        print(f"ingest: file is empty: {p}", file=sys.stderr)
        return 2
    return _do_ingest_single(args, text)


# ── 디렉토리 batch (F1) ───────────────────────────
def _ingest_dir(args) -> int:
    d = Path(args.dir_path)
    if not d.is_dir():
        print(f"ingest: not a directory: {d}", file=sys.stderr)
        return 2
    files = sorted(d.glob(args.pattern))
    if not files:
        print(f"ingest: no files match {args.pattern!r} in {d}",
              file=sys.stderr)
        return 1

    loop = _make_loop(args)
    texts: list[str] = []
    valid_files: list[Path] = []
    for f in files:
        try:
            t = f.read_text(encoding="utf-8").strip()
            if t:
                texts.append(t)
                valid_files.append(f)
        except Exception as e:
            print(f"  [skip] {f.name}: {e}", file=sys.stderr)

    if not texts:
        print("ingest: no readable files")
        return 1

    print(f"ingest: {len(texts)} files → batch")
    results = loop.ingest_batch(texts, source=args.source)

    for i, (f, r) in enumerate(zip(valid_files, results["success"]), 1):
        print(f"  [{i}/{len(valid_files)}] {f.name} ✓ (id={r.entry.id[:8]})")
        if args.tag:
            loop.add_tags(r.entry.id, args.tag)

    if results["errors"]:
        print(f"\n{len(results['errors'])} errors:")
        for err in results["errors"]:
            print(f"  - {err['text']}: {err['error']}")

    print(f"\n✓ {len(results['success'])} saved, "
          f"{len(results['errors'])} skipped")
    return 0
