"""htp.knowledge CLI 진입점 (L2 sidequest session-3 — cli/ 패키지로 위임).

Design Ref: docs/02-design/features/htp-knowledge-cli-polish.design.md §2.6
이전 (sub-1): argparse 가 이 파일에 직접 있었음.
현재: cli/__init__.py 의 main() 호출만.

사용:
    python -m htp.knowledge {ingest,query,discover,list,delete,edit,tag,export,migrate}
"""
from __future__ import annotations

import sys

from .cli import main


if __name__ == "__main__":
    sys.exit(main())
