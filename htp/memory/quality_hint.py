"""
quality_hint — LLM interpretation 의 통찰 깊이 heuristic.

Design Ref: docs/02-design/features/htp-conflict-memory.design.md §2 M3
양적 검증 출처: docs/03-analysis/conflict_quant_summary.md §3 + §5

키워드는 양적 검증 50건의 고품질 응답에서 빈출, 저품질에서 적게 등장한 단어들.
완벽한 metric 은 아니지만 recall best-match 정렬 기준으로 충분.

DAG: 외부 의존 없음 (string 만 다룸).
"""
from __future__ import annotations


# 양적 검증 50건 분석 기반
# 고품질 (mechanism mapping / 차원 분해) 에서 빈출:
QUALITY_KEYWORDS: "list[str]" = [
    # 영어
    "mechanism", "axis", "dimension", "scope", "layer",
    "complementarity", "framing", "trade-off", "two-axis", "tradeoff",
    "asymmetr", "parallel", "decomp", "analog",
    # 한국어
    "메커니즘", "차원", "관점", "보완", "축", "상보",
    "trade", "vs", "대비", "분해",
]


def quality_hint(interpretation: str) -> float:
    """Structural keyword count 기반 통찰 품질 추정.

    Returns
    -------
    float in [0.0, 1.0]
      0.0 — 빈 문자열 또는 키워드 0개
      ~0.33 — 키워드 1개 (낮은 신뢰)
      ~0.67 — 키워드 2개 (중간)
      1.0 — 키워드 3개 이상 (높음)

    양적 검증 결과 ()
      - 고품질 7건 평균 ~2-4개 키워드 → 0.67-1.0
      - 저품질 8건 평균 0-1개 키워드 → 0.0-0.33
    """
    if not interpretation:
        return 0.0
    text = interpretation.lower()
    hits = sum(1 for kw in QUALITY_KEYWORDS if kw.lower() in text)
    return min(1.0, hits / 3.0)


__all__ = ["quality_hint", "QUALITY_KEYWORDS"]
