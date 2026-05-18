"""
Conflict interpretation prompt 구성.

Design Ref: docs/02-design/features/htp-conflict-interpretation.design.md §2 M2
Plan SC: htp-conflict-interpretation.plan §2 Stage 2

KnowledgeLoop 가 escalate=True 시 호출. LLMRegion 의 system prompt + per-call
user prompt 두 부분으로 분리.

DAG: htp/knowledge/conflict_prompt.py — 외부 의존 없음 (string 만 다룸).
"""
from __future__ import annotations


# LLMRegion 의 system prompt — JSON 반환 형식 강제.
SYSTEM_PROMPT = (
    "You are an analyst integrating cross-domain knowledge. "
    "When given a new statement that conflicts with existing knowledge, "
    "identify the precise nature of the conflict and propose a hypothesis "
    "that integrates both perspectives. "
    "Return JSON with keys 'interpretation' (1-2 sentences explaining the "
    "conflict and possible integration) and 'hypothesis' (one synthesis "
    "idea, may be empty if no synthesis is possible)."
)


def build_conflict_prompt(
    new_text:   str,
    new_source: str,
    existing:   "list[tuple[str, str]]",
    coherence:  float,
    conflict:   float,
) -> str:
    """Conflict interpretation prompt (user side) 생성.

    Parameters
    ----------
    new_text, new_source : 새로 ingest 한 entry 의 본문/출처.
    existing             : top-3 이웃 (text, source) 리스트.
    coherence, conflict  : PairwiseCoherenceGate 결과의 두 메트릭.

    Returns
    -------
    str : LLMRegion.run() 에 전달할 prompt 문자열.
    """
    lines = [
        f"Conflict detected (coherence={coherence:.2f}, conflict={conflict:.2f}).",
        "",
        f"New statement ({new_source}):",
        f"  {new_text}",
        "",
        "Existing related knowledge:",
    ]
    if not existing:
        lines.append("  (none)")
    else:
        for i, (text, source) in enumerate(existing, 1):
            lines.append(f"  {i}. [{source}] {text}")
    lines.append("")
    lines.append(
        "Explain the conflict precisely and propose an integration "
        "hypothesis. Return JSON."
    )
    return "\n".join(lines)


__all__ = ["SYSTEM_PROMPT", "build_conflict_prompt"]
