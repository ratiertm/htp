"""htp-conflict-interpretation — KnowledgeLoop ↔ LLMRegion 연결 검증.

Design Ref: docs/02-design/features/htp-conflict-interpretation.design.md
Plan SC: SC1-SC6
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from htp.knowledge import KnowledgeLoop, KnowledgeStore
from htp.knowledge.embedding import EmbeddingBridge
from htp.knowledge.encoder   import TfidfJLEncoder
from htp.knowledge.types     import KnowledgeEntry
from htp.knowledge.conflict_prompt import build_conflict_prompt, SYSTEM_PROMPT
from htp.knowledge.migrate   import migrate_add_interpretation
from htp.llm.llm_region      import LLMRegion


# ── Architecture B: Auto Mock default ─────────────────

def test_default_creates_mock_interpreter():
    """SC3: conflict_interpreter=None 이면 자동 MockLLMRegion 생성."""
    with tempfile.TemporaryDirectory() as td:
        store = KnowledgeStore(Path(td) / "log.jsonl")
        loop  = KnowledgeLoop(encoder=TfidfJLEncoder(dim=32), store=store)
        assert loop.conflict_interpreter is not None
        assert isinstance(loop.conflict_interpreter, LLMRegion)
        assert loop.conflict_interpreter.use_mock is True


def test_user_provided_interpreter_used():
    """사용자가 명시 LLMRegion 넘기면 그것 사용 (Mock 자동 생성 안 함)."""
    with tempfile.TemporaryDirectory() as td:
        store = KnowledgeStore(Path(td) / "log.jsonl")
        custom = LLMRegion(region_name="custom_interp",
                           specialty="emotion", use_mock=True)
        loop  = KnowledgeLoop(
            encoder=TfidfJLEncoder(dim=32),
            store=store,
            conflict_interpreter=custom,
        )
        assert loop.conflict_interpreter is custom
        assert loop.conflict_interpreter.region_name == "custom_interp"


# ── escalate 분기 ─────────────────────────────────────

def _make_embed_loop(tmp: Path, **kwargs) -> KnowledgeLoop:
    store = KnowledgeStore(tmp / "log.jsonl")
    return KnowledgeLoop(encoder=EmbeddingBridge(), store=store, **kwargs)


@pytest.mark.skipif(
    __import__("os").environ.get("HTP_SKIP_HF_DOWNLOAD", "").lower() in ("1", "true"),
    reason="HF download skipped",
)
def test_escalate_true_triggers_interpretation():
    """SC2: escalate=True 시 entry.interpretation 채움."""
    with tempfile.TemporaryDirectory() as td:
        loop = _make_embed_loop(Path(td))

        # 한 도메인 누적 후 이질 ingest → escalate=True 기대
        for t in ["해마 CA3 패턴 완성 시냅스 recurrent",
                  "시냅스 가소성 헵 학습",
                  "감마 진동 뇌파 의식"]:
            loop.ingest(t, source="뇌과학")

        result = loop.ingest(
            "Redis LRU 캐시 eviction 전략 nginx 로드밸런서",
            source="인프라",
        )
        # escalate=True 인 경우 interpretation 채워져야 함
        if result.coherence_info and result.coherence_info["escalate"]:
            assert result.entry.interpretation is not None, (
                f"escalate=True 인데 interpretation 가 None"
            )
            assert isinstance(result.entry.interpretation, str)
            assert len(result.entry.interpretation) > 0


def test_escalate_false_skips_interpretation():
    """일관 / 첫 ingest → interpretation=None."""
    with tempfile.TemporaryDirectory() as td:
        store = KnowledgeStore(Path(td) / "log.jsonl")
        loop  = KnowledgeLoop(encoder=TfidfJLEncoder(dim=32), store=store)
        # 빈 cache 에 첫 ingest → neighbors < 2 → coherence_info=None → skip
        result = loop.ingest("최초 지식", source="test")
        assert result.entry.interpretation is None


def test_max_interpretations_cap():
    """SC5: max_interpretations 도달 후 더 이상 호출 안 함."""
    with tempfile.TemporaryDirectory() as td:
        store = KnowledgeStore(Path(td) / "log.jsonl")
        loop  = KnowledgeLoop(
            encoder=TfidfJLEncoder(dim=32),
            store=store,
            max_interpretations=2,
        )
        loop._interpretations_count = 2   # cap 도달 시뮬레이션
        assert loop._can_interpret() is False


def test_can_interpret_blocked_by_cost_router():
    """CostRouter.should_block=True 면 호출 안 함."""
    with tempfile.TemporaryDirectory() as td:
        store = KnowledgeStore(Path(td) / "log.jsonl")
        loop  = KnowledgeLoop(encoder=TfidfJLEncoder(dim=32), store=store)
        # pressure 인위적으로 끌어올림
        loop.conflict_interpreter.router._ema_cost = (
            loop.conflict_interpreter.router.budget * 5.0
        )
        assert loop.conflict_interpreter.router.should_block()
        assert loop._can_interpret() is False


# ── JSONL round-trip ─────────────────────────────────

def test_jsonl_round_trip_with_interpretation():
    """SC4: interpretation 필드가 jsonl save → load 시 보존."""
    with tempfile.TemporaryDirectory() as td:
        log = Path(td) / "log.jsonl"
        store = KnowledgeStore(log)

        # 수동으로 interpretation 채운 entry 작성
        import numpy as np
        from datetime import datetime, timezone
        entry = KnowledgeEntry(
            text="test",
            vec=np.array([0.1, 0.2], dtype=np.float64),
            source="src",
            timestamp=datetime.now(timezone.utc).isoformat(),
            interpretation="이것은 테스트 해석",
        )
        store.append(entry)

        store2 = KnowledgeStore(log)
        loaded = store2.load_all()
        assert len(loaded) == 1
        assert loaded[0].interpretation == "이것은 테스트 해석"


def test_jsonl_legacy_entries_load_with_none():
    """기존 jsonl (interpretation 필드 없음) 도 로드 가능 — None default."""
    with tempfile.TemporaryDirectory() as td:
        log = Path(td) / "log.jsonl"
        # legacy 라인 직접 작성 — interpretation 필드 누락
        legacy_rec = {
            "text": "legacy",
            "vec":  [0.1, 0.2],
            "source": "src",
            "timestamp": "2026-05-19T00:00:00+00:00",
            "neighbors": [],
            "conflict_count": 0,
            "id": "legacy-uuid-1",
            "tags": [],
            # interpretation 필드 없음
        }
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(json.dumps(legacy_rec) + "\n", encoding="utf-8")

        store = KnowledgeStore(log)
        entries = store.load_all()
        assert len(entries) == 1
        assert entries[0].interpretation is None


# ── Prompt template ──────────────────────────────────

def test_build_conflict_prompt_includes_inputs():
    """prompt 가 new + existing + 메트릭을 모두 포함."""
    prompt = build_conflict_prompt(
        new_text="새 지식",
        new_source="뇌과학",
        existing=[("기존 지식 1", "AI"), ("기존 지식 2", "인프라")],
        coherence=0.85,
        conflict=0.15,
    )
    assert "새 지식" in prompt
    assert "뇌과학" in prompt
    assert "기존 지식 1" in prompt
    assert "AI" in prompt
    assert "0.85" in prompt
    assert "0.15" in prompt


def test_build_conflict_prompt_empty_existing():
    """existing=[] 도 graceful."""
    prompt = build_conflict_prompt(
        new_text="x", new_source="s",
        existing=[], coherence=0.5, conflict=0.5,
    )
    assert "(none)" in prompt


def test_system_prompt_contains_json_keyword():
    """LLM 이 JSON 반환하도록 system prompt 가 지시."""
    assert "JSON" in SYSTEM_PROMPT


# ── Migration ────────────────────────────────────────

def test_migrate_add_interpretation_field():
    """jsonl 에 interpretation=null 명시 추가 후 보존."""
    with tempfile.TemporaryDirectory() as td:
        log = Path(td) / "log.jsonl"
        legacy = {
            "text": "x", "vec": [0.1], "source": "s",
            "timestamp": "t", "neighbors": [], "conflict_count": 0,
            "id": "uuid-1", "tags": [],
        }
        log.write_text(json.dumps(legacy) + "\n", encoding="utf-8")

        result = migrate_add_interpretation(log)
        assert result["migrated"] == 1
        assert result["had_interpretation"] == 0
        assert Path(result["backup_path"]).exists()

        # 마이그레이션 후 명시적으로 interpretation 필드 있음
        new_content = log.read_text()
        assert "interpretation" in new_content
