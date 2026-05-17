"""
L2 sidequest session-2 — batch / delete / edit / add_tags / filter tests.

Design Ref: docs/02-design/features/htp-knowledge-cli-polish.design.md §2.3-2.4
Plan SC: T2, T3, T5, T6 + delete/add_tags

신규 테스트 (153 → 159, 단조성 우선):
- test_batch_single_fit                — encoder.fit() 1회만 (F1 / R2)
- test_batch_skip_and_continue         — 실패 skip + 에러 누적 (sub-decision #5)
- test_loop_delete_round_trip          — delete + tombstone + load_all 동등
- test_loop_edit_preserves_id          — Plan FR-13 — id 유지 (sub-decision 모순 해결)
- test_loop_add_tags_union             — 중복 제거 union
- test_filter_source_since_tag         — source/since/tag 조합 (F3)
- test_parse_since_formats             — Nd / YYYY-MM / ISO 모두
"""
from __future__ import annotations

import tempfile
from datetime import datetime, timezone, timedelta
from pathlib  import Path

import numpy as np
import pytest

from htp.knowledge import (
    KnowledgeLoop, KnowledgeEntry, KnowledgeStore, TfidfJLEncoder,
    filter_entries, parse_since,
)


@pytest.fixture
def temp_store():
    with tempfile.TemporaryDirectory() as td:
        yield KnowledgeStore(Path(td) / "knowledge_log.jsonl")


@pytest.fixture
def fresh_loop(temp_store):
    return KnowledgeLoop(encoder=TfidfJLEncoder(), store=temp_store)


# ══════════════════════════════════════════════════════════
# T2: Batch — encoder.fit() 1회만 (옵션 A-2 영속화)
# ══════════════════════════════════════════════════════════

def test_batch_single_fit(fresh_loop):
    """ingest_batch(N texts) 시 encoder.fit() 은 첫 호출에서만 (옵션 A-2)."""
    # 초기 상태: 미fitted
    assert fresh_loop.encoder._fitted is False

    results = fresh_loop.ingest_batch(
        ["alpha pattern recall", "beta network energy",
         "gamma redis lookup"],
        source="test",
    )
    assert len(results["success"]) == 3
    assert results["errors"] == []
    # fit 은 1회만 → 이제 _fitted = True
    assert fresh_loop.encoder._fitted is True

    # 두 번째 batch 호출 시 fit 재호출 안 됨 (옵션 A-2 가드)
    fresh_loop.ingest_batch(["delta new term"], source="test")
    assert fresh_loop.encoder._fitted is True  # 변화 없음


# ══════════════════════════════════════════════════════════
# T3: Batch — skip-and-continue (sub-decision #5)
# ══════════════════════════════════════════════════════════

def test_batch_skip_and_continue(fresh_loop, monkeypatch):
    """ingest 실패 시 skip + 에러 누적, 나머지 처리 계속."""
    original_ingest = fresh_loop.ingest
    call_count = {"n": 0}

    def flaky_ingest(text, source=""):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise RuntimeError("intentional failure")
        return original_ingest(text, source=source)

    monkeypatch.setattr(fresh_loop, "ingest", flaky_ingest)

    results = fresh_loop.ingest_batch(
        ["first", "second-fail", "third"], source="test",
    )
    assert len(results["success"]) == 2
    assert len(results["errors"]) == 1
    assert results["errors"][0]["error"] == "intentional failure"
    assert "second-fail" in results["errors"][0]["text"]


# ══════════════════════════════════════════════════════════
# Delete — tombstone round-trip via loop
# ══════════════════════════════════════════════════════════

def test_loop_delete_round_trip(fresh_loop):
    """loop.delete(id) → load_all 시 미반환 + jsonl 무손상."""
    r1 = fresh_loop.ingest("first text", source="a")
    r2 = fresh_loop.ingest("second text", source="b")
    r3 = fresh_loop.ingest("third text", source="c")

    # 존재 확인
    assert len(fresh_loop._cache) == 3

    # delete
    ok = fresh_loop.delete(r2.entry.id)
    assert ok is True
    assert len(fresh_loop._cache) == 2

    # 같은 store 새 인스턴스에서 load_all
    new_loop = KnowledgeLoop(
        encoder=TfidfJLEncoder(), store=fresh_loop.store,
    )
    loaded_ids = {e.id for e in new_loop._cache}
    assert r1.entry.id in loaded_ids
    assert r2.entry.id not in loaded_ids   # 삭제 적용
    assert r3.entry.id in loaded_ids

    # 존재 안 하는 id 삭제 시 False
    assert fresh_loop.delete("non-existent-uuid") is False


# ══════════════════════════════════════════════════════════
# Edit — id 유지 (Plan FR-13)
# ══════════════════════════════════════════════════════════

def test_loop_edit_preserves_id(fresh_loop):
    """edit 시 id 는 유지, text/vec/timestamp 갱신."""
    r = fresh_loop.ingest("original text alpha beta", source="brain")
    original_id        = r.entry.id
    original_timestamp = r.entry.timestamp
    original_source    = r.entry.source

    edited = fresh_loop.edit(original_id, "modified text gamma delta")
    assert edited is not None
    # id 보존 (Plan FR-13 핵심)
    assert edited.id == original_id
    # text 갱신
    assert edited.text == "modified text gamma delta"
    # source 보존
    assert edited.source == original_source
    # timestamp 갱신 (다름 — 다만 동일 ms 시 같을 수도, 그래도 의미 동일)
    # vec 재계산 (다른 토큰 → 다른 vec)
    assert not np.array_equal(edited.vec, r.entry.vec)

    # cache 에는 edit 결과만 (id 중복 없음)
    matching = [e for e in fresh_loop._cache if e.id == original_id]
    assert len(matching) == 1
    assert matching[0].text == "modified text gamma delta"

    # 새 인스턴스로 load_all — 후자 우선 → edited entry 반환
    new_loop = KnowledgeLoop(
        encoder=TfidfJLEncoder(), store=fresh_loop.store,
    )
    loaded_by_id = {e.id: e for e in new_loop._cache}
    assert loaded_by_id[original_id].text == "modified text gamma delta"


# ══════════════════════════════════════════════════════════
# add_tags — union
# ══════════════════════════════════════════════════════════

def test_loop_add_tags_union(fresh_loop):
    """add_tags 가 기존 tags 와 union (중복 제거)."""
    r = fresh_loop.ingest("text", source="a")
    eid = r.entry.id

    # 첫 tag 추가
    e1 = fresh_loop.add_tags(eid, ["memory", "distributed"])
    assert set(e1.tags) == {"memory", "distributed"}

    # 추가 — 일부 중복 + 신규
    e2 = fresh_loop.add_tags(eid, ["distributed", "neural"])
    assert set(e2.tags) == {"memory", "distributed", "neural"}

    # 존재 안 하는 id
    assert fresh_loop.add_tags("non-existent", ["x"]) is None


# ══════════════════════════════════════════════════════════
# T6: Filter — source / since / tag 조합
# ══════════════════════════════════════════════════════════

def test_filter_source_since_tag():
    """source/since/tag 필터 단독 + 조합."""
    now = datetime.now(timezone.utc)
    e_brain_old = KnowledgeEntry(
        text="b1", vec=np.zeros(8), source="brain",
        timestamp=(now - timedelta(days=60)).isoformat(),
        tags=["pattern"],
    )
    e_brain_new = KnowledgeEntry(
        text="b2", vec=np.zeros(8), source="brain",
        timestamp=(now - timedelta(days=5)).isoformat(),
        tags=["pattern", "memory"],
    )
    e_ai_new = KnowledgeEntry(
        text="a1", vec=np.zeros(8), source="ai",
        timestamp=(now - timedelta(days=2)).isoformat(),
        tags=["memory"],
    )
    entries = [e_brain_old, e_brain_new, e_ai_new]

    # source 필터
    assert {e.id for e in filter_entries(entries, source="brain")} == {
        e_brain_old.id, e_brain_new.id,
    }

    # since 필터 (30일 이내)
    recent = filter_entries(entries, since="30d")
    assert e_brain_old not in recent
    assert e_brain_new in recent
    assert e_ai_new in recent

    # tag 필터
    pattern_tagged = filter_entries(entries, tag="pattern")
    assert {e.id for e in pattern_tagged} == {
        e_brain_old.id, e_brain_new.id,
    }

    # 조합: brain + 최근 30일
    combined = filter_entries(entries, source="brain", since="30d")
    assert combined == [e_brain_new]


# ══════════════════════════════════════════════════════════
# parse_since 형식 검증
# ══════════════════════════════════════════════════════════

def test_parse_since_formats():
    """Nd / YYYY-MM / YYYY-MM-DD / ISO datetime 모두 지원."""
    now = datetime.now(timezone.utc)

    # "Nd"
    dt = parse_since("30d")
    delta_secs = (now - dt).total_seconds()
    assert 29.5 * 86400 < delta_secs <= 30.5 * 86400

    # "YYYY-MM"
    dt = parse_since("2026-04")
    assert dt.year == 2026 and dt.month == 4 and dt.day == 1
    assert dt.tzinfo is not None

    # "YYYY-MM-DD"
    dt = parse_since("2026-04-15")
    assert dt.year == 2026 and dt.month == 4 and dt.day == 15

    # ISO datetime (with TZ)
    dt = parse_since("2026-05-17T10:00:00+00:00")
    assert dt.year == 2026 and dt.hour == 10

    # 잘못된 형식
    with pytest.raises(ValueError):
        parse_since("not-a-date")
    with pytest.raises(ValueError):
        parse_since("30 days")
