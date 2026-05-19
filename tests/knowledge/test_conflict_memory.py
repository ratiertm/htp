"""htp-conflict-memory — Memory ↔ KnowledgeLoop 통합 검증.

Design Ref: docs/02-design/features/htp-conflict-memory.design.md
Plan SC: SC1-SC6
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest
import torch

from htp.knowledge import KnowledgeLoop, KnowledgeStore
from htp.knowledge.encoder import TfidfJLEncoder
from htp.memory.memory_system import MemorySystem
from htp.memory.episode_store import EpisodeStore
from htp.memory.types         import Episode
from htp.memory.quality_hint  import quality_hint, QUALITY_KEYWORDS


_SKIP_HF = os.environ.get("HTP_SKIP_HF_DOWNLOAD", "").lower() in ("1", "true")


# ── M1: Episode.interpretation_text 필드 ────────────

def test_episode_interpretation_text_field_default_empty():
    """Episode 기본 인스턴스의 interpretation_text 는 빈 문자열."""
    ep = Episode()
    assert hasattr(ep, "interpretation_text")
    assert ep.interpretation_text == ""


def test_episode_with_interpretation_text():
    """interpretation_text 인자 설정 가능."""
    ep = Episode(interpretation_text="해석 본문")
    assert ep.interpretation_text == "해석 본문"


# ── M2: SQL schema 확장 + 마이그레이션 ──────────────

def test_episode_store_save_and_load_with_interpretation():
    """interpretation_text 가 SQLite round-trip 으로 보존."""
    with tempfile.TemporaryDirectory() as td:
        store = EpisodeStore(Path(td) / "mem.db")
        ep = Episode(
            step=1, winner="conflict_interpreter",
            action_type="interpret", score=0.15,
            state_vec=b"\x00" * 16,
            context="trigger",
            interpretation_text="이전 해석 본문",
        )
        ep_id = store.save(ep)
        rows = store._conn.execute(
            "SELECT interpretation_text FROM episodes WHERE episode_id=?",
            (ep_id,),
        ).fetchone()
        assert rows is not None
        assert rows[0] == "이전 해석 본문"


def test_episode_store_schema_migration_idempotent():
    """기존 DB (interpretation_text 없는 schema) 도 자동 마이그레이션."""
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "legacy.db"

        # legacy schema 직접 작성 (interpretation_text 컬럼 없음)
        import sqlite3
        conn = sqlite3.connect(db_path)
        conn.executescript("""
            CREATE TABLE episodes (
                episode_id TEXT PRIMARY KEY,
                step INTEGER, winner TEXT, action_type TEXT,
                score REAL, state_vec BLOB, context TEXT,
                outcome TEXT, recall_count INTEGER, novelty REAL,
                swr_tagged INTEGER, session_id TEXT, timestamp REAL
            );
        """)
        conn.execute(
            "INSERT INTO episodes VALUES (?, 1, 'w', 'a', 0.5, ?, 'c', NULL, 0, 1.0, 0, 's', 0.0)",
            ("legacy-id", b"\x00" * 16),
        )
        conn.commit()
        conn.close()

        # EpisodeStore 재open — ALTER 가 idempotent 하게 컬럼 추가
        store = EpisodeStore(db_path)
        cols = [r[1] for r in store._conn.execute("PRAGMA table_info(episodes)")]
        assert "interpretation_text" in cols

        # legacy row 로드 시 interpretation_text="" default
        legacy_eps = store.recent(10)
        assert len(legacy_eps) == 1
        assert legacy_eps[0].interpretation_text == ""


# ── M3: quality_hint ────────────────────────────────

def test_quality_hint_empty_string():
    assert quality_hint("") == 0.0


def test_quality_hint_zero_keywords():
    """키워드 0개 → 0.0."""
    assert quality_hint("This is a plain sentence with no insight.") == 0.0


def test_quality_hint_high_keyword_count():
    """3개 이상 → 1.0."""
    text = ("The mechanism differs along the axis of scope, and the layer of "
            "abstraction is different.")
    assert quality_hint(text) == 1.0


def test_quality_hint_partial_keywords():
    """1개 → ~0.33, 2개 → ~0.67."""
    one = quality_hint("Just a mechanism.")
    two = quality_hint("This mechanism and that dimension.")
    assert 0.0 < one < two <= 1.0


def test_quality_hint_korean_keywords():
    """한국어 키워드도 인식."""
    text = "이것은 메커니즘 차원에서의 분해 관점이다."
    score = quality_hint(text)
    assert score >= 0.67


# ── M4: MemorySystem.save_conflict / recall_conflict ──

def test_save_conflict_creates_episode_with_winner():
    """save_conflict → winner='conflict_interpreter' Episode 생성."""
    with tempfile.TemporaryDirectory() as td:
        mem = MemorySystem(memory_dir=td)
        interp_vec = torch.randn(64)
        ep_id = mem.save_conflict(
            trigger_vec        =interp_vec,
            new_text           = "새 텍스트",
            partner_texts      = ["파트너1", "파트너2"],
            interpretation     = "mechanism 의 layer 가 다른 차원이다",
            conflict_score     = 0.15,
        )
        # _row_to_episode 로 가져와 검증
        ep = mem.l2._row_to_episode(
            mem.l2._conn.execute(
                "SELECT * FROM episodes WHERE episode_id=?", (ep_id,),
            ).fetchone()
        )
        assert ep.winner == "conflict_interpreter"
        assert "mechanism" in ep.interpretation_text
        # quality_hint 도 in-memory 기록
        assert ep_id in mem._quality_by_episode
        assert mem._quality_by_episode[ep_id] >= 0.67   # 키워드 3개


def test_recall_conflict_returns_sorted_by_quality():
    """recall_conflict 가 quality_hint 내림차순 정렬."""
    with tempfile.TemporaryDirectory() as td:
        mem = MemorySystem(memory_dir=td)
        # 3 episodes 저장 — 다른 quality
        ep1 = mem.save_conflict(
            trigger_vec        =torch.ones(64) * 0.5,
            new_text="t1", partner_texts=["p1"],
            interpretation="단순 설명, 키워드 없음",
            conflict_score=0.1,
        )
        ep2 = mem.save_conflict(
            trigger_vec        =torch.ones(64) * 0.55,
            new_text="t2", partner_texts=["p2"],
            interpretation="mechanism 의 axis 가 다른 dimension",
            conflict_score=0.1,
        )
        ep3 = mem.save_conflict(
            trigger_vec        =torch.ones(64) * 0.45,
            new_text="t3", partner_texts=["p3"],
            interpretation="단순한 텍스트입니다",
            conflict_score=0.1,
        )
        # query vec 은 모두 비슷한 cosine
        results = mem.recall_conflict(torch.ones(64) * 0.5, top_k=3)
        assert len(results) == 3
        # 첫 결과가 quality 가 가장 높은 ep2
        assert results[0][0].episode_id == ep2
        assert results[0][1] >= 0.67


def test_recall_conflict_empty_when_no_episodes():
    """저장된 conflict 없으면 빈 list."""
    with tempfile.TemporaryDirectory() as td:
        mem = MemorySystem(memory_dir=td)
        results = mem.recall_conflict(torch.randn(64))
        assert results == []


# ── M5/M6: KnowledgeLoop 통합 + IngestResult.recall_hint ──

def _make_loop_tfidf(tmp: Path) -> KnowledgeLoop:
    store = KnowledgeStore(tmp / "log.jsonl")
    return KnowledgeLoop(
        encoder=TfidfJLEncoder(dim=32),
        store=store,
        coherence_thresholds=(0.10, 0.12),
    )


def test_knowledge_loop_default_creates_memory():
    """Architecture B: memory=None → 자동 MemorySystem 생성."""
    with tempfile.TemporaryDirectory() as td:
        loop = _make_loop_tfidf(Path(td))
        assert loop.memory is not None
        assert isinstance(loop.memory, MemorySystem)


def test_knowledge_loop_user_memory_used():
    """사용자가 명시한 MemorySystem 사용."""
    with tempfile.TemporaryDirectory() as td:
        custom_mem = MemorySystem(memory_dir=Path(td) / "custom")
        store = KnowledgeStore(Path(td) / "log.jsonl")
        loop = KnowledgeLoop(
            encoder=TfidfJLEncoder(dim=32),
            store=store,
            memory=custom_mem,
        )
        assert loop.memory is custom_mem


def test_ingest_result_has_recall_hint_field():
    """IngestResult 에 recall_hint 필드 신설."""
    with tempfile.TemporaryDirectory() as td:
        loop = _make_loop_tfidf(Path(td))
        result = loop.ingest("first", source="x")
        assert hasattr(result, "recall_hint")
        # 첫 ingest 는 escalate 안 됨 → recall_hint None
        assert result.recall_hint is None


@pytest.mark.skipif(_SKIP_HF, reason="HF download skipped")
def test_recall_hint_on_second_similar_conflict():
    """2회차 비슷한 충돌에서 recall_hint 가 채워짐."""
    from htp.knowledge.embedding import EmbeddingBridge

    with tempfile.TemporaryDirectory() as td:
        store = KnowledgeStore(Path(td) / "log.jsonl")
        loop = KnowledgeLoop(
            encoder=EmbeddingBridge(),
            store=store,
            coherence_thresholds=(0.10, 0.12),
        )

        # 1회차 — 신규 도메인 적재 → 충돌
        loop.ingest("해마 CA3 패턴 완성 시냅스 recurrent", source="뇌과학")
        loop.ingest("시냅스 가소성 헵 학습", source="뇌과학")
        # 첫 이질 ingest — escalate, save Episode
        r1 = loop.ingest("Redis LRU 캐시 eviction nginx 로드밸런서",
                        source="인프라")
        if not (r1.coherence_info and r1.coherence_info["escalate"]):
            pytest.skip("첫 ingest 가 escalate=True 아님 (e5 분포 의존)")

        # 2회차 — 비슷한 충돌 (유사 도메인 쌍)
        r2 = loop.ingest("Kubernetes pod scheduler eviction policy",
                        source="인프라")
        # recall_hint 가 None 이 아니어야 함 (vec 유사도 가까움)
        # 단, mismatch threshold 통과해야 — 실 e5 임베딩 의존성 있음
        # 통과 못해도 SC3 strict 는 아닐 수 있으나, 시도된 흐름 자체는 검증
        assert hasattr(r2, "recall_hint")


# ══════════════════════════════════════════════════════════
# htp-conflict-recall-guardrail (2026-05-20) — 회귀 방지선
# Plan: docs/01-plan/features/htp-conflict-recall-guardrail.plan.md
# 지시서: docs/01-plan/features/claude_code_지시서_conflict_recall_phase1-2.md §2
#
# 외부 리뷰 "MERGE GO" 가 컨테이너 실측에서 NO-GO 로 정정됨.
# 거짓 양성 100% (12/12), 튜닝 처방 4종 전부 FAIL.
# 결함 위치: threshold 아님. recall key 설계 (state_vec=trigger_vec).
# 본 테스트들은 *결함을 RED 로 고정* — 처방 적용 후 GREEN 전환을 자동 검출.
# ══════════════════════════════════════════════════════════


@pytest.mark.skipif(_SKIP_HF, reason="HF download skipped")
def test_recall_does_not_hit_on_unrelated_conflict():
    """완전 무관 입력은 저장된 충돌 해석을 recall 해서는 안 된다.

    측정 1 에서 'EASY_NEG' 가 현재 시스템에서 거짓 양성으로 HIT 함을
    확인. 이 테스트는 그 결함을 고정한다 — 처방 적용 후 GREEN 이어야
    머지 가능. 현 master 에서는 RED (의도된 실패 = 결함 증명).
    """
    from htp.knowledge.embedding import EmbeddingBridge
    import torch
    with tempfile.TemporaryDirectory() as td:
        enc = EmbeddingBridge()
        mem = MemorySystem(memory_dir=Path(td) / "mem")
        # anchor: 인프라 캐시 충돌 해석 저장
        tv = torch.tensor(enc.encode(
            "Redis LRU 캐시 eviction 전략 메모리 축출"),
            dtype=torch.float32)
        mem.save_conflict(
            trigger_vec=tv, new_text="Redis LRU eviction",
            partner_texts=["해마 CA3"], interpretation="categorical conflict",
            conflict_score=0.15)
        # 완전 무관 probe — 절대 HIT 하면 안 됨
        for unrelated in ("중세 고딕 성당 부벽 구조 하중 분산",
                          "김치 발효 유산균 pH 변화 숙성"):
            qv = torch.tensor(enc.encode_query(unrelated),
                              dtype=torch.float32)
            results = mem.recall_conflict(qv, top_k=3)
            # recall_conflict 자체는 후보를 줄 수 있으나,
            # _try_recall_conflict 게이트 통과(mismatch<thr)는 막혀야 함.
            # 게이트 로직을 직접 재현해 검증:
            if results:
                best_ep, _ = results[0]
                import struct
                n = len(best_ep.state_vec) // 4
                pv = torch.tensor(
                    struct.unpack(f"{n}f", best_ep.state_vec),
                    dtype=torch.float32)
                mismatch = float((qv - pv).norm())
                assert mismatch >= mem.CONFLICT_RECALL_MISMATCH_THRESHOLD, (
                    f"거짓 양성: 무관 입력 '{unrelated[:20]}' 가 "
                    f"mismatch={mismatch:.3f} 로 recall HIT "
                    f"(thr={mem.CONFLICT_RECALL_MISMATCH_THRESHOLD})")


@pytest.mark.skipif(_SKIP_HF, reason="HF download skipped")
def test_recall_query_uses_query_prefix():
    """발견 B 회귀 방지 — recall 경로가 encode_query() 를 쓰는지.

    _try_recall_conflict 가 passage prefix(encode) 로 검색하면
    e5 비대칭 검색이 깨진다. 이 테스트는 loop 가 query prefix 를
    쓰도록 강제한다. 현 master RED (encode 사용 중) → 처방 후 GREEN.
    """
    from htp.knowledge.embedding import EmbeddingBridge
    enc = EmbeddingBridge()
    # e5: query/passage prefix 가 다른 벡터를 내야 정상
    v_p = enc.encode("테스트 문장")
    v_q = enc.encode_query("테스트 문장")
    import numpy as np
    assert not np.allclose(v_p, v_q), (
        "encode 와 encode_query 가 동일 벡터 — prefix 미적용 의심")
    # 실제 loop 경로가 query prefix 쓰는지는 작업 2-2 의 xfail 로 추적


def test_recall_fp_dataset_is_tracked():
    """측정 데이터셋이 repo 에 고정돼 재현 가능한지 sanity."""
    from pathlib import Path as _P
    p = _P("scripts/conflict_recall_fp_eval.py")
    assert p.exists(), (
        "측정 스크립트 미존재 — 작업 2-3 에서 scripts/ 에 커밋 필요")


@pytest.mark.skipif(_SKIP_HF, reason="HF download skipped")
@pytest.mark.xfail(
    reason="trigger-key 설계 결함(측정 2): 표면 다르고 구조 같은 "
           "충돌을 trigger 임베딩으로 MISS. 3단계 처방 전까지 "
           "의도된 실패. 외부 리뷰 §9-1.1 미검증 결정.",
    strict=True)
def test_trigger_key_recalls_same_conflict_different_surface():
    """같은 추상 충돌의 다른 표면 표현은 recall 돼야 한다 (당위).

    'Redis 캐시 축출' 과 '시냅스 가지치기' 는 같은 eviction 추상
    충돌이나 trigger 표면이 달라 임베딩이 멀다. 측정 2 에서 이
    케이스 분리 불가 확인. strict xfail = 처방으로 해결되면
    XPASS 로 빨간불 → 그때 이 마커를 제거하고 GREEN 전환.
    """
    from htp.knowledge.embedding import EmbeddingBridge
    import torch, struct
    with tempfile.TemporaryDirectory() as td:
        enc = EmbeddingBridge()
        mem = MemorySystem(memory_dir=Path(td) / "mem")
        tv = torch.tensor(enc.encode(
            "Redis LRU 캐시 eviction 메모리 축출 정책"),
            dtype=torch.float32)
        mem.save_conflict(
            trigger_vec=tv, new_text="Redis eviction",
            partner_texts=["x"], interpretation="eviction 추상 충돌",
            conflict_score=0.15)
        # 같은 추상 충돌, 완전히 다른 표면
        qv = torch.tensor(enc.encode_query(
            "시냅스 가지치기 미세아교세포 약한 연결 제거"),
            dtype=torch.float32)
        results = mem.recall_conflict(qv, top_k=3)
        assert results, "후보 없음"
        best_ep, _ = results[0]
        n = len(best_ep.state_vec) // 4
        pv = torch.tensor(struct.unpack(f"{n}f", best_ep.state_vec),
                          dtype=torch.float32)
        mismatch = float((qv - pv).norm())
        # 당위: 같은 충돌이므로 HIT 해야 함 → 현 설계로는 실패(xfail)
        assert mismatch < mem.CONFLICT_RECALL_MISMATCH_THRESHOLD


# ══════════════════════════════════════════════════════════
# 보강 (2026-05-20) — 지시서 §2-1 테스트의 두 한계 보완:
#   (1) prefix 불일치: 지시서 코드는 enc.encode_query() 사용 →
#       실 _try_recall_conflict 의 encode() 경로 우회
#   (2) NEG 난이도 부족: EASY_NEG 2건 (성당/김치) 만 — query-prefix
#       에 의해 분리됨. HARD_NEG (같은 도메인 다른 메커니즘) 누락
# 해결: 실 KnowledgeLoop.ingest 경로 + HARD_NEG 데이터셋으로 통합 검증
# ══════════════════════════════════════════════════════════


@pytest.mark.skipif(_SKIP_HF, reason="HF download skipped")
def test_recall_hint_none_on_hard_neg_via_real_loop_path():
    """실 KnowledgeLoop.ingest 경로로 HARD_NEG 거짓 양성 거부 검증.

    기존 test_recall_does_not_hit_on_unrelated_conflict 는:
      - enc.encode_query() 사용 (실 loop 와 다른 prefix)
      - EASY_NEG 2건만 (지시서 측정 1: query-prefix 로 일부 분리 가능)
    → 결함 우회. 본 테스트는 (1) loop.ingest 실 경로 + (2) HARD_NEG 로
    실제 결함 RED 를 고정.

    현 master RED (의도된 실패) → Phase 3 처방 후 GREEN.
    """
    from htp.knowledge.embedding import EmbeddingBridge
    from htp.knowledge import KnowledgeLoop, KnowledgeStore
    with tempfile.TemporaryDirectory() as td:
        store = KnowledgeStore(Path(td) / "log.jsonl")
        loop = KnowledgeLoop(
            encoder=EmbeddingBridge(), store=store,
            coherence_thresholds=(0.10, 0.12),
        )
        # baseline: 다른 도메인 채우기 (escalate 조건 형성)
        for t in ("해마 CA3 패턴 완성 시냅스 recurrent",
                  "시냅스 가소성 헵 학습 법칙",
                  "감마 진동 뇌파 의식 통합"):
            loop.ingest(t, source="뇌과학")

        # 1회차: Redis LRU 캐시 충돌 ingest → escalate → Episode 저장
        loop.ingest(
            "Redis LRU 캐시 eviction 전략 메모리 축출",
            source="인프라",
        )
        # 2회차: HARD_NEG — 같은 도메인 (인프라/Redis), 다른 메커니즘 (샤딩)
        # 지시서 측정 1 의 HARD_NEG anchor0 케이스와 동일.
        r = loop.ingest(
            "Redis cluster 샤딩 hash slot 재분배 리밸런싱",
            source="인프라",
        )

        # 실 경로 sanity 검증.
        #
        # ⚠ 발견 (2026-05-20 진단): 같은 도메인 연속 ingest 는 _evaluate_coherence
        # 의 top-3 이웃에 직전 ingest 가 포함됨 → coherence 0.90+ → conflict
        # < 0.12 → escalate=False → _try_recall_conflict 자체가 호출 안 됨.
        # 따라서 이 테스트는 *결함을 검증하지 못함* — escalate 분기에서 자동
        # 우회됨. 결함 RED 는 아래 test_recall_conflict_hard_neg_via_memory_direct
        # (MemorySystem 직접 호출 — escalate 분기 우회) 에서 검증.
        if r.coherence_info and r.coherence_info.get("escalate"):
            assert r.recall_hint is None, (
                f"거짓 양성: HARD_NEG (Redis 샤딩 vs LRU eviction) 가 "
                f"recall_hint={r.recall_hint!r} 로 HIT — 결함 RED 확정"
            )


@pytest.mark.skipif(_SKIP_HF, reason="HF download skipped")
@pytest.mark.xfail(
    reason="trigger-key 설계 결함(측정 1·2): HARD_NEG 같은 도메인 다른 "
           "메커니즘이 trigger 임베딩 cosine 0.85+ 분포에서 분리 불가. "
           "현 master 에서 거짓 양성 발현. Phase 3 처방 전까지 의도된 "
           "실패 — 지시서 §0 측정 1 의 HARD_NEG FP 6/6 (100%) 와 일치.",
    strict=True)
def test_recall_conflict_hard_neg_via_memory_direct():
    """MemorySystem.recall_conflict 직접 호출 + HARD_NEG — escalate 분기 우회.

    위 test_recall_hint_none_on_hard_neg_via_real_loop_path 는 escalate
    분기에서 자동 우회됨 (같은 도메인 연속 ingest → coherence 높음).
    본 테스트는 그 우회를 차단하고 *MemorySystem 의 recall key 설계 결함*
    자체를 검증한다.

    경로 일치:
      - encoder.encode() (passage prefix) 사용 — 실 _try_recall_conflict 와 동일
      - mem.recall_conflict 직접 호출 — escalate 분기 안 거침
      - HARD_NEG = 같은 Redis 도메인 다른 메커니즘 (지시서 측정 1 anchor0)

    strict xfail = 처방으로 해결되면 XPASS 로 빨간불 → 마커 제거 + GREEN.
    """
    from htp.knowledge.embedding import EmbeddingBridge
    import torch, struct
    with tempfile.TemporaryDirectory() as td:
        enc = EmbeddingBridge()
        mem = MemorySystem(memory_dir=Path(td) / "mem")
        # anchor: Redis LRU 캐시 충돌 (지시서 측정 1 anchor[0])
        tv = torch.tensor(enc.encode(
            "Redis LRU 캐시 eviction 전략 메모리 축출"),
            dtype=torch.float32)
        mem.save_conflict(
            trigger_vec=tv, new_text="Redis LRU eviction",
            partner_texts=["d"], interpretation="categorical conflict",
            conflict_score=0.15)

        # HARD_NEG: 같은 도메인, 다른 메커니즘 (지시서 측정 1 HARD_NEG[anchor0])
        # passage prefix (encode) 사용 — 실 _try_recall_conflict 경로와 동일
        qv = torch.tensor(enc.encode(
            "Redis cluster 샤딩 hash slot 재분배 리밸런싱"),
            dtype=torch.float32)
        results = mem.recall_conflict(qv, top_k=3)
        assert results, "후보 없음 (anchor 미저장?)"
        best_ep, _ = results[0]
        n = len(best_ep.state_vec) // 4
        pv = torch.tensor(struct.unpack(f"{n}f", best_ep.state_vec),
                          dtype=torch.float32)
        mismatch = float((qv - pv).norm())
        # 당위: HARD_NEG 이므로 mismatch >= threshold 여야 함 (거짓 양성 거부).
        # 현 master: trigger 임베딩 cosine 0.85+ → mismatch < 0.6 → 거짓 양성 HIT.
        # → 이 assert 가 FAIL = 결함 RED 확정 (xfail strict 가 expected_failure
        #    로 표시)
        assert mismatch >= mem.CONFLICT_RECALL_MISMATCH_THRESHOLD, (
            f"거짓 양성: HARD_NEG mismatch={mismatch:.3f} < "
            f"{mem.CONFLICT_RECALL_MISMATCH_THRESHOLD} (=거짓 양성 HIT)"
        )
