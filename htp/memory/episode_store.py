"""
L2 EpisodeStore — SQLite + SWR novelty × reward 태깅
=====================================================

설계 문서: design/htp_memory_design_final.md §3
"""
from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from .types import Episode, bytes_to_tensor


SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes (
    episode_id          TEXT PRIMARY KEY,
    step                INTEGER,
    winner              TEXT,
    action_type         TEXT,
    score               REAL,
    state_vec           BLOB,
    context             TEXT,
    outcome             TEXT,
    recall_count        INTEGER DEFAULT 0,
    novelty             REAL    DEFAULT 1.0,
    swr_tagged          INTEGER DEFAULT 0,
    session_id          TEXT,
    timestamp           REAL,
    interpretation_text TEXT    DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_winner    ON episodes(winner);
CREATE INDEX IF NOT EXISTS idx_swr       ON episodes(swr_tagged);
CREATE INDEX IF NOT EXISTS idx_timestamp ON episodes(timestamp);
CREATE INDEX IF NOT EXISTS idx_outcome   ON episodes(outcome);
"""


class EpisodeStore:
    """SQLite 기반 L2 에피소드 저장소."""

    def __init__(self, db_path: str | Path = "htp_memory.db"):
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.executescript(SCHEMA)
        # async 환경 대비 WAL 모드
        self._conn.execute("PRAGMA journal_mode=WAL")
        # htp-conflict-memory (2026-05-19): 기존 DB 마이그레이션.
        # interpretation_text 컬럼 존재 확인 후 idempotent ALTER.
        cols = [r[1] for r in self._conn.execute("PRAGMA table_info(episodes)")]
        if "interpretation_text" not in cols:
            self._conn.execute(
                "ALTER TABLE episodes ADD COLUMN interpretation_text TEXT DEFAULT ''"
            )
            self._conn.commit()

    # ── 저장 ─────────────────────────────────────

    def save(self, ep: Episode) -> str:
        if not ep.episode_id:
            ep.episode_id = str(uuid.uuid4())
        # 14 columns (interpretation_text 추가)
        self._conn.execute("""
            INSERT OR REPLACE INTO episodes
            (episode_id, step, winner, action_type, score, state_vec,
             context, outcome, recall_count, novelty, swr_tagged,
             session_id, timestamp, interpretation_text)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            ep.episode_id, ep.step, ep.winner, ep.action_type,
            ep.score, ep.state_vec, ep.context, ep.outcome,
            ep.recall_count, ep.novelty, int(ep.swr_tagged),
            ep.session_id, ep.timestamp,
            ep.interpretation_text,
        ))
        self._conn.commit()
        return ep.episode_id

    # ── 업데이트 ─────────────────────────────────

    def feedback(self, episode_id: str, outcome: str):
        """사후 결과 기록 — L3 패턴 개선에 반영."""
        self._conn.execute(
            "UPDATE episodes SET outcome=? WHERE episode_id=?",
            (outcome, episode_id),
        )
        self._conn.commit()

    def increment_recall(self, episode_id: str):
        """CA1 재활성화 횟수 증가."""
        self._conn.execute(
            "UPDATE episodes SET recall_count = recall_count + 1 WHERE episode_id=?",
            (episode_id,),
        )
        self._conn.commit()

    # ── SWR 태깅 ─────────────────────────────────

    def tag_swr(self,
                priority_threshold: float = 0.5,
                conflict_map: "dict[str, float] | None" = None):
        """
        SWR 태깅 — LeCun/Yang & Buzsáki 2024 + sub-3 Plan FR-15:
          priority = novelty × score × (1 + conflict_magnitude)
          tagged   = priority >= threshold

        conflict_map: episode_id → conflict_magnitude. None 또는 미존재 시 0.
                       기존 호출자 (conflict_map=None) 는 기존 식 동등 (회귀 보호).
        """
        rows = self._conn.execute(
            "SELECT episode_id, score, novelty FROM episodes WHERE outcome IS NOT NULL"
        ).fetchall()
        for ep_id, score, novelty in rows:
            conflict = (conflict_map.get(ep_id, 0.0)
                        if conflict_map is not None else 0.0)
            priority = (novelty or 0.0) * (score or 0.0) * (1.0 + conflict)
            tagged   = 1 if priority >= priority_threshold else 0
            self._conn.execute(
                "UPDATE episodes SET swr_tagged=? WHERE episode_id=?",
                (tagged, ep_id),
            )
        self._conn.commit()

    # ── 조회 ─────────────────────────────────────

    def search_similar(self,
                       state_vec: torch.Tensor,
                       top_k: int = 5,
                       winner_filter: Optional[str] = None,
                       swr_only: bool = False) -> list[Episode]:
        """
        64-dim 코사인 유사도 검색.
        winner_filter: CA3 prior 로 특정 winner 만 필터
        swr_only: consolidation 후보만 검색
        """
        query = "SELECT * FROM episodes WHERE 1=1"
        params: list = []
        if winner_filter:
            query += " AND winner=?"
            params.append(winner_filter)
        if swr_only:
            query += " AND swr_tagged=1"

        rows = self._conn.execute(query, params).fetchall()
        if not rows:
            return []

        scored: list[tuple[float, tuple]] = []
        for row in rows:
            blob = row[5]
            if not blob:
                continue
            vec = bytes_to_tensor(blob)
            if vec.shape[0] != state_vec.shape[0]:
                continue       # 차원 불일치 (구 8-dim 데이터) 스킵
            sim = F.cosine_similarity(
                state_vec.unsqueeze(0), vec.unsqueeze(0),
            ).item()
            scored.append((sim, row))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [self._row_to_episode(r) for _, r in scored[:top_k]]

    def recent(self, n: int = 100) -> list[Episode]:
        rows = self._conn.execute(
            "SELECT * FROM episodes ORDER BY timestamp DESC LIMIT ?", (n,),
        ).fetchall()
        return [self._row_to_episode(r) for r in rows]

    def get_swr_tagged(self, limit: int = 200) -> list[Episode]:
        """SWR 태깅된 에피소드들을 유사도 검색 없이 직접 조회."""
        rows = self._conn.execute(
            "SELECT * FROM episodes WHERE swr_tagged=1 ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [self._row_to_episode(r) for r in rows]

    def count(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0])

    def close(self):
        self._conn.close()

    # ── 내부 ─────────────────────────────────────

    @staticmethod
    def _row_to_episode(row) -> Episode:
        # htp-conflict-memory: interpretation_text (row[13]) — legacy row 는 14열 미만
        interpretation = row[13] if len(row) > 13 else ""
        return Episode(
            episode_id=row[0], step=row[1], winner=row[2],
            action_type=row[3], score=row[4], state_vec=row[5],
            context=row[6], outcome=row[7], recall_count=row[8] or 0,
            novelty=row[9] or 1.0, swr_tagged=bool(row[10]),
            session_id=row[11] or "", timestamp=row[12] or 0.0,
            interpretation_text=interpretation or "",
        )
