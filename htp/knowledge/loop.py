"""
KnowledgeLoop — 최소 지식 입출력 루프.

Design Ref: docs/02-design/features/htp-thalamus-car.design.md §4.3
Plan SC: FR-05.4, FR-05.5 (ingest/query/discover + dataclass)

DAG: htp/knowledge/ — htp/runtime/* 미참조.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime    import datetime, timezone

import numpy as np

from .encoder     import TextEncoder
from .persistence import KnowledgeStore


# ══════════════════════════════════════════════════════════
# Dataclass 정의
# ══════════════════════════════════════════════════════════

@dataclass
class KnowledgeEntry:
    text: str
    vec: np.ndarray
    source: str
    timestamp: str
    neighbors: list = field(default_factory=list)
    conflict_count: int = 0


@dataclass
class Neighbor:
    entry_id: int
    similarity: float


@dataclass
class IngestResult:
    entry: KnowledgeEntry
    neighbors: list
    conflicts: list
    resonances: list


@dataclass
class QueryResult:
    question: str
    relevant: list
    cluster_count: int


@dataclass
class Discovery:
    entry_a_id: int
    entry_b_id: int
    source_a: str
    source_b: str
    similarity: float
    insight: str


# ══════════════════════════════════════════════════════════
# Helper
# ══════════════════════════════════════════════════════════

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)


# ══════════════════════════════════════════════════════════
# KnowledgeLoop
# ══════════════════════════════════════════════════════════

class KnowledgeLoop:
    """최소 지식 입출력 루프 (Stage 0.5 MVP).

    Design Ref: §4.3 — ingest / query / discover.
    """

    def __init__(self,
                 encoder: TextEncoder,
                 store: KnowledgeStore | None = None,
                 conflict_threshold: float = 0.3,
                 resonance_threshold: float = 0.7,
                 discover_threshold: float = 0.6):
        self.encoder = encoder
        self.store   = store or KnowledgeStore.default()
        self.conflict_threshold  = conflict_threshold
        self.resonance_threshold = resonance_threshold
        self.discover_threshold  = discover_threshold
        self._cache: list[KnowledgeEntry] = self.store.load_all()

    # ── ingest ────────────────────────────────────────────
    def ingest(self, text: str, source: str = "") -> IngestResult:
        corpus = [e.text for e in self._cache] + [text]
        self.encoder.fit(corpus)
        vec = self.encoder.encode(text)

        # 기존 캐시는 새 encoder 로 재-encode 해야 동일 공간 비교 가능
        for e in self._cache:
            e.vec = self.encoder.encode(e.text)

        neighbors = self._find_neighbors(vec, top_k=5)
        conflicts  = [n for n in neighbors if n.similarity < self.conflict_threshold]
        resonances = [n for n in neighbors if n.similarity > self.resonance_threshold]

        entry = KnowledgeEntry(
            text=text, vec=vec, source=source,
            timestamp=datetime.now(timezone.utc).isoformat(),
            neighbors=[(n.entry_id, n.similarity) for n in neighbors],
            conflict_count=len(conflicts),
        )
        self._cache.append(entry)
        self.store.append(entry)

        return IngestResult(entry=entry, neighbors=neighbors,
                            conflicts=conflicts, resonances=resonances)

    # ── query ─────────────────────────────────────────────
    def query(self, question: str) -> QueryResult:
        if not self._cache:
            return QueryResult(question=question, relevant=[], cluster_count=0)
        q_vec = self.encoder.encode(question)
        relevant = self._find_neighbors(q_vec, top_k=10)
        clusters = self._count_clusters(relevant)
        return QueryResult(question=question, relevant=relevant,
                           cluster_count=clusters)

    # ── discover ──────────────────────────────────────────
    def discover(self) -> list[Discovery]:
        discoveries: list[Discovery] = []
        n = len(self._cache)
        for i in range(n):
            a = self._cache[i]
            for j in range(i + 1, n):
                b = self._cache[j]
                if a.source == b.source:
                    continue
                sim = _cosine(a.vec, b.vec)
                if sim > self.discover_threshold:
                    discoveries.append(Discovery(
                        entry_a_id=i, entry_b_id=j,
                        source_a=a.source, source_b=b.source,
                        similarity=sim,
                        insight=(f"'{a.source}'와 '{b.source}'가 "
                                 f"{self.encoder.dim}-dim 공간에서 "
                                 f"{sim:.2f} 유사도로 연결됨"),
                    ))
        discoveries.sort(key=lambda d: d.similarity, reverse=True)
        return discoveries[:10]

    # ── 내부 ──────────────────────────────────────────────
    def _find_neighbors(self, vec: np.ndarray, top_k: int) -> list[Neighbor]:
        if not self._cache:
            return []
        sims = [
            Neighbor(entry_id=i, similarity=_cosine(vec, e.vec))
            for i, e in enumerate(self._cache)
        ]
        sims.sort(key=lambda n: n.similarity, reverse=True)
        return sims[:top_k]

    def _count_clusters(self, neighbors: list[Neighbor],
                        cluster_threshold: float = 0.6) -> int:
        if not neighbors:
            return 0
        groups: list[list[Neighbor]] = []
        for n in neighbors:
            assigned = False
            cur = self._cache[n.entry_id].vec
            for g in groups:
                rep = self._cache[g[0].entry_id].vec
                if _cosine(rep, cur) > cluster_threshold:
                    g.append(n)
                    assigned = True
                    break
            if not assigned:
                groups.append([n])
        return len(groups)


__all__ = [
    "KnowledgeLoop", "KnowledgeEntry", "Neighbor",
    "IngestResult", "QueryResult", "Discovery",
]
