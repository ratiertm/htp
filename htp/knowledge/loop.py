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
from .types       import KnowledgeEntry, Tombstone   # L2 sidequest session-1

# Design Ref: docs/02-design/features/htp-bridge-integration-design.md §2-4 (S1-S3)
# 시스템 A (htp/thalamus) → 시스템 B (htp/knowledge) 단방향 import.
# 역방향 import 금지 (test_no_circular_deps.py 가 영구 검증).
from htp.thalamus.signature import RegionSignature
from htp.thalamus.coherence.pairwise import PairwiseCoherenceGate
from htp.thalamus.types     import RegionResponse
from htp.thalamus.router.vector_router import VectorRouter
from htp.thalamus.region_signal import RegionSignal


# ══════════════════════════════════════════════════════════
# Dataclass 정의
# (KnowledgeEntry 는 types.py 로 이동 — session-1, sub-decision #3)
# 이 모듈에서 backward-compat 위해 re-export.
# ══════════════════════════════════════════════════════════

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
    coherence_info: "dict | None" = None   # Bridge §3 (S2): {coherence, conflict, escalate}


@dataclass
class QueryResult:
    question: str
    relevant: list
    cluster_count: int
    routing_info: "dict | None" = None   # Bridge §4 (S3): VectorRouter.last_metrics
    mode: str = "flat"                    # Bridge §4 (S3): "flat" | "routed"


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

        # Critical Gap #3 옵션 A-2: encoder state 영속화.
        # CLI 다중 호출 시 동일 임베딩 공간 보장 — fit 결과를 디스크에 저장/복원.
        self._encoder_state_path = self.store.path.parent / "encoder_state.pkl"
        if hasattr(self.encoder, "load"):
            self.encoder.load(self._encoder_state_path)

        # Bridge Integration §2: source 별 RegionSignature (시스템 A 학습 단위).
        # 캐시 vec 으로부터 centroid 를 EMA 로 재구축 — 영속화 불필요 (vec 은 JSONL 보존).
        self._signatures: dict[str, RegionSignature] = {}
        self._rebuild_signatures()

        # Bridge Integration §3 (S2): CoherenceGate — ingest 시 정합성 검사.
        self._coherence = PairwiseCoherenceGate(
            conflict_threshold   = 0.3,
            escalation_threshold = 0.7,
        )

        # Bridge Integration §4 (S3): VectorRouter — query 시 source 범위 축소.
        self._router = VectorRouter(beta=0.5)

    # ── Bridge Integration §2 (S1) — RegionSignature ─────
    def _rebuild_signatures(self) -> None:
        """기존 _cache 에서 source 별 RegionSignature 를 재구축.

        Bridge Design §2-2. encoder.fit 미완 상태에서 호출되어도 안전 —
        cache 의 vec 은 저장 시점에 encode 된 것이라 그대로 사용 가능.
        """
        for entry in self._cache:
            self._update_signature(entry.source, entry.vec)

    def _update_signature(self, source: str, vec: np.ndarray) -> None:
        """source 별 RegionSignature 점진 학습 (Hebbian EMA)."""
        sig = self._signatures.get(source)
        if sig is None or sig.dim != len(vec):
            # dim 변경 시 (encoder 교체 등) 재초기화. centroid 인자로 dim 자동 추론.
            sig = RegionSignature(centroid=np.zeros(len(vec), dtype=np.float64),
                                  count=0, dim=len(vec))
            self._signatures[source] = sig
        sig.update(vec.astype(np.float64))

    # ── ingest ────────────────────────────────────────────
    def ingest(self, text: str, source: str = "") -> IngestResult:
        # Critical Gap #3 (옵션 A-2): encoder.fit() 1회 freeze + 영속화.
        # 첫 ingest 시점에만 fit, 이후엔 동일 임베딩 공간 유지.
        # CLI 다중 프로세스에서도 .htp/encoder_state.pkl 로 복원되어 동일 공간.
        # 트레이드오프: 새 어휘 미반영 — sub-5 EmbeddingBridge 에서 본질 해결.
        if not getattr(self.encoder, "_fitted", False):
            corpus = [e.text for e in self._cache] + [text]
            self.encoder.fit(corpus)
            # fit 직후 영속화 — 다음 프로세스에서 동일 state 복원
            if hasattr(self.encoder, "save"):
                self.encoder.save(self._encoder_state_path)

        vec = self.encoder.encode(text)

        # encoder 가 freeze 되므로 _cache 재-encode 불필요 (이전 옵션 B 제거).

        # Bridge Integration §2 (S1): source 별 RegionSignature 점진 학습.
        self._update_signature(source, vec)

        neighbors = self._find_neighbors(vec, top_k=5)
        conflicts  = [n for n in neighbors if n.similarity < self.conflict_threshold]
        resonances = [n for n in neighbors if n.similarity > self.resonance_threshold]

        # Bridge Integration §3 (S2): CoherenceGate — 새 vec + 상위 이웃 정합성 측정.
        coherence_info = self._evaluate_coherence(vec, source, neighbors)

        entry = KnowledgeEntry(
            text=text, vec=vec, source=source,
            timestamp=datetime.now(timezone.utc).isoformat(),
            neighbors=[(n.entry_id, n.similarity) for n in neighbors],
            conflict_count=len(conflicts),
        )
        self._cache.append(entry)
        self.store.append(entry)

        return IngestResult(entry=entry, neighbors=neighbors,
                            conflicts=conflicts, resonances=resonances,
                            coherence_info=coherence_info)

    # ── Bridge Integration §3 (S2) — CoherenceGate ────────
    def _evaluate_coherence(
        self,
        vec: np.ndarray,
        source: str,
        neighbors: "list[Neighbor]",
    ) -> "dict | None":
        """새 vec + 상위 이웃 (top 3) 을 RegionResponse 로 변환 → CoherenceGate.bind.

        Bridge Design §3-2. neighbors < 2 면 정합성 비교 자체가 무의미 → None.
        반환 dict: {coherence, conflict, escalate}.
        """
        if len(neighbors) < 2:
            return None

        responses = [
            RegionResponse(
                region_id  = f"new_{source}",
                output_vec = np.asarray(vec, dtype=np.float64),
                precision  = 1.0,
            )
        ]
        for n in neighbors[:3]:
            ne = self._cache[n.entry_id]
            responses.append(RegionResponse(
                region_id  = f"existing_{ne.source}",
                output_vec = np.asarray(ne.vec, dtype=np.float64),
                precision  = 1.0,
            ))

        bound = self._coherence.bind(responses)
        return {
            "coherence": float(bound.coherence),
            "conflict":  float(bound.conflict),
            "escalate":  bool(bound.escalate_to_pfc),
        }

    # ── batch ingest (L2 sidequest F1) ────────────────────
    def ingest_batch(self, texts: list[str], source: str = ""
                    ) -> dict:
        """N 개 텍스트 일괄 ingest — encoder.fit() 1회만 (옵션 A-2 영속화).

        Sub-decision #5: skip-and-continue 정책.

        반환: {"success": [IngestResult], "errors": [{"text", "error"}]}
        """
        results = {"success": [], "errors": []}
        for text in texts:
            try:
                r = self.ingest(text, source=source)
                results["success"].append(r)
            except Exception as e:
                results["errors"].append({
                    "text":  (text[:80] + "...") if len(text) > 80 else text,
                    "error": str(e),
                })
        return results

    # ── delete / edit / add_tags (L2 sidequest F4) ────────
    def delete(self, entry_id: str) -> bool:
        """tombstone 패턴 delete. 성공 시 True, ID 미존재 시 False."""
        found = next((e for e in self._cache if e.id == entry_id), None)
        if found is None:
            return False
        self._cache = [e for e in self._cache if e.id != entry_id]
        self.store.append_tombstone(Tombstone(
            kind      = "delete",
            ref_id    = entry_id,
            timestamp = datetime.now(timezone.utc).isoformat(),
        ))
        return True

    def edit(self, entry_id: str, new_text: str) -> KnowledgeEntry | None:
        """본문 수정 — id 유지 (Plan FR-13). 같은 id 로 새 entry append.

        load_all 의 '후자 우선' 로직으로 최신 entry 가 반환됨.
        text/vec/timestamp 갱신, tags/source 보존.
        """
        target = next((e for e in self._cache if e.id == entry_id), None)
        if target is None:
            return None

        new_vec = self.encoder.encode(new_text)
        new_entry = KnowledgeEntry(
            text           = new_text,
            vec            = new_vec,
            source         = target.source,
            timestamp      = datetime.now(timezone.utc).isoformat(),
            neighbors      = [],
            conflict_count = 0,
            id             = entry_id,
            tags           = list(target.tags),
        )
        # cache 업데이트
        idx = self._cache.index(target)
        self._cache[idx] = new_entry
        # jsonl 에 추가 — 같은 id 두 라인이 되지만 load_all 의 후자 우선
        self.store.append(new_entry)
        return new_entry

    def add_tags(self, entry_id: str, tags: list[str]
                ) -> KnowledgeEntry | None:
        """entry 에 tags 추가 (중복 제거 union)."""
        target = next((e for e in self._cache if e.id == entry_id), None)
        if target is None:
            return None
        merged = list({*target.tags, *tags})
        new_entry = KnowledgeEntry(
            text           = target.text,
            vec            = target.vec,
            source         = target.source,
            timestamp      = datetime.now(timezone.utc).isoformat(),
            neighbors      = list(target.neighbors),
            conflict_count = target.conflict_count,
            id             = entry_id,
            tags           = merged,
        )
        idx = self._cache.index(target)
        self._cache[idx] = new_entry
        self.store.append(new_entry)
        return new_entry

    # ── query ─────────────────────────────────────────────
    def query(self, question: str, mode: str = "flat") -> QueryResult:
        """검색.

        mode:
          "flat"   — 기존 전체 _cache 순회 (backward-compat 기본값).
          "routed" — VectorRouter 가 관련 source 선택 후 그 안에서 검색 (Bridge §4).
        """
        if not self._cache:
            return QueryResult(question=question, relevant=[],
                               cluster_count=0, mode=mode)
        # sub-5 merge plan §2: query mode encoding (e5 prefix 활용)
        # encode_query 미지원 인코더는 encode 와 동일 동작
        encode_q = getattr(self.encoder, "encode_query", None) or self.encoder.encode
        q_vec = encode_q(question)

        routing_info = None
        if mode == "routed" and self._signatures:
            candidates, routing_info = self._routed_candidates(q_vec)
        else:
            candidates = list(range(len(self._cache)))

        relevant = self._find_neighbors_among(q_vec, candidates, top_k=10)
        clusters = self._count_clusters(relevant)
        return QueryResult(question=question, relevant=relevant,
                           cluster_count=clusters,
                           routing_info=routing_info, mode=mode)

    # ── Bridge Integration §4 (S3) — VectorRouter ─────────
    def _routed_candidates(
        self, q_vec: np.ndarray,
    ) -> "tuple[list[int], dict]":
        """VectorRouter 로 활성 source 선택 → 해당 source 의 entry index 만 반환.

        Bridge Design §4-2. 모든 source score=0 (cold start 또는 무관) → fallback
        으로 전체 cache 반환.
        """
        # torch import 는 placeholder output_vec 용 — RegionSignal 이 필수 필드로 요구.
        # 시스템 A (RegionSignal) 를 수정하지 않기 위한 회피책 (Design §부록).
        import torch
        placeholder = torch.zeros(1)

        # source 별 RegionSignal 합성 — VectorRouter 는 region_signature 만 참조.
        signals = [
            RegionSignal(
                region_id        = source,
                hub_strength     = 0.0,
                fire_rate        = 0.0,
                top_hubs         = [],
                overload         = False,
                output_vec       = placeholder,
                precision        = 1.0,
                region_signature = sig,
            )
            for source, sig in self._signatures.items()
        ]
        scores = self._router.score(None, q_vec.astype(np.float64), signals)

        selected = {s.region_id for s in scores if s.score > 1e-8}
        metrics = dict(self._router.last_metrics)   # snapshot

        if not selected:
            # 모든 source 가 0 → fallback (전체 순회)
            metrics["fallback"] = "all_zero"
            return list(range(len(self._cache))), metrics

        idxs = [i for i, e in enumerate(self._cache) if e.source in selected]
        metrics["selected_sources"] = sorted(selected)
        metrics["candidate_count"]  = len(idxs)
        return idxs, metrics

    def _find_neighbors_among(
        self, vec: np.ndarray, indices: "list[int]", top_k: int,
    ) -> "list[Neighbor]":
        """indices 내에서만 cosine 검색."""
        if not indices:
            return []
        sims = [
            Neighbor(entry_id=i, similarity=_cosine(vec, self._cache[i].vec))
            for i in indices
        ]
        sims.sort(key=lambda n: n.similarity, reverse=True)
        return sims[:top_k]

    # ── query_v2 (sub-5 merge plan 작업 3 — I5 confidence) ───
    def query_v2(self, question: str, top_k: int = 5,
                 gap_threshold: "float | None" = None,
                 mode: str = "flat"):
        """confidence (top-1 vs top-2 gap) 포함 query — 신규.

        반환: QueryResultV2 (results + confidence + has_match).

        gap_threshold: None 이면 DEFAULT_GAP_THRESHOLD (0.005) 사용.
        mode: "flat" | "routed" (Bridge §4 S3 — VectorRouter 활성 source 사전 필터).
        """
        from .confidence import QueryResultV2, ScoredResult, DEFAULT_GAP_THRESHOLD

        if gap_threshold is None:
            gap_threshold = DEFAULT_GAP_THRESHOLD

        if not self._cache:
            return QueryResultV2(question=question, results=[],
                                 confidence=0.0, has_match=False)

        encode_q = getattr(self.encoder, "encode_query", None) or self.encoder.encode
        q_vec = encode_q(question)

        if mode == "routed" and self._signatures:
            candidates, _ = self._routed_candidates(q_vec)
            neighbors = self._find_neighbors_among(q_vec, candidates, top_k=top_k)
        else:
            neighbors = self._find_neighbors(q_vec, top_k=top_k)

        sims = [n.similarity for n in neighbors]
        gap, has_match = QueryResultV2.compute_confidence(sims, gap_threshold)

        results = [
            ScoredResult(
                entry_id   = self._cache[n.entry_id].id,
                text       = self._cache[n.entry_id].text,
                source     = self._cache[n.entry_id].source,
                similarity = n.similarity,
                rank       = i + 1,
            )
            for i, n in enumerate(neighbors)
        ]

        return QueryResultV2(
            question   = question,
            results    = results,
            confidence = gap,
            has_match  = has_match,
        )

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
    "KnowledgeLoop", "KnowledgeEntry", "Tombstone", "Neighbor",
    "IngestResult", "QueryResult", "Discovery",
]
