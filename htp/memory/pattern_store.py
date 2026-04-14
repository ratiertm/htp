"""
L3 PatternStore — Online Hebbian EMA + Go-CLS + CA3 pattern completion
=======================================================================

설계 문서: design/htp_memory_design_final.md §4
LeCun 수정: K-Means 배치 → Online Hebbian EMA (뇌는 전체 재계산 안 함)
Go-CLS: count ≥ 3 ∧ snr ≥ 1.5 ∧ swr_tagged → 패턴 승격
CA3: pattern completion (노이즈 입력 → 가까운 centroid 로 수렴)
"""
from __future__ import annotations

import dataclasses
import json
import time
import uuid
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from .types import Episode, Pattern, bytes_to_tensor, tensor_to_bytes


class PatternStore:
    """L3 패턴 메모리 — 신피질 일반화 표현."""

    SIMILARITY_THRESHOLD = 0.75   # 같은 패턴으로 볼 코사인 기준
    GO_CLS_MIN_COUNT     = 3      # 패턴 승격 최소 에피소드 수
    GO_CLS_MIN_SNR       = 1.5    # 패턴 승격 최소 SNR
    CA3_ALPHA            = 0.7    # CA3 completion 혼합 비율 (원본 vs centroid)

    def __init__(self, path: str | Path = "htp_patterns.json"):
        self.path = Path(path)
        self._patterns: dict[str, Pattern] = {}
        self._buffer:   dict[str, list]    = {}   # 새 패턴 후보 버퍼
        self._load()

    # ── CA3: Pattern Completion ──────────────────

    def complete(self, state_vec: torch.Tensor) -> tuple[torch.Tensor, Optional[Pattern]]:
        """
        CA3 pattern completion — 부분/노이즈 입력을 가장 가까운 centroid 로 완성.
        반환: (completed_vec, matched_pattern | None)
        """
        if not self._patterns:
            return state_vec.clone(), None

        best_sim, best_pat, best_centroid = -1.0, None, state_vec.clone()
        for pat in self._patterns.values():
            centroid = bytes_to_tensor(pat.centroid_vec)
            if centroid.shape != state_vec.shape:
                continue
            sim = F.cosine_similarity(
                state_vec.unsqueeze(0), centroid.unsqueeze(0),
            ).item()
            if sim > best_sim:
                best_sim, best_pat, best_centroid = sim, pat, centroid

        if best_sim >= self.SIMILARITY_THRESHOLD and best_pat is not None:
            completed = self.CA3_ALPHA * state_vec + (1 - self.CA3_ALPHA) * best_centroid
            return completed, best_pat
        return state_vec.clone(), None

    def match_confidence(self, state_vec: torch.Tensor) -> float:
        """L3 매칭 신뢰도 — SWR novelty 계산용 (1 - confidence)."""
        _, pat = self.complete(state_vec)
        if pat is None:
            return 0.0
        centroid = bytes_to_tensor(pat.centroid_vec)
        return F.cosine_similarity(
            state_vec.unsqueeze(0), centroid.unsqueeze(0),
        ).item()

    # ── Online Hebbian Consolidation ──────────────

    def consolidate(self, episodes: list[Episode]):
        """
        SWR 태깅된 에피소드만 → 기존 패턴 EMA 업데이트 또는 새 패턴 버퍼에 적재.
        Online 방식 (뇌는 전체 재계산 안 함).
        """
        tagged = [e for e in episodes if e.swr_tagged]
        if not tagged:
            return

        for ep in tagged:
            if not ep.state_vec:
                continue
            vec = bytes_to_tensor(ep.state_vec)
            _, matched = self.complete(vec)
            if matched:
                self._update_pattern(matched, ep, vec)
            else:
                self._create_or_buffer(ep, vec)

        self._save()

    def _update_pattern(self, pat: Pattern, ep: Episode, vec: torch.Tensor):
        """Online Hebbian EMA centroid 업데이트."""
        centroid = bytes_to_tensor(pat.centroid_vec)
        lr = 1.0 / (pat.episode_count + 1)
        new_centroid = (1 - lr) * centroid + lr * vec

        if ep.outcome == "success":
            pat.winner_dist[ep.winner] = pat.winner_dist.get(ep.winner, 0) + 1

        pat.episode_count += 1
        pat.centroid_vec   = tensor_to_bytes(new_centroid)
        if pat.winner_dist:
            pat.best_winner  = max(pat.winner_dist, key=pat.winner_dist.get)
            pat.success_rate = pat.winner_dist[pat.best_winner] / pat.episode_count
        pat.snr        = self._compute_snr([ep.score])
        pat.updated_at = time.time()
        self._patterns[pat.pattern_id] = pat

    # ── 패턴 버퍼 + 승격 ────────────────────────

    def _create_or_buffer(self, ep: Episode, vec: torch.Tensor):
        """Go-CLS 조건 충족 시만 패턴 승격."""
        matched_key = None
        for key, buf_list in self._buffer.items():
            buf_vec = bytes_to_tensor(buf_list[0]["vec"])
            if buf_vec.shape != vec.shape:
                continue
            sim = F.cosine_similarity(
                vec.unsqueeze(0), buf_vec.unsqueeze(0),
            ).item()
            if sim >= self.SIMILARITY_THRESHOLD:
                matched_key = key
                break

        item = {"vec": tensor_to_bytes(vec), "ep": dataclasses.asdict(ep)}
        if matched_key:
            self._buffer[matched_key].append(item)
            if len(self._buffer[matched_key]) >= self.GO_CLS_MIN_COUNT:
                self._promote_to_pattern(matched_key)
        else:
            self._buffer[str(uuid.uuid4())] = [item]

    def _promote_to_pattern(self, key: str):
        items  = self._buffer[key]
        vecs   = [bytes_to_tensor(i["vec"]) for i in items]
        scores = [i["ep"].get("score", 0.0) for i in items]

        snr = self._compute_snr(scores)
        if snr < self.GO_CLS_MIN_SNR:
            del self._buffer[key]
            return

        centroid    = torch.stack(vecs).mean(dim=0)
        winner_dist: dict[str, int] = {}
        for i in items:
            if i["ep"].get("outcome") == "success":
                w = i["ep"].get("winner", "")
                if w:
                    winner_dist[w] = winner_dist.get(w, 0) + 1

        if not winner_dist:
            del self._buffer[key]
            return

        best_winner  = max(winner_dist, key=winner_dist.get)
        success_rate = winner_dist[best_winner] / len(items)

        pat = Pattern(
            pattern_id    = str(uuid.uuid4()),
            centroid_vec  = tensor_to_bytes(centroid),
            best_winner   = best_winner,
            success_rate  = success_rate,
            episode_count = len(items),
            winner_dist   = winner_dist,
            snr           = snr,
            generalize_ok = True,
        )
        self._patterns[pat.pattern_id] = pat
        del self._buffer[key]

    # ── SNR ──────────────────────────────────────

    @staticmethod
    def _compute_snr(scores: list[float]) -> float:
        if len(scores) < 2:
            return 0.0
        t = torch.tensor(scores, dtype=torch.float32)
        mu  = t.mean().item()
        sig = t.std().item() + 1e-8
        return mu / sig

    # ── 조회 ─────────────────────────────────────

    def pattern_count(self) -> int:
        return len(self._patterns)

    def buffer_count(self) -> int:
        return len(self._buffer)

    # ── I/O ──────────────────────────────────────

    def _load(self):
        try:
            with self.path.open() as f:
                data = json.load(f)
        except FileNotFoundError:
            self._patterns = {}
            self._buffer   = {}
            return

        self._patterns = {}
        for k, v in data.get("patterns", {}).items():
            v["centroid_vec"] = bytes.fromhex(v["centroid_vec"]) if isinstance(v.get("centroid_vec"), str) else v.get("centroid_vec", b"")
            self._patterns[k] = Pattern(**v)
        self._buffer = data.get("buffer", {})
        # buffer 안의 vec 를 bytes 로 복원
        for key, items in self._buffer.items():
            for i in items:
                if isinstance(i.get("vec"), str):
                    i["vec"] = bytes.fromhex(i["vec"])

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)

        def _pat_to_dict(p: Pattern) -> dict:
            d = dataclasses.asdict(p)
            d["centroid_vec"] = p.centroid_vec.hex() if isinstance(p.centroid_vec, bytes) else p.centroid_vec
            return d

        buffer_serializable: dict[str, list[dict]] = {}
        for key, items in self._buffer.items():
            buffer_serializable[key] = []
            for i in items:
                item = {"ep": i["ep"]}
                v = i["vec"]
                item["vec"] = v.hex() if isinstance(v, bytes) else v
                buffer_serializable[key].append(item)

        with self.path.open("w") as f:
            json.dump({
                "patterns": {k: _pat_to_dict(v) for k, v in self._patterns.items()},
                "buffer":   buffer_serializable,
            }, f)
