# HTP Memory System — 최종 설계 (LeCun 검토 반영)

## 수정사항 요약

| 항목 | 기존 | 수정 (LeCun 반영) | 근거 논문 |
|------|------|------|------|
| state_vec 차원 | 8-dim | 64-dim | 해마 place cell sparse 표현 |
| L3 클러스터링 | K-Means 배치 | Online Hebbian EMA | Hopfield / Sparse Distributed Memory |
| SWR 공식 | score + recall_count | novelty × reward | Yang & Buzsáki Science 2024 |
| recall 구조 | L3 필터→L2 검색 | CA3 완성→CA1 불일치 감지 | CA3-CA1 pattern completion/separation |
| consolidation 트리거 | 50 에피소드마다 | CUSUM overload 발생 시 | CUSUM = 피질 과부하 = 수면 필요 신호 |

---

## 1. state_vec 차원 확장

### 수정: thalamus.py

```python
# 기존
compress_dim: int = 8

# 수정 — 해마 place cell은 고차원 sparse 표현 사용
# 8-dim: 코사인 유사도 평균 0.3~0.5 → 검색 의미 없음
# 64-dim: 충분한 표현 공간 확보
compress_dim: int = 64
```

**수학적 근거 (JL Lemma):**
```
k = O(log(n) / ε²)
n = 에피소드 수 (목표 ~1000개)
ε = 0.1 허용 오차
→ k ≈ 64 적절
```

---

## 2. 데이터 구조

### types.py

```python
from __future__ import annotations
import math
import uuid
import time
from dataclasses import dataclass, field
from typing import Any
import torch


@dataclass
class Episode:
    """L2 에피소드 메모리 단위 — 해마 단기 에피소드 인코딩"""
    episode_id:   str          # UUID
    step:         int          # BrainRuntime._step
    winner:       str          # 이긴 Region 이름
    action_type:  str          # "execute" | "inhibit"
    score:        float        # PFC combined score (= reward 신호)
    state_vec:    bytes        # 64-dim Tensor → blob
    context:      str          # 입력 요약 50자
    outcome:      str = None   # "success" | "fail" (사후 평가)
    recall_count: int = 0      # CA1 재활성화 횟수
    novelty:      float = 1.0  # SWR 태깅용 — L3 mismatch 거리
    swr_tagged:   bool = False # consolidation 대상 여부
    session_id:   str = ""
    timestamp:    float = field(default_factory=time.time)


@dataclass
class Pattern:
    """L3 패턴 메모리 — 신피질 일반화 표현"""
    pattern_id:    str
    centroid_vec:  bytes       # 64-dim EMA 중심
    best_winner:   str         # 가장 성공한 Region
    success_rate:  float
    episode_count: int
    winner_dist:   dict        # {region: count}
    snr:           float       # Go-CLS 패턴 신뢰도 μ/σ
    generalize_ok: bool        # 일반화 조건 통과 (≥3 에피, snr≥1.5)
    updated_at:    float = field(default_factory=time.time)


@dataclass
class MemoryContext:
    """recall() 반환값 — CA3-CA1 양방향 처리 결과"""
    # CA3: pattern completion 결과
    completed_vec:   torch.Tensor    # CA3가 복원한 완성 벡터
    mismatch:        float           # CA1 불일치 거리

    # CA1: 가치 기반 선택
    candidates:      list            # 검색된 유사 에피소드
    recommendation:  str | None      # best_winner 추천
    confidence:      float           # 예측 신뢰도

    # 메타
    pattern:         Pattern | None  # 매칭된 L3 패턴
    is_novel:        bool            # 새로운 상황 여부
```

---

## 3. L2 — EpisodeStore (SWR 태깅 수정)

### episode_store.py

```python
import sqlite3
import uuid
import math
import torch
import torch.nn.functional as F
from .types import Episode

SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes (
    episode_id   TEXT PRIMARY KEY,
    step         INTEGER,
    winner       TEXT,
    action_type  TEXT,
    score        REAL,
    state_vec    BLOB,         -- 64-dim float32
    context      TEXT,
    outcome      TEXT,
    recall_count INTEGER DEFAULT 0,
    novelty      REAL    DEFAULT 1.0,
    swr_tagged   BOOLEAN DEFAULT FALSE,
    session_id   TEXT,
    timestamp    REAL
);
CREATE INDEX IF NOT EXISTS idx_winner    ON episodes(winner);
CREATE INDEX IF NOT EXISTS idx_swr       ON episodes(swr_tagged);
CREATE INDEX IF NOT EXISTS idx_timestamp ON episodes(timestamp);
CREATE INDEX IF NOT EXISTS idx_outcome   ON episodes(outcome);
"""

class EpisodeStore:
    def __init__(self, db_path: str = "htp_memory.db"):
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.executescript(SCHEMA)

    def save(self, ep: Episode) -> str:
        ep.episode_id = ep.episode_id or str(uuid.uuid4())
        self._conn.execute("""
            INSERT INTO episodes VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            ep.episode_id, ep.step, ep.winner, ep.action_type,
            ep.score, ep.state_vec, ep.context, ep.outcome,
            ep.recall_count, ep.novelty, ep.swr_tagged,
            ep.session_id, ep.timestamp
        ))
        self._conn.commit()
        return ep.episode_id

    def feedback(self, episode_id: str, outcome: str):
        self._conn.execute(
            "UPDATE episodes SET outcome=? WHERE episode_id=?",
            (outcome, episode_id)
        )
        self._conn.commit()

    def increment_recall(self, episode_id: str):
        """CA1 재활성화 기록"""
        self._conn.execute(
            "UPDATE episodes SET recall_count = recall_count + 1 WHERE episode_id=?",
            (episode_id,)
        )
        self._conn.commit()

    def tag_swr(self):
        """
        SWR 태깅 — LeCun 수정: novelty × reward
        
        생물학 (Yang & Buzsáki Science 2024):
          - 보상 소비 시 SWR 발생
          - 새롭고(novelty) 보상이 높은(reward) 경험 선택
          - 이미 자주 본 것은 적응(adaptation)으로 태깅 약화
        
        수학:
          priority = novelty × score
          (곱셈: 둘 다 높아야 태깅 — AND 조건)
          swr_tagged = priority >= 0.5
        """
        rows = self._conn.execute(
            "SELECT episode_id, score, novelty FROM episodes WHERE outcome IS NOT NULL"
        ).fetchall()

        for ep_id, score, novelty in rows:
            priority   = novelty * score          # novelty × reward
            swr_tagged = priority >= 0.5
            self._conn.execute(
                "UPDATE episodes SET swr_tagged=? WHERE episode_id=?",
                (swr_tagged, ep_id)
            )
        self._conn.commit()

    def search_similar(
        self,
        state_vec: torch.Tensor,
        top_k: int = 5,
        winner_filter: str = None,
        swr_only: bool = False,
    ) -> list[Episode]:
        """
        64-dim 코사인 유사도 기반 검색
        winner_filter: CA3 prior 적용 시 특정 winner로 필터
        swr_only: consolidation 후보 검색 시
        """
        query = "SELECT * FROM episodes WHERE 1=1"
        params = []
        if winner_filter:
            query += " AND winner=?"
            params.append(winner_filter)
        if swr_only:
            query += " AND swr_tagged=TRUE"

        rows = self._conn.execute(query, params).fetchall()
        if not rows:
            return []

        scored = []
        for row in rows:
            vec = torch.frombuffer(row[5], dtype=torch.float32).clone()
            if vec.shape[0] != state_vec.shape[0]:
                continue
            sim = F.cosine_similarity(
                state_vec.unsqueeze(0), vec.unsqueeze(0)
            ).item()
            scored.append((sim, row))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [self._row_to_episode(r) for _, r in scored[:top_k]]

    def recent(self, n: int = 100) -> list[Episode]:
        rows = self._conn.execute(
            "SELECT * FROM episodes ORDER BY timestamp DESC LIMIT ?", (n,)
        ).fetchall()
        return [self._row_to_episode(r) for r in rows]

    def _row_to_episode(self, row) -> Episode:
        return Episode(
            episode_id=row[0], step=row[1], winner=row[2],
            action_type=row[3], score=row[4], state_vec=row[5],
            context=row[6], outcome=row[7], recall_count=row[8],
            novelty=row[9], swr_tagged=bool(row[10]),
            session_id=row[11], timestamp=row[12]
        )
```

---

## 4. L3 — PatternStore (Online Hebbian + Go-CLS)

### pattern_store.py

```python
import json
import math
import uuid
import time
import torch
import torch.nn.functional as F
from .types import Pattern, Episode

class PatternStore:
    """
    L3 패턴 메모리 — 신피질 일반화 표현
    
    LeCun 수정:
      K-Means(배치) → Online Hebbian EMA(스트리밍)
      이유: 뇌는 전체 재계산 안 함 — attractor 기반 점진적 업데이트
    
    Go-CLS 조건:
      유사 에피소드 ≥ 3 AND snr ≥ 1.5 AND swr_tagged
      → 일반화에 도움될 때만 consolidate
    
    CA3 역할:
      pattern completion — 부분 입력(state_vec)으로 centroid 복원
    """

    SIMILARITY_THRESHOLD = 0.75  # 같은 패턴으로 볼 거리

    def __init__(self, path: str = "htp_patterns.json"):
        self.path = path
        self._patterns: dict[str, Pattern] = {}
        self._load()

    # ── CA3: Pattern Completion ─────────────────────────────

    def complete(self, state_vec: torch.Tensor) -> tuple[torch.Tensor, Pattern | None]:
        """
        CA3 pattern completion:
        부분/노이즈 있는 입력 → 가장 가까운 centroid로 완성
        
        반환: (completed_vec, matched_pattern | None)
        """
        if not self._patterns:
            return state_vec.clone(), None

        best_sim   = -1.0
        best_pat   = None
        best_centroid = state_vec.clone()

        for pat in self._patterns.values():
            centroid = torch.frombuffer(pat.centroid_vec, dtype=torch.float32).clone()
            if centroid.shape != state_vec.shape:
                continue
            sim = F.cosine_similarity(
                state_vec.unsqueeze(0), centroid.unsqueeze(0)
            ).item()
            if sim > best_sim:
                best_sim      = sim
                best_pat      = pat
                best_centroid = centroid

        if best_sim >= self.SIMILARITY_THRESHOLD:
            # 알려진 패턴으로 완성 (attractor로 수렴)
            alpha   = 0.7   # 원본 vs 패턴 혼합 비율
            completed = alpha * state_vec + (1 - alpha) * best_centroid
            return completed, best_pat
        else:
            # 새로운 패턴 — 원본 그대로
            return state_vec.clone(), None

    def match_confidence(self, state_vec: torch.Tensor) -> float:
        """L3 매칭 신뢰도 — SWR novelty 계산용"""
        _, pat = self.complete(state_vec)
        if pat is None:
            return 0.0   # 완전 새로움 → novelty = 1.0
        centroid = torch.frombuffer(pat.centroid_vec, dtype=torch.float32).clone()
        return F.cosine_similarity(
            state_vec.unsqueeze(0), centroid.unsqueeze(0)
        ).item()

    # ── Online Hebbian Consolidation ───────────────────────

    def consolidate(self, episodes: list[Episode]):
        """
        LeCun 수정: Online Hebbian EMA — 배치 K-Means 대신
        
        처리 순서:
          1. SWR 태깅된 것만
          2. 기존 패턴에 흡수 또는 새 패턴 생성
          3. EMA로 centroid 업데이트 (online)
          4. Go-CLS 조건: snr ≥ 1.5 AND count ≥ 3
        """
        tagged = [e for e in episodes if e.swr_tagged]
        if not tagged:
            return

        for ep in tagged:
            vec = torch.frombuffer(ep.state_vec, dtype=torch.float32).clone()
            _, matched = self.complete(vec)

            if matched:
                # 기존 패턴 EMA 업데이트 (Hebbian: 함께 활성화 → 강화)
                self._update_pattern(matched, ep, vec)
            else:
                # 새 패턴 후보 생성
                self._create_or_buffer(ep, vec)

        self._save()

    def _update_pattern(self, pat: Pattern, ep: Episode, vec: torch.Tensor):
        """Online Hebbian EMA centroid 업데이트"""
        centroid = torch.frombuffer(pat.centroid_vec, dtype=torch.float32).clone()

        # EMA: 새 에피소드를 centroid에 점진적 반영
        lr = 1.0 / (pat.episode_count + 1)  # 에피소드 많을수록 보수적
        new_centroid = (1 - lr) * centroid + lr * vec

        # winner 분포 업데이트
        if ep.outcome == "success":
            pat.winner_dist[ep.winner] = pat.winner_dist.get(ep.winner, 0) + 1

        pat.episode_count += 1
        pat.centroid_vec   = new_centroid.numpy().tobytes()
        pat.best_winner    = max(pat.winner_dist, key=pat.winner_dist.get)
        pat.success_rate   = pat.winner_dist.get(pat.best_winner, 0) / pat.episode_count
        pat.updated_at     = time.time()

        # SNR 재계산
        scores = [ep.score]  # 간단히 현재 score로
        pat.snr = self._compute_snr(scores)

        self._patterns[pat.pattern_id] = pat

    # ── 버퍼: 새 패턴 후보 관리 ───────────────────────────

    def __init__(self, path="htp_patterns.json"):
        self.path = path
        self._patterns: dict[str, Pattern] = {}
        self._buffer:   dict[str, list]    = {}  # 새 패턴 후보 버퍼
        self._load()

    def _create_or_buffer(self, ep: Episode, vec: torch.Tensor):
        """
        Go-CLS 조건 충족 시만 패턴 승격
        조건: 유사 에피소드 ≥ 3 AND snr ≥ 1.5
        """
        # 버퍼에서 유사한 후보 찾기
        matched_key = None
        for key, buf_list in self._buffer.items():
            buf_vec = torch.frombuffer(buf_list[0]["vec"], dtype=torch.float32).clone()
            sim = F.cosine_similarity(vec.unsqueeze(0), buf_vec.unsqueeze(0)).item()
            if sim >= self.SIMILARITY_THRESHOLD:
                matched_key = key
                break

        if matched_key:
            self._buffer[matched_key].append({"vec": vec.numpy().tobytes(), "ep": ep})
        else:
            new_key = str(uuid.uuid4())
            self._buffer[new_key] = [{"vec": vec.numpy().tobytes(), "ep": ep}]

        # Go-CLS 조건 체크: ≥ 3개면 패턴 승격
        if matched_key and len(self._buffer[matched_key]) >= 3:
            self._promote_to_pattern(matched_key)

    def _promote_to_pattern(self, key: str):
        """버퍼 → L3 패턴 승격 (Go-CLS 조건 통과)"""
        items  = self._buffer[key]
        vecs   = [torch.frombuffer(i["vec"], dtype=torch.float32).clone() for i in items]
        eps    = [i["ep"] for i in items]
        scores = [e.score for e in eps]

        snr = self._compute_snr(scores)
        if snr < 1.5:
            del self._buffer[key]
            return  # 노이즈 많은 패턴 제외

        centroid     = torch.stack(vecs).mean(dim=0)
        winner_dist  = {}
        for e in eps:
            if e.outcome == "success":
                winner_dist[e.winner] = winner_dist.get(e.winner, 0) + 1

        if not winner_dist:
            del self._buffer[key]
            return

        best_winner  = max(winner_dist, key=winner_dist.get)
        success_rate = winner_dist[best_winner] / len(eps)

        pattern = Pattern(
            pattern_id    = str(uuid.uuid4()),
            centroid_vec  = centroid.numpy().tobytes(),
            best_winner   = best_winner,
            success_rate  = success_rate,
            episode_count = len(eps),
            winner_dist   = winner_dist,
            snr           = snr,
            generalize_ok = True,
        )
        self._patterns[pattern.pattern_id] = pattern
        del self._buffer[key]

    @staticmethod
    def _compute_snr(scores: list[float]) -> float:
        """Go-CLS SNR = μ / σ — 패턴 신뢰도"""
        if len(scores) < 2:
            return 0.0
        t = torch.tensor(scores)
        mu  = t.mean().item()
        sig = t.std().item() + 1e-8
        return mu / sig

    def _load(self):
        try:
            with open(self.path) as f:
                data = json.load(f)
            self._patterns = {
                k: Pattern(**v) for k, v in data.get("patterns", {}).items()
            }
            self._buffer = data.get("buffer", {})
        except FileNotFoundError:
            self._patterns = {}
            self._buffer   = {}

    def _save(self):
        import dataclasses
        with open(self.path, "w") as f:
            json.dump({
                "patterns": {
                    k: dataclasses.asdict(v)
                    for k, v in self._patterns.items()
                },
                "buffer": self._buffer,
            }, f, default=str)
```

---

## 5. MemorySystem — CA3-CA1 양방향 recall

### memory_system.py

```python
import math
import uuid
import time
import torch
import torch.nn.functional as F
from .types import Episode, MemoryContext
from .episode_store import EpisodeStore
from .pattern_store import PatternStore


class MemorySystem:
    """
    HTP 통합 메모리 시스템
    
    L1: PFCRuntime.working_memory (deque[7]) — 세션 내, 이미 구현됨
    L2: EpisodeStore (SQLite) — 세션 간 에피소드
    L3: PatternStore (JSON)   — 장기 패턴 일반화
    
    LeCun 수정사항 모두 반영:
      1. state_vec 64-dim
      2. Online Hebbian EMA (K-Means 대신)
      3. SWR = novelty × reward
      4. CA3 완성 → CA1 불일치 감지
      5. CUSUM overload → consolidation 트리거
    """

    CA1_MISMATCH_THRESHOLD = 0.3  # CA1 불일치 임계값

    def __init__(
        self,
        db_path:      str   = "htp_memory.db",
        pattern_path: str   = "htp_patterns.json",
        session_id:   str   = None,
    ):
        self.l2         = EpisodeStore(db_path)
        self.l3         = PatternStore(pattern_path)
        self.session_id = session_id or str(uuid.uuid4())
        self._last_episode_id: str | None = None

    # ── 저장 ──────────────────────────────────────────────

    def save(
        self,
        thal_out,           # ThalamusOutput
        action,             # Action
        context: str,
        score:   float,
    ) -> str:
        """
        에피소드 저장 + SWR novelty 계산
        
        novelty = 1.0 - L3 매칭 신뢰도
        → L3에 패턴 없을수록 새롭고 태깅 우선순위 높음
        """
        state_vec = thal_out.state_vec  # 64-dim Tensor
        novelty   = 1.0 - self.l3.match_confidence(state_vec)

        ep = Episode(
            step        = thal_out.step,
            winner      = action.winner,
            action_type = action.type,
            score       = score,
            state_vec   = state_vec.numpy().tobytes(),
            context     = context[:50],
            novelty     = novelty,
            session_id  = self.session_id,
        )
        ep_id = self.l2.save(ep)
        self._last_episode_id = ep_id
        return ep_id

    # ── CA3-CA1 recall ─────────────────────────────────────

    def recall(self, state_vec: torch.Tensor) -> MemoryContext:
        """
        CA3-CA1 양방향 recall (LeCun 수정)
        
        CA3: pattern completion
          부분/노이즈 입력 → 가장 가까운 L3 centroid로 완성
          (Hopfield attractor 수렴)
        
        CA1: mismatch detection
          원본 vs CA3 완성본 불일치 측정
          - 불일치 작음 → 알려진 상황 → L3 추천 사용
          - 불일치 큼   → 새로운 상황 → L2 직접 탐색
        """
        # ── CA3: Pattern Completion ──────────────────────
        completed_vec, pattern = self.l3.complete(state_vec)

        # ── CA1: Mismatch Detection ──────────────────────
        mismatch = (state_vec - completed_vec).norm().item()

        if mismatch < self.CA1_MISMATCH_THRESHOLD:
            # 알려진 패턴 → L3 prior로 L2 필터
            # (CA3 완성본의 winner를 prior로 사용)
            candidates = self.l2.search_similar(
                state_vec,
                top_k       = 10,
                winner_filter = pattern.best_winner if pattern else None,
            )
        else:
            # 새로운 상황 → 필터 없이 L2 전체 탐색
            candidates = self.l2.search_similar(state_vec, top_k=5)

        # ── CA1: 가치 기반 선택 ──────────────────────────
        # CA1 가치 함수: score × log(recall_count + 2)
        # 생물학: CA1이 보상 정보(score) + 재활성화 빈도로 가치 평가
        recommendation = None
        confidence     = 0.0

        if candidates:
            for ep in candidates:
                self.l2.increment_recall(ep.episode_id)

            best = max(
                candidates,
                key=lambda e: e.score * math.log(e.recall_count + 2)
            )
            recommendation = best.winner
            confidence     = best.score

        return MemoryContext(
            completed_vec  = completed_vec,
            mismatch       = mismatch,
            candidates     = candidates,
            recommendation = recommendation,
            confidence     = confidence,
            pattern        = pattern,
            is_novel       = mismatch >= self.CA1_MISMATCH_THRESHOLD,
        )

    # ── CUSUM 기반 consolidation 트리거 ───────────────────

    def on_overload(self, region_id: str):
        """
        LeCun 수정: CUSUM overload = 수면 신호 = consolidation 트리거
        
        생물학:
          뇌가 과부하(피로) → 수면 필요
          수면 중 SWR → 해마→신피질 consolidation
          HTP: Region CUSUM 임계값 초과 → consolidation 발동
        """
        # SWR 태깅 먼저 (보상×신규성 기준)
        self.l2.tag_swr()

        # Go-CLS + Online Hebbian consolidation
        tagged_episodes = self.l2.search_similar(
            torch.zeros(64),  # 전체 검색
            top_k    = 200,
            swr_only = True,
        )
        self.l3.consolidate(tagged_episodes)

    # ── 사후 평가 ─────────────────────────────────────────

    def feedback(self, outcome: str, episode_id: str = None):
        """행동 결과를 에피소드에 기록 — L3 패턴 개선에 반영"""
        ep_id = episode_id or self._last_episode_id
        if ep_id:
            self.l2.feedback(ep_id, outcome)
```

---

## 6. BrainRuntime 연동 (최소 변경)

### brain_runtime.py 수정 부분

```python
from ..memory.memory_system import MemorySystem
from ..thalamus.top_down import TopDownSignal

class BrainRuntime:
    def __init__(self, pfc_config=None):
        ...
        self.memory = MemorySystem()        # ① 추가

    def run(self, data: Any) -> Action:
        self._ensure_thalamus()
        self._step += 1

        # ② recall: 이전 state_vec로 기억 조회
        mem_ctx = None
        if self._step > 1 and hasattr(self, '_last_state_vec'):
            mem_ctx = self.memory.recall(self._last_state_vec)

            # ③ CA1 추천을 top-down hint로 주입
            if mem_ctx.recommendation and not mem_ctx.is_novel:
                self._last_td = self._inject_memory_hint(
                    self._last_td,
                    mem_ctx.recommendation,
                    mem_ctx.confidence,
                )

        # 기존 Region 실행
        for region in self.regions.values():
            if region._nodes:
                try:
                    region.run(data)
                except Exception as e:
                    print(f"  [warn] {region.region_name}: {e}")

        thal_out      = self.thalamus.step(data, top_down=self._last_td)
        action, td    = self.pfc.decide(thal_out, regions=self.regions)
        self._last_td = td

        # ④ state_vec 보존 (다음 스텝 recall용)
        self._last_state_vec = thal_out.state_vec.clone()

        # ⑤ 에피소드 저장
        score = self._extract_score(action)
        self.memory.save(thal_out, action, str(data)[:50], score)

        # ⑥ CUSUM overload → consolidation 트리거 (수면 메커니즘)
        for name, region in self.regions.items():
            if region._cusum_S > region._cusum_h:
                self.memory.on_overload(name)
                region._cusum_S = 0.0   # 수면 후 초기화

        # 기존 억제/CC 처리
        for rid, strength in thal_out.suppressed.items():
            if rid in self.regions and strength > 0:
                self.regions[rid].apply_suppression(strength)
        if self._cc:
            self._cc.apply(thal_out)

        winner_region = self.regions.get(action.winner)
        if winner_region:
            last = getattr(winner_region, "_last_result", None)
            if last:
                action.result = last.outputs.get(action.winner)

        return action

    def _inject_memory_hint(self, td, recommended: str, confidence: float):
        """CA1 추천 → Thalamus top-down bias 주입"""
        bias_strength = confidence * 0.5  # 신뢰도에 비례한 힌트
        if td is None:
            return TopDownSignal(
                biases   = {recommended: bias_strength},
                strength = 0.3,
                step     = self._step,
            )
        td.biases[recommended] = max(
            td.biases.get(recommended, 0.0),
            bias_strength
        )
        return td

    def _extract_score(self, action) -> float:
        """PFC score 추출 — reason 문자열에서 파싱"""
        try:
            # "score=0.723 (cos=..." 형식에서 파싱
            part = action.reason.split("score=")[1].split(" ")[0]
            return float(part)
        except Exception:
            return 0.5
```

---

## 7. 전체 데이터 흐름

```
step N:
  ┌─────────────────────────────────────────────────────────┐
  │ RECALL (이전 state_vec 기반)                             │
  │                                                          │
  │  CA3: L3.complete(prev_state_vec)                        │
  │       → completed_vec + matched_pattern                  │
  │                                                          │
  │  CA1: mismatch = ||prev - completed||                    │
  │       < 0.3 → 알려진 상황 → L3 winner로 L2 필터         │
  │       ≥ 0.3 → 새로운 상황 → L2 전체 탐색                │
  │                                                          │
  │  CA1 가치: best = argmax(score × log(recall+2))          │
  │  → recommendation → top-down hint 주입                   │
  └─────────────────────────────────────────────────────────┘
          ↓
  Thalamus.step(top_down = td + memory_hint)
          ↓
  PFC.decide() → action + score
          ↓
  ┌─────────────────────────────────────────────────────────┐
  │ SAVE                                                     │
  │  novelty = 1.0 - L3.match_confidence(state_vec)         │
  │  L2.save(Episode(score=reward, novelty=novelty))         │
  └─────────────────────────────────────────────────────────┘
          ↓
  ┌─────────────────────────────────────────────────────────┐
  │ CONSOLIDATION (CUSUM overload 시)                        │
  │                                                          │
  │  SWR 태깅: priority = novelty × score ≥ 0.5            │
  │                                                          │
  │  Go-CLS + Online Hebbian:                               │
  │    유사 그룹 ≥ 3 AND snr ≥ 1.5 → L3 패턴 승격           │
  │    기존 패턴: EMA centroid 업데이트                       │
  │                                                          │
  │  Region._cusum_S = 0.0 (수면 후 초기화)                  │
  └─────────────────────────────────────────────────────────┘
```

---

## 8. 수학 요약

```
state_vec 차원:    64-dim  (JL k ≈ log(1000)/0.1² ≈ 64)

SWR 태깅:          priority  = novelty × score
                   novelty   = 1.0 - L3.match_confidence
                   tagged    = priority ≥ 0.5

CA3 완성:          completed = α×input + (1-α)×centroid  (α=0.7)
                   EMA lr    = 1 / (episode_count + 1)

CA1 불일치:        mismatch  = ||input - completed||₂
                   novel     = mismatch ≥ 0.3

CA1 가치함수:      value     = score × log(recall_count + 2)

Go-CLS SNR:        snr       = μ(scores) / σ(scores)
승격 조건:         count ≥ 3 AND snr ≥ 1.5 AND swr_tagged

consolidation:     CUSUM_S > CUSUM_h → on_overload() 발동
```

---

## 9. 파일 구조

```
htp/memory/
  __init__.py
  types.py          # Episode, Pattern, MemoryContext
  episode_store.py  # L2: SQLite + SWR 태깅
  pattern_store.py  # L3: Online Hebbian + Go-CLS + CA3
  memory_system.py  # CA3-CA1 통합 + CUSUM 트리거
```

---

## 10. thalamus.py 수정 (compress_dim)

```python
# Thalamus.__init__
compress_dim: int = 64   # 8 → 64 변경
```

---

*LeCun 검토 4가지 모두 반영 완료*
*참고: Yang & Buzsáki Science 2024, Go-CLS Nature Neuroscience 2023,*
*Hopfield Networks, Sparse Distributed Memory (Kanerva 1988)*
EOF