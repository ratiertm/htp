---
template: design
version: 1.3
feature: htp-thalamus-car
date: 2026-05-17
author: Mindbuild
project: HTP (Hub Topology Programming)
status: Draft
---

# htp-thalamus-car Design Document

> **Summary**: 설계서 v4 (Rev 1.3) 9 Stage 를 **6개 Phase Sub-Cycles** 로 묶어 진행. Stage 0.5 TextEncoder MVP 는 `sklearn TfidfVectorizer + GaussianRandomProjection`, knowledge_log 는 `.htp/knowledge_log.jsonl` 로 영속화. 매 sub-cycle 종료 시 Match Rate 90% 강제.
>
> **Project**: HTP
> **Version**: post-`201f0f2`
> **Author**: Mindbuild
> **Date**: 2026-05-17
> **Status**: Draft
> **Planning Doc**: [htp-thalamus-car.plan.md](../../01-plan/features/htp-thalamus-car.plan.md) (Rev 0.2)
> **선행 설계서**: `htp_thalamus_car_design v4.md` (Rev 1.3)

---

## Context Anchor

> Copied from Plan document (Rev 0.2).

| Key | Value |
|-----|-------|
| **WHY** | 태그 매칭이 HTP 4대 원칙과 모순. Phase 3-4 확장 전 토대 정리 + Stage 7까지 가서 "실제로는 다르게 동작해야 했다" 발견 회피 |
| **WHO** | HTP 개발자 본인 + Stage 0.5 이후 본인이 "매일 쓰는 도구" 사용자 |
| **RISK** | (1) 회귀 깨짐 (2) CoherenceGate O(N²) (3) RegionSignature 냉시작 (4) EmbeddingBridge sLLM 의존 (5) vec↔prompt 품질 (6) **MVP TF-IDF 가 cross-domain similarity 못 찾을 위험 — 조기 발견 자체가 가치** |
| **SUCCESS** | Stage 0.5 cross-domain 발견 시나리오 Go + 회귀 57+46 + 본선 27-30 + 실험 4 + isolated-50% + throughput 1.5× + tag↔vector 동등 또는 우위 |
| **SCOPE** | Stage 0-7 (9 Stage, 6 sub-cycles 로 묶음). Stage 6은 `experiment/embedding-bridge` 브랜치 |

---

## 1. Overview

### 1.1 Design Goals

1. 9 Stage 를 **6 sub-cycle** 로 의미 단위로 묶어, 매 sub-cycle 종료 시 Match Rate 90% 강제 — 검증 빈도 6배 증가
2. **Stage 0.5 Knowledge Loop MVP 를 단일 sub-cycle 의 *마지막* 단계로 배치** — 토대(Stage 0) 위에 루프(Stage 0.5)를 닫고 sub-cycle 1 완료
3. **TextEncoder Protocol 단일 정의** (`htp/knowledge/encoder.py`) — `KnowledgeLoop`/`LLMRegion`/`RegionSignature` 모두 이 Protocol 사용. Stage 6 EmbeddingBridge 완성 시 단일 구현 교체로 전 시스템 품질 동시 향상
4. **회귀 보호 유지** — 매 sub-cycle 직후 회귀 57 + 이전 사이클 unit 46 = 103 + 신규 누적 통과
5. **DAG 확장** — `htp/knowledge/` 도 `htp/runtime/` 미참조

### 1.2 Design Principles

- **토대 먼저** — Config (Stage 0) 가 sub-cycle 1의 진입점
- **루프를 먼저 닫는다** (v4 신규 원칙) — Stage 0.5 가 sub-cycle 1의 마무리, 이후 sub-cycle 의 검증 토대
- **본선/실험 분리** — Stage 6 는 별도 브랜치, sub-cycle 5 의 Go/No-Go 통과 시에만 본선 머지
- **각 sub-cycle Match Rate 90%** — 통과 못 하면 iterate, 다음 sub-cycle 진입 금지
- **TextEncoder 인터페이스 격리** — 단일 Protocol, 구현 교체 가능

### 1.3 Selected Strategy

**옵션 C — Phase Sub-Cycles (6 sub-cycles)** + **옵션 α — sklearn TF-IDF + GaussianRandomProjection** + **JSON 파일 저장**

### 1.4 sub-cycle 매핑

| sub-cycle | 포함 Stage | 의미 | 누적 테스트 |
|-----------|-----------|------|-----------|
| **sub-1** | Stage 0 + 0.5 | 토대 + 루프 폐쇄 | 65 |
| **sub-2** | Stage 1 + 2 | Vector routing 도입 + Hybrid 검증 | 78 |
| **sub-3** | Stage 3 | CoherenceGate + Memory 연동 | 84 |
| **sub-4** | Stage 4 + 5 | LLMRegion + Pipeline | 89 |
| **sub-5** | Stage 6 (실험 브랜치) | EmbeddingBridge | 별도 4 |
| **sub-6** | Stage 7 | vector default 전환 | 89 재실행 |

각 sub-cycle 은 자체 plan-design-do-check-report 사이클을 진행 (PDCA mini-cycle). 본 Design 문서는 **sub-1 의 Design 도 포함**하고, sub-2~sub-6 은 sub-1 완료 후 각각의 sub-cycle Design 문서를 별도 생성한다.

---

## 2. Architecture Overview

### 2.1 sub-1 Target Module Structure (Stage 0 + 0.5)

```
htp/
├── __init__.py                       (변경 없음 — 공개 API 표면)
├── core/
│   └── config.py                     +RoutingConfig/CoherenceConfig/LLMBridgeConfig/PipelineConfig
│                                     (Stage 0, 이전 사이클 facade 확장)
├── knowledge/                        ★ NEW (Stage 0.5)
│   ├── __init__.py                   public exports
│   ├── encoder.py                    TextEncoder Protocol + TfidfJLEncoder 구현
│   ├── loop.py                       KnowledgeLoop + dataclass
│   ├── persistence.py                JSONL append/load 헬퍼
│   └── __main__.py                   CLI dispatcher (argparse)
└── ... (나머지 변경 없음)

tests/
├── knowledge/                        ★ NEW (Stage 0.5)
│   ├── __init__.py
│   └── test_loop.py                  5 tests
└── ... (나머지 변경 없음)

.htp/                                 ★ NEW (Stage 0.5 런타임 디렉토리)
└── knowledge_log.jsonl               appendable knowledge entries
```

### 2.2 Dependency Direction (DAG 확장)

```
htp/__init__.py
    │
    └── htp/runtime/* ──→ htp/core/*  + htp/knowledge/*  + ...
                            ↑           ↑
                           torch     sklearn (TfidfVectorizer,
                                              GaussianRandomProjection)

규칙:
  - htp/knowledge/*.py  ∈ {sklearn, numpy, dataclasses, json, pathlib} 만 import
  - htp/knowledge/*.py 는 htp/runtime, htp/thalamus, htp/memory 미참조
  - htp/runtime/*.py 가 htp/knowledge/* import 가능 (단방향)
  - htp/thalamus/signature.py (Stage 1) 가 htp/knowledge/encoder.py 의 TextEncoder Protocol import (단방향)
```

`tests/unit/test_no_circular_deps.py` (이전 사이클) 의 검사 범위를 `htp/knowledge/`로 확장.

---

## 3. Stage 0 Detailed Design — HTPConfig sub-config 분리

### 3.1 신규 sub-config 정의

```python
# htp/core/config.py (확장)
@dataclass
class RoutingConfig:
    """Content-Addressable Routing 설정"""
    mode:            str   = "tag"         # "tag" | "vector" | "hybrid"
    alpha:           float = 0.5           # hybrid 모드의 vector 비중
    threshold_beta:  float = 0.5           # θ = μ + β×σ
    warmup_steps:    int   = 10            # signature 냉시작 보호 (Region 호출 횟수)

@dataclass
class CoherenceConfig:
    """CoherenceGate 설정 (sub-3 에서 활성화)"""
    conflict_threshold:  float = 0.3
    agreement_threshold: float = 0.7
    novelty_boost:       float = 1.0
    lsh_transition_n:    int   = 16

@dataclass
class LLMBridgeConfig:
    """EmbeddingBridge + CostRouter 설정 (sub-4, sub-5 에서 활성화)"""
    embedding_model:           str   = "BAAI/bge-small-ko-v1.5"
    embed_dim:                 int   = 384
    cost_level_thresholds:     tuple = (0.2, 0.5, 0.8)
    budget_pressure_threshold: float = 0.8

@dataclass
class PipelineConfig:
    """PipelinedBrainRuntime 설정 (sub-4 에서 활성화)"""
    buffer_size: int = 3
```

### 3.2 HTPConfig facade 확장 (이전 사이클 패턴 재사용)

```python
class HTPConfig:
    __slots__ = ("n_nodes", "device", "hub", "prune", "activation",
                 "routing", "coherence", "llm_bridge", "pipeline")  # +4

    def __init__(self, ..., routing=None, coherence=None,
                 llm_bridge=None, pipeline=None, **kwargs):
        # ... 기존 (Phase 1 sub-configs) ...
        object.__setattr__(self, "routing",    routing    or RoutingConfig())
        object.__setattr__(self, "coherence",  coherence  or CoherenceConfig())
        object.__setattr__(self, "llm_bridge", llm_bridge or LLMBridgeConfig())
        object.__setattr__(self, "pipeline",   pipeline   or PipelineConfig())
        # ... flat kwarg dispatch 도 4개 신규 sub-config 포함하도록 확장 ...
```

기존 `__getattr__`/`__setattr__` 위임 로직에 신규 4개 sub-config 추가만 하면 자동 작동 — 이전 사이클의 facade 패턴이 그대로 확장됨.

### 3.3 Stage 0 변경 영향

| 파일 | 변경 라인 (예상) |
|------|---------------|
| `htp/core/config.py` | +90 (4 dataclass + facade slot/dispatch 확장) |
| `tests/unit/test_config_isolation.py` (이전) | +3 (RoutingConfig 독립 생성, CoherenceConfig 독립 생성, deprecated warning) |

---

## 4. Stage 0.5 Detailed Design — Knowledge Loop MVP

### 4.1 TextEncoder Protocol

```python
# htp/knowledge/encoder.py
from typing import Protocol
import numpy as np

class TextEncoder(Protocol):
    """텍스트 → 64-dim 벡터 인코딩 프로토콜.

    Design Ref: §1.1 Goal 3 — KnowledgeLoop / LLMRegion / RegionSignature 공유.
    Stage 6 EmbeddingBridge 가 같은 Protocol 의 다른 구현이 됨.
    """

    @property
    def dim(self) -> int:
        """출력 벡터 차원. MVP/Bridge 모두 64 고정."""
        ...

    def encode(self, text: str) -> np.ndarray:
        """text → np.ndarray shape (dim,)"""
        ...

    def fit(self, corpus: list[str]) -> None:
        """선택적: TF-IDF 같은 모델은 어휘 학습 필요. EmbeddingBridge 는 no-op."""
        ...
```

### 4.2 MVP 구현 — TfidfJLEncoder (옵션 α 선택)

```python
# htp/knowledge/encoder.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.random_projection      import GaussianRandomProjection
import numpy as np

class TfidfJLEncoder:
    """TF-IDF + Gaussian Random Projection.

    Design Ref: Plan §8.2 Decision — "TF-IDF+JL (의도적 조잡)" 선택.
    어휘 일치 + JL 차원 보존 보조 정리 ε ≈ √(8 ln(N)/k) 보장.

    의존성: scikit-learn 만 (이미 ML 프로젝트 표준).
    품질: 의도적으로 낮음. Stage 6 EmbeddingBridge 로 교체 가능.
    """

    def __init__(self, dim: int = 64, max_features: int = 5000,
                 random_state: int = 42):
        self._dim = dim
        self._tfidf = TfidfVectorizer(
            max_features = max_features,
            ngram_range  = (1, 2),         # uni + bigram (Korean 부분 보강)
            lowercase    = True,
            token_pattern = r"(?u)\b\w+\b",  # 한/영 모두 매칭
        )
        self._jl = GaussianRandomProjection(
            n_components = dim,
            random_state = random_state,
        )
        self._fitted = False

    @property
    def dim(self) -> int:
        return self._dim

    def fit(self, corpus: list[str]) -> None:
        """전체 코퍼스로 어휘 + JL 행렬 학습 (1회)."""
        if not corpus:
            return
        X_sparse = self._tfidf.fit_transform(corpus)  # (N_docs, max_features)
        # JL 은 GRP fit 시 components_ 행렬 생성 — 입력 차원 필요
        self._jl.fit(X_sparse)
        self._fitted = True

    def encode(self, text: str) -> np.ndarray:
        """text → 64-dim 벡터 (L2 정규화)."""
        if not self._fitted:
            # 단일 텍스트로 emergency fit (cold-start 보호)
            self.fit([text])
        X_sparse = self._tfidf.transform([text])      # (1, max_features)
        x_proj   = self._jl.transform(X_sparse)       # (1, dim)
        x        = np.asarray(x_proj).flatten()
        norm     = np.linalg.norm(x)
        return x / norm if norm > 1e-8 else x
```

### 4.3 KnowledgeLoop 클래스

```python
# htp/knowledge/loop.py
from dataclasses import dataclass, field
from datetime    import datetime
from pathlib     import Path

import numpy as np

from .encoder     import TextEncoder
from .persistence import KnowledgeStore


@dataclass
class KnowledgeEntry:
    text: str
    vec: np.ndarray
    source: str
    timestamp: str  # ISO 8601 (JSON 직렬화 친화적)
    neighbors: list = field(default_factory=list)
    conflict_count: int = 0


@dataclass
class Neighbor:
    entry_id: int       # knowledge_log 내 인덱스
    similarity: float


@dataclass
class IngestResult:
    entry: KnowledgeEntry
    neighbors: list[Neighbor]
    conflicts: list[Neighbor]   # similarity < 0.3
    resonances: list[Neighbor]  # similarity > 0.7


@dataclass
class QueryResult:
    question: str
    relevant: list[Neighbor]
    cluster_count: int


@dataclass
class Discovery:
    entry_a_id: int
    entry_b_id: int
    source_a: str
    source_b: str
    similarity: float
    insight: str        # 템플릿 기반 (Stage 4 LLMRegion 에서 자연어 해석)


class KnowledgeLoop:
    """최소 지식 입출력 루프.

    Design Ref: §4 Stage 0.5 — "텍스트 입력 → 벡터 → 저장 → 발견 → 출력".
    Plan SC: FR-05.4 (ingest/query/discover 3 method).
    """

    def __init__(self,
                 encoder: TextEncoder,
                 store: KnowledgeStore = None,
                 conflict_threshold: float = 0.3,
                 resonance_threshold: float = 0.7,
                 discover_threshold: float = 0.6):
        self.encoder = encoder
        self.store   = store or KnowledgeStore.default()
        self.conflict_threshold = conflict_threshold
        self.resonance_threshold = resonance_threshold
        self.discover_threshold = discover_threshold
        self._cache: list[KnowledgeEntry] = self.store.load_all()

    def ingest(self, text: str, source: str = "") -> IngestResult:
        # 1) vec 변환 — 누적 코퍼스로 fit 후 transform
        corpus = [e.text for e in self._cache] + [text]
        self.encoder.fit(corpus)
        vec = self.encoder.encode(text)

        # 2) 이웃 탐색
        neighbors = self._find_neighbors(vec, top_k=5)
        conflicts = [n for n in neighbors if n.similarity < self.conflict_threshold]
        resonances = [n for n in neighbors if n.similarity > self.resonance_threshold]

        # 3) entry 생성 + 저장
        entry = KnowledgeEntry(
            text=text, vec=vec, source=source,
            timestamp=datetime.utcnow().isoformat() + "Z",
            neighbors=[(n.entry_id, n.similarity) for n in neighbors],
            conflict_count=len(conflicts),
        )
        self._cache.append(entry)
        self.store.append(entry)

        return IngestResult(entry=entry, neighbors=neighbors,
                            conflicts=conflicts, resonances=resonances)

    def query(self, question: str) -> QueryResult:
        if not self._cache:
            return QueryResult(question=question, relevant=[], cluster_count=0)
        q_vec = self.encoder.encode(question)
        relevant = self._find_neighbors(q_vec, top_k=10)
        clusters = self._count_clusters(relevant)
        return QueryResult(question=question, relevant=relevant,
                           cluster_count=clusters)

    def discover(self) -> list[Discovery]:
        """다른 source 간 high-similarity 쌍 발견."""
        discoveries = []
        for i, a in enumerate(self._cache):
            for j, b in enumerate(self._cache[i+1:], start=i+1):
                if a.source == b.source:
                    continue
                sim = float(np.dot(a.vec, b.vec) /
                            (np.linalg.norm(a.vec) * np.linalg.norm(b.vec) + 1e-8))
                if sim > self.discover_threshold:
                    discoveries.append(Discovery(
                        entry_a_id=i, entry_b_id=j,
                        source_a=a.source, source_b=b.source,
                        similarity=sim,
                        insight=f"'{a.source}'와 '{b.source}'가 64-dim 공간에서 "
                                f"{sim:.2f} 유사도로 연결됨"
                    ))
        discoveries.sort(key=lambda d: d.similarity, reverse=True)
        return discoveries[:10]

    def _find_neighbors(self, vec: np.ndarray, top_k: int) -> list[Neighbor]:
        if not self._cache:
            return []
        sims = []
        for i, e in enumerate(self._cache):
            denom = np.linalg.norm(vec) * np.linalg.norm(e.vec) + 1e-8
            sim = float(np.dot(vec, e.vec) / denom)
            sims.append(Neighbor(entry_id=i, similarity=sim))
        sims.sort(key=lambda n: n.similarity, reverse=True)
        return sims[:top_k]

    def _count_clusters(self, neighbors: list[Neighbor],
                        cluster_threshold: float = 0.6) -> int:
        """간이 단일 패스 클러스터링 — 첫 패스에선 그룹 수만 반환."""
        if not neighbors:
            return 0
        groups = []
        for n in neighbors:
            assigned = False
            for g in groups:
                # 그룹 대표와의 유사도로 판정
                rep = self._cache[g[0].entry_id].vec
                cur = self._cache[n.entry_id].vec
                denom = np.linalg.norm(rep) * np.linalg.norm(cur) + 1e-8
                if float(np.dot(rep, cur) / denom) > cluster_threshold:
                    g.append(n)
                    assigned = True
                    break
            if not assigned:
                groups.append([n])
        return len(groups)
```

### 4.4 KnowledgeStore — JSONL 영속화 (옵션 JSON 선택)

```python
# htp/knowledge/persistence.py
import json
from pathlib import Path
import numpy as np

from .loop import KnowledgeEntry


class KnowledgeStore:
    """JSONL append-only 영속 저장소.

    Design Ref: Plan §8.2 — JSON 파일 선택. 설치/마이그레이션 불필요.
    파일: .htp/knowledge_log.jsonl  (이전 사이클 .htp/ 디렉토리 재사용)
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def default(cls) -> "KnowledgeStore":
        return cls(Path(".htp/knowledge_log.jsonl"))

    def append(self, entry: KnowledgeEntry) -> None:
        """JSONL 1줄 append. vec 은 list 로 직렬화."""
        rec = {
            "text":           entry.text,
            "vec":            entry.vec.tolist(),
            "source":         entry.source,
            "timestamp":      entry.timestamp,
            "neighbors":      entry.neighbors,
            "conflict_count": entry.conflict_count,
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def load_all(self) -> list[KnowledgeEntry]:
        if not self.path.exists():
            return []
        entries = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                entries.append(KnowledgeEntry(
                    text=rec["text"],
                    vec=np.array(rec["vec"], dtype=np.float64),
                    source=rec["source"],
                    timestamp=rec["timestamp"],
                    neighbors=rec.get("neighbors", []),
                    conflict_count=rec.get("conflict_count", 0),
                ))
        return entries
```

### 4.5 CLI (`htp/knowledge/__main__.py`)

```python
# python -m htp.knowledge ingest --source <src> "<text>"
# python -m htp.knowledge query "<question>"
# python -m htp.knowledge discover

import argparse
import sys

from .encoder import TfidfJLEncoder
from .loop    import KnowledgeLoop


def main() -> int:
    parser = argparse.ArgumentParser(prog="htp.knowledge",
                                     description="HTP Knowledge Loop MVP")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ingest = sub.add_parser("ingest")
    p_ingest.add_argument("--source", required=True)
    p_ingest.add_argument("text")

    p_query = sub.add_parser("query")
    p_query.add_argument("question")

    sub.add_parser("discover")

    args = parser.parse_args()

    loop = KnowledgeLoop(encoder=TfidfJLEncoder())

    if args.cmd == "ingest":
        result = loop.ingest(args.text, source=args.source)
        print(f"✓ 저장 완료  (vec norm: {float(__import__('numpy').linalg.norm(result.entry.vec)):.2f})")
        print(f"◆ 유사 지식 {len(result.neighbors)}건:")
        for n in result.neighbors:
            entry = loop._cache[n.entry_id]
            marker = "←" if n.similarity < 0.3 else "✓"
            print(f"  [{n.similarity:.2f}] {entry.text[:60]}... ({entry.source}) {marker}")
        if result.resonances:
            top = result.resonances[0]
            entry = loop._cache[top.entry_id]
            print(f"⚡ 발견: '{args.source}'와 '{entry.source}'가 {top.similarity:.2f} 유사도")
        return 0

    if args.cmd == "query":
        result = loop.query(args.question)
        if not result.relevant:
            print("저장된 지식이 없습니다. 먼저 ingest 하세요.")
            return 0
        print(f"◆ '{args.question}' 관련 지식 {len(result.relevant)}건 "
              f"({result.cluster_count}개 클러스터):")
        for n in result.relevant:
            entry = loop._cache[n.entry_id]
            print(f"  [{n.similarity:.2f}] {entry.text[:80]}... ({entry.source})")
        return 0

    if args.cmd == "discover":
        discoveries = loop.discover()
        if not discoveries:
            print("Cross-domain 발견 없음. 다른 source 의 지식을 추가하세요.")
            return 0
        print(f"⚡ Top 발견 {len(discoveries)}건:")
        for d in discoveries:
            a = loop._cache[d.entry_a_id]
            b = loop._cache[d.entry_b_id]
            print(f"  [{d.similarity:.2f}] {d.source_a}:\"{a.text[:50]}...\"")
            print(f"           ↔ {d.source_b}:\"{b.text[:50]}...\"")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
```

### 4.6 Public API (`htp/knowledge/__init__.py`)

```python
"""htp.knowledge — Knowledge Loop MVP (Stage 0.5).

Design Ref: htp_thalamus_car_design v4.md §0.5 (Rev 1.3).
"""
from .encoder import TextEncoder, TfidfJLEncoder
from .loop    import (
    KnowledgeLoop,
    KnowledgeEntry,
    Neighbor,
    IngestResult,
    QueryResult,
    Discovery,
)
from .persistence import KnowledgeStore

__all__ = [
    "TextEncoder", "TfidfJLEncoder",
    "KnowledgeLoop", "KnowledgeEntry", "Neighbor",
    "IngestResult", "QueryResult", "Discovery",
    "KnowledgeStore",
]
```

---

## 5. sub-1 Testing Strategy

### 5.1 회귀 보호

- `pytest tests/regression/ tests/unit/` — 57 + 46 = **103 통과** 매 step 직후
- 외부 의존: `from htp.runtime.htp_runtime import ...` (server.py 사용 경로) — 영향 0

### 5.2 신규 본선 테스트 (sub-1 누적 60 → 65)

#### Stage 0 (+3, 이전 사이클 test_config_isolation.py 확장)

| 테스트 | 검증 |
|--------|------|
| `test_routing_config_isolation` | `RoutingConfig` 독립 생성 + 기본값 |
| `test_coherence_config_isolation` | `CoherenceConfig` 독립 생성 + LSH threshold 16 |
| `test_subconfig_flat_kwarg_dispatch` | `HTPConfig(threshold_beta=0.7)` 가 `routing.threshold_beta` 로 위임 |

#### Stage 0.5 (+5, `tests/knowledge/test_loop.py` 신규)

| 테스트 | 검증 |
|--------|------|
| `test_loop_ingest_basic` | 텍스트 입력 → 64-dim 벡터 생성 → store 저장 |
| `test_loop_query_neighbor` | 유사 텍스트 질의 → relevant[0].similarity > 0.5 |
| `test_loop_discover_cross_domain` | **3 source 시나리오 — 뇌과학-AI > 뇌과학-인프라** (Stage 0.5 Go/No-Go 핵심) |
| `test_loop_text_encoder_interface` | `TfidfJLEncoder` 가 `TextEncoder` Protocol 준수 + `encode(str)` shape == (64,) |
| `test_loop_empty_state` | 0건 상태에서 query/discover 빈 결과 (에러 없음) |

#### test_no_circular_deps.py 확장 (이전 사이클 자산)

```python
# htp/knowledge/* 도 htp/runtime 미참조 검증
_DAG_EXEMPT 에 추가 없음 (knowledge 는 DAG 안전 강제)
```

### 5.3 CLI 수동 검증 (Stage 0.5 Go/No-Go 핵심 시나리오)

```bash
python -m htp.knowledge ingest --source "뇌과학" "뇌의 기억은 주소가 아니라 내용으로 저장·인출된다"
python -m htp.knowledge ingest --source "AI" "Hopfield network는 에너지 최소화로 패턴을 인출한다"
python -m htp.knowledge ingest --source "인프라" "Redis는 key로 value를 조회한다"
python -m htp.knowledge discover

# 기대 결과: 뇌과학-AI 유사도 > 뇌과학-인프라 유사도
```

---

## 6. Risks (sub-1 한정)

| 위험 | Impact | 완화 |
|------|--------|------|
| TF-IDF가 cross-domain 발견 못 함 (Plan §6 위험 6) | High | (a) `ngram_range=(1,2)` 로 bigram 보강 (b) 영문 술어 공유 가정에 의존 (c) No-Go 시 δ(Mecab+sklearn) 또는 γ(sentence-transformers) 로 즉시 교체 |
| `TextEncoder` Protocol 교체 시 깨짐 | Med | encoder.py 에 단일 정의, 다른 모듈은 import 만 |
| `.htp/knowledge_log.jsonl` 손상 | Low | append-only, 손상 라인 skip 로직 추가 가능 (sub-2 이후) |
| Stage 0.5 의존성 추가 (scikit-learn) | Low | `requirements.txt` 에 명시. 이미 ML 프로젝트 표준 |
| 회귀 깨짐 | Low | 매 step 직후 pytest, `routing_mode="tag"` 유지 |

---

## 7. Public API (sub-1 후)

### 7.1 변경 없는 경로

`htp/__init__.py` 공개 28 symbols — **변경 없음**. 사용자 코드 무영향.

### 7.2 신규 import 경로 (선택적)

```python
from htp.knowledge import KnowledgeLoop, TfidfJLEncoder, TextEncoder
from htp.core.config import RoutingConfig, CoherenceConfig, LLMBridgeConfig, PipelineConfig
```

`htp/__init__.py` 에 export 추가 여부는 sub-cycle 1 Do 단계에서 결정 — **기본 미추가** (사용자 표면 최소화, Stage 6 EmbeddingBridge 까지 완료 후 일괄 노출 검토).

---

## 8. Test Plan (sub-1 한정)

| Level | 적용 | 대체 |
|-------|----|----|
| L1 (HTTP API) | N/A | `python -m htp.knowledge ingest/query/discover` CLI 1 round-trip 수동 검증 |
| L2 (UI) | N/A | — |
| L3 (E2E) | N/A | CLI 시나리오 = 뇌과학/AI/인프라 3-source discover |
| 회귀 | 57 + 46 + 3 + 5 = 111 | pytest |

---

## 9. Decision Record (sub-cycle 1)

| Decision | Selected | Rationale |
|----------|---------|-----------|
| Cycle 운영 모델 | **C — Phase Sub-Cycles (6)** | 의미 단위 + 매 sub-cycle Match Rate 검증 |
| sub-1 Stage 묶음 | **Stage 0 + Stage 0.5** | 토대 + 루프 폐쇄가 한 의미 단위 |
| TextEncoder MVP | **α — sklearn TfidfVectorizer + GaussianRandomProjection** | 설계서 v4 Rev 1.3 명시 선호. sklearn 일 의존, ~20줄 |
| TF-IDF ngram | **(1, 2)** | unigram + bigram. 영문 술어 공유 + 약간의 phrase 보강 |
| TF-IDF token_pattern | `r"(?u)\b\w+\b"` | 한/영 모두 매칭 (의미적 토큰화는 Stage 6 sentence-transformers 에 위임) |
| knowledge_log 저장 | **JSONL `.htp/knowledge_log.jsonl`** | append-only, 설치 불필요, 수동 검사 가능 |
| `htp/knowledge/` 위치 | **htp/core/, htp/runtime/, htp/thalamus/ 와 동급 패키지** | DAG 의존 방향: 외부 라이브러리만 import, htp/runtime 미참조 |
| public API export | **sub-1 에서는 미추가** | 사용자 표면 최소화. Stage 6 후 일괄 검토 |
| CLI dispatcher | **argparse + `__main__.py`** | 표준 라이브러리, 의존성 0 |
| Stage 0.5 의존성 정책 | **scikit-learn 허용** | 의도적 조잡 MVP + 검증된 라이브러리. 표준 ML 의존성 |

---

## 10. Architecture Considerations

### 10.1 Project Level

Research / Library — bkit 3-level 외. 이전 사이클과 동일.

### 10.2 Key Tools

| Category | Used |
|----------|------|
| Language | Python 3.10+ |
| ML 의존 | PyTorch (기존), **scikit-learn (Stage 0.5 신규)** |
| Test | pytest |
| CLI | argparse (표준 라이브러리) |
| 영속화 | JSON + jsonl (표준 라이브러리) |

---

## 11. Implementation Guide

### 11.1 Recommended Order (sub-1)

1. **step-0a**: `htp/core/config.py` 에 4 sub-config 추가 + facade slot/dispatch 확장
2. **step-0b**: `tests/unit/test_config_isolation.py` 에 3 테스트 추가 → 회귀+unit 통과 확인 (Stage 0 완료)
3. **step-0.5a**: `htp/knowledge/__init__.py` + `encoder.py` (TextEncoder Protocol + TfidfJLEncoder)
4. **step-0.5b**: `htp/knowledge/persistence.py` (KnowledgeStore JSONL)
5. **step-0.5c**: `htp/knowledge/loop.py` (KnowledgeLoop + 5 dataclass)
6. **step-0.5d**: `htp/knowledge/__main__.py` (argparse CLI)
7. **step-0.5e**: `tests/knowledge/test_loop.py` (5 테스트)
8. **step-0.5f**: CLI 수동 시나리오 (뇌과학/AI/인프라) — Go/No-Go 검증
9. **step-0.5g**: `requirements.txt` 에 `scikit-learn` 추가
10. **step-0.5h**: `tests/unit/test_no_circular_deps.py` 확장 (htp/knowledge DAG 검증)
11. **step-1-final**: 회귀 + unit + knowledge 통합 통과 확인 (목표 65)

### 11.2 Key Files Reference (sub-1)

| Step | Files Created | Files Modified |
|------|---------------|----------------|
| 0a | — | `htp/core/config.py` |
| 0b | — | `tests/unit/test_config_isolation.py` |
| 0.5a | `htp/knowledge/__init__.py`, `htp/knowledge/encoder.py` | — |
| 0.5b | `htp/knowledge/persistence.py` | — |
| 0.5c | `htp/knowledge/loop.py` | — |
| 0.5d | `htp/knowledge/__main__.py` | — |
| 0.5e | `tests/knowledge/__init__.py`, `tests/knowledge/test_loop.py` | — |
| 0.5g | — | `requirements.txt` (+`scikit-learn`) |
| 0.5h | — | `tests/unit/test_no_circular_deps.py` |

### 11.3 Session Guide (sub-1 Module Map)

| Module Key | Description | Time | Dependencies |
|-----------|------------|------|--------------|
| `stage-0` | Stage 0 — config sub-config 4종 + facade | ~1h | 이전 사이클 facade |
| `stage-0.5-core` | encoder + persistence + loop 핵심 클래스 | ~2h | stage-0 |
| `stage-0.5-cli` | `__main__.py` + 수동 시나리오 검증 | ~1h | stage-0.5-core |
| `stage-0.5-test` | 5 신규 테스트 + DAG 확장 + 회귀 확인 | ~1.5h | stage-0.5-core, stage-0.5-cli |

**권장 세션 분할**:
- Session A: `stage-0` + `stage-0.5-core` (인프라 + 핵심 클래스, ~3h) — `/pdca do htp-thalamus-car --scope stage-0,stage-0.5-core`
- Session B: `stage-0.5-cli` + `stage-0.5-test` (CLI + 테스트 + Go/No-Go, ~2.5h)
- (sub-1 완료 후) `/pdca analyze htp-thalamus-car` — Match Rate 검증

---

## 12. Next Steps (sub-1)

1. [ ] `/pdca do htp-thalamus-car --scope stage-0,stage-0.5-core` (Session A)
2. [ ] Session A 회귀 확인 후 Session B 진행 (`--scope stage-0.5-cli,stage-0.5-test`)
3. [ ] Session B 후 CLI 수동 시나리오 (Go/No-Go 핵심)
4. [ ] `/pdca analyze htp-thalamus-car` — sub-1 Match Rate
5. [ ] Match Rate ≥ 90% + Stage 0.5 시나리오 Go → sub-2 Design 진입
6. [ ] No-Go 시 TextEncoder 옵션 δ(Mecab) 또는 γ(sentence-transformers) 로 교체

---

## 13. sub-2 ~ sub-6 미리보기 (별도 sub-cycle Design 에서 상세화)

| sub-cycle | Stage | 핵심 산출물 | 예상 Match Rate 검증 시점 |
|-----------|-------|-----------|----------------------|
| sub-2 | 1 + 2 | RegionSignature, `_gate_vector`, `_gate_hybrid` | 78 tests |
| sub-3 | 3 | CoherenceGate, BoundResponse, swr_priority 확장 | 84 tests |
| sub-4 | 4 + 5 | ExternalRegion, LLMRegion, PipelinedBrainRuntime, CostRouter `select_level` | 89 tests |
| sub-5 | 6 (실험) | EmbeddingBridge (브랜치 `experiment/embedding-bridge`) | 별도 4 |
| sub-6 | 7 | vector default 전환, tag 코드 archive | 89 재실행 |

각 sub-cycle 시작 시 `/pdca design htp-thalamus-car` 재호출 또는 별도 sub-feature 이름 (예: `htp-thalamus-car-sub2`) 으로 분리 — sub-1 완료 후 결정.

---

## Version History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 0.1 | 2026-05-17 | Initial — 6 sub-cycles 운영 모델, TfidfJLEncoder MVP, JSONL 저장 선택 | Mindbuild |
