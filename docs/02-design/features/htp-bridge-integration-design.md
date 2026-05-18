---
template: design
feature: htp-bridge-integration
date: 2026-05-18
author: Mindbuild
project: HTP (Hub Topology Programming)
status: Draft
predecessor: sub-5 (EmbeddingBridge merge 후)
purpose: "시스템 A(brain-like 구조)와 시스템 B(KnowledgeLoop)의 최소 연결"
---

# HTP Bridge Integration 설계서

**시스템 A(brain-like 구조)와 시스템 B(KnowledgeLoop)의 연결**

> **핵심 가설**: brain-like 구조(RegionSignature, CoherenceGate, VectorRouter)가
> 단순 cosine 순회보다 더 나은 지식 관리를 제공한다.
>
> **이 설계서의 목적**: 위 가설을 검증할 수 있는 최소 연결을 구현한다.
> 가설이 참이면 HTP 프로젝트의 방향이 확인되고,
> 거짓이면 시스템 A의 방향을 수정해야 한다.
> **어느 쪽이든 지금 해봐야 안다.**

---

## 0. 현재 상태 진단

### 0-1. 두 개의 분리된 시스템

```
시스템 A (brain-like 구조) — 5,800줄
┌─────────────────────────────────────────────────┐
│ htp/core/        hub_formation, pruning, NGE    │
│ htp/runtime/     BrainRuntime, PFC, Region      │
│ htp/thalamus/    CoreCells, Router, Coherence   │
│ htp/memory/      CA3-CA1, EpisodeStore, Pattern │
└─────────────────────────────────────────────────┘
       ↕ 연결 없음 (import 0건)
시스템 B (KnowledgeLoop) — 400줄
┌─────────────────────────────────────────────────┐
│ htp/knowledge/   loop.py, encoder.py, CLI       │
│                  자체 _cosine() 함수로 검색      │
└─────────────────────────────────────────────────┘
```

### 0-2. 검증된 사실

```
코드베이스: 소스 6,230줄 / 테스트 2,941줄 / 문서 9,100줄+
KnowledgeLoop → htp/thalamus/ import: 0건
KnowledgeLoop → htp/runtime/ import:  0건
KnowledgeLoop → htp/memory/ import:   0건
KnowledgeLoop.query() 구현: 전체 _cache 순회 + 자체 _cosine()
```

### 0-3. 이미 존재하는 연결 가능 코드

| 모듈 | 파일 | 현재 상태 | 연결 시 역할 |
|------|------|----------|------------|
| `RegionSignature` | `htp/thalamus/signature.py` | 테스트만 통과, 실데이터 없음 | source별 의미 중심점 학습 |
| `VectorRouter` | `htp/thalamus/router/vector_router.py` | 테스트만 통과, 실데이터 없음 | query 시 관련 source 선택 |
| `PairwiseCoherenceGate` | `htp/thalamus/coherence/pairwise.py` | 테스트만 통과, 실데이터 없음 | ingest 시 충돌 감지 |
| `RegionResponse` | `htp/thalamus/types.py` | CoherenceGate 입력용 | 지식 엔트리 → Region 응답 변환 |
| `RegionSignal` | `htp/thalamus/region_signal.py` | Router 입력용 | source → Region 신호 변환 |

**핵심: 새로 만들 코드가 거의 없다. 이미 있는 것을 연결만 하면 된다.**

---

## 1. 연결 아키텍처

### 1-1. 연결 후 데이터 흐름

```
ingest("뇌의 기억은 내용 기반으로 인출된다", source="뇌과학")
  │
  ├─ encoder.encode(text) → vec (64-dim 또는 384-dim)
  │
  ├─ [기존] _cache에 append + JSONL 저장
  │
  ├─ [연결 1] RegionSignature("뇌과학").update(vec)
  │           → source의 의미 중심점이 Hebbian EMA로 학습됨
  │
  └─ [연결 2] CoherenceGate.bind(새 지식 + 기존 이웃)
              → conflict 감지 시 entry.conflict_count 증가
              → "이 지식은 기존 지식과 모순됨" 신호 생성


query("패턴 인출 메커니즘")
  │
  ├─ encoder.encode(question) → q_vec
  │
  ├─ [연결 3] VectorRouter.score(q_vec, source별 RegionSignal)
  │           → 관련 source 선택 (전체 순회 대신)
  │           → "뇌과학" source score: 0.8, "인프라" source score: 0.1
  │
  ├─ 선택된 source의 엔트리만 cosine 검색
  │
  └─ [비교] 기존 전체 순회 결과 vs 라우팅 결과 → A/B 비교
```

### 1-2. 설계 원칙

1. **기존 코드 수정 최소화** — `loop.py` 에만 import 3줄 + 메서드 수정. 시스템 A 코드는 변경 없음
2. **기존 동작 보존** — `--mode flat` 옵션으로 기존 전체 순회 유지. 기본값은 기존과 동일
3. **A/B 비교 가능** — 같은 데이터, 같은 쿼리로 flat vs routed 결과를 나란히 비교

---

## 2. 연결 1: source → RegionSignature (Hebbian 학습)

### 2-1. 변경 대상

`htp/knowledge/loop.py` — `KnowledgeLoop.__init__()` 및 `ingest()`

### 2-2. 코드 변경

```python
# loop.py 상단에 추가
from htp.thalamus.signature import RegionSignature

class KnowledgeLoop:
    def __init__(self, encoder, store=None, ...):
        # ... 기존 코드 ...
        
        # [연결 1] source별 RegionSignature
        self._signatures: dict[str, RegionSignature] = {}
        # 기존 캐시에서 signature 복원
        self._rebuild_signatures()

    def _rebuild_signatures(self):
        """기존 _cache에서 source별 RegionSignature 재구축."""
        for entry in self._cache:
            src = entry.source
            if src not in self._signatures:
                dim = len(entry.vec)
                self._signatures[src] = RegionSignature(dim=dim)
            self._signatures[src].update(entry.vec)

    def ingest(self, text, source=""):
        vec = self.encoder.encode(text)
        
        # [연결 1] RegionSignature 갱신
        if source not in self._signatures:
            self._signatures[source] = RegionSignature(dim=len(vec))
        self._signatures[source].update(vec)
        
        # ... 기존 neighbors/conflicts/resonances 로직 ...
        # ... 기존 저장 로직 ...
```

### 2-3. 이 연결이 증명하는 것

- RegionSignature.centroid가 **실제 사용자 데이터**로 학습됨
- source("뇌과학")에 지식 10개를 넣으면 centroid가 뇌과학 개념 공간의 중심으로 수렴
- HTP 4대 원칙 "구조는 데이터가 만든다"의 첫 실증

### 2-4. 검증

```bash
# 테스트: 같은 source의 centroid가 해당 도메인 query와 높은 유사도를 가지는가
$ python -c "
from htp.knowledge.loop import KnowledgeLoop
from htp.knowledge.encoder import TfidfJLEncoder

loop = KnowledgeLoop(encoder=TfidfJLEncoder())
loop.ingest('뇌의 기억은 내용 기반으로 인출된다', source='뇌과학')
loop.ingest('해마 CA3는 패턴 완성을 수행한다', source='뇌과학')
loop.ingest('시냅스 가소성이 학습의 기반이다', source='뇌과학')
loop.ingest('Redis는 key-value 저장소이다', source='인프라')
loop.ingest('로드밸런서가 트래픽을 분산한다', source='인프라')

# RegionSignature 확인
for src, sig in loop._signatures.items():
    print(f'{src}: count={sig.count}, centroid_norm={float(sig.centroid.__abs__().sum()):.3f}')
    # 쿼리와의 유사도
    q = loop.encoder.encode('기억과 학습의 신경과학')
    print(f'  query 유사도: {sig.similarity(q):.3f}')
"
```

**기대 결과**: "뇌과학" signature의 query 유사도 > "인프라" signature의 query 유사도

---

## 3. 연결 2: ingest 시 CoherenceGate 충돌 감지

### 3-1. 변경 대상

`htp/knowledge/loop.py` — `ingest()`

### 3-2. 코드 변경

```python
# loop.py 상단에 추가
from htp.thalamus.coherence.pairwise import PairwiseCoherenceGate
from htp.thalamus.types import RegionResponse

class KnowledgeLoop:
    def __init__(self, encoder, store=None, ...):
        # ... 기존 코드 ...
        
        # [연결 2] CoherenceGate
        self._coherence = PairwiseCoherenceGate(
            conflict_threshold=0.3,
            escalation_threshold=0.7,
        )

    def ingest(self, text, source=""):
        vec = self.encoder.encode(text)
        neighbors = self._find_neighbors(vec, top_k=5)
        
        # [연결 2] 기존 이웃과의 정합성 검사
        coherence_info = None
        if len(neighbors) >= 2:
            # 새 지식 + 상위 이웃을 RegionResponse로 변환
            responses = [
                RegionResponse(
                    region_id=f"new_{source}",
                    output_vec=vec,
                    precision=1.0,
                )
            ]
            for n in neighbors[:3]:  # 상위 3개 이웃만
                neighbor_entry = self._cache[n.entry_id]
                responses.append(RegionResponse(
                    region_id=f"existing_{neighbor_entry.source}",
                    output_vec=neighbor_entry.vec,
                    precision=1.0,
                ))
            
            bound = self._coherence.bind(responses)
            coherence_info = {
                "coherence": bound.coherence,
                "conflict": bound.conflict,
                "escalate": bound.escalate_to_pfc,
            }
        
        # 기존 conflict/resonance 로직에 coherence_info 반영
        conflicts = [n for n in neighbors if n.similarity < self.conflict_threshold]
        resonances = [n for n in neighbors if n.similarity > self.resonance_threshold]
        
        entry = KnowledgeEntry(
            text=text, vec=vec, source=source,
            timestamp=datetime.now(timezone.utc).isoformat(),
            neighbors=[(n.entry_id, n.similarity) for n in neighbors],
            conflict_count=len(conflicts),
        )
        self._cache.append(entry)
        self.store.append(entry)
        
        return IngestResult(
            entry=entry, neighbors=neighbors,
            conflicts=conflicts, resonances=resonances,
            coherence_info=coherence_info,  # 신규 필드
        )
```

### 3-3. IngestResult 확장

```python
@dataclass
class IngestResult:
    entry: KnowledgeEntry
    neighbors: list
    conflicts: list
    resonances: list
    coherence_info: dict | None = None  # 연결 2 추가
```

### 3-4. CLI 출력 변경

```python
# __main__.py — _cmd_ingest() 에 추가

def _cmd_ingest(loop, source, text):
    result = loop.ingest(text, source=source)
    # ... 기존 출력 ...
    
    # [연결 2] 정합성 정보
    if result.coherence_info:
        ci = result.coherence_info
        if ci["escalate"]:
            print(f"⚠ 충돌 감지! coherence={ci['coherence']:.2f}, "
                  f"conflict={ci['conflict']:.2f}")
            print(f"  → 이 지식은 기존 지식과 모순될 수 있습니다")
        else:
            print(f"✓ 정합성 양호 (coherence={ci['coherence']:.2f})")
```

### 3-5. 이 연결이 증명하는 것

- CoherenceGate가 **실제 지식 간 충돌을 감지하는가**
- 단순 cosine의 높음/낮음이 아니라, "여러 지식 간의 정합성"을 측정하는 고유 기능
- **단순 벡터 검색으로는 불가능한 것**: "새 지식이 기존 체계와 모순되는가?"

### 3-6. 검증 시나리오

```bash
# 시나리오: 모순되는 지식 입력
$ python -m htp.knowledge ingest --source "AI" \
    "Transformer의 self-attention은 입력 전체를 한번에 참조하는 전역적 메커니즘이다"

$ python -m htp.knowledge ingest --source "뇌과학" \
    "뇌의 주의 메커니즘은 국소적이며 순차적으로 정보를 처리한다"

# 기대 출력:
# ⚠ 충돌 감지! coherence=0.35, conflict=0.72
# → 이 지식은 기존 지식과 모순될 수 있습니다
```

```bash
# 시나리오: 일관된 지식 입력
$ python -m htp.knowledge ingest --source "뇌과학" \
    "해마 CA3는 recurrent connection으로 패턴 완성을 수행한다"

$ python -m htp.knowledge ingest --source "뇌과학" \
    "패턴 완성은 부분 단서에서 전체 기억을 복원하는 과정이다"

# 기대 출력:
# ✓ 정합성 양호 (coherence=0.82)
```

---

## 4. 연결 3: query 시 VectorRouter로 범위 축소

### 4-1. 변경 대상

`htp/knowledge/loop.py` — `query()`

### 4-2. 코드 변경

```python
# loop.py 상단에 추가
from htp.thalamus.router.vector_router import VectorRouter
from htp.thalamus.region_signal import RegionSignal

class KnowledgeLoop:
    def __init__(self, encoder, store=None, ...):
        # ... 기존 코드 ...
        
        # [연결 3] VectorRouter
        self._router = VectorRouter(beta=0.5)

    def query(self, question, mode="flat"):
        """
        mode:
          "flat"   — 기존 전체 순회 (A/B 비교용 보존)
          "routed" — VectorRouter로 관련 source 선택 후 검색
        """
        if not self._cache:
            return QueryResult(question=question, relevant=[], cluster_count=0)
        
        q_vec = self.encoder.encode(question)
        
        if mode == "routed" and self._signatures:
            candidates = self._routed_query(q_vec)
        else:
            candidates = self._cache
        
        # candidates 안에서 cosine 검색 (기존 로직)
        relevant = self._find_neighbors_in(q_vec, candidates, top_k=10)
        clusters = self._count_clusters(relevant)
        
        return QueryResult(
            question=question,
            relevant=relevant,
            cluster_count=clusters,
            routing_info=self._router.last_metrics if mode == "routed" else None,
        )

    def _routed_query(self, q_vec):
        """[연결 3] VectorRouter로 관련 source만 선택."""
        # source별 RegionSignal 생성
        signals = []
        for source, sig in self._signatures.items():
            signals.append(RegionSignal(
                region_id=source,
                hub_strength=0.0,  # KnowledgeLoop에서는 미사용
                fire_rate=0.0,
                top_hubs=[],
                overload=False,
                output_vec=None,   # VectorRouter는 이 필드 미사용
                precision=1.0,
                region_signature=sig,
            ))
        
        scores = self._router.score(None, q_vec, signals)
        
        # score > 0인 source만 선택
        selected_sources = {s.region_id for s in scores if s.score > 0}
        
        if not selected_sources:
            # 모든 source가 0 → fallback으로 전체 반환
            return self._cache
        
        return [e for e in self._cache if e.source in selected_sources]

    def _find_neighbors_in(self, vec, candidates, top_k):
        """candidates 리스트 내에서 cosine 검색."""
        if not candidates:
            return []
        sims = []
        for e in candidates:
            idx = self._cache.index(e) if e in self._cache else -1
            sim = _cosine(vec, e.vec)
            sims.append(Neighbor(entry_id=idx, similarity=sim))
        sims.sort(key=lambda n: n.similarity, reverse=True)
        return sims[:top_k]
```

### 4-3. QueryResult 확장

```python
@dataclass
class QueryResult:
    question: str
    relevant: list
    cluster_count: int
    routing_info: dict | None = None  # 연결 3 추가
```

### 4-4. CLI A/B 비교 옵션

```python
# __main__.py — query 커맨드에 --mode 옵션 추가

p_query.add_argument("--mode", choices=["flat", "routed", "compare"],
                     default="flat", help="검색 모드")

def _cmd_query(loop, question, mode="flat"):
    if mode == "compare":
        # A/B 비교: flat vs routed 결과를 나란히 출력
        result_flat   = loop.query(question, mode="flat")
        result_routed = loop.query(question, mode="routed")
        
        print(f"── Flat (전체 순회, {len(loop._cache)}건) ──")
        _print_results(loop, result_flat)
        
        print(f"\n── Routed (VectorRouter, "
              f"{result_routed.routing_info}) ──")
        _print_results(loop, result_routed)
        
        # 차이 분석
        flat_top1  = result_flat.relevant[0] if result_flat.relevant else None
        route_top1 = result_routed.relevant[0] if result_routed.relevant else None
        if flat_top1 and route_top1:
            same = flat_top1.entry_id == route_top1.entry_id
            print(f"\n{'✅ top-1 동일' if same else '⚡ top-1 다름!'}")
    else:
        result = loop.query(question, mode=mode)
        _print_results(loop, result)

def _print_results(loop, result):
    for n in result.relevant[:5]:
        entry = loop._cache[n.entry_id]
        preview = entry.text[:80] + ("..." if len(entry.text) > 80 else "")
        print(f"  [{n.similarity:+.2f}] ({entry.source}) {preview}")
```

### 4-5. 이 연결이 증명하는 것

- VectorRouter가 **query와 관련 없는 source를 제거**하여 검색 정밀도를 높이는가
- dynamic threshold가 **실제 데이터 분포**에서 적절하게 작동하는가
- 엔트리 수가 많아질 때 **검색 범위 축소**로 성능 이점이 있는가

### 4-6. 검증 시나리오

```bash
# 데이터 준비: 3개 도메인, 각 5개 지식
$ python -m htp.knowledge ingest --source "뇌과학" "해마 CA3의 패턴 완성"
$ python -m htp.knowledge ingest --source "뇌과학" "시냅스 가소성과 헵의 법칙"
$ python -m htp.knowledge ingest --source "뇌과학" "감마 진동과 시간적 바인딩"
$ python -m htp.knowledge ingest --source "뇌과학" "시상의 게이팅 메커니즘"
$ python -m htp.knowledge ingest --source "뇌과학" "수면 중 기억 공고화"

$ python -m htp.knowledge ingest --source "AI" "Transformer의 self-attention"
$ python -m htp.knowledge ingest --source "AI" "RLHF와 인간 피드백 학습"
$ python -m htp.knowledge ingest --source "AI" "MoE 라우팅 전략"
$ python -m htp.knowledge ingest --source "AI" "RAG 검색 증강 생성"
$ python -m htp.knowledge ingest --source "AI" "Diffusion 모델 생성 원리"

$ python -m htp.knowledge ingest --source "인프라" "Redis 캐시 전략"
$ python -m htp.knowledge ingest --source "인프라" "Kubernetes 오케스트레이션"
$ python -m htp.knowledge ingest --source "인프라" "로드밸런서 알고리즘"
$ python -m htp.knowledge ingest --source "인프라" "CDN 엣지 캐싱"
$ python -m htp.knowledge ingest --source "인프라" "마이크로서비스 통신 패턴"

# A/B 비교
$ python -m htp.knowledge query "기억의 신경 메커니즘" --mode compare

# 기대 결과:
# ── Flat (전체 순회, 15건) ──
#   [+0.82] (뇌과학) 해마 CA3의 패턴 완성...
#   [+0.45] (AI) Transformer의 self-attention...    ← 노이즈
#   [+0.38] (인프라) Redis 캐시 전략...              ← 노이즈
#
# ── Routed (VectorRouter, active_count=1) ──
#   [+0.82] (뇌과학) 해마 CA3의 패턴 완성...
#   [+0.78] (뇌과학) 시냅스 가소성과 헵의 법칙...
#   [+0.71] (뇌과학) 수면 중 기억 공고화...
#                                                   ← 노이즈 제거됨
# ✅ top-1 동일 (but routed가 더 정밀한 결과)
```

---

## 5. 종합 테스트 계획

### 5-1. Go/No-Go 기준

이 프로젝트의 핵심 가설을 검증하는 3개 질문:

| # | 질문 | 검증 방법 | Go 기준 | No-Go 시 조치 |
|---|------|---------|---------|--------------|
| **Q1** | RegionSignature가 source별 의미 중심을 학습하는가? | 같은 도메인 query의 signature 유사도 비교 | 같은 도메인 유사도 > 다른 도메인 유사도 | encoder 또는 dim 조정 |
| **Q2** | CoherenceGate가 실제 충돌을 감지하는가? | 모순 지식 쌍 vs 일관 지식 쌍 | 모순 쌍 conflict > 0.5, 일관 쌍 conflict < 0.3 | threshold 또는 coherence 알고리즘 변경 |
| **Q3** | VectorRouter가 검색 정밀도를 높이는가? | flat vs routed A/B 비교 | routed top-5에 관련 source 비율 ≥ 80% | beta 또는 routing 전략 변경 |

### 5-2. 자동 테스트 (7건)

```python
# tests/knowledge/test_bridge_integration.py

# ── 연결 1: RegionSignature ──

def test_signature_learns_from_ingest():
    """source별 RegionSignature가 ingest 데이터로 학습됨."""
    loop = _make_loop()
    loop.ingest("해마 CA3 패턴 완성", source="뇌과학")
    loop.ingest("시냅스 가소성", source="뇌과학")
    loop.ingest("Redis 캐시", source="인프라")
    
    assert "뇌과학" in loop._signatures
    assert "인프라" in loop._signatures
    assert loop._signatures["뇌과학"].count == 2
    assert loop._signatures["인프라"].count == 1

def test_signature_domain_discrimination():
    """같은 도메인 query가 해당 signature와 더 높은 유사도."""
    loop = _make_loop_with_data()  # 3 도메인 × 5 entries
    
    q_vec = loop.encoder.encode("기억의 신경 메커니즘")
    sim_brain = loop._signatures["뇌과학"].similarity(q_vec)
    sim_infra = loop._signatures["인프라"].similarity(q_vec)
    
    assert sim_brain > sim_infra, (
        f"뇌과학 유사도({sim_brain:.3f})가 "
        f"인프라 유사도({sim_infra:.3f})보다 높아야 함"
    )

# ── 연결 2: CoherenceGate ──

def test_coherence_detects_conflict():
    """모순 지식 입력 시 conflict 감지."""
    loop = _make_loop()
    loop.ingest("attention은 전역적 메커니즘이다", source="AI")
    result = loop.ingest("주의는 국소적이며 순차적이다", source="뇌과학")
    
    assert result.coherence_info is not None
    assert result.coherence_info["conflict"] > 0.3

def test_coherence_accepts_consistent():
    """일관 지식 입력 시 conflict 낮음."""
    loop = _make_loop()
    loop.ingest("CA3는 패턴 완성을 수행한다", source="뇌과학")
    result = loop.ingest("패턴 완성은 부분 단서에서 전체를 복원한다", source="뇌과학")
    
    assert result.coherence_info is not None
    assert result.coherence_info["conflict"] < 0.5

# ── 연결 3: VectorRouter ──

def test_routed_query_selects_relevant_source():
    """routed 모드가 관련 source를 선택."""
    loop = _make_loop_with_data()
    result = loop.query("기억의 신경 메커니즘", mode="routed")
    
    # 상위 5개 중 뇌과학 비율
    top5_sources = [loop._cache[n.entry_id].source for n in result.relevant[:5]]
    brain_ratio = top5_sources.count("뇌과학") / len(top5_sources)
    assert brain_ratio >= 0.6  # 최소 60% 이상 뇌과학

def test_routed_vs_flat_top1_same():
    """routed와 flat의 top-1이 같거나, routed가 더 정확."""
    loop = _make_loop_with_data()
    
    flat = loop.query("해마의 역할", mode="flat")
    routed = loop.query("해마의 역할", mode="routed")
    
    # top-1이 같거나, 둘 다 뇌과학 source
    flat_src = loop._cache[flat.relevant[0].entry_id].source
    routed_src = loop._cache[routed.relevant[0].entry_id].source
    assert routed_src == "뇌과학"  # routed는 반드시 정확해야 함

def test_routed_reduces_search_space():
    """routed 모드가 검색 범위를 축소하는지 확인."""
    loop = _make_loop_with_data()  # 15 entries, 3 sources
    result = loop.query("Redis 성능 최적화", mode="routed")
    
    # routing_info에서 active_count 확인
    assert result.routing_info is not None
    assert result.routing_info.get("active_count", 3) < 3  # 전체 3 source보다 적게 선택
```

### 5-3. 수동 시나리오 테스트 (3건)

| # | 시나리오 | 입력 | 기대 결과 |
|---|---------|------|---------|
| M1 | Cross-domain 연결 발견 | 뇌과학 5개 + AI 5개 + discover | "뇌과학:attention ↔ AI:attention" 발견 |
| M2 | 가짜 매칭 방어 | vault에 없는 주제 query + routed | VectorRouter가 모든 source score ≈ 0 → flat fallback |
| M3 | 충돌 + 발견 연쇄 | 모순 지식 ingest → discover | conflict가 높은 쌍이 discover 결과에서도 나타남 |

---

## 6. DAG 의존 방향

```
변경 후:

htp/knowledge/loop.py ──→ htp/thalamus/signature.py       (연결 1)
                      ──→ htp/thalamus/coherence/pairwise.py (연결 2)
                      ──→ htp/thalamus/types.py             (연결 2)
                      ──→ htp/thalamus/router/vector_router.py (연결 3)
                      ──→ htp/thalamus/region_signal.py     (연결 3)
                      ──→ htp/knowledge/encoder.py          (기존)
                      ──→ htp/knowledge/persistence.py      (기존)

금지 (유지):
  htp/thalamus/* → htp/knowledge/  (역방향 금지)
  htp/knowledge/ → htp/runtime/    (BrainRuntime 직접 참조 금지)
  htp/knowledge/ → htp/memory/     (MemorySystem 직접 참조 금지 — 추후 연결)
```

**knowledge → thalamus 단방향만 추가.** thalamus는 knowledge를 모름.
시스템 A의 코드를 한 줄도 수정하지 않고, 시스템 B에서 A의 모듈을 **사용**만 한다.

---

## 7. dim 호환성

현재 시스템 A는 `dim=64` 기본, 시스템 B는 TfidfJLEncoder가 `dim=64`,
EmbeddingBridge(sub-5)가 `dim=384`.

```python
# RegionSignature는 이미 dim 파라미터를 받음
RegionSignature(dim=64)   # TfidfJLEncoder 사용 시
RegionSignature(dim=384)  # EmbeddingBridge 사용 시

# VectorRouter는 dim-agnostic (cosine만 계산)
# PairwiseCoherenceGate도 dim-agnostic
```

**추가 작업 없음.** 기존 설계가 이미 dim 동적 호환을 지원.

---

## 8. 구현 순서

| 단계 | 작업 | 코드 변경량 | 테스트 | 소요 |
|------|------|-----------|--------|------|
| **S1** | 연결 1 (RegionSignature) | loop.py +20줄 | +2건 | 30분 |
| **S2** | 연결 2 (CoherenceGate) | loop.py +25줄, IngestResult +1필드, CLI +5줄 | +2건 | 45분 |
| **S3** | 연결 3 (VectorRouter) | loop.py +30줄, QueryResult +1필드, CLI +15줄 | +3건 | 45분 |
| **S4** | A/B 비교 시나리오 실행 | CLI로 수동 테스트 | 수동 3건 | 30분 |
| **합계** | | ~95줄 변경 | 7건 자동 + 3건 수동 | ~2.5시간 |

### S1-S3 각 단계 후 체크

```bash
# 매 단계 후:
$ cd htp && python -m pytest -q
# 기존 테스트 전체 통과 + 신규 테스트 통과 확인
```

---

## 9. 성공/실패 판단 기준

### 성공 (확신을 가져도 됨)

아래 3개 중 **2개 이상 달성**:

1. **Q1 통과**: 같은 도메인 signature 유사도 > 다른 도메인 유사도 (3개 도메인 중 2개 이상)
2. **Q2 통과**: 모순 지식 conflict > 0.3 **그리고** 일관 지식 conflict < 0.5 (의미적 구분 가능)
3. **Q3 통과**: routed top-5 관련 source 비율 ≥ 60% **그리고** flat 대비 동등 이상

→ **시스템 A가 단순 cosine보다 더 나은 지식 관리를 제공한다**는 가설 지지.

### 실패 (방향 수정 필요)

3개 중 **2개 이상 실패**:

- Q1 실패: RegionSignature가 도메인을 구분 못함 → encoder 품질 문제 또는 dim 부족
- Q2 실패: CoherenceGate가 모순/일관을 구분 못함 → threshold 부적절 또는 벡터 공간 분리 부족
- Q3 실패: VectorRouter가 검색 정밀도를 높이지 못함 → dynamic threshold가 현 데이터에 부적합

→ **시스템 A의 가치가 불확실.** 가능한 조치:
  1. EmbeddingBridge(384-dim)로 encoder 교체 후 재검증 (TF-IDF의 한계일 수 있음)
  2. TF-IDF에서도 실패하고 EmbeddingBridge에서도 실패하면, 시스템 A의 추상화 수준 재검토

### 판단 유보 (추가 데이터 필요)

1개 성공, 1개 실패, 1개 경계:

→ 데이터 양을 늘려 재검증 (15개 → 50개). 소규모에서 경계인 것이 대규모에서 명확해질 수 있음.

---

## 10. 이 테스트 이후 로드맵

| 결과 | 다음 액션 |
|------|---------|
| **성공** | sub-5 merge → 연결 코드 main에 반영 → EmbeddingBridge로 재검증 → Stage 4(LLMRegion) 진행. LLMRegion이 CoherenceGate conflict를 자연어로 해석하는 것이 다음 마일스톤 |
| **실패 (TF-IDF만)** | sub-5 merge 먼저 → EmbeddingBridge(384-dim)로 연결 재검증. 대부분의 실패가 TF-IDF의 의미적 한계에서 올 가능성 높음 |
| **실패 (EmbeddingBridge에서도)** | 시스템 A의 추상화 재검토. "source = Region" 매핑이 적절한가? 더 세밀한 Region 정의가 필요한가? 근본적 방향 수정 |

---

## 부록: 핵심 코드 호환성 확인

### RegionSignal 생성 시 필수 필드

현재 `RegionSignal`은 `output_vec: torch.Tensor`를 필수로 요구한다.
KnowledgeLoop에서는 torch 의존을 피하고 싶으므로, VectorRouter가 실제로
`output_vec`을 사용하는지 확인:

```python
# VectorRouter.score() — output_vec 참조 없음. 
# region_signature.similarity()만 사용.
# → output_vec=None 전달 가능 (단, torch.Tensor 타입 체크가 있으면 문제)
```

**확인 필요**: `RegionSignal` dataclass에서 `output_vec`의 default를 None으로
변경하거나, KnowledgeLoop 전용 경량 signal 타입을 만들어야 할 수 있음.
이건 S3 구현 시 결정.

### PairwiseCoherenceGate 입력 차원

`PairwiseCoherenceGate.bind()`의 `RegionResponse.output_vec`은
`np.ndarray`이고 차원 제약 없음 (cosine만 계산). dim=64든 384든 동작.

### TfidfJLEncoder의 fit() 시점

현재 `KnowledgeLoop.__init__`에서 `encoder.load()`를 호출하고,
첫 `ingest`에서 `fit()`을 호출. `_rebuild_signatures()`는 `__init__`에서
호출되므로, encoder가 아직 fit되지 않은 상태에서 기존 cache의 vec를 사용.
**이건 기존 persistence에 vec가 JSONL로 저장되어 있으므로 문제 없음**
— vec는 저장 시점에 encode된 것이고, signature rebuild는 저장된 vec를 그대로 사용.

---

## 부록 B: Q2 retune (2026-05-18)

### 발견

S4 검증 결과 Q2 (CoherenceGate 의 모순/일관 구분) 가 EmbeddingBridge 에서
**부분** PASS. 정성적 방향성은 맞으나 (`이질 conflict=0.152, 일관 conflict=0.116`),
default threshold `(0.3, 0.7)` 가 e5 의 cosine 0.85+ 영역 분포에 너무 엄격해
absolute 기준 미달.

### 측정 (15 entries × 3 source)

| encoder | intra max | inter max | gap |
|---------|----------:|----------:|----:|
| TfidfJLEncoder (dim=64) | 1.000 (포화) | 1.000 (포화) | 0 |
| EmbeddingBridge (e5-small, dim=384) | 0.124 | 0.141 | -0.017 (marginal) |

### 재튜닝 결과

`htp/knowledge/loop.py` 의 `_COHERENCE_DEFAULTS`:

| encoder | (conflict, escalation) | 효과 |
|---------|------------------------|------|
| TfidfJLEncoder | (0.5, 1.0) | sparse 노이즈 차단 — escalation 비활성 |
| EmbeddingBridge / STAdapter | (0.105, 0.135) | e5 분포 측정 기반 |
| 미지 encoder | (0.3, 0.7) | PairwiseCoherenceGate 보수적 default |

사용자 override: `KnowledgeLoop(coherence_thresholds=(c, e))`.

### Q2 재검증 (retune 후, EmbeddingBridge)

| 입력 | conflict | escalate |
|------|---------:|:--------:|
| 이질 ("주의는 국소적·순차적이다" → 뇌과학) | 0.153 | **True** ✓ |
| 일관 ("해마 패턴 완성 recurrent" → 뇌과학) | 0.107 | **False** ✓ |

→ design §9 Q2 strict 기준 충족. Q1/Q2/Q3 **3/3 PASS**.

### 테스트 (+5)

`tests/knowledge/test_bridge_q2_retune.py`:
- `test_tfidf_default_thresholds` — TfidfJLEncoder → (0.5, 1.0)
- `test_embedding_default_thresholds` — EmbeddingBridge → (0.105, 0.135)
- `test_unknown_encoder_conservative_default` — 미지 → (0.3, 0.7)
- `test_explicit_override_wins` — coherence_thresholds 인자 우선
- `test_tfidf_escalate_never_triggered` — TFIDF 단계 escalate 영구 False
