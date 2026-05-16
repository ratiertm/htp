# HTP Thalamus CAR 설계서

**Content-Addressable Parallel Router로의 Thalamus 재설계**

기준: 2026-05-16 · 커밋 `6be8746` 기반 · 회귀 테스트 57/57 위에 적층  
선행 문서: `htp_memory_design_final.md`, `htp_friston_review.md`, `htp_multimodal_design.md`  
프로젝트 리뷰: `htp-project-review.md` §3-C 미반영 항목 중 2건 해소 대상  
**Rev 1.1**: 교차 리뷰 피드백 반영 — HTPConfig DI 조기 처리, CoherenceGate LSH 임계값 확정(N=16), vec↔prompt 3단계 전략 + CostRouter 4-Level 확장  
**Rev 1.2**: Stage 순서 재정렬 — Config 분리를 Stage 0으로 선행, EmbeddingBridge를 실험 트랙으로 분리, 전 Stage Go/No-Go 기준 추가

---

## 0. 동기 (Why Now)

### 0-1. 신경과학적 근거

뇌의 기억은 주소가 아니라 내용으로 저장·인출된다. "사과"라는 기억은 시각피질(빨간색, 둥근 형태) + 측두엽(음운) + 변연계(감정)에 분산된 뉴런 앙상블의 동시 활성화 패턴이며, 사과 향기라는 부분 단서 하나만으로 CA3의 recurrent connection이 전체 패턴을 수십 ms 안에 병렬 복원한다. 이것이 content-addressable memory이다.

이 병렬 인출을 가능하게 하는 세 조건:
1. **분산 표상** — 하나의 기억 = 다수 뉴런의 동시 활성화 패턴 (앙상블 코딩)
2. **시냅스 가소성** — Hebb's rule로 동시 발화 뉴런 간 연결 강화 → 엔그램 형성
3. **시간적 바인딩** — 감마(30–100Hz)/세타(4–8Hz) 진동이 분산된 뉴런 집단의 발화를 동기화하여 하나의 통합 경험으로 묶음

현재 HTP는 (1)과 (2)를 64-dim 상태 벡터 + Hebbian EMA로 구현했으나, 라우팅이 문자열 태그 매칭(= 주소 기반)이고 (3) 시간적 바인딩이 부재하다. 이 설계서는 Thalamus를 이 세 조건을 모두 충족하는 content-addressable parallel router로 재설계한다.

### 0-2. 프로젝트 리뷰에서 도출된 3대 갭

| # | 갭 | 리뷰 근거 | 이 설계서의 해소 방법 |
|---|---|----------|-------------------|
| G1 | 시맨틱 라우팅 부재 | §3-C "임베딩 기반 시맨틱 라우팅" 미반영 | CoreCells tag→벡터 유사도 전환 |
| G2 | Temporal binding 없음 | 다중 Region 응답의 의미적 묶음 부재 | Coherence gate 도입 |
| G3 | LLMNode 그래프 고립 | §부록A "176 isolated 노드" | LLMRegionRuntime 추상 경계 재정의 |

### 0-3. 설계 원칙 (기존 4대 원칙과의 정합성)

- **원칙 1 "구조는 데이터가 만든다"** → 태그 매칭을 벡터 유사도로 바꾸면, 라우팅 경로가 데이터의 의미 구조에서 자동 결정됨
- **원칙 2 "허브는 창발한다"** → 벡터 공간에서 자주 매칭되는 Region이 자연스럽게 허브로 승격
- **원칙 4 "판단은 위임한다"** → LLM 노드가 동일한 벡터 인터페이스로 참여, 별도 경로 불필요

---

## 1. Content-Addressable Routing (G1 해소)

### 1-1. 현재 구조의 문제

```
현재: input.tags = {"vision", "red"} → CoreCells가 tag ∩ region.tags로 매칭
문제: "붉은 둥근 과일"은 "vision" 태그가 없으면 시각 Region에 도달 불가
본질: 이것은 주소 기반 접근이다
```

### 1-2. 목표 구조

```
목표: input → embed(input) → 64-dim 벡터 q
      각 Region → region_signature 64-dim 벡터 k_i (학습됨)
      CoreCells: score_i = cosine(q, k_i) × precision_i
      threshold 이상인 모든 Region에 동시 라우팅
```

핵심 전환: **태그 집합의 교집합 연산 → 벡터 공간의 유사도 연산**

### 1-3. Region Signature 벡터

각 Region은 자신이 처리한 입력들의 통계로부터 64-dim signature를 유지한다.

```python
class RegionSignature:
    """Region의 content-addressable 키. 처리한 입력의 EMA."""
    
    def __init__(self, dim: int = 64):
        self.centroid = np.zeros(dim)      # 처리한 입력의 EMA 중심
        self.count = 0
        self.lr_fn = lambda n: 1.0 / (n + 1)  # 기존 Hebbian EMA와 동일
    
    def update(self, input_vec: np.ndarray):
        """입력 처리 후 signature 갱신. Online Hebbian EMA."""
        lr = self.lr_fn(self.count)
        self.centroid = (1 - lr) * self.centroid + lr * input_vec
        self.count += 1
    
    def similarity(self, query: np.ndarray) -> float:
        """content-addressable lookup. 코사인 유사도."""
        norm_q = np.linalg.norm(query)
        norm_c = np.linalg.norm(self.centroid)
        if norm_q < 1e-8 or norm_c < 1e-8:
            return 0.0
        return float(np.dot(query, self.centroid) / (norm_q * norm_c))
```

**기존 Hebbian EMA(`lr = 1/(count+1)`)를 그대로 재활용.** L3 PatternStore에서 검증된 수식이므로 새로운 하이퍼파라미터 도입 없음.

### 1-4. CoreCells 라우팅 전환

```python
# AS-IS (tag 매칭)
def gate(self, signals: list[RegionSignal]) -> RegionSignal:
    scores = []
    for sig in signals:
        tag_overlap = len(sig.tags & self.current_input_tags)
        biased = sig.precision * tag_overlap + self.td_bias.get(sig.region_id, 0)
        scores.append(biased)
    winner = signals[argmax(scores)]
    return winner

# TO-BE (content-addressable)
def gate(self, signals: list[RegionSignal], query_vec: np.ndarray) -> list[RegionSignal]:
    scored = []
    for sig in signals:
        sim = sig.region_signature.similarity(query_vec)
        biased = sig.precision * sim + self.td_bias.get(sig.region_id, 0)
        scored.append((sig, biased))
    
    # winner-take-all → threshold-based multi-select
    threshold = self._dynamic_threshold(scored)
    selected = [(sig, s) for sig, s in scored if s >= threshold]
    return selected  # 복수 Region 동시 선택 가능
```

**핵심 변경 3가지:**
1. `tag_overlap` (이산) → `cosine_similarity` (연속)
2. `winner = argmax` (단일 승자) → `threshold` (복수 선택) — 병렬 라우팅의 전제
3. `query_vec`가 입력의 64-dim 임베딩 — RegionSignal에 이미 `state_vec`가 있으므로 인터페이스 일관

### 1-5. Dynamic Threshold

고정 threshold는 Region 수에 따라 민감도가 변한다. Precision-weighted 동적 임계값:

```
θ_dynamic = μ(scores) + β × σ(scores)
```

- `β = 0.5` 기본값 (HTPConfig에 파라미터화)
- 모든 score가 비슷하면 (σ 작음) → 다수 Region 활성화 (분산 처리)
- 하나가 압도적이면 (σ 큼) → 소수 Region만 활성화 (집중 처리)

이 동작은 뇌의 thalamic gating과 대응: 명확한 자극은 특정 피질 영역으로 집중, 모호한 자극은 여러 영역에 분산.

### 1-6. 하위 호환성 전략

tag 매칭을 즉시 제거하지 않는다.

```python
class CoreCells:
    def __init__(self, config: HTPConfig):
        self.routing_mode = config.routing_mode  # "tag" | "vector" | "hybrid"
    
    def gate(self, signals, query_vec=None):
        if self.routing_mode == "tag":
            return self._gate_tag(signals)        # 기존 로직 보존
        elif self.routing_mode == "vector":
            return self._gate_vector(signals, query_vec)
        else:  # hybrid
            tag_score = self._tag_score(signals)
            vec_score = self._vec_score(signals, query_vec)
            combined = self.alpha * vec_score + (1 - self.alpha) * tag_score
            return self._threshold_select(signals, combined)
```

**마이그레이션 경로**: tag → hybrid(α=0.3) → hybrid(α=0.7) → vector. 각 단계에서 회귀 테스트 통과 확인.

### 1-7. Modern Hopfield 해석

위 구조의 수학적 의미를 명확히 한다.

Region signature `k_i`를 Key, 입력 벡터 `q`를 Query, Region 출력을 Value로 놓으면:

```
attention(q, K, V) = softmax(q · K^T / √d) · V
```

CoreCells의 content-addressable routing은 single-head attention의 특수 사례이다. precision 가중치는 Key별 temperature에 해당하고, dynamic threshold는 sparse attention mask에 해당한다.

이 해석이 중요한 이유: 향후 multi-head attention으로 확장하면, 하나의 입력이 여러 "관점"에서 동시에 라우팅될 수 있다. 뇌에서 같은 시각 자극이 what pathway(측두엽)와 where pathway(두정엽)에 동시에 라우팅되는 것과 대응.

---

## 2. 병렬 실행과 Temporal Binding (G2 해소)

### 2-1. 현재 병렬성의 한계

`AsyncBrainRuntime`의 `asyncio.gather`가 LLM 노드를 동시 호출하지만, 두 가지가 빠져 있다:

1. **결과 바인딩**: 여러 Region의 응답이 돌아올 때 "이것들이 같은 입력에 대한 응답"이라는 시간적 묶음이 없음
2. **비정합 탐지**: Region A는 "사과"로 해석하고 Region B는 "토마토"로 해석했을 때, 이 불일치를 감지하는 메커니즘 없음

### 2-2. Coherence Gate 설계

뇌의 감마 진동(30–100Hz)은 분산된 뉴런 집단의 발화 타이밍을 동기화하여, 관련 정보끼리 하나의 "묶음"으로 엮는다. 소프트웨어에서 이 시간적 바인딩을 벡터 정합성(coherence)으로 대체한다.

```python
class CoherenceGate:
    """
    다중 Region 응답의 시간적 바인딩.
    뇌의 감마 동기화 → 벡터 공간 정합성 검사.
    """
    
    def __init__(self, config: HTPConfig):
        self.conflict_threshold = config.coherence_conflict_threshold  # 기본 0.3
        self.agreement_threshold = config.coherence_agreement_threshold  # 기본 0.7
    
    def bind(self, responses: list[RegionResponse]) -> BoundResponse:
        """
        복수 Region 응답을 하나의 통합 응답으로 바인딩.
        
        3단계:
        1. 쌍별 코사인 유사도 계산 → coherence matrix
        2. 클러스터링: 높은 coherence끼리 그룹
        3. conflict가 있으면 PFC에 에스컬레이션
        """
        n = len(responses)
        if n <= 1:
            return BoundResponse(responses, coherence=1.0, conflict=False)
        
        # 1) Pairwise coherence matrix
        vecs = [r.state_vec for r in responses]
        coherence_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                coherence_matrix[i][j] = cosine_similarity(vecs[i], vecs[j])
                coherence_matrix[j][i] = coherence_matrix[i][j]
        
        # 2) Mean coherence
        mean_coherence = coherence_matrix[np.triu_indices(n, k=1)].mean()
        
        # 3) Conflict detection
        min_pair = coherence_matrix[np.triu_indices(n, k=1)].min()
        has_conflict = min_pair < self.conflict_threshold
        
        if has_conflict:
            # PFC 에스컬레이션: 목표 정렬 기준으로 승자 결정
            return BoundResponse(
                responses, coherence=mean_coherence,
                conflict=True, escalate_to_pfc=True
            )
        
        # 4) Precision-weighted fusion
        weights = np.array([r.precision for r in responses])
        weights /= weights.sum()
        fused_vec = sum(w * v for w, v in zip(weights, vecs))
        
        return BoundResponse(
            responses, coherence=mean_coherence,
            conflict=False, fused_vec=fused_vec
        )
```

### 2-3. CA1 Mismatch와의 연결

현재 Memory의 CA3→CA1 경로에 `mismatch` L2 거리 분기가 있다. CoherenceGate의 conflict detection은 이것의 상위 레벨 버전이다:

```
CA3-CA1 mismatch: 기억 패턴 내부의 불일치 (저장 vs 인출)
CoherenceGate conflict: Region 간 해석의 불일치 (동시 처리 결과 간)
```

둘 다 "예측과 현실의 차이"를 감지한다는 점에서 Friston의 prediction error와 일관된다. CoherenceGate의 conflict signal을 Memory의 novelty score에 반영하면, "Region 간 의견 불일치가 큰 입력"이 SWR consolidation에서 높은 우선순위를 받는다.

```python
# Memory 연동
novelty_boost = 1.0 + bound_response.conflict_magnitude  # conflict가 클수록 novelty↑
swr_priority = novelty_boost * novelty * reward  # 기존 SWR 공식 확장
```

### 2-4. 3-Level 병렬성 아키텍처

| 레벨 | 대상 | 메커니즘 | 뇌 대응 |
|------|-----|---------|---------|
| **L1: 벡터 연산** | 유사도 계산, Hebbian 업데이트, pattern completion | PyTorch CUDA 배치 연산 | 시냅스 수준 동시 연산 |
| **L2: Region 병렬** | 다중 Region 동시 실행 | `asyncio.gather` + CoherenceGate | 피질 영역 간 병렬 처리 |
| **L3: 파이프라인** | 연속 입력의 겹침 처리 | Thalamus t₂ 라우팅 ∥ Region t₁ 처리 | V1→V2→V4 파이프라인 |

### 2-5. L1 벡터 연산 배치화

현재 signature similarity가 Region마다 순차 계산된다. 이를 단일 행렬 연산으로:

```python
# AS-IS: for loop
scores = [sig.region_signature.similarity(query_vec) for sig in signals]

# TO-BE: batch matrix op
K = torch.stack([sig.region_signature.centroid for sig in signals])  # (N, 64)
q = torch.tensor(query_vec).unsqueeze(0)  # (1, 64)
scores = F.cosine_similarity(q, K, dim=1)  # (N,) — 단일 연산
```

Region 수가 작을 때(< 10) 차이가 미미하지만, 멀티모달 확장으로 Region이 수십 개로 늘어나면 이 배치화가 필수가 된다.

### 2-6. L3 파이프라인 병렬성

```
시간축 →    t₁         t₂         t₃
Thalamus:  route(x₁)  route(x₂)  route(x₃)
Region A:             process(x₁) process(x₂)
Region B:             process(x₁) process(x₂)
Coherence:                        bind(x₁)    bind(x₂)
PFC:                                           decide(x₁)
```

`AsyncBrainRuntime`에 입력 큐를 추가하면, Thalamus가 다음 입력을 라우팅하는 동안 Region들은 이전 입력을 처리할 수 있다. latency는 줄지 않지만 throughput이 Region 수에 비례하여 증가.

```python
class PipelinedBrainRuntime:
    """L3 파이프라인 병렬성. 연속 입력 스트림 처리."""
    
    async def run_stream(self, inputs: AsyncIterator[np.ndarray]):
        pipeline = asyncio.Queue(maxsize=3)  # 3-stage 파이프라인 버퍼
        
        async def router_stage():
            async for x in inputs:
                routes = self.thalamus.gate(x)
                await pipeline.put((x, routes))
        
        async def process_stage():
            while True:
                x, routes = await pipeline.get()
                results = await asyncio.gather(
                    *[r.region.process(x) for r in routes]
                )
                bound = self.coherence_gate.bind(results)
                yield bound
        
        # 두 stage가 동시 실행
        asyncio.create_task(router_stage())
        async for bound in process_stage():
            yield self.pfc.decide(bound)
```

---

## 3. LLM 통합 경계 재정의 (G3 해소)

### 3-1. 현재 문제

`LLMRegionRuntime(RegionRuntime)` 상속이 PageRank 허브 형성, Hebbian 가소성, NGE 분열 등 Region 내부 메커니즘을 전부 상속한다. 프로젝트 리뷰의 질문: "LLM 노드에 PageRank 허브 형성이 정말 필요한가?"

답: **필요 없다.** LLM은 내부에 이미 자체 attention 메커니즘이 있다. HTP가 관리해야 하는 것은 LLM의 내부가 아니라, LLM과 나머지 시스템 사이의 인터페이스이다.

### 3-2. 새로운 추상 계층

```
RegionRuntime (HTPRuntime 상속)
  ├── NativeRegion    — 기존 4-engine 피질 영역. 내부에 허브/가지치기/발생 있음
  └── ExternalRegion  — 외부 처리기 래퍼. 내부 구조 없음, 인터페이스만 준수
        ├── LLMRegion     — LLM API 호출
        ├── SensorRegion   — 센서 입력 (멀티모달 확장 대비)
        └── ToolRegion     — 외부 도구 호출 (향후)
```

```python
class ExternalRegion:
    """
    외부 처리기를 Region 인터페이스로 래핑.
    
    RegionRuntime을 상속하지 않음 — 대신 같은 프로토콜을 구현:
    - input: 64-dim state_vec
    - output: RegionSignal (state_vec + precision + metadata)
    - signature: RegionSignature (content-addressable key)
    """
    
    def __init__(self, config: HTPConfig):
        self.signature = RegionSignature(dim=config.compress_dim)
        self.cost_router = CostRouter(config)  # LLM 전용: 비용 관리
    
    async def process(self, query_vec: np.ndarray) -> RegionSignal:
        """프로토콜 준수: 입력 벡터 → RegionSignal"""
        raise NotImplementedError
    
    def update_signature(self, input_vec: np.ndarray):
        """처리 후 signature 갱신 — content-addressable routing에 참여"""
        self.signature.update(input_vec)


class LLMRegion(ExternalRegion):
    """LLM API 호출을 Region 프로토콜로 래핑."""
    
    async def process(self, query_vec: np.ndarray) -> RegionSignal:
        # 1) 벡터 → 텍스트 프롬프트 변환 (decoder)
        prompt = self.vec_to_prompt(query_vec)
        
        # 2) LLM 호출 (CostRouter가 모델 선택)
        model = self.cost_router.select_model()
        response = await self.llm_client.complete(model, prompt)
        
        # 3) 텍스트 → 벡터 변환 (encoder)
        output_vec = self.prompt_to_vec(response)
        
        # 4) precision 추정 (응답 확신도)
        precision = self._estimate_precision(response)
        
        return RegionSignal(
            state_vec=output_vec,
            precision=precision,
            region_id=self.region_id,
            region_signature=self.signature
        )
```

### 3-3. 그래프 고립 해소

이 변경으로 LLMNode/CostRouter가 고립되는 이유가 사라진다:

- **기존**: `LLMRegionRuntime` → (상속) → `RegionRuntime` — 상속이 유일한 연결점이었고, 실제로 Region의 내부 메커니즘을 사용하지 않음
- **변경 후**: `LLMRegion` → (프로토콜) → `RegionSignal`, `RegionSignature`, `CostRouter` — 실제 사용하는 것만 연결

Knowledge Graph에서 `LLMRegion`은 `RegionSignal`(C4), `RegionSignature`(신규), `CostRouter`(C6)와 직접 연결되어, 기존 176개 isolated 노드 중 LLM 관련 노드가 그래프 본체에 합류한다.

### 3-4. vec ↔ prompt 변환: 3단계 전략

LLMRegion의 가장 큰 실용적 병목은 `vec_to_prompt`와 `prompt_to_vec`이다. 64-dim 벡터를 LLM이 이해하는 텍스트로 변환하고 다시 돌려야 한다. 이 병목의 근본 원인은 **LLM이 텍스트 공간에서 작동하고 HTP가 벡터 공간에서 작동하기 때문**이다. 두 공간 사이의 왕복이 비용의 본질이다.

```
현재 경로:
벡터 → [vec_to_prompt] → 텍스트 → LLM → 텍스트 → [prompt_to_vec] → 벡터
         ↑ 병목 1                              ↑ 병목 2
```

#### 3-4-1. 단기 전략: Dim-Tag 매핑 (Stage 4)

```python
def vec_to_prompt(self, vec: np.ndarray) -> str:
    # top-k 활성 차원을 의미 태그로 매핑
    # 기존 tag 시스템을 차원-태그 사전으로 재활용
    top_dims = np.argsort(np.abs(vec))[-5:]
    tags = [self.dim_tag_map[d] for d in top_dims]
    return f"Context: {', '.join(tags)}. Task: ..."
```

손실이 크지만 동작한다. 이 단계의 목적은 `LLMRegion` 인터페이스를 확정하는 것이지, 변환 품질을 높이는 게 아니다.

#### 3-4-2. 중기 전략: EmbeddingBridge (Stage 6 실험 브랜치)

sLLM 전체를 쓰는 것보다 **sLLM의 임베딩 레이어만 추출**해서 양방향 프로젝션으로 활용하는 것이 핵심이다. sLLM 전체 추론(수백 ms)이 아니라 인코딩 + 선형 프로젝션(수 ms)만 필요하므로 비용과 지연이 극적으로 줄어든다.

```python
class EmbeddingBridge:
    """
    sLLM 임베딩 레이어를 양방향 프로젝션으로 활용.
    
    구조: HTP 64-dim ↔ Linear Projection ↔ sLLM 384-dim ↔ 텍스트
    비용: 인코딩 + 선형 프로젝션 ~5ms (sLLM 전체 추론 ~100ms 대비)
    학습: project_down/up은 학습 가능 — HTP 운영 중 변환 품질 점진 개선 (가소성)
    """
    
    def __init__(self, model_name: str = "BAAI/bge-small-ko-v1.5"):
        # 한국어 특화 소형 임베딩 모델 (~33M 파라미터)
        self.encoder = SentenceTransformer(model_name)
        self.embed_dim = 384  # bge-small 출력 차원
        
        # 384-dim ↔ 64-dim 선형 프로젝션 (학습 가능)
        self.project_down = nn.Linear(384, 64)   # embed → HTP 벡터
        self.project_up = nn.Linear(64, 384)     # HTP 벡터 → embed
    
    def vec_to_prompt(self, htp_vec: np.ndarray) -> str:
        """64-dim → 384-dim → 가장 가까운 텍스트 표현"""
        embed = self.project_up(torch.tensor(htp_vec))
        return self._embed_to_prompt(embed)
    
    def prompt_to_vec(self, text: str) -> np.ndarray:
        """텍스트 → 384-dim → 64-dim"""
        embed = self.encoder.encode(text)
        return self.project_down(torch.tensor(embed)).detach().numpy()
    
    def train_step(self, text: str, htp_vec: np.ndarray):
        """
        온라인 학습: 실제 사용 데이터로 프로젝션 품질 개선.
        HTP의 Hebbian 가소성과 동일한 철학 — 사용할수록 정확해진다.
        """
        embed = torch.tensor(self.encoder.encode(text))
        projected = self.project_down(embed)
        target = torch.tensor(htp_vec)
        loss = F.mse_loss(projected, target)
        loss.backward()
        # optimizer.step() — 외부에서 관리
```

**sLLM 모델 선택 기준:**
- 한국어 지원 필수 → `BAAI/bge-small-ko-v1.5` (384-dim, ~33M)
- 영어 전용 환경 → `all-MiniLM-L6-v2` (384-dim, ~22M)
- 다국어 → `paraphrase-multilingual-MiniLM-L12-v2` (384-dim, ~118M)
- 모두 로컬 실행 가능, GPU 불필요

#### 3-4-3. 장기 전략: Continuous Token — 변환 제거 (ModalEncoder 통합 후)

궁극적 목표는 **변환 자체를 없애는 것**이다. `htp_multimodal_design.md`의 ModalEncoder가 완성되면, 텍스트도 벡터도 모두 동일한 64-dim 공간의 모달리티가 된다.

```
장기 경로:
벡터 → LLM(벡터 입출력 모드) → 벡터
         ↑ 변환 없음
```

LLM의 입력 임베딩 레이어를 건너뛰고, 64-dim 벡터를 LLM의 hidden dimension으로 직접 프로젝션하는 "continuous token" 또는 "soft prompt" 방식이다. 이 경우 로컬 sLLM이 필수 — API 기반 LLM은 continuous input을 받지 않는다.

```
단기 (dim-tag):     벡터 → 이산 태그 → 텍스트 → LLM → 텍스트 → 태그 → 벡터
중기 (bridge):      벡터 → 선형 프로젝션 → 임베딩 → 텍스트 → LLM → 텍스트 → 임베딩 → 선형 프로젝션 → 벡터
장기 (continuous):  벡터 → 선형 프로젝션 → LLM hidden → 선형 프로젝션 → 벡터
```

변환 단계가 줄어들수록 정보 손실과 지연이 감소한다.

### 3-5. CostRouter 4-Level 의사결정 트리

vec ↔ prompt 3단계 전략에 맞춰 CostRouter를 확장한다. 기존 EMA 비용 압박 메커니즘이 4단계 위에서 그대로 작동한다.

```python
class CostRouter:
    """
    확장된 비용 라우팅. 변환 유형에 따라 4단계 의사결정.
    
    대부분의 호출이 Level 1-2에서 처리되면 API 비용이 드라마틱하게 감소.
    기존 EMA 비용 압박은 Level 3-4 사이에서 작동.
    """
    
    # Level  용도              실행 위치   비용    지연
    # ─────────────────────────────────────────────────
    # 1      vec↔prompt 변환   로컬        0      ~5ms    ← EmbeddingBridge
    # 2      간단한 추론        로컬 sLLM   0      ~100ms  ← 로컬 소형 모델
    # 3      복잡한 추론        API 소형    낮음   ~300ms  ← 기존 CostRouter 하한
    # 4      고품질 추론        API 대형    높음   ~1s     ← 기존 CostRouter 상한
    
    def select_level(self, task_complexity: float, budget_pressure: float) -> int:
        """
        task_complexity: Region이 추정한 입력 복잡도 [0, 1]
        budget_pressure: EMA 비용 압박 [0, 1] (기존 메커니즘)
        """
        if task_complexity < 0.2:
            return 1  # 변환만 필요
        elif task_complexity < 0.5 or budget_pressure > 0.8:
            return 2  # 로컬 sLLM으로 충분하거나 예산 압박
        elif task_complexity < 0.8:
            return 3  # API 소형 모델
        else:
            return 4  # 고품질 필요
```

**Level 1–2가 처리하는 비율 목표: 전체 호출의 70% 이상.** 이 비율이 달성되면 API 비용이 현재 대비 ~70% 절감된다.

---

## 4. 통합 아키텍처

### 4-1. 변경 후 데이터 흐름

```
입력 x
  │
  ▼
Encoder: x → q (64-dim query vector)
  │
  ▼
Thalamus CoreCells: content-addressable routing
  │  q · K^T (K = Region signatures 행렬)
  │  precision weighting + top-down bias
  │  dynamic threshold → 복수 Region 선택
  │
  ├──────────┬──────────┐
  ▼          ▼          ▼          ← asyncio.gather (L2 병렬)
Region_A   Region_B   LLMRegion
  │          │          │
  ▼          ▼          ▼
  └──────────┴──────────┘
             │
             ▼
      CoherenceGate: temporal binding
        │  pairwise coherence → conflict detection
        │  precision-weighted fusion (or PFC escalation)
        │
        ▼
      PFCRuntime: goal alignment + action selection
        │  WM deque[7] 업데이트
        │  top-down bias 갱신 → Thalamus로 피드백
        │
        ▼
      Memory: consolidation
        │  novelty × reward × (1 + conflict_magnitude) → SWR priority
        │  CA3 pattern completion, L3 Go-CLS 승격
        │
        ▼
      Action 출력 + Region signature 업데이트
```

### 4-2. 새로운/변경 클래스 목록

| 클래스 | 위치 | 신규/변경 | 역할 |
|--------|------|---------|------|
| `RegionSignature` | `htp/thalamus/signature.py` | **신규** | Region의 content-addressable 키 |
| `CoherenceGate` | `htp/thalamus/coherence.py` | **신규** | 다중 Region 응답의 temporal binding |
| `BoundResponse` | `htp/thalamus/types.py` | **신규** | CoherenceGate 출력 데이터 |
| `ExternalRegion` | `htp/runtime/external_region.py` | **신규** | 외부 처리기 프로토콜 |
| `LLMRegion` | `htp/llm/llm_region.py` | **신규** (기존 `LLMRegionRuntime` 대체) | LLM을 Region 프로토콜로 래핑 |
| `EmbeddingBridge` | `htp/llm/embedding_bridge.py` | **신규** | sLLM 임베딩 기반 vec↔prompt 양방향 프로젝션 |
| `CoreCells.gate()` | `htp/thalamus/core_cells.py` | **변경** | tag→벡터 유사도 라우팅 |
| `RegionSignal` | `htp/thalamus/types.py` | **변경** | `region_signature` 필드 추가 |
| `CostRouter` | `htp/llm/cost_router.py` | **변경** | 4-Level 의사결정 트리 확장 |
| `HTPConfig` | `htp/config.py` | **변경** | 서브 Config 분리 (§4-3 참조) |
| `PipelinedBrainRuntime` | `htp/runtime/pipelined_brain.py` | **신규** | L3 파이프라인 병렬성 |

### 4-3. HTPConfig 서브 Config 분리 (Stage 0 — 최우선)

프로젝트 리뷰에서 41 엣지 god-object로 지적된 `HTPConfig`에 추가 파라미터를 투입하면 비대화가 가속된다. **Config 분리를 모든 기능 구현보다 먼저 수행**하여, 이후 Stage에서 추가되는 파라미터가 처음부터 올바른 서브 Config에 들어가도록 한다. 토대가 정리되지 않은 상태에서 기능을 쌓으면 나중에 다시 손대야 한다.

```python
@dataclass
class RoutingConfig:
    """Content-Addressable Routing 관련 설정"""
    mode: str = "hybrid"                    # "tag" | "vector" | "hybrid"
    alpha: float = 0.5                      # hybrid 모드에서 vector 비중
    threshold_beta: float = 0.5             # dynamic threshold = μ + β×σ
    warmup_steps: int = 10                  # signature 냉시작 보호

@dataclass
class CoherenceConfig:
    """CoherenceGate 관련 설정"""
    conflict_threshold: float = 0.3         # 이하 → conflict
    agreement_threshold: float = 0.7        # 이상 → strong agreement
    novelty_boost: float = 1.0              # conflict → SWR priority 증폭 계수
    lsh_transition_n: int = 16              # Region ≥ N이면 LSH 근사 전환

@dataclass
class LLMBridgeConfig:
    """EmbeddingBridge + CostRouter 관련 설정"""
    embedding_model: str = "BAAI/bge-small-ko-v1.5"
    embed_dim: int = 384
    cost_level_thresholds: tuple = (0.2, 0.5, 0.8)  # Level 1/2/3/4 경계
    budget_pressure_threshold: float = 0.8            # Level 강제 하향 임계

@dataclass
class PipelineConfig:
    """PipelinedBrainRuntime 관련 설정"""
    buffer_size: int = 3                    # 파이프라인 스테이지 버퍼

@dataclass
class HTPConfig:
    """
    최상위 Config. 서브 Config 합성.
    
    기존 Phase 1-5 파라미터는 그대로 유지.
    CAR 신규 파라미터는 서브 Config에 격리.
    
    엣지 분석: 기존 41 → 서브 Config 합성으로 직접 엣지 ~35로 감소 예상.
    (서브 Config가 중간 노드로 의존성을 흡수)
    """
    # ... 기존 Phase 1-5 파라미터 유지 ...
    
    routing: RoutingConfig = field(default_factory=RoutingConfig)
    coherence: CoherenceConfig = field(default_factory=CoherenceConfig)
    llm_bridge: LLMBridgeConfig = field(default_factory=LLMBridgeConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
```

**엣지 절감 효과**: CAR 신규 파라미터가 서브 Config를 경유하므로, `CoreCells`는 `HTPConfig`가 아닌 `RoutingConfig`에만 의존한다. graphify 기준 `HTPConfig` 직접 엣지가 41 → ~35로 감소하고, CAR 파라미터 추가에도 불구하고 god-object 문제가 악화되지 않는다.

---

## 5. 회귀 테스트 전략

### 5-1. 기존 57개 테스트 보호

`routing_mode = "tag"` 기본값으로 모든 기존 테스트가 변경 없이 통과해야 한다. 새 코드는 `routing_mode = "vector"` 또는 `"hybrid"`일 때만 활성화.

### 5-2. 신규 테스트 — 본선 트랙 (예상 22–25개)

| Stage | 그룹 | 테스트 | 검증 대상 |
|-------|------|--------|----------|
| **0** | **Config DI** | config_sub_routing | RoutingConfig 독립 생성 및 접근 |
| | | config_sub_coherence | CoherenceConfig 독립 생성 및 접근 |
| | | config_backward_compat | 기존 HTPConfig 인터페이스 하위 호환 |
| **1** | **Signature** | signature_update_ema | EMA 수렴: 동일 입력 N회 → centroid ≈ input |
| | | signature_similarity_orthogonal | 직교 벡터 → similarity ≈ 0 |
| | | signature_similarity_identical | 동일 벡터 → similarity ≈ 1 |
| | | signature_zero_vector | 영벡터 입력 → similarity = 0 (division by zero 방어) |
| | **CAR** | car_single_match | 1개 Region만 threshold 초과 → 단일 라우팅 |
| | | car_multi_match | 복수 Region threshold 초과 → 병렬 라우팅 |
| | | car_no_match | 모든 Region threshold 미달 → fallback 동작 |
| | | car_precision_weighting | 높은 precision Region이 낮은 similarity에도 선택 |
| | | car_topdown_bias | td_bias가 routing 결과에 영향 |
| | | car_dynamic_threshold | σ 큰 경우 소수 선택, σ 작은 경우 다수 선택 |
| **2** | **Hybrid** | hybrid_alpha_zero | α=0 → tag 전용 (기존 동작 동일) |
| | | hybrid_alpha_one | α=1 → vector 전용 |
| | | hybrid_migration | α 0.1→0.9 점진 증가 시 결과 연속적 변화 (급변 없음) |
| **3** | **Coherence** | coherence_identical | 동일 응답들 → coherence ≈ 1, conflict=False |
| | | coherence_conflict | 직교 응답들 → conflict=True, escalate=True |
| | | coherence_partial | 2/3 정합, 1/3 불일치 → conflict 감지 |
| | | coherence_precision_fusion | precision 높은 응답이 fusion에서 높은 가중치 |
| | | coherence_swr_boost | conflict=True → SWR priority 증가 |
| | | coherence_lsh_fallback | Region ≥ 16 시 LSH 근사 전환 확인 |
| **4** | **LLMRegion** | llm_region_protocol | LLMRegion이 NativeRegion 없이 RegionSignal 프로토콜 준수 |
| | | llm_region_signature_update | 처리 후 signature 갱신 |
| | | llm_region_cost_level | CostRouter 4-Level 의사결정 검증 |
| **5** | **Pipeline** | pipeline_throughput | 연속 입력 시 throughput > 단순 순차의 1.5× |
| | | pipeline_order_preservation | 결과 순서가 입력 순서와 일치 |

### 5-3. 신규 테스트 — 실험 트랙 (EmbeddingBridge, 별도 브랜치)

| 그룹 | 테스트 | 검증 대상 |
|------|--------|----------|
| **EmbeddingBridge** | bridge_roundtrip | vec→prompt→vec 왕복 시 코사인 유사도 > 0.8 |
| | bridge_train_step | 온라인 학습 후 프로젝션 loss 감소 |
| | bridge_zero_vec | 영벡터 입력 방어 |
| | bridge_vs_dimtag | dim-tag 매핑 대비 라우팅 품질 A/B 비교 |

실험 트랙 테스트는 본선 회귀 테스트 스위트에 포함하지 않는다. `tests/experimental/` 디렉토리에 격리하여, 본선 CI가 EmbeddingBridge 의존성(SentenceTransformer 등)에 영향받지 않도록 한다.

---

## 6. 구현 단계 (PDCA)

### 6-0. Stage 순서 원칙

1. **토대 먼저** — Config가 정리되지 않은 상태에서 기능을 쌓으면 나중에 다시 손댄다
2. **본선/실험 분리** — EmbeddingBridge는 외부 의존성(sLLM)이 크므로 별도 브랜치에서 검증
3. **각 Stage에 Go/No-Go** — 테스트 개수뿐 아니라 제품적 성공 기준을 명시

### 6-1. Go/No-Go 기준 총괄표

| Stage | 이름 | 누적 테스트 | Go 기준 | No-Go 시 조치 |
|-------|------|-----------|---------|--------------|
| **0** | Config 분리 | 57+3 = 60 | 기존 57/57 통과 + 기존 API 100% 하위 호환 | 서브 Config 접근 패턴 재설계 |
| **1** | Vector routing | 60+10 = 70 | tag mode 대비 routing miss 감소, empty route 0건 | RegionSignature 초기화/warmup 로직 재조정 |
| **2** | Hybrid 검증 | 70+3 = 73 | α=0.1→0.9 변화에 결과 급변(∆cosine>0.5) 없음 | α 스케줄링 또는 blending 함수 교체 |
| **3** | CoherenceGate | 73+6 = 79 | 의도적 conflict case에서 90% 이상 감지 | conflict_threshold 조정 또는 감지 알고리즘 변경 |
| **4** | ExternalRegion | 79+3 = 82 | LLMRegion이 NativeRegion 없이 RegionSignal 프로토콜 100% 준수 | 프로토콜 인터페이스 재정의 |
| **5** | Pipeline | 82+2 = 84 | throughput ≥ 1.5× 순차 대비, 결과 순서 보존 100% | 파이프라인 버퍼/동기화 로직 재검토 |
| **6** | EmbeddingBridge *(실험)* | 별도 | roundtrip cosine > 0.8, dim-tag 대비 품질 우위 | 모델 교체 또는 프로젝션 차원 조정 |
| **7** | Vector default | 84 재실행 | 실제 샘플 입력에서 tag mode보다 품질 우위 확인 | hybrid 모드 유지, 전환 연기 |

---

### Stage 0: HTPConfig 서브 Config 분리

- **Plan**: §4-3 구현. `RoutingConfig`, `CoherenceConfig`, `LLMBridgeConfig`, `PipelineConfig` 4개 서브 Config 추출
- **Do**: 기존 HTPConfig 인터페이스 하위 호환 유지. 신규 파라미터는 서브 Config에만 추가. 기존 Phase 1-5 파라미터는 이동하지 않음 (breaking change 방지)
- **Check**: 기존 57 테스트 통과 + Config DI 3개 = 60 테스트. graphify 재실행 → HTPConfig 엣지 수 41 → ~35 확인
- **Go/No-Go**: `config.routing.mode`, `config.coherence.conflict_threshold` 등 서브 접근이 정상 작동하고, `config.routing_mode` 같은 기존 flat 접근이 deprecated warning과 함께 여전히 동작하면 Go

**이 Stage를 최우선으로 수행하는 이유**: 이후 모든 Stage에서 서브 Config에 파라미터를 추가한다. Config 구조가 흔들리면 이후 전체가 흔들린다.

### Stage 1: RegionSignature + CoreCells 벡터 모드

- **Plan**: §1 구현. `RegionSignature` 클래스 + CoreCells에 `_gate_vector()` 추가. `RoutingConfig`에서 파라미터 로드
- **Do**: `RoutingConfig.mode = "tag"` 기본값 유지. vector 모드는 테스트에서만 활성화
- **Check**: Signature 4개 + CAR 6개 = 70 테스트
- **Go/No-Go**: vector mode에서 tag mode 대비 routing miss가 같거나 적고, empty route(아무 Region도 선택되지 않는 경우)가 0건이면 Go. empty route가 발생하면 warmup 로직 또는 dynamic threshold fallback 재조정

### Stage 2: Hybrid Routing 검증

- **Plan**: hybrid 모드 구현 및 안정성 검증. α를 0.1에서 0.9까지 변화시키며 라우팅 결과의 연속성 확인
- **Do**: `RoutingConfig.mode = "hybrid"`, α=0.3으로 데모 실행. 기존 Phase 1 routing demo와 결과 비교
- **Check**: Hybrid 3개 테스트 = 73 테스트
- **Go/No-Go**: α를 0.1 단위로 변경할 때 선택된 Region 집합의 변화가 급격하지 않음 (연속된 α 값에서 cosine similarity of selected set > 0.5). 급변이 발생하면 blending 함수를 linear에서 sigmoid로 교체 검토

### Stage 3: CoherenceGate + Memory 연동

- **Plan**: §2 구현. CoherenceGate 클래스 + SWR priority 확장. `CoherenceConfig`에서 파라미터 로드
- **Do**: BrainRuntime에 CoherenceGate 삽입 (Region 응답 수집 후, PFC 전달 전)
- **Check**: Coherence 6개 테스트 (LSH fallback 포함) = 79 테스트
- **Go/No-Go**: 의도적으로 직교 벡터(cosine < 0.1)를 가진 Region 응답 쌍 10개를 생성하여 conflict 감지율 ≥ 90%. 정합 응답 쌍 10개에서 false positive ≤ 10%. 미달 시 `conflict_threshold` 조정 또는 단순 min-pair 대신 클러스터링 기반 감지로 전환

### Stage 4: ExternalRegion + LLMRegion 리팩토링

- **Plan**: §3 구현. `LLMRegionRuntime` → `LLMRegion` 전환. 단기 vec↔prompt (dim-tag 매핑)
- **Do**: 기존 `LLMRegionRuntime`을 `archive/deprecated_phase4/`로 이동 (기존 관행 유지)
- **Check**: LLMRegion 3개 테스트 = 82 테스트
- **Go/No-Go**: `LLMRegion`이 `NativeRegion` 인스턴스 없이 단독으로 `RegionSignal`을 생성할 수 있으면 Go. graphify 재실행하여 LLM 관련 isolated 노드 수가 감소했으면 Go. 프로토콜 불일치가 있으면 `ExternalRegion` 인터페이스 재정의

### Stage 5: PipelinedBrainRuntime

- **Plan**: §2-6 구현. 파이프라인 병렬성. `PipelineConfig`에서 버퍼 크기 로드
- **Do**: 기존 `AsyncBrainRuntime` 보존, `PipelinedBrainRuntime` 신규 추가
- **Check**: Pipeline 2개 테스트 (throughput + order preservation) = 84 테스트
- **Go/No-Go**: 연속 입력 10개 기준 throughput ≥ 1.5× 순차 대비. 결과 순서가 입력 순서와 100% 일치. throughput 미달 시 파이프라인 스테이지 수 또는 버퍼 크기 조정

### Stage 6: EmbeddingBridge (실험 브랜치)

- **브랜치**: `experiment/embedding-bridge` — 본선 `main`과 분리
- **Plan**: §3-4-2 구현. sLLM 임베딩 기반 양방향 프로젝션
- **Do**: `EmbeddingBridge` 클래스 구현. `tests/experimental/`에 테스트 격리. 본선 CI에 영향 없음
- **Check**: bridge_roundtrip cosine > 0.8, bridge_vs_dimtag A/B 비교
- **Go/No-Go (본선 머지 기준)**: roundtrip cosine > 0.8 **그리고** dim-tag 매핑 대비 실제 라우팅 시나리오에서 품질 우위 확인. 미달 시 sLLM 모델 교체 또는 프로젝션 차원(384→64) 조정. 본선에 머지하지 않고 실험 브랜치에 유지하는 것도 유효한 결정

**실험 트랙으로 분리하는 이유**: EmbeddingBridge는 외부 의존성(`sentence-transformers`, 특정 sLLM 모델)이 크고, 모델 선택·프로젝션 차원·학습률 등 자체적으로 탐색해야 할 하이퍼파라미터가 많다. 본선의 진행 속도에 영향을 주지 않으면서 독립적으로 품질을 끌어올리는 것이 합리적이다.

### Stage 7: hybrid → vector 기본값 전환

- **Plan**: Stage 1-5 완료 + 충분한 시뮬레이션 후 `RoutingConfig.mode` 기본값을 "vector"로 전환
- **Do**: tag 관련 코드를 `archive/`로 이동
- **Check**: 전체 84개 테스트를 vector 모드로 재실행
- **Go/No-Go**: 실제 샘플 입력 세트(최소 20개)에서 vector mode가 tag mode 대비 routing 정확도 우위 확인. 동등하면 hybrid 유지, 열위이면 전환 연기. 프로젝트 리뷰 갱신, §3-C에서 "임베딩 기반 시맨틱 라우팅" 해소 표기

---

## 7. 위험과 완화

| 위험 | 영향 | 완화 |
|------|------|------|
| RegionSignature 냉시작 | 초기에 centroid가 영벡터 → 모든 similarity ≈ 0 | 최초 N회(`RoutingConfig.warmup_steps=10`)는 tag 모드 강제, 이후 hybrid 전환 |
| Dynamic threshold 불안정 | Region 수가 2–3개일 때 σ가 불안정 | N < 4이면 threshold를 고정값(0.3)으로 fallback |
| CoherenceGate O(N²) | Region 수 증가 시 pairwise 비용 | **N=16을 LSH 전환 임계값으로 확정** (`CoherenceConfig.lsh_transition_n=16`). 16×16=256 쌍은 CUDA에서 ~0.1ms. N=32(1,024 쌍)부터 레이턴시 가시화. N≥16 도달 시점은 멀티모달 확장 이후이므로, ModalEncoder 통합과 LSH 도입을 동일 Phase에서 처리 |
| HTPConfig 비대화 | 41 → 49 엣지 예상 | **Stage 0에서 서브 Config 분리를 최우선 처리.** 4개 서브 Config(`RoutingConfig`, `CoherenceConfig`, `LLMBridgeConfig`, `PipelineConfig`)가 의존성 중간 노드로 기능하여 직접 엣지 ~35로 감소 예상 |
| vec ↔ prompt 변환 품질 | LLMRegion의 실용성 좌우 | 3단계 전략: (단기) dim-tag 매핑 → (중기, Stage 6 실험 브랜치) EmbeddingBridge sLLM 임베딩 프로젝션 ~5ms → (장기) continuous token으로 변환 제거. CostRouter 4-Level에서 Level 1-2 처리 비율 70% 목표 |
| EmbeddingBridge 모델 의존성 | sLLM 모델 업데이트/폐기 시 프로젝션 재학습 필요 | **실험 브랜치에서 독립 검증.** `LLMBridgeConfig.embedding_model`로 모델 교체 가능. 본선 머지 기준: roundtrip cosine > 0.8 + dim-tag 대비 품질 우위. 미달 시 본선에 머지하지 않는 것도 유효한 결정 |
| 기존 테스트 깨짐 | routing_mode 기본값 변경 시 | tag 기본값 유지, 전환은 Stage 7에서만 |

---

## 8. 미래 확장점 (이 설계서 스코프 밖)

| 확장 | 연결되는 기존 미반영 항목 | 시기 |
|------|----------------------|------|
| Multi-head attention 라우팅 | what/where pathway 분리 | Stage 7 완료 후 |
| ModalEncoder 통합 | `htp_multimodal_design.md` V-JEPA | Stage 4 이후 |
| Continuous token (변환 제거) | §3-4-3 장기 전략. 로컬 sLLM 필수 | ModalEncoder 완성 + Stage 6 실험 결과 반영 후 |
| Predictive Coding `PredictiveRegion` | `htp_friston_review.md` 장기 항목 | CoherenceGate(Stage 3)가 prediction error 인프라 제공 |
| LSH 근사 도입 | CoherenceGate O(N²) 스케일링 | Region ≥ 16 도달 시 (ModalEncoder 통합과 동시) |
| Incremental PCA (64-dim 적응) | LeCun 리뷰 미반영 | Region 수 N > 1000 도달 시 |
| Lateral Inhibition 국소화 | LeCun 리뷰 미반영 | NativeRegion 내부 개선 |

**참고**: HTPConfig DI 리팩토링은 기존 리뷰 §5 약점 #2에서 지적된 항목이었으나, 이 설계서의 Stage 0으로 편입하여 **최우선으로 해소**한다.

---

## 부록 A. 신경과학 ↔ 소프트웨어 매핑 전체표

| 뇌 메커니즘 | 기능 | HTP 구현 (현재) | HTP 구현 (이 설계 후) |
|------------|------|----------------|---------------------|
| 분산 표상 (앙상블 코딩) | 하나의 기억 = 다수 뉴런의 패턴 | 64-dim state_vec | ← 동일 |
| Hebb's rule | 동시 발화 → 연결 강화 | Hebbian EMA (lr=1/(n+1)) | ← 동일 + RegionSignature에 재활용 |
| Content-addressable memory | 내용의 일부 = 검색 키 | ✗ (tag 매칭) | **벡터 유사도 라우팅** |
| 패턴 완성 (CA3 recurrent) | 부분 단서 → 전체 복원 | CA3 complete() α=0.7 | ← 동일 |
| 감마 동기화 (temporal binding) | 분산 정보를 하나로 묶음 | ✗ | **CoherenceGate** |
| 시상 게이팅 | 피질 영역 선택적 활성화 | CoreCells tag 매칭 | **CoreCells 벡터 유사도** |
| 시상 → 피질 동시 전파 | 병렬 활성화 | asyncio.gather | ← 동일 + dynamic threshold 다중 선택 |
| 피질-피질 연합 영역 | 모달리티 간 변환 | ✗ | **EmbeddingBridge** (sLLM 임베딩 프로젝션) |
| SWR 공고화 | 해마 → 피질 기억 전이 | SWR priority = novelty × reward | **+ conflict_magnitude 반영** |
| 수면 공고화 (replay) | 기억 재분배 | L2→L3 Go-CLS | ← 동일 |
| PFC 목표 정렬 | top-down 제어 | TopDownBias softmax | + **conflict 에스컬레이션** 수신 |

---

## 부록 B. Modern Hopfield Network ↔ HTP 대응

Ramsauer et al. (2020)이 보여준 Modern Hopfield Network와 Transformer attention의 동치성을 HTP에 매핑한다.

```
Modern Hopfield          Transformer              HTP CAR
─────────────           ─────────────            ──────────
저장 패턴 ξ_i           Key K                    RegionSignature.centroid
쿼리 상태 ξ             Query Q                  input query_vec (64-dim)
에너지 최소화            softmax(QK^T/√d)         cosine_sim × precision + td_bias
인출된 패턴              Value V                  선택된 Region의 출력
저장 용량 exponential    context window            Region 수 (동적 증가 가능)
```

이 대응이 의미하는 것: HTP의 Thalamus content-addressable routing은 단일 attention head의 특수화이며, 이론적으로 잘 정의된 에너지 함수 위에서 작동한다. 패턴 인출의 수렴 보장이 Hopfield network의 수학적 성질로부터 상속된다.
