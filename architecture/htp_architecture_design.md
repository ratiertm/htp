# HTP (Hub Topology Programming) — Architecture Design Document
version: 0.1 | status: design

---

## 1. 개요 및 철학

### 1.1 문제 의식

기존 프로그래밍 패러다임은 인간이 모든 로직을 사전에 설계해야 한다.
허브 토폴로지 프로그래밍은 데이터가 흐르면서 구조 자체가 창발하는 패러다임이다.

뇌의 원리:
- 시냅스 가지치기: 안 쓰는 연결 제거
- 신경발생: 과부하 영역에 새 뉴런 생성
- 헤비안 학습: 함께 발화하는 노드는 함께 연결
- 시상 게이팅: 어떤 피질 영역을 활성화할지 동적 결정

### 1.2 핵심 원칙

1. **구조는 데이터가 만든다** — 개발자가 if/else로 라우팅을 설계하지 않음
2. **허브는 창발한다** — 자주 쓰이는 노드가 자연스럽게 허브로 승격
3. **네트워크는 살아있다** — 노드가 생기고, 분열하고, 소멸함
4. **판단은 위임한다** — LLM/Agent를 노드로 쓰면 인간이 감당 불가한 동적 판단을 위임 가능

---

## 2. 전체 아키텍처

```
외부 입력 (데이터 스트림)
        │
        ▼
┌───────────────────────────────────────────┐
│              Region Layer                  │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│  │언어Region│ │기억Region│ │감정Region│  │
│  │HTPRuntime│ │HTPRuntime│ │HTPRuntime│  │
│  └──────────┘ └──────────┘ └──────────┘  │
└───────────────────────────────────────────┘
        │ (각 Region 출력 신호)
        ▼
┌───────────────────────────────────────────┐
│                 Thalamus                   │
│  Core cells: 내용 게이팅                   │
│  Matrix cells: Winner-take-all             │
│  NGE 트리거: 과부하 → 신경발생 신호        │
└───────────────────────────────────────────┘
        │ (압축 상태벡터)
        ▼
┌───────────────────────────────────────────┐
│              PFCRuntime                    │
│  Working memory · Top-down 억제 · 최종결정 │
└───────────────────────────────────────────┘
        │
        ▼
    행동 출력 (Action)
```

---

## 3. 컴포넌트 설계

### 3.1 WeightMatrix

```python
class WeightMatrix:
    """
    W[u][v] = u→v 연결 강도 (단일 소유)
    세 엔진(HFE, PE, NGE)이 참조를 공유.
    쓰기: HFE(헤비안) + PE(제거)
    읽기: ActivationEngine(전파 계산)
    """
    W: torch.Tensor         # [N x N] 가중치 행렬
    n: int                  # 현재 노드 수 (동적 확장)
    fire_history: list      # 발화 이력 (usage pruning용)

    def set(u, v, w)        # 엣지 설정
    def in_strength(v)      # 노드 v의 입력 강도 합
    def edge_count()        # 활성 엣지 수
    def record_fire(fired)  # 발화 기록
    def recent_fire_rate(node_id, window)  # 최근 발화율
```

### 3.2 HubFormationEngine (HFE)

```python
class HubFormationEngine:
    """
    헤비안 학습 + 허브 승격
    생물학: 시냅스 강화 (LTP)
    """
    # 매 스텝:
    # 1. 행 정규화 전파 → 발화 결정
    # 2. co-activation → 헤비안 업데이트 (포화 방지)
    # 3. in_strength > hub_threshold → 허브 승격

    is_hub: torch.BoolTensor    # 허브 마스크
    fire_count: torch.Tensor    # 누적 발화 횟수
    hub_events: list            # 승격/강등 이력

    def step(signal) -> fired_mask
    def hub_indices() -> list[int]
    def top_hubs(k) -> list[(id, strength)]
```

### 3.3 PruningEngine (PE)

```python
class PruningEngine:
    """
    3가지 독립 전략
    생물학: 미세아교세포(Microglia)가 시냅스 제거
    """

    # [1] Decay Pruning (매 스텝)
    #     W *= (1 - decay_rate)
    #     W < prune_threshold → 제거
    #     뇌: 시냅스 시간 감쇠

    # [2] Usage Pruning (N스텝마다)
    #     최근 window 내 발화율 낮은 노드의 엣지 약화
    #     뇌: 활동 의존적 가지치기

    # [3] Redundancy Pruning (M스텝마다)
    #     두 노드 입력 패턴 코사인 유사도 > threshold
    #     → 약한 쪽 엣지 제거
    #     뇌: 중복 경로 정리

    def run_all(step) -> dict[str, int]  # 세 전략 실행
    def report() -> str
```

### 3.4 NodeGenerationEngine (NGE)

```python
class NodeGenerationEngine:
    """
    3가지 노드 생성 전략
    생물학: 신경발생(Neurogenesis) — 해마 치상회
    핵심: 시상이 NGE 트리거 신호를 보냄 (NRXN1 메커니즘)
    """

    # [1] Hub Split (허브 과부하)
    #     in_strength > split_threshold AND calls > min_calls
    #     → 부모 노드를 두 자식으로 분열
    #     → 연결을 강도 기준으로 절반씩 상속
    #     → 자식은 immature 상태로 시작 (성숙 전 재분열 금지)
    #     뇌: 피질 컬럼 분화

    # [2] Sprout (패턴 미매칭)
    #     강한 신호인데 N번 연속 발화 없음
    #     → 새 탐색 노드를 약한 연결로 생성
    #     뇌: 수상돌기 발아

    # [3] Interpolate (라우팅 실패)
    #     특정 노드 쌍 사이 N번 연속 실패
    #     → 중계 노드 삽입
    #     뇌: 인터뉴런 생성

    # 정제된 제약 조건:
    maturity_calls: int     # 자식 노드 최소 호출 수 (재분열 전)
    global_cooldown: int    # 전체 시스템 쿨다운 (연쇄 분열 방지)
    max_gen_per_run: int    # run() 당 최대 생성 수 = 1
    immature: set           # 미성숙 노드 ID 집합

    def check_split(step) -> list[Node]
    def check_sprout(signal, fired_ids, step) -> Node | None
    def check_interpolate(step) -> list[Node]
    def _create_node(name, fn, in_edges, out_edges) -> Node
```

### 3.5 ActivationEngine (AE)

```python
class ActivationEngine:
    """
    캐스케이드 전파 + 시맨틱 배제 라우팅
    """

    # 캐스케이드 전파:
    # entry 노드 실행 → output
    # _make_signal(output, prev, visited) → sig
    # HFE.step(sig) → fired
    # _semantic_filter(fired, prev, data) → final
    # 발화 노드 실행 → output → 반복

    # 시맨틱 배제:
    # label="error" → to_alert 태그 매칭 → to_cache 억제

    # 데코레이터:
    # @rt.node   — 함수를 노드로 등록
    # @tag(...)  — 시맨틱 라우팅 태그
    # @terminal  — 캐스케이드 종착점

    def run(data, entry, max_depth) -> (path, outputs, hub_ids)
```

### 3.6 HTPRuntime (단일 영역)

```python
class HTPRuntime:
    """
    4엔진 통합. 하나의 뇌 피질 영역.
    """
    wm:  WeightMatrix
    hfe: HubFormationEngine
    pe:  PruningEngine
    nge: NodeGenerationEngine
    ae:  ActivationEngine

    def node(fn) -> fn          # 데코레이터
    def connect(src, dst, w)    # 초기 연결
    def run(data, entry) -> RunResult
    def status()                # 상태 출력
```

---

## 4. Thalamus 설계

### 4.1 RegionSignal (Region → Thalamus 통신 단위)

```python
@dataclass
class RegionSignal:
    region_id:    str
    hub_strength: float     # 최대 허브 강도
    fire_rate:    float     # 최근 발화율
    top_hubs:     list      # 상위 허브 (id, strength)
    overload:     bool      # 과부하 여부
    output_vec:   tensor    # 최근 출력 임베딩 (압축용)
```

### 4.2 CoreCells (내용 게이팅)

```python
class CoreCells:
    """
    생물학: 특정 시상핵 (VPM, dLGN 등)
    각 Region의 특정 노드를 ON/OFF
    """
    def gate(signals: list[RegionSignal]) -> GatingMask:
        # 허브 강도 × 발화율 → 활성화 점수
        # 점수 기반 소프트 게이팅
        # 반환: {region_id: {node_id: gate_strength}}
```

### 4.3 MatrixCells (상태 게이팅)

```python
class MatrixCells:
    """
    생물학: 중심매체핵, 내측배측핵
    Winner-take-all: 어떤 Region이 우세한지 결정
    """
    def compete(signals: list[RegionSignal]) -> CompetitionResult:
        # 전체 Region 신호 집계
        # Softmax 경쟁 → 승자 Region 선택
        # 패자 Region 억제 강도 계산
        # 반환: winner_id, suppression_map
```

### 4.4 NGETrigger (신경발생 트리거)

```python
class NGETrigger:
    """
    생물학: NRXN1 신호 → outer radial glia → 신경발생
    (Thalamic NRXN1-Mediated Input, bioRxiv 2025)
    과부하 Region의 NGE에 split 명령 전달
    """
    def fire(region_id: str, overload_strength: float):
        # 해당 Region의 NGE.check_split() 강제 트리거
        # 임계값 일시적 낮춤 → 즉각 분열 유도
```

### 4.5 Thalamus

```python
class Thalamus:
    """
    Core + Matrix 이중 구조
    생물학: 시상 (Neuron 2024 — 압축·게이팅·재구성)
    """
    regions:     list[RegionRuntime]
    core:        CoreCells
    matrix:      MatrixCells
    nge_trigger: NGETrigger
    overload_threshold: float = 4.0

    def step(data) -> ThalamusOutput:
        # 1. 각 Region 병렬 활성화
        signals = [r.collect_signal() for r in regions]

        # 2. Core: 특정 노드 게이팅
        gating = core.gate(signals)

        # 3. Matrix: Winner-take-all
        competition = matrix.compete(signals)

        # 4. 과부하 감지 → NGE 트리거
        for sig in signals:
            if sig.overload:
                nge_trigger.fire(sig.region_id, sig.hub_strength)

        # 5. 고차원 → 저차원 압축 (PFC 전달용)
        state_vec = compress(competition, gating)

        return ThalamusOutput(
            winner      = competition.winner_id,
            state_vec   = state_vec,      # 저차원 맥락 벡터
            gating      = gating,         # 노드별 게이팅 마스크
            suppressed  = competition.suppression_map
        )

@dataclass
class ThalamusOutput:
    winner:     str             # 승자 Region ID
    state_vec:  tensor          # 압축 상태벡터 → PFC 입력
    gating:     GatingMask      # Core 게이팅 결과
    suppressed: dict            # 억제할 Region 목록
```

---

## 5. PFCRuntime 설계

```python
class PFCRuntime(HTPRuntime):
    """
    생물학: 전전두엽 (PFC)
    장기 목표 유지 + Top-down 억제 + 최종 결정
    """
    working_memory: deque[ThalamusOutput]  # 최근 N개 상태
    long_term_goals: list[Goal]            # 고정 목표 (외부 설정)
    inhibition_threshold: float = 0.4

    def decide(thal_out: ThalamusOutput) -> Action:
        # 1. Working memory 업데이트
        working_memory.append(thal_out)

        # 2. 시상 압축벡터를 자체 허브 네트워크에 통과
        #    → PFC 내부에도 HTPRuntime 존재
        result = self.run(thal_out.state_vec, entry=pfc_entry)

        # 3. 장기 목표와 정렬도 계산
        alignment = check_alignment(result, long_term_goals)

        # 4. 실행 or 억제
        if alignment > inhibition_threshold:
            return Action.EXECUTE(thal_out.winner, result)
        else:
            return Action.INHIBIT(
                target  = thal_out.winner,
                reason  = alignment.gap,
                redirect= find_alternative(working_memory)
            )
```

---

## 6. BrainRuntime (최상위 통합)

```python
class BrainRuntime:
    """
    전체 시스템 오케스트레이터
    RegionRuntimes + Thalamus + PFCRuntime
    """
    regions:  dict[str, RegionRuntime]
    thalamus: Thalamus
    pfc:      PFCRuntime

    def run(data) -> Action:
        # 1. 모든 Region 비동기 처리
        #    (각 Region은 독립적으로 허브 형성 중)
        for r in regions.values():
            r.activate_async(data)

        # 2. Thalamus 게이팅
        thal_out = thalamus.step(data)

        # 3. PFC 최종 결정
        action = pfc.decide(thal_out)

        # 4. 억제 신호 피드백
        for region_id in thal_out.suppressed:
            regions[region_id].apply_suppression(
                thal_out.suppressed[region_id]
            )

        return action
```

---

## 7. LLM-as-Node 확장 설계

### 7.1 핵심 통찰

현재 LLM(Claude 등)은 이미 내부적으로 동일한 구조로 작동한다:
- Attention → 동적 허브 형성
- FFN → 전문화된 뉴런 그룹 (사실상 MoE)
- 레이어 → 캐스케이드 전파

차이: LLM은 런타임에 토폴로지가 고정. HTP는 동적으로 변함.

### 7.2 LLMNode

```python
class LLMNode(Node):
    """
    함수 대신 LLM API call이 노드가 됨
    """
    model:       str            # "claude-sonnet-4-6" 등
    system:      str            # 이 노드의 전문 역할
    temperature: float = 0.3

    def run(self, data) -> dict:
        prompt = format(data)
        response = anthropic.messages.create(
            model    = self.model,
            system   = self.system,
            messages = [{"role": "user", "content": prompt}]
        )
        return parse(response)
```

### 7.3 사용 예시

```python
rt = HTPRuntime()

@rt.node
@tag("language", "text", "parse")
def language_llm(data):
    return LLMNode(
        model  = "claude-sonnet-4-6",
        system = "텍스트를 파싱하고 핵심 의도를 추출하라"
    ).run(data)

@rt.node
@tag("code", "debug", "implement")
def code_llm(data):
    return LLMNode(
        model  = "claude-sonnet-4-6",
        system = "코드를 분석하고 구현하라"
    ).run(data)

@rt.node
@tag("memory", "recall", "context")
def memory_agent(data):
    # RAG + VectorDB 조합
    return MemoryAgent(vectordb=chroma).run(data)
```

### 7.4 BrainRuntime with LLMs

```python
brain = BrainRuntime()

# 각 Region에 전문화된 LLM 배치
brain.add_region("language", LLMRegionRuntime(
    specialty = "language",
    model     = "claude-sonnet-4-6",
    system    = "언어 이해 및 생성 전담"
))
brain.add_region("code", LLMRegionRuntime(
    specialty = "code",
    model     = "claude-sonnet-4-6",
    system    = "코드 분석 및 구현 전담"
))
brain.add_region("memory", RAGRegionRuntime(
    specialty = "memory",
    vectordb  = chroma
))

# Thalamus가 자동으로:
# - 어떤 Region을 발화시킬지 결정
# - 과부하 Region에 새 LLM 인스턴스 생성 (신경발생)
# - 안 쓰는 Region 가중치 감쇠 (가지치기)

result = brain.run("이 코드의 버그를 찾아줘")
```

---

## 8. 구현 순서 (로드맵)

### Phase 1 — 단일 영역 완성 (현재)
- [x] HubFormationEngine
- [x] PruningEngine (3전략)
- [x] ActivationEngine (캐스케이드 + 시맨틱)
- [x] NodeGenerationEngine (split + sprout + interpolate)
- [x] HTPRuntime 통합

### Phase 2 — 다중 영역 + Thalamus
- [ ] RegionRuntime (HTPRuntime 확장)
- [ ] RegionSignal 인터페이스
- [ ] CoreCells
- [ ] MatrixCells (Winner-take-all)
- [ ] NGETrigger
- [ ] Thalamus 통합

### Phase 3 — PFC + BrainRuntime
- [ ] PFCRuntime
- [ ] Working memory
- [ ] BrainRuntime 오케스트레이터
- [ ] Top-down 억제 피드백 루프

### Phase 4 — LLM-as-Node
- [ ] LLMNode 추상화
- [ ] LLMRegionRuntime
- [ ] 비동기 처리 (asyncio)
- [ ] 비용/지연 기반 라우팅

### Phase 5 — 센서 통합 (장기)
- [ ] 멀티모달 입력 (이미지, 오디오)
- [ ] 지속적 메모리 (영구 가중치)
- [ ] 자기보존 없는 Stakes 설계

---

## 9. 미해결 설계 질문

1. **Thalamus 자체가 학습해야 하는가?**
   현재 설계: CoreCells/MatrixCells는 규칙 기반
   질문: 이것도 HebbianLearning 적용해야 하는가?

2. **Region 간 직접 통신 허용하는가?**
   현재: Region → Thalamus → Region (간접)
   대안: Region끼리 약한 직접 연결 허용 (cortico-cortical)

3. **PFC Working memory 크기?**
   뇌: 7±2 청크 (GWT)
   구현: deque maxlen = ?

4. **LLM 노드의 비용 vs 성능 트레이드오프?**
   과부하로 새 LLM 인스턴스 생성 = API 비용 증가
   어느 시점에 억제 vs 생성을 선택하는가?

5. **4D 시각화를 실제 개발 도구로?**
   허브 형성 과정을 실시간으로 보면서 디버깅
   Claude Code 연동 가능한가?

---

## 10. 참고 문헌

- Baars, B.J. (1988). Global Workspace Theory
- Dehaene, S. et al. (2003). Global Neuronal Workspace
- Goyal, A. & Bengio, Y. (2021). Recurrent Independent Mechanisms
- Thalamic contributions to consciousness. Neuron 2024
- Thalamic NRXN1-Mediated Neurogenesis. bioRxiv 2025
- Adult hippocampal neurogenesis. Nature Comm 2024
- Synaptic pruning by microglia. Frontiers 2025
