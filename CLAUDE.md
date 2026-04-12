# HTP (Hub Topology Programming) — Claude Code Context

이 파일은 Claude Code에서 프로젝트를 이어가기 위한 맥락 문서다.
채팅에서 논의한 내용을 바탕으로 Phase 2 구현을 시작하면 된다.

---

## 프로젝트 요약

뇌의 허브 노드 구조를 프로그래밍 패러다임으로 구현한 시스템.
핵심 아이디어: 개발자가 if/else로 라우팅을 설계하는 게 아니라,
데이터가 흐르면서 허브 구조가 창발하고, 노드가 생기고 죽고 분열한다.

생물학적 근거:
- 헤비안 학습 (시냅스 강화)
- 미세아교세포 가지치기 (시냅스 제거)
- 신경발생 (새 뉴런 생성 — 해마 치상회)
- 시상 게이팅 (Neuron 2024 — 압축·재구성·트리거)
- NRXN1 신호 → 피질 신경발생 유도 (bioRxiv 2025)

---

## 현재 구현 상태 (Phase 1 완료)

### 파일 구조
```
hub_formation_engine.py     헤비안 학습 + 허브 감지
htp_runtime.py              WeightMatrix + 4엔진 통합 (HFE + PE + NGE + AE)
activation_engine.py        캐스케이드 전파 + 시맨틱 배제 라우팅
node_generation_engine.py   동적 노드 생성 (split + sprout + interpolate)
htp_architecture_design.md  전체 아키텍처 설계 문서
```

### 핵심 클래스

```python
# WeightMatrix — W[u][v] 단일 소유, 세 엔진이 참조 공유
# HubFormationEngine — 헤비안 + 허브 승격
# PruningEngine — Decay / Usage / Redundancy 3전략
# NodeGenerationEngine — Split / Sprout / Interpolate 3전략
# ActivationEngine — 캐스케이드 전파 + 시맨틱 배제
# HTPRuntime — 4엔진 통합 오케스트레이터

# 데코레이터
@rt.node              # 함수를 노드로 등록
@tag("success", ...)  # 시맨틱 라우팅 태그
@terminal             # 캐스케이드 종착점
```

### 검증된 동작
- 12/12 라우팅 정확도 (success → to_cache, error → to_alert)
- 허브 분열: 30회 데이터 후 classify 노드 자동 분열
- 3가지 가지치기 전략 작동 확인
- 성숙 조건 + 전역 쿨다운으로 연쇄 분열 방지

---

## 다음 단계: Phase 2 — Thalamus 구현

### 구현 목표
단일 HTPRuntime → 다중 RegionRuntime + Thalamus + PFCRuntime

### 전체 구조
```
외부 입력
    ↓
RegionRuntime × N  (각 영역이 독립적 HTPRuntime)
    ↓ (RegionSignal)
Thalamus
  ├── CoreCells     특정 노드 ON/OFF 게이팅
  ├── MatrixCells   Winner-take-all 경쟁
  └── NGETrigger    과부하 → 신경발생 신호
    ↓ (ThalamusOutput: 압축 상태벡터)
PFCRuntime
  ├── working_memory  최근 N개 상태
  └── long_term_goals 장기 목표
    ↓
Action (최종 출력)
```

### 구현할 클래스 (우선순위 순)

**1. RegionSignal** — Region → Thalamus 통신 단위
```python
@dataclass
class RegionSignal:
    region_id:    str
    hub_strength: float   # 최대 허브 강도
    fire_rate:    float   # 최근 발화율
    top_hubs:     list    # [(id, strength), ...]
    overload:     bool    # hub_strength > threshold
    output_vec:   tensor  # 최근 출력 임베딩
```

**2. RegionRuntime** — HTPRuntime 확장
```python
class RegionRuntime(HTPRuntime):
    region_name: str
    specialty:   str   # "language" | "memory" | "emotion" | "sensor"

    def collect_signal(self) -> RegionSignal
    def apply_suppression(self, strength: float)  # Thalamus 억제 수신
    def activate_async(self, data)                # 비동기 처리용
```

**3. CoreCells** — 내용 게이팅
```python
class CoreCells:
    # 허브 강도 × 발화율 → 활성화 점수
    # 점수 기반 소프트 게이팅
    def gate(signals: list[RegionSignal]) -> GatingMask
```

**4. MatrixCells** — 상태 게이팅
```python
class MatrixCells:
    # Softmax 경쟁 → 승자 Region
    # 패자 Region 억제 강도 계산
    def compete(signals: list[RegionSignal]) -> CompetitionResult
```

**5. NGETrigger** — 신경발생 트리거
```python
class NGETrigger:
    # 과부하 Region NGE.check_split() 강제 트리거
    # 생물학: NRXN1 신호 → 피질 신경발생
    def fire(region_id: str, overload_strength: float)
```

**6. Thalamus** — 통합
```python
class Thalamus:
    regions:     list[RegionRuntime]
    core:        CoreCells
    matrix:      MatrixCells
    nge_trigger: NGETrigger

    def step(data) -> ThalamusOutput:
        # 1. 각 Region 신호 수집
        # 2. Core 게이팅
        # 3. Matrix 경쟁
        # 4. 과부하 → NGE 트리거
        # 5. 압축 → ThalamusOutput
```

**7. PFCRuntime** — 최종 결정
```python
class PFCRuntime(HTPRuntime):
    working_memory:  deque      # 최근 N개 상태 (뇌: 7±2)
    long_term_goals: list

    def decide(thal_out: ThalamusOutput) -> Action
```

**8. BrainRuntime** — 최상위 오케스트레이터
```python
class BrainRuntime:
    regions:  dict[str, RegionRuntime]
    thalamus: Thalamus
    pfc:      PFCRuntime

    def run(data) -> Action
```

---

## 미해결 설계 질문 (구현 전 결정 필요)

1. **Thalamus CoreCells/MatrixCells가 규칙 기반인가 학습 기반인가?**
   - 규칙 기반: 빠르고 예측 가능, 하지만 정적
   - 학습 기반: HebbianLearning 적용, 더 동적이지만 복잡

2. **Region 간 직접 통신을 허용하는가?**
   - 현재 설계: Region → Thalamus → Region (간접만)
   - 대안: cortico-cortical 직접 약한 연결 허용

3. **PFC Working memory 크기?**
   - 뇌: 7±2 청크 (GWT)
   - deque maxlen = ?

4. **LLM 노드 비용 vs 생성 트레이드오프?**
   - 과부하 → 새 LLM 인스턴스 = API 비용 증가
   - 어느 시점에 억제 vs 생성 선택?

---

## Phase 4 장기 목표: LLM-as-Node

```python
# 현재 LLM(Claude)은 이미 동일 구조로 작동함:
# Attention → 동적 허브 / FFN → 전문화 뉴런 그룹 (사실상 MoE)
# 차이: LLM은 런타임에 토폴로지 고정. HTP는 동적으로 변함.

# LLMNode: 함수 대신 LLM API call이 노드
class LLMNode(Node):
    model:  str   # "claude-sonnet-4-6"
    system: str   # 이 노드의 전문 역할

# BrainRuntime with LLMs:
# RegionRuntime("language") → Claude API
# RegionRuntime("code")     → Claude Code API
# RegionRuntime("memory")   → RAG + VectorDB
# Thalamus → 어떤 Region/LLM을 발화시킬지 동적 결정
# NGE → 과부하 시 새 LLM 인스턴스 생성 (신경발생)
```

---

## 참고 문헌 (생물학)

- Thalamic contributions to consciousness. Neuron 2024
  → 시상은 중계가 아닌 압축·게이팅·재구성 수행
- Thalamic NRXN1-Mediated Neurogenesis. bioRxiv 2025
  → 시상 신호가 피질 신경발생을 직접 유도
- Adult neurogenesis improves spatial information. Nature Comm 2024
  → 새 뉴런은 새로운 sparse code 제공 (pattern separation)
- Synaptic pruning by microglia. Frontiers 2025
  → 발달 중 시냅스의 절반이 활동 의존적으로 제거됨
- Global Workspace Theory (Baars 1988, Dehaene 2003)
- Recurrent Independent Mechanisms (Goyal & Bengio 2021)

---

## 시작 명령

```
Phase 2 Thalamus 구현 시작.
미해결 설계 질문 1번부터 결정하고 RegionSignal → RegionRuntime 순서로 구현해줘.
```
