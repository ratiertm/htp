---
template: plan
feature: htp-thalamus-car
sub_cycle: sub-5 merge
date: 2026-05-18
author: Mindbuild
project: HTP (Hub Topology Programming)
status: Plan
reviewer: Claude / Gemini
base_branch: experiment/embedding-bridge
base_tests: 189/189 PASS
target_tests: 195/195 PASS
---

# HTP sub-5 Merge 수정 계획

> **목적**: experiment/embedding-bridge → main merge 전후 보완 작업 3건.
> 두 리뷰어(Claude, Gemini) 피드백의 합의점 + 차이점을 종합한 실행 계획.
>
> **작업 총량**: 3건, 예상 2-3시간
> **테스트 변화**: 189 → **195** (+6)
> **merge 시점**: 작업 1, 2 완료 후 (192/192 확인 시점)

---

## 0. 리뷰 합의/차이 요약

### 합의 (별도 논의 불필요)

| 항목 | Claude | Gemini | 판단 |
|------|--------|--------|------|
| D1-D4 적절성 | ✅ | ✅ | 보호 충분 |
| Hopfield 실패 해석 | cosine similarity 본질적 한계 | 동일 | 정확 |
| I5 최우선 | ✅ | ✅ | 합의 |
| 모델 선택 (e5-small) | 적합 | 탁월한 타협점 | 유지 |
| main merge | Go | 조건부 Go | 아래 차이 참조 |

### 차이 → 해소 방안

| 항목 | Claude | Gemini | 이 계획의 결정 |
|------|--------|--------|--------------|
| Adversarial test | 불필요 → 수정: 필요 | merge 조건 | **merge 전 추가** (Gemini 수용, Claude 재동의) |
| I5 시점 | merge 직후 당일 | merge 조건 | **merge 직후 당일** (Claude 유지. I5는 EmbeddingBridge와 독립 로직) |
| e5 prefix | 후속 확인 권장 | 미언급 | **merge 전 적용** (Claude 제안. 비용 낮고 품질 개선 가능) |

---

## 1. 작업 1: Adversarial Test 추가

**시점**: merge 전
**소요**: 30분
**테스트**: +2건 (189 → 191)

### 동기

weights hash 테스트는 "결과적으로 weights가 변했는가"를 검증하지만,
"누군가 `requires_grad=True`로 바꾸고 학습을 시도했을 때 즉시 차단되는가"를 검증하지 않는다.
실제 위험 시나리오: fine-tune 코드를 추가했지만 학습 데이터가 적어서
weights 변화가 hash 정밀도 안에 숨는 경우. Adversarial test가 이를 잡아야 한다.

### 파일

`tests/knowledge/test_sub5_adversarial.py` (신규)

### 코드

```python
"""D1 원칙 adversarial 검증.

의도적으로 D1을 위반하려 시도했을 때 방어가 작동하는지 확인.
이 테스트가 존재하는 한, D1 위반 코드가 PR에 들어와도 CI에서 차단된다.
"""
import numpy as np
import pytest


def test_d1_adversarial_grad_enable_blocked():
    """D1 위반 시도: requires_grad를 True로 바꿔도 no_grad 컨텍스트가 방어."""
    from htp.knowledge.embedding.embedding_bridge import EmbeddingBridge

    bridge = EmbeddingBridge()
    weights_before = bridge.weights_hash()

    # 의도적 위반 시도: grad 활성화
    for p in bridge._adapter._model.parameters():
        p.requires_grad = True

    # encode 호출 — no_grad 컨텍스트가 여전히 방어하는지
    vec = bridge.encode("test adversarial input")

    # 결과: 벡터는 정상 생성되지만 weights는 불변
    assert vec.shape == (bridge.dim,)
    assert bridge.weights_hash() == weights_before

    # 정리: 원래 상태 복원 (다른 테스트에 영향 방지)
    for p in bridge._adapter._model.parameters():
        p.requires_grad = False


def test_d1_adversarial_fit_with_data_noop():
    """D1 위반 시도: fit()에 대량 데이터를 넣어도 weights 불변."""
    from htp.knowledge.embedding.embedding_bridge import EmbeddingBridge

    bridge = EmbeddingBridge()
    weights_before = bridge.weights_hash()

    # 의도적 위반 시도: 실제 학습 데이터를 fit에 주입
    corpus = ["pattern completion", "memory recall", "neural network",
              "hippocampal replay", "synaptic plasticity"] * 100
    bridge.fit(corpus)  # D1: no-op이어야 함

    assert bridge.weights_hash() == weights_before
```

### 검증 기준

- 두 테스트 모두 PASS
- weights_hash 비교가 동일
- 기존 189개 테스트 영향 없음

---

## 2. 작업 2: e5 Prefix 적용

**시점**: merge 전
**소요**: 30분
**테스트**: +1건 (191 → 192)

### 동기

`intfloat/multilingual-e5` 시리즈는 query에 `"query: "`, 문서에 `"passage: "` prefix를 붙여야
최적 성능이 나온다. 현재 prefix 없이 13/14를 달성했다면, 추가 시 경계 사례
(Hopfield 가짜 매칭 등) 구분력이 개선될 가능성이 있다.

### 변경 파일 4개

#### 2-1. STAdapter (핵심 변경)

`htp/knowledge/embedding/st_adapter.py`

```python
class STAdapter:
    # ... 기존 __init__ 동일 ...

    def encode_one(self, text: str, *, is_query: bool = False) -> np.ndarray:
        """D1 ③: no_grad context.

        e5 모델은 query/passage prefix로 최적 성능:
        - is_query=True:  검색 질의 시 ("query: " prefix)
        - is_query=False: 문서 저장 시 ("passage: " prefix)
        """
        prefix = "query: " if is_query else "passage: "
        with torch.no_grad():
            return np.asarray(
                self._model.encode(prefix + text, normalize_embeddings=True),
                dtype=np.float64,
            )
```

#### 2-2. EmbeddingBridge (메서드 추가)

`htp/knowledge/embedding/embedding_bridge.py`

```python
class EmbeddingBridge:
    # ... 기존 코드 ...

    def encode(self, text: str) -> np.ndarray:
        """문서 저장용 인코딩 (passage mode)."""
        return self._adapter.encode_one(text, is_query=False)

    def encode_query(self, text: str) -> np.ndarray:
        """검색 질의용 인코딩 (query mode)."""
        return self._adapter.encode_one(text, is_query=True)
```

#### 2-3. TextEncoder Protocol 확장

`htp/knowledge/encoder.py`

```python
class TextEncoder(Protocol):
    @property
    def dim(self) -> int: ...
    def encode(self, text: str) -> np.ndarray: ...
    def encode_query(self, text: str) -> np.ndarray: ...  # 신규
    def fit(self, corpus: list[str]) -> None: ...
    # save/load 등 기존 메서드 유지
```

#### 2-4. TfidfJLEncoder 호환

`htp/knowledge/tfidf_encoder.py`

```python
class TfidfJLEncoder:
    # ... 기존 코드 ...

    def encode_query(self, text: str) -> np.ndarray:
        """TF-IDF는 query/passage 구분 없음. encode()와 동일."""
        return self.encode(text)
```

### KnowledgeLoop 연동

`htp/knowledge/loop.py`

```python
class KnowledgeLoop:
    def ingest(self, text, ...):
        vec = self.encoder.encode(text)            # passage mode (기존과 동일)

    def query(self, question, ...):
        q_vec = self.encoder.encode_query(question) # query mode (변경)

    def discover(self, ...):
        # discover는 저장된 벡터 간 비교이므로 변경 없음
```

### 테스트

`tests/knowledge/test_sub5_e5_prefix.py` (신규)

```python
def test_e5_prefix_query_vs_passage():
    """query prefix와 passage prefix가 다른 벡터를 생성하는지 확인."""
    from htp.knowledge.embedding.embedding_bridge import EmbeddingBridge
    import numpy as np

    bridge = EmbeddingBridge()
    text = "pattern completion in hippocampus"

    vec_passage = bridge.encode(text)
    vec_query = bridge.encode_query(text)

    # 같은 텍스트라도 prefix가 다르면 벡터가 다름
    cosine = float(np.dot(vec_passage, vec_query) /
                   (np.linalg.norm(vec_passage) * np.linalg.norm(vec_query)))
    assert cosine < 0.999, "query와 passage 벡터가 동일하면 prefix 미적용"
    assert cosine > 0.8, "같은 텍스트이므로 의미적으로는 가까워야 함"
```

### 검증 기준

- `test_e5_prefix_query_vs_passage` PASS
- 기존 3 테스트셋(Journal, Paper, Vault) 재실행하여 성능 유지 또는 개선 확인
- Vault "Hopfield" cosine 변화 측정 (개선 여부 기록, 개선되지 않아도 blocking 아님)

---

## 3. 작업 3: I5 Confidence Score MVP

**시점**: merge 직후 (당일)
**소요**: 1시간
**테스트**: +3건 (192 → 195)

### 동기

Vault 테스트에서 "Hopfield network 패턴 인출" 쿼리가 vault에 없는 주제인데도
0.85+ cosine으로 가짜 top-1을 반환. 지식 저장소의 신뢰도를 파괴하는 문제.
Top-1 vs Top-2 gap으로 confidence를 측정하여 "확신 없음"을 명시적으로 표현한다.

### 신규 파일

`htp/knowledge/confidence.py`

```python
"""Query 결과의 confidence 측정.

핵심 아이디어: top-1과 top-2의 cosine gap이 좁으면 확신이 없다.
- gap 큼 → top-1이 명확히 우세 → 진짜 매칭
- gap 좁음 → 비슷한 후보가 여럿 → 매칭 없거나 모호
"""
from dataclasses import dataclass


@dataclass
class ScoredResult:
    """query 결과 단일 항목 + 메타데이터."""
    text: str
    source: str
    similarity: float
    rank: int


@dataclass
class QueryResult:
    """query 반환값. confidence 포함."""
    question: str
    results: list[ScoredResult]
    confidence: float       # top-1 vs top-2 gap
    has_match: bool          # confidence > threshold

    @staticmethod
    def compute_confidence(
        similarities: list[float],
        gap_threshold: float = 0.02,
    ) -> tuple[float, bool]:
        """
        Top-1 vs Top-2 cosine gap으로 confidence 측정.

        Parameters
        ----------
        similarities : list[float]
            검색 결과의 cosine similarity 리스트 (정렬 불필요).
        gap_threshold : float
            이 값 이하이면 "확신 없음".
            0.02는 multilingual-e5-small의 밀집 분포에서
            경험적으로 적절한 기본값.

        Returns
        -------
        (gap, has_match) : tuple[float, bool]
            gap: top-1과 top-2의 cosine 차이.
            has_match: gap > threshold.
        """
        if len(similarities) < 2:
            return 0.0, False
        sorted_sims = sorted(similarities, reverse=True)
        gap = sorted_sims[0] - sorted_sims[1]
        return gap, gap > gap_threshold
```

### KnowledgeLoop.query() 수정

`htp/knowledge/loop.py`

```python
from htp.knowledge.confidence import QueryResult, ScoredResult

class KnowledgeLoop:
    def query(self, question: str, top_k: int = 5) -> QueryResult:
        q_vec = self.encoder.encode_query(question)  # 작업 2의 prefix 적용
        neighbors = self._find_neighbors(q_vec, top_k=top_k)

        similarities = [n.similarity for n in neighbors]
        confidence, has_match = QueryResult.compute_confidence(similarities)

        results = [
            ScoredResult(
                text=n.entry.text,
                source=n.entry.source,
                similarity=n.similarity,
                rank=i + 1,
            )
            for i, n in enumerate(neighbors)
        ]

        return QueryResult(
            question=question,
            results=results,
            confidence=confidence,
            has_match=has_match,
        )
```

### CLI 출력 변경

`htp/knowledge/__main__.py` (query 커맨드 부분)

```python
# AS-IS
# [0.85] vault-projects/HwpxViewer Work Log

# TO-BE
def _format_query_result(result: QueryResult) -> str:
    lines = []

    if not result.has_match:
        lines.append(f"⚠ Low confidence (gap={result.confidence:.3f})"
                     f" — 확실한 매칭 없음")

    for r in result.results:
        marker = "  " if result.has_match else "  ?"
        lines.append(f"{marker}[{r.similarity:.3f}] {r.source}: {r.text[:60]}")

    return "\n".join(lines)
```

```
# 매칭 있을 때 (gap > 0.02):
  [0.91] vault-topics: V-JEPA world model self-supervised...
  [0.82] vault-ai: World Model history...

# 매칭 없을 때 (gap ≤ 0.02):
⚠ Low confidence (gap=0.008) — 확실한 매칭 없음
  ?[0.853] vault-projects: HwpxViewer Work Log...
  ?[0.845] vault-projects: blogautomation Work Log...
```

### 테스트

`tests/knowledge/test_sub5_confidence.py` (신규)

```python
"""I5 Confidence Score 검증."""
from htp.knowledge.confidence import QueryResult


def test_confidence_clear_match():
    """명확한 매칭: top-1 >> top-2 → confidence 높음."""
    sims = [0.92, 0.78, 0.71, 0.65]
    gap, has_match = QueryResult.compute_confidence(sims)
    assert gap > 0.1
    assert has_match is True


def test_confidence_no_match():
    """매칭 없음: 모든 similarity 비슷 → confidence 낮음."""
    sims = [0.86, 0.855, 0.85, 0.845]
    gap, has_match = QueryResult.compute_confidence(sims)
    assert gap < 0.02
    assert has_match is False


def test_confidence_single_entry():
    """entry 1개: confidence 0 → has_match=False."""
    gap, has_match = QueryResult.compute_confidence([0.9])
    assert gap == 0.0
    assert has_match is False
```

### 검증 기준

- 3 테스트 PASS
- Vault "Hopfield network 패턴 인출" 재실행 → `has_match=False` 확인
- Vault "HTP thalamus router 라우팅" 재실행 → `has_match=True` 확인 (실제 매칭은 살아있어야 함)
- 전체 195/195 PASS

---

## 4. 실행 타임라인

```
Day 1 (merge 당일):

  09:00  작업 1: Adversarial test 2건 추가
         → pytest 실행 → 191/191 확인

  09:30  작업 2: e5 prefix 적용
         → STAdapter.encode_one(is_query) 수정
         → EmbeddingBridge.encode_query() 추가
         → TextEncoder Protocol 확장
         → TfidfJLEncoder.encode_query() 호환 추가
         → KnowledgeLoop.query() 연동
         → test 1건 추가
         → pytest 실행 → 192/192 확인

  10:00  기존 3 테스트셋 재실행 (prefix 적용 후)
         → Journal / Paper / Vault 성능 유지/개선 확인
         → Vault "Hopfield" cosine 변화 기록

  10:30  ──── main merge (experiment/embedding-bridge → main) ────

  11:00  작업 3: I5 confidence score MVP
         → confidence.py 신규
         → KnowledgeLoop.query() 수정
         → CLI 출력 변경
         → test 3건 추가
         → pytest 실행 → 195/195 확인

  12:00  Vault "Hopfield" 재검증
         → has_match=False 확인
         → "HTP thalamus" has_match=True 확인

  12:30  commit + push to main
```

---

## 5. 체크리스트

| # | 항목 | 작업 | merge 전/후 | 완료 |
|---|------|------|:---------:|:---:|
| 1 | Adversarial test | 2건 추가 | **전** | ☐ |
| 2 | e5 prefix | `encode_query()` + 1건 테스트 | **전** | ☐ |
| 3 | 전체 회귀 | 192/192 PASS | **전** | ☐ |
| 4 | 3 테스트셋 재실행 | prefix 적용 후 성능 확인 | **전** | ☐ |
| 5 | **main merge** | experiment → main | **실행** | ☐ |
| 6 | I5 confidence | `compute_confidence()` + 3건 | **직후** | ☐ |
| 7 | Hopfield 재검증 | has_match=False 확인 | **직후** | ☐ |
| 8 | 테스트 최종 | 195/195 PASS | **직후** | ☐ |

---

## 6. 리스크

| Risk | 영향 | 완화 |
|------|------|------|
| e5 prefix 적용 후 기존 테스트 regression | ingest된 벡터(passage)와 query 벡터(query)가 다른 공간 → 기존 similarity 수치 변화 | 기존 테스트가 absolute threshold가 아닌 ranking 기반이면 영향 없음. threshold 기반 테스트가 있으면 수치 재조정 |
| encode_query() Protocol 추가가 기존 TextEncoder 구현체 깨짐 | Protocol에 메서드 추가 → 기존 구현체가 미구현 시 isinstance 실패 | TfidfJLEncoder에 encode_query() = encode() 래퍼 즉시 추가 |
| I5 gap_threshold=0.02가 e5 prefix 적용 후 부적절 | prefix 적용으로 cosine 분포가 변하면 gap 분포도 변함 | 작업 2 완료 후, 작업 3 전에 Vault 결과로 gap 분포 확인하여 threshold 조정 |
| merge conflict | experiment branch가 오래 분리되어 main과 충돌 가능 | 작업 1, 2를 빠르게 끝내고 당일 merge. I5는 main에서 진행하므로 conflict 위험 없음 |

---

## 7. 후속 (이 계획 스코프 밖)

| 항목 | 우선순위 | 시기 |
|------|---------|------|
| I3 relative ranking (top-k 내 min-max 정규화) | 2순위 | I5 완료 후 |
| I1 recursive glob (서브디렉토리 ingest) | 3순위 | 다음 sub-cycle |
| I4 Obsidian frontmatter tags 추출 | 3순위 | I1과 통합 |
| I2 frontmatter strip (list preview) | 4순위 | UX 개선 일괄 |
| Stage 4+5 LLMRegion + Pipeline (sub-4) | 본선 | merge 안정화 후 |

---

## 8. Sign-off

| 항목 | 값 |
|------|---|
| 계획 기준 브랜치 | `experiment/embedding-bridge` |
| 기준 테스트 | 189/189 PASS |
| 목표 테스트 | **195/195 PASS** |
| merge 전 작업 | 2건 (adversarial + prefix) |
| merge 후 작업 | 1건 (I5 confidence) |
| 예상 소요 | 2-3시간 (Day 1 오전 완료) |
| 리뷰 근거 | Claude 리뷰 + Gemini 리뷰 합의 |
