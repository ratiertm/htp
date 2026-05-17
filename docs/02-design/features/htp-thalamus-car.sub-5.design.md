---
template: design
feature: htp-thalamus-car
sub_cycle: sub-5 (Stage 6 EmbeddingBridge)
date: 2026-05-17
author: Mindbuild
project: HTP (Hub Topology Programming)
status: Draft
selected_option: B — Modular (embedding/ 패키지)
branch: experiment/embedding-bridge
---

# htp-thalamus-car sub-5 Design — EmbeddingBridge

> **Summary**: Plan FR-01~17 + D1-D4 원칙을 Option B (Modular) 로 구현. `htp/knowledge/embedding/` 패키지에 `EmbeddingBridge` (TextEncoder Protocol 구현체) + multiple model adapters. `RegionSignature.dim` dynamic 확장으로 384/768 차원 임베딩 지원. D1-D4 검증 테스트가 강제.
>
> **Selected Architecture**: Option B — Modular (sub-2 router/, sub-3 coherence/ 패턴 일관)
> **Predecessor**: sub-5 Plan (commit `e4c58fa`, experiment/embedding-bridge branch)
> **Test Target**: 172 → **185** (Plan §5.3 명시)

---

## Context Anchor (Plan 에서 전파)

| Key | Value |
|-----|-------|
| **WHY** | TF-IDF + JL 의 본질 한계 정량 증명 (시나리오 D: top-1 0/4) — 데이터로 해결 불가. 본질 해결 = 사전학습 임베딩. |
| **WHO** | HTP 개발자 + Knowledge Loop 사용자. D2 Protocol 호환으로 기존 호출자 무영향. |
| **RISK** | (R4) HTP 의 LLM 종속화 — D1-D4 강제로 방지. (R5) RegionSignature dim 변경 backward-compat. (R7) 한국어 모델 품질. |
| **SUCCESS** | top-1 ≥ 3/4, discover ≥ 75%, 한국어 PASS, cross-language PASS, 회귀 172 유지. |
| **SCOPE** | `experiment/embedding-bridge` 브랜치만. Go/No-Go 통과 시 main merge. |

---

## 1. Architecture: Option B — Modular (`htp/knowledge/embedding/`)

```
htp/knowledge/
├── encoder.py                  무변경 (TfidfJLEncoder 보존 — D3)
├── embedding/                  [신규 패키지] sub-2 router/ 패턴 일관
│   ├── __init__.py             공개 export
│   ├── base.py                 BaseEmbeddingModel Protocol + 공통 헬퍼
│   ├── bridge.py               EmbeddingBridge(TextEncoder) — Protocol 어댑터
│   ├── st_adapter.py           sentence-transformers 어댑터
│   └── hf_adapter.py           transformers (HuggingFace) raw 어댑터 — fallback
└── ...

htp/thalamus/
└── signature.py                수정 — dim 동적 + assertion 완화 (D4 보조)

requirements.txt                수정 — sentence-transformers>=2.7 추가
```

---

## 2. 6 Sub-Decision 결정

| # | 항목 | 결정 | 근거 |
|---|------|------|------|
| 1 | Default 모델 | **`intfloat/multilingual-e5-small`** (118MB, 384-dim) | Plan §5.2 권장. 한국어 지원 + 작음 |
| 2 | Architecture | **Option B Modular** (`embedding/` 패키지) | sub-2 router/ + sub-3 coherence/ 패턴 일관성 |
| 3 | dim 호환 | **Dynamic** — RegionSignature(dim=384) | Plan FR-10. 향후 다른 모델 (768/1024) 도 지원 |
| 4 | CLI `--encoder` 옵션 | **추가** (`tfidf` / `embedding`) | D3 사용자 명시적 선택 가능 |
| 5 | 캐시 정책 | **HuggingFace 기본 캐시** (`~/.cache/huggingface/`) | OS 표준 + .htp/models/ 중복 방지. 사용자 `HF_HOME` 환경변수로 override 가능 |
| 6 | D1 검증 방식 | **3중**: `model.eval()` + `torch.no_grad()` (encode 내부) + weights hash 비교 (test) | 강제 + 검증 |

---

## 3. Architecture Detail

### 3.1 BaseEmbeddingModel Protocol (`embedding/base.py`)

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class BaseEmbeddingModel(Protocol):
    """사전학습 모델 어댑터 인터페이스.

    EmbeddingBridge 내부에서 모델 종류별 어댑터 다형성.
    예: STAdapter (sentence-transformers) / HFAdapter (transformers raw)
    """
    @property
    def dim(self) -> int: ...
    @property
    def model_name(self) -> str: ...
    def encode_one(self, text: str) -> "np.ndarray": ...
    def encode_batch(self, texts: list[str]) -> "np.ndarray": ...
```

### 3.2 EmbeddingBridge (TextEncoder Protocol 어댑터)

```python
class EmbeddingBridge:
    """사전학습 모델을 TextEncoder Protocol 로 wrapping (D2 Protocol 호환).

    D1 Frozen: model.eval() + torch.no_grad() 강제
    D2 Protocol: TextEncoder Protocol 완전 준수 (encode/fit/dim/save/load)
    D3 Fallback: TfidfJLEncoder 와 독립 — 둘 다 KnowledgeLoop 가 받음
    D4 학습 분리: 이 클래스는 *encode 만*. HTP 학습은 RegionSignature/CA3 에서.
    """
    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-small",
        adapter:    "BaseEmbeddingModel | None" = None,
    ):
        if adapter is None:
            from .st_adapter import STAdapter
            adapter = STAdapter(model_name)
        self._adapter = adapter
        self._dim = adapter.dim
        self._fitted = True   # 사전학습 — 항상 True

    @property
    def dim(self) -> int:
        return self._dim

    def fit(self, corpus: list[str]) -> None:
        """no-op — 사전학습 모델은 fit 불필요 (D1 frozen)."""
        pass

    def encode(self, text: str) -> "np.ndarray":
        """D1: 추론 시 grad 비활성 + eval mode."""
        return self._adapter.encode_one(text)

    def save(self, path) -> None:
        """metadata 만 pickle (모델 자체는 HF 캐시 사용)."""
        ...

    def load(self, path) -> bool:
        """metadata 복원 + 같은 model_name 으로 adapter 재로드."""
        ...
```

### 3.3 STAdapter (`embedding/st_adapter.py`) — Default

```python
class STAdapter:
    """sentence-transformers 어댑터.

    D1: SentenceTransformer 가 내부적으로 eval mode + no_grad.
    """
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name)
        # D1: 명시적 eval + grad 비활성
        self._model.eval()
        for p in self._model.parameters():
            p.requires_grad = False
        self._dim = self._model.get_sentence_embedding_dimension()
        self._model_name = model_name

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return self._model_name

    def encode_one(self, text: str) -> "np.ndarray":
        import torch
        with torch.no_grad():
            vec = self._model.encode(text, normalize_embeddings=True)
        return np.asarray(vec, dtype=np.float64)

    def encode_batch(self, texts: list[str]) -> "np.ndarray":
        import torch
        with torch.no_grad():
            vecs = self._model.encode(texts, normalize_embeddings=True,
                                       batch_size=8, show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float64)
```

### 3.4 RegionSignature dim 동적 (D4 보조)

```python
@dataclass
class RegionSignature:
    centroid: np.ndarray = field(default=None)
    count:    int = 0
    dim:      int = 64    # default 유지 (backward-compat)

    def __post_init__(self):
        if self.centroid is None:
            self.centroid = np.zeros(self.dim, dtype=np.float64)
        elif self.centroid.shape != (self.dim,):
            # dim 자동 추론 (centroid 가 명시 주어진 경우)
            self.dim = self.centroid.shape[0]
    # ... update / similarity 무변경 (dim 무관)
```

기존 `dim=64` 호출자 무영향, 새 `RegionSignature(dim=384)` 가능.

### 3.5 CLI `--encoder` 옵션 (D3)

```python
# cli/__init__.py 의 _build_parser:
parser.add_argument(
    "--encoder",
    choices=["tfidf", "embedding"],
    default="tfidf",     # 회귀 보호 — 기본은 기존
    help="인코더 선택 (tfidf: 빠름, embedding: 정확)",
)

# 각 subcommand 의 _make_loop():
def _make_loop(encoder_type: str = "tfidf"):
    if encoder_type == "embedding":
        from ..embedding import EmbeddingBridge
        enc = EmbeddingBridge()
    else:
        enc = TfidfJLEncoder()
    return KnowledgeLoop(encoder=enc)
```

---

## 4. DAG 의존 방향 (확장)

```
htp/knowledge/embedding/* ──→ sentence-transformers / transformers / numpy
                          ──→ htp.knowledge.encoder.TextEncoder Protocol (interface)

금지: htp/knowledge/embedding/* → htp.runtime / htp.thalamus / htp.memory
        (sub-1 의 knowledge DAG 규칙 유지)
```

`test_no_circular_deps.py` 의 `_KNOWLEDGE_DIR.rglob` 가 자동으로 `embedding/` 도 검사 (session-3 의 rglob 변경 활용).

---

## 5. Session Plan

| Session | Scope | 누적 테스트 | 소요 |
|---------|-------|----------|------|
| **session-1** | embedding/ 패키지 + EmbeddingBridge + STAdapter + D1/D2/D3 검증 | 172 → 178 | ~3h |
| **session-2** | 시나리오 D 재현 + 한국어 + cross-language + CLI `--encoder` | 178 → 183 | ~3h |
| **session-3** | RegionSignature dim 동적 + D4 검증 + Go/No-Go 보고 | 183 → 185 | ~2h |

---

## 6. Test Plan (sub-5: +13 신규)

### 6.1 회귀 (172)
- TfidfJLEncoder fallback path — 모든 기존 테스트 무영향
- `--encoder tfidf` default 보존

### 6.2 신규 (172 → 185)

**Session 1 — embedding/ + D1/D2/D3 (+6)**
- `test_embedding_bridge_protocol_compliance` — isinstance(TextEncoder)
- `test_embedding_bridge_frozen_weights` — **D1** weights hash 변경 안 됨 (encode 후 hash 동일)
- `test_embedding_bridge_no_grad` — **D1** torch.is_grad_enabled() == False during encode
- `test_embedding_bridge_fit_is_noop` — fit(corpus) 후 weights hash 동일
- `test_embedding_bridge_save_load_round_trip` — metadata pickle round-trip
- `test_tfidf_fallback_still_works` — **D3** TfidfJLEncoder 회귀 동등

**Session 2 — 시나리오 검증 (+5)**
- `test_scenario_d_query_top1` — 20-paper query top-1 ≥ 3/4
- `test_scenario_d_discover_quality` — 합리적 매칭 ≥ 6/8
- `test_korean_semantic_match` — "기억은" ↔ "기억이" similarity ≥ 0.5
- `test_cross_language_hub` — "attention" ↔ "어텐션" ≥ 0.5
- `test_cli_encoder_option` — `--encoder embedding` 동작

**Session 3 — D4 + dim 동적 (+2)**
- `test_region_signature_dim_dynamic` — RegionSignature(dim=384) 동작
- `test_d4_htp_structure_learns_post_embedding` — RegionSignature.update / centroid 변화 → HTP 학습은 위에서만

---

## 7. Risks (Plan §6 보강)

| ID | Risk | Mitigation |
|----|------|----------|
| R1 | sentence-transformers 미설치 | **try/except ImportError** + 명확한 에러 메시지 (`pip install sentence-transformers`) |
| R2 | 첫 호출 시 모델 다운로드 ~30s | 콘솔 진행 표시 + .htp 또는 HF 캐시 활용 |
| R4 | LLM 종속화 | **D1 3중 검증** (eval + no_grad + weights hash) |
| R5 | dim 변경 backward-compat | RegionSignature(dim=64) 기존 호출자 무영향 |
| R8 | 시나리오 D Go/No-Go 미달 | sub-decision #1 모델 재선정 — bge-m3 fallback |

---

## 8. Decision Record

| Decision | Choice | Rationale |
|----------|--------|----------|
| Architecture | Option B Modular | sub-2/3 패턴 일관 |
| Default 모델 | multilingual-e5-small | 한국어 지원 + 작음 (118MB) |
| Adapter pattern | STAdapter (default) + HFAdapter (fallback) | 의존성 유연 |
| dim 호환 | Dynamic | 향후 모델 확장 보장 |
| CLI 옵션 | `--encoder tfidf|embedding` | D3 명시 |
| 캐시 | HuggingFace 기본 | OS 표준 |
| D1 검증 | 3중 (eval + no_grad + hash) | 강제 |

---

## 9. Out-of-Scope (sub-5)

- LLM full reasoning — sub-4 분리
- 임베딩 fine-tune — D1 위반
- GPU optimization / batch 가속 — 별도 cycle
- HTP 핵심 (Hub/Memory/NGE) 수정 — D4 위반
- Multimodal (image/audio) — separate cycle

---

## 10. Next: `/pdca do --scope session-1`
