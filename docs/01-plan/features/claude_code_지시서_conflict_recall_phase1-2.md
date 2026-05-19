# Claude Code 작업 지시서 — htp-conflict-memory recall 결함 대응 (Phase 1·2)

> **출처**: 컨테이너 실측 3회 (외부 리뷰 §5 후속). 커밋 `3d2d611` 기준.
> **범위**: 이 지시서는 **1·2단계만** 다룬다. 3단계(처방 적용)는
> 별도 측정이 끝난 뒤 별도 지시서로 내려온다. **3단계를 미리 하지 말 것.**
> **원칙**: 모든 작업은 "고쳐라"가 아니라 "고치고 측정/테스트로 증명하라".

---

## 0. 배경 — 무엇이 확정됐나 (측정 결과)

외부 리뷰 문서(`htp-conflict-memory-실사용검증-외부리뷰용.md`)는
**"Full MERGE GO"** 로 판정했다. 컨테이너 실측 결과 이 판정은 **NO-GO** 다.
아래는 추측이 아니라 `intfloat/multilingual-e5-small` 실 임베딩으로
측정한 수치다.

### 측정 1 — 거짓 양성 (anchor 6 / probe 14)

| 모드 | 거짓 양성(NEG인데 HIT) | 참 양성 |
|---|---|---|
| 현재 시스템 (절대 L2 < 0.6) | **12/12 (100%)** | 2/2 |
| query-prefix 적용 | 10/12 (83%) | 2/2 |

- "중세 고딕 성당 구조", "김치 발효" 같은 **완전 무관 입력도 충돌 해석을
  recall** 함. threshold 0.6 은 사실상 아무것도 거르지 않는다.
- 시연 1건(eviction 2회)이 HIT 한 건 시스템이 작동해서가 아니라
  **threshold 가 너무 느슨해 무엇이든 HIT** 하기 때문이었다.

### 측정 2 — 처방 비교 (튜닝 레이어 4종)

| 처방 | HARD_NEG FP | TRUE_POS TP | 판정 |
|---|---|---|---|
| Baseline (절대 cos≥0.82) | 6/6 | 2/2 | FAIL |
| P1 상대마진 (gap≥0.05) | 0/6 | 1/2 | FAIL |
| P2 분포컷 (cos≥0.867) | 1/6 | 1/2 | FAIL |
| P3 결합 | 0/6 | 1/2 | FAIL |

- **네 방법 전부 FAIL.** TRUE_POS cos(0.865~0.890) 와 HARD_NEG
  cos(0.848~0.869) 분포가 겹쳐, 절대컷·마진·분포컷 어느 자(尺)로도
  분리 불가. 신호 자체가 trigger 임베딩에 없다.
- 결함 위치는 **threshold 가 아니라 recall key 설계**
  (외부 리뷰 §9-1.1 의 `state_vec = trigger_vec` 결정).

### 측정 3 — 코드 리딩 발견 검증

- **발견 A (cosine 선택 / L2 컷 지표 불일치)**: **반증됨.**
  e5 는 L2 정규화 출력이라 corr(cos,L2) = -0.997, 두 지표 단조 일치.
  지표 불일치는 거짓 양성 원인이 **아니다.**
- **발견 B (query-prefix 누락)**: **확인됨.** `_try_recall_conflict` 가
  `encode()`(passage prefix)로 검색. `encode_query()`(query prefix)
  미사용. 적용 시 거짓 양성 12→10 로 부분 개선(완전 해결 아님).

---

## 1. 절대 금지 항목 (먼저 읽을 것)

다음은 외부 리뷰 §10·Q2 가 권하지만 **측정으로 무효 판정된** 작업이다.
**Claude Code 는 다음을 수행하지 말 것:**

1. ❌ `CONFLICT_RECALL_MISMATCH_THRESHOLD` 값 변경 / 동적 조정 함수 작성
   (`_threshold_for_dim(dim)` 류 포함). → 측정 2 에서 튜닝 레이어
   전부 FAIL. 헛수고 확정.
2. ❌ recall key 설계 변경 (interpretation-key, hybrid, 구조화 키 등).
   → 3단계 영역. 측정 미완. 지금 손대면 추측 적용.
3. ❌ 외부 리뷰의 "MERGE GO" 를 근거로 한 master 머지 / PR 생성.
4. ❌ 발견 A 관련 "cosine/L2 정합" 리팩토링. → 측정으로 반증됨.

이 4가지를 건드리면 지난 3회 측정이 무의미해진다.

---

## 2. Phase 1 — 회귀 방지선 (코드 수정 전 먼저)

**목표**: 거짓 양성을 잡는 테스트를 repo 에 심는다. §6 Bug 1·2 가
단위 테스트를 비껴간 이유 = 무관 쌍 테스트 부재. 그 구멍을 막는다.

### 작업 2-1. 무관 충돌 거짓 양성 테스트 추가

**파일**: `tests/knowledge/test_conflict_memory.py` (기존 파일에 append)

**기존 패턴 준수** (이미 파일에 있음):
- `_SKIP_HF = os.environ.get("HTP_SKIP_HF_DOWNLOAD", "")...` 가드 사용
- `@pytest.mark.skipif(_SKIP_HF, reason="HF download skipped")`
- `from htp.knowledge.embedding import EmbeddingBridge` (함수 내 import)
- `from htp.memory.memory_system import MemorySystem`

**추가할 테스트** (3개):

```python
@pytest.mark.skipif(_SKIP_HF, reason="HF download skipped")
def test_recall_does_not_hit_on_unrelated_conflict():
    """완전 무관 입력은 저장된 충돌 해석을 recall 해서는 안 된다.

    측정 1 에서 'EASY_NEG' 가 현재 시스템에서 거짓 양성으로 HIT 함을
    확인. 이 테스트는 그 결함을 고정한다 — 처방 적용 후 GREEN 이어야
    머지 가능. 현 master 에서는 RED (의도된 실패 = 결함 증명).
    """
    from htp.knowledge.embedding import EmbeddingBridge
    import torch
    with tempfile.TemporaryDirectory() as td:
        enc = EmbeddingBridge()
        mem = MemorySystem(memory_dir=Path(td) / "mem")
        # anchor: 인프라 캐시 충돌 해석 저장
        tv = torch.tensor(enc.encode(
            "Redis LRU 캐시 eviction 전략 메모리 축출"),
            dtype=torch.float32)
        mem.save_conflict(
            trigger_vec=tv, new_text="Redis LRU eviction",
            partner_texts=["해마 CA3"], interpretation="categorical conflict",
            conflict_score=0.15)
        # 완전 무관 probe — 절대 HIT 하면 안 됨
        for unrelated in ("중세 고딕 성당 부벽 구조 하중 분산",
                          "김치 발효 유산균 pH 변화 숙성"):
            qv = torch.tensor(enc.encode_query(unrelated),
                              dtype=torch.float32)
            results = mem.recall_conflict(qv, top_k=3)
            # recall_conflict 자체는 후보를 줄 수 있으나,
            # _try_recall_conflict 게이트 통과(mismatch<thr)는 막혀야 함.
            # 게이트 로직을 직접 재현해 검증:
            if results:
                best_ep, _ = results[0]
                import struct
                n = len(best_ep.state_vec) // 4
                pv = torch.tensor(
                    struct.unpack(f"{n}f", best_ep.state_vec),
                    dtype=torch.float32)
                mismatch = float((qv - pv).norm())
                assert mismatch >= mem.CONFLICT_RECALL_MISMATCH_THRESHOLD, (
                    f"거짓 양성: 무관 입력 '{unrelated[:20]}' 가 "
                    f"mismatch={mismatch:.3f} 로 recall HIT "
                    f"(thr={mem.CONFLICT_RECALL_MISMATCH_THRESHOLD})")


@pytest.mark.skipif(_SKIP_HF, reason="HF download skipped")
def test_recall_query_uses_query_prefix():
    """발견 B 회귀 방지 — recall 경로가 encode_query() 를 쓰는지.

    _try_recall_conflict 가 passage prefix(encode) 로 검색하면
    e5 비대칭 검색이 깨진다. 이 테스트는 loop 가 query prefix 를
    쓰도록 강제한다. 현 master RED (encode 사용 중) → 처방 후 GREEN.
    """
    from htp.knowledge.embedding import EmbeddingBridge
    enc = EmbeddingBridge()
    # e5: query/passage prefix 가 다른 벡터를 내야 정상
    v_p = enc.encode("테스트 문장")
    v_q = enc.encode_query("테스트 문장")
    import numpy as np
    assert not np.allclose(v_p, v_q), (
        "encode 와 encode_query 가 동일 벡터 — prefix 미적용 의심")
    # 실제 loop 경로가 query prefix 쓰는지는 작업 2-2 의 xfail 로 추적


def test_recall_fp_dataset_is_tracked():
    """측정 데이터셋이 repo 에 고정돼 재현 가능한지 sanity."""
    from pathlib import Path as _P
    p = _P("scripts/conflict_recall_fp_eval.py")
    assert p.exists(), (
        "측정 스크립트 미존재 — 작업 2-3 에서 scripts/ 에 커밋 필요")
```

### 작업 2-2. trigger-key 결함을 xfail 로 명문화

**목적**: 첫 외부 리뷰 §2 "열린 질문과 잠긴 코드의 모순" 을 코드
레벨에서 해소. 결정이 잠겼지만 미검증임을 테스트 스위트가 알게 한다.

**파일**: 동일 파일에 append.

```python
@pytest.mark.skipif(_SKIP_HF, reason="HF download skipped")
@pytest.mark.xfail(
    reason="trigger-key 설계 결함(측정 2): 표면 다르고 구조 같은 "
           "충돌을 trigger 임베딩으로 MISS. 3단계 처방 전까지 "
           "의도된 실패. 외부 리뷰 §9-1.1 미검증 결정.",
    strict=True)
def test_trigger_key_recalls_same_conflict_different_surface():
    """같은 추상 충돌의 다른 표면 표현은 recall 돼야 한다 (당위).

    'Redis 캐시 축출' 과 '시냅스 가지치기' 는 같은 eviction 추상
    충돌이나 trigger 표면이 달라 임베딩이 멀다. 측정 2 에서 이
    케이스 분리 불가 확인. strict xfail = 처방으로 해결되면
    XPASS 로 빨간불 → 그때 이 마커를 제거하고 GREEN 전환.
    """
    from htp.knowledge.embedding import EmbeddingBridge
    import torch, struct
    with tempfile.TemporaryDirectory() as td:
        enc = EmbeddingBridge()
        mem = MemorySystem(memory_dir=Path(td) / "mem")
        tv = torch.tensor(enc.encode(
            "Redis LRU 캐시 eviction 메모리 축출 정책"),
            dtype=torch.float32)
        mem.save_conflict(
            trigger_vec=tv, new_text="Redis eviction",
            partner_texts=["x"], interpretation="eviction 추상 충돌",
            conflict_score=0.15)
        # 같은 추상 충돌, 완전히 다른 표면
        qv = torch.tensor(enc.encode_query(
            "시냅스 가지치기 미세아교세포 약한 연결 제거"),
            dtype=torch.float32)
        results = mem.recall_conflict(qv, top_k=3)
        assert results, "후보 없음"
        best_ep, _ = results[0]
        n = len(best_ep.state_vec) // 4
        pv = torch.tensor(struct.unpack(f"{n}f", best_ep.state_vec),
                          dtype=torch.float32)
        mismatch = float((qv - pv).norm())
        # 당위: 같은 충돌이므로 HIT 해야 함 → 현 설계로는 실패(xfail)
        assert mismatch < mem.CONFLICT_RECALL_MISMATCH_THRESHOLD
```

### 작업 2-3. 측정 스크립트를 repo 에 고정

컨테이너에서 작성된 두 스크립트를 `scripts/` 에 커밋해 재현 가능하게:
- `scripts/conflict_recall_fp_eval.py` (거짓 양성 측정)
- `scripts/conflict_recall_remedy_eval.py` (처방 비교)

> 이 파일들의 내용은 별첨으로 전달된다. **로직을 수정하지 말고
> 그대로 커밋.** 재현성이 목적이므로 임의 개선 금지.

---

## 3. Phase 2 — 검증 게이트 (Phase 1 완료 후)

### 작업 3-1. 회귀 baseline 갱신 확인

```
HTP_SKIP_HF_DOWNLOAD=1 python -m pytest tests/ -q
```
- 기존 303 PASS 유지 (HF skip 모드). 신규 비-HF 테스트
  (`test_recall_fp_dataset_is_tracked`) 1개 추가 → 304.
- **깨진 회귀 0건 필수.** 1건이라도 깨지면 중단하고 보고.

### 작업 3-2. HF 모드 결과를 정확히 보고

```
python -m pytest tests/knowledge/test_conflict_memory.py -q -rxX
```
**기대 결과** (현 master = 결함 상태):
- `test_recall_does_not_hit_on_unrelated_conflict` → **FAIL (RED)**
  ← 이게 정상. 결함이 실재함을 증명. 고치지 말 것.
- `test_trigger_key_recalls_same_conflict_different_surface`
  → **xfail (예상된 실패)**
- `test_recall_query_uses_query_prefix` → PASS

> RED/xfail 을 "고쳐서 GREEN 만들기" 금지. 이건 결함의 *증거*이지
> 버그가 아니다. GREEN 전환은 3단계 처방에서만.

### 작업 3-3. CLAUDE.md 에 의사결정 기록

`CLAUDE.md` 에 다음을 append (기존 형식 따름):

```markdown
## htp-conflict-memory recall — 측정 확정 (외부 리뷰 후속)

- 외부 리뷰 "Full MERGE GO" 판정 → 실측 결과 **NO-GO**
- 거짓 양성: 현재 시스템 100% (threshold 0.6 무력)
- 튜닝 레이어(threshold/margin/dist-cut) 전부 FAIL — 측정 2
- 결함 위치: threshold 아님. recall key 설계(§9-1.1 trigger-key)
- 발견 A(지표 불일치) 반증 / 발견 B(query-prefix) 확인
- trigger-key = 미검증 LOCK → xfail 로 명문화 (test_trigger_key_*)
- 3단계 처방(interpretation-key/hybrid/구조화) 측정 대기 중
- 금지: threshold 동적조정, recall key 변경, MERGE — 본 지시서 §1
```

---

## 4. 완료 보고 형식

작업 종료 시 다음만 보고 (산문 금지, 표로):

| 항목 | 결과 |
|---|---|
| 신규 테스트 추가 | n개 (파일:라인) |
| 비-HF 회귀 (303 기준) | PASS/FAIL 수 |
| HF 모드: unrelated FP 테스트 | RED 확인 여부 |
| HF 모드: trigger-key xfail | xfail 확인 여부 |
| 측정 스크립트 커밋 | 2개 경로 |
| CLAUDE.md 갱신 | 라인 |
| 금지항목 §1 위반 | 없음 (필수) |

---

## 5. 한 줄 요약 (Claude Code 가 가장 먼저 읽을 것)

> **고치는 작업이 아니다.** 결함을 *테스트로 고정*하고, trigger-key
> 미검증을 *xfail 로 명문화*하고, 측정 스크립트를 *재현 가능하게
> 커밋*하는 것이 전부다. RED/xfail 은 정상이며 건드리지 않는다.
> threshold·recall key·머지는 §1 에 의해 금지. 처방은 다음 지시서.
