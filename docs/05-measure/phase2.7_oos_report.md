# Phase 2.7 측정 결과 — 분포컷 out-of-sample 일반화 검정

**작성일**: 2026-05-20
**지시서**: `docs/05-measure/claude_code_측정지시서_phase2.7_oos_generalization.md`
**스크립트**: `scripts/conflict_recall_oos_eval.py` (신규, split 라우팅)
**Raw**: `docs/03-analysis/conflict_recall_oos_raw.{BAAI_bge-m3,default}.json`

---

## Phase 2.7 측정 결과

데이터셋: anchor 6 / TRUE_POS 10 / HARD_NEG 12 / MID_NEG 4 / EASY_NEG 2
split: calib NEG 9 / eval NEG 9 + TRUE_POS 전수 10
라벨 검수: **Y** (사용자 검수 통과 2026-05-20)

| 모델 | split | TRUE_POS 재현율 | HARD_NEG FP율 | 판정 |
|------|-------|:---------------:|:-------------:|:----:|
| bge-m3 | calib (참고) | 0/0 (TP 없음) | 1/6 (17%) | - |
| **bge-m3** | **eval** | **2/10 (20%)** | **0/6 (0%)** | **FAIL** |
| e5-small | calib (대조) | 0/0 | 0/6 (0%) | - |
| **e5-small** | **eval** | **1/10 (10%)** | **0/6 (0%)** | **FAIL 예상대로** |

**산출 컷 (bge-m3 calib 9 NEG, 95th pct)**: cos≥**0.6241** (Phase 2.6 의 0.6213 과 거의 동일 — calib 일관성 확인)
**calib→eval 일반화 격차 (bge-m3)**: FP -16.7%p (eval 에서 HARD_NEG 더 잘 거부)
**판정 (지시서 §1.M3 표 셋째 행)**: **대**
- eval 재현율 20% << 80% → in-sample PASS 가 과적합이었음

---

## 의미 분석

Phase 2.6 PASS 가 가졌던 **두 약점이 측정으로 확정**:

| Phase 2.6 PASS | Phase 2.7 OOS 결과 |
|----------------|---------------------|
| TP=2 (분모 작음) | TP 10 으로 늘리니 재현율 **20%** (8건 MISS) |
| 0.621 컷이 같은 NEG 로 산출 | 같은 컷을 신규 NEG/TP 에 적용 → 컷 자체는 안 변함 (0.624) 단 TP 들이 anchor 와 cos 미달 |

**핵심 — bge-m3 의 분포컷이 일반화 못 하는 이유**:
- TP 신규 8건 (예: "OS 페이지 교체 알고리즘", "그래프 신경망 message passing", "분산 합의 Paxos") 이 anchor (예: "Redis LRU eviction") 와 cos 0.624 미만
- 즉 *같은 추상 충돌의 다른 표면 표현* 을 cos 가 못 잡음
- HARD_NEG FP 는 0/6 으로 완벽한데 TP 도 같이 못 잡음 = 컷이 *전반적으로 너무 멀음*

**Phase 2.5 "Y 패러다임 한계" 가 옳았다**:
- dense sentence embedding 은 표면 어휘 일치 위주로 매칭
- abstraction-level 매칭 (다른 도메인 같은 충돌) 능력 부재
- 모델 크기 키워도 (e5-large 1024d), 패러다임 바꿔도 (bge-m3) 한계 동일
- Phase 2.6 "중" 은 매우 좁은 in-sample 표본이 만든 환상

---

## Phase 2.5 / 2.6 / 2.7 의 진폭

| Phase | 자동 판정 | 실제 결론 |
|-------|----------|----------|
| 2.5 | 3/3 모델 raw FAIL → 대 | 옳음 (raw cos 분리 불가) |
| 2.6 | bge-m3 + P2 PASS in-sample → 중 | **잘못** (in-sample 표본 작음) |
| 2.7 | bge-m3 OOS 20% recall → 대 | **확정** (OOS 일반화 실패) |

→ Phase 2.6 의 "중" 단정이 측정 설계 결함이었음. Phase 2.7 이 보정.

---

## Phase 3 = 대

**작업 범위 (별도 지시서 대기)**:
- **임베딩 키 폐기**. trigger-vec / interpretation-vec / hybrid / 모델교체 + 분포컷 — 모두 폐기
- **비-임베딩 키 신설**:
  - LLM 으로 충돌 유형 라벨 추출 (예: "eviction pattern", "global-attention parallelism", "linearizable consensus")
  - 라벨을 sparse 키 또는 구조화 인덱스로 저장
  - recall 시 *라벨 매칭* (cosine 의존 0)

---

## 스크립트 패치 요약

```diff
scripts/conflict_recall_fp_eval.py:
+ PROBES_V2 = [...]   # 4-tuple (kind, text, anchor_idx, split)
+                     # 기존 PROBES 무변경 (Phase 2.5/2.6 재현성 보존)

scripts/conflict_recall_oos_eval.py (신규):
+ split 라우팅 (calib NEG → 95th pct cut / eval NEG+TP → 성능 측정)
+ MARGIN=0.05, percentile=95, score 형태 무변경 (remedy_eval 와 동일)
```

anchor 6 무변경 / MARGIN / percentile / 판정 형태 모두 무변경. **split 라우팅만 추가**.

---

## 영구 산출

| 파일 | 경로 |
|------|------|
| 데이터셋 확장 | `scripts/conflict_recall_fp_eval.py` (PROBES_V2 추가) |
| OOS 측정 스크립트 (신규) | `scripts/conflict_recall_oos_eval.py` |
| raw JSON bge-m3 | `docs/03-analysis/conflict_recall_oos_raw.BAAI_bge-m3.json` |
| raw JSON e5-small | `docs/03-analysis/conflict_recall_oos_raw.default.json` |
| 측정 지시서 | `docs/05-measure/claude_code_측정지시서_phase2.7_oos_generalization.md` |
| 본 리포트 | `docs/05-measure/phase2.7_oos_report.md` |

---

## 금지 §2 위반 확인

| 금지 | 위반 |
|------|:----:|
| anchor 6 변경, MARGIN/percentile/score 형태 변경 | ✓ 없음 |
| calib/eval probe 중복 (leakage) | ✓ 없음 (split 명시) |
| EmbeddingBridge default / recall_conflict 코드 수정 | ✓ 없음 |
| Phase 3 측정 전 단정 | ✓ 없음 (출력만) |
| xfail 2건 GREEN 전환 | ✓ 없음 |
| HARD_NEG "쉽게" 작성 | ✓ 사용자 검수 통과 |
| LLM 리뷰로 측정 대체 | ✓ 수치만 |

---

## 한 줄 요약

> **bge-m3 분포컷 OOS 재현율 20% — 일반화 실패. Phase 2.6 "중" 반증.
> Phase 3 = 대 (비-임베딩 키 / LLM 라벨 추출) 최종 확정.**
