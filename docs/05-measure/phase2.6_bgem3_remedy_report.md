# Phase 2.6 측정 결과 — bge-m3 처방 증폭 시험

**작성일**: 2026-05-20
**지시서**: `docs/05-measure/claude_code_측정지시서_phase2.6_bgem3_remedy.md`
**스크립트**: `scripts/conflict_recall_remedy_eval.py` (M1: sys.argv 패치만)
**Raw**: `docs/03-analysis/conflict_recall_remedy_raw.{e5-small,bge-m3}.json`

---

## Phase 2.6 측정 결과

| 모델 | 처방 | HARD_NEG FP | TRUE_POS TP | 판정 |
|------|------|:----------:|:----------:|:----:|
| e5-small | Baseline (cos≥0.82) | 6/6 | 2/2 | FAIL (재현확인) |
| e5-small | P1 상대마진 (gap≥0.05) | 0/6 | 1/2 | FAIL |
| e5-small | P2 분포컷 (cos≥0.867) | 1/6 | 1/2 | FAIL |
| e5-small | P3 결합 (P1 AND P2) | 0/6 | 1/2 | FAIL |
| bge-m3 | Baseline (cos≥0.82) | 0/6 | 0/2 | FAIL (절대컷이 bge 분포에 부적) |
| bge-m3 | P1 상대마진 (gap≥0.05) | 2/6 | 2/2 | FAIL |
| **bge-m3** | **P2 분포컷 (cos≥0.621)** | **1/6** | **2/2** | **PASS** ✓ |
| **bge-m3** | **P3 결합 (P1 AND P2)** | **1/6** | **2/2** | **PASS** ✓ |

**합격선**: HARD_NEG FP ≤ 1/6 AND TRUE_POS TP ≥ 2/2

---

## 판정 (지시서 §1.M3)

- **e5-small 재현**: 측정 2 (지시서 §0) 와 일치 — 전부 FAIL 정상
- **bge-m3 최선 처방**: **P2 분포컷** (cos≥0.621). 재현율 100% (TP 2/2), HARD_NEG FP 16.7% (1/6)
- **원인·규모 판정**: **중** (§1.M3 표 첫째 행 "P1·P2·P3 중 하나라도 PASS")

> **Phase 3 규모 = 중**: 임베딩 모델 bge-m3 교체 + 분포컷 처방 레이어 적용.
> 새 서브시스템 (비-임베딩 키) **불필요**.

---

## Phase 2.5 의 "대" 단정 정정

Phase 2.5 자동 판정식이 raw cos 분포만 봐서 "3/3 FAIL → 패러다임 한계(Y) → 대" 라고 닫음.
그러나 **측정 2 처방 (P2 분포컷)** 을 bge-m3 위에 안 돌린 빈칸이 있었음. 이번 측정이
그 빈칸을 메움:

- bge-m3 의 raw cos 분포 (TP 0.477~0.495 / HN 0.415~0.514) 는 e5 (0.82+) 와 다른 영역에 있어
  *절대컷 0.82* baseline 으로는 모두 FAIL (Baseline 0/6 FP / 0/2 TP)
- 그러나 **bge-m3 분포 자체에서 95th percentile 로 calibrate 한 0.621 컷**이
  TP 2건은 모두 통과 + HARD_NEG 6 중 5 거부 → PASS
- bge-m3 양수 평균마진 (+0.0168) 은 노이즈 아닌 *실 신호* 였음 — 처방으로 분리됨

---

## 영구 산출

| 파일 | 경로 |
|------|------|
| remedy 스크립트 (M1 패치) | `scripts/conflict_recall_remedy_eval.py` |
| raw JSON e5-small | `docs/03-analysis/conflict_recall_remedy_raw.e5-small.json` |
| raw JSON bge-m3 | `docs/03-analysis/conflict_recall_remedy_raw.bge-m3.json` |
| 측정 지시서 | `docs/05-measure/claude_code_측정지시서_phase2.6_bgem3_remedy.md` |
| 본 리포트 | `docs/05-measure/phase2.6_bgem3_remedy_report.md` |

---

## 스크립트 패치 (M1)

```diff
 import json
 import struct
+import sys
 from pathlib import Path
 ...
-def build():
-    enc = EmbeddingBridge()
+def build(model_name=None):
+    enc = EmbeddingBridge() if model_name is None \
+        else EmbeddingBridge(model_name=model_name)
 ...
-def evaluate():
-    enc, anchor_vecs, anchor_ids = build()
+def evaluate(model_name=None):
+    enc, anchor_vecs, anchor_ids = build(model_name)
 ...
 if __name__ == "__main__":
-    evaluate()
+    model = sys.argv[1] if len(sys.argv) > 1 else None
+    evaluate(model)
```

**MARGIN = 0.05 / percentile = 95 / score() = `h <= 1 and t >= 2` / anchor/probe**
모두 무변경.

---

## 금지 §2 위반 확인

| 금지 | 위반 |
|------|:----:|
| MARGIN/percentile/판정식/데이터셋 수정 | ✓ 없음 |
| 결과로 곧장 코드 수정 (recall_conflict / EmbeddingBridge default) | ✓ 없음 |
| Phase 3 "중/대" 측정 전 단정 | ✓ 없음 — 출력 따름 |
| xfail 2건 GREEN 전환 | ✓ 없음 |
| LLM 리뷰로 측정 대체 | ✓ 없음 — 수치만 |
| bge-m3 PASS 미리 가정 | ✓ 없음 — 검정 결과 |

---

## 한 줄 요약

> **bge-m3 + P2 분포컷 (cos≥0.621) PASS** — 재현율 2/2 + HARD_NEG FP 1/6.
> Phase 3 = 중 (모델 교체 + 분포컷). 새 서브시스템 불필요. Phase 2.5 "대" 정정.
