"""
결정적 측정 — Phase 3 규모를 가르는 단일 질문.

질문: 문제가 키의 *내용*(trigger vs interpretation)인가,
      키의 *공간*(e5-small dense 384d)인가?

측정 2 에서 trigger 임베딩은 TRUE_POS/HARD_NEG cos 분포가 겹쳐 FAIL.
이번엔 동일 anchor/probe 에 대해 **interpretation 텍스트를 e5 로 임베딩**,
같은 분포 분리 테스트를 한다.

  - 분리됨  → 내용 문제. (가) interpretation-key 가 처방. Phase 3 = 키 교체(소).
  - 안 됨   → 공간 문제. (가)(나) 폐기. (다) 구조화 키만 생존. Phase 3 = 라벨추출(대).

판정 기준 (측정 2 와 동일한 자):
  분리 성공 = TRUE_POS 최소 cos > HARD_NEG 최대 cos (분포 미겹침)
  부분      = 평균은 분리되나 꼬리 겹침 (마진/컷으로 구제 가능)
  실패      = 분포 겹침 (측정 2 와 동일 — 공간 문제 확정)

실행: PYTHONPATH=. python scripts/conflict_recall_keyspace_eval.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from htp.knowledge.embedding import EmbeddingBridge
from scripts.conflict_recall_fp_eval import ANCHORS, PROBES


def _cos(a, b):
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


# 각 anchor 의 "정답 interpretation" (ANCHORS 의 해석 본문).
# TRUE_POS probe 는 그 anchor 와 같은 충돌의 다른 표현이므로,
# probe 의 *예상 해석* 도 같은 의미여야 한다.
# 여기선 probe 자체를 interpretation-공간으로 인코딩할 수 없으므로
# (probe 는 입력 텍스트지 해석이 아님), 두 시나리오를 분리 측정한다:
#
#  S1. anchor.interp 끼리의 분리도 (저장 키들이 서로 구분되는가)
#  S2. probe(trigger) vs anchor.interp 교차 (실제 recall 상황:
#      입력 trigger 로 저장된 interpretation 키를 검색)


def main():
    # Phase 2.5 (2026-05-20): 모델 인자 파라미터화 — 지시서 §1.M2.
    # anchor/probe/판정식 로직은 무수정. 모델만 sys.argv[1] 로 받음.
    model_name = sys.argv[1] if len(sys.argv) > 1 else EmbeddingBridge.DEFAULT_MODEL
    enc = EmbeddingBridge(model_name=model_name)
    print(f"# encoder: {enc._model_name} dim={enc.dim}")

    interp_vecs = [
        torch.tensor(enc.encode(interp), dtype=torch.float32)
        for _, interp in ANCHORS
    ]
    trig_vecs = [
        torch.tensor(enc.encode(trig), dtype=torch.float32)
        for trig, _ in ANCHORS
    ]

    # ── S1: interpretation 키들끼리 분리되는가 ──
    print(f"\n{'='*64}\n## S1 — 저장 interpretation 키 상호 분리도\n{'='*64}")
    n = len(ANCHORS)
    offdiag = []
    for i in range(n):
        for j in range(n):
            if i != j:
                offdiag.append(_cos(interp_vecs[i], interp_vecs[j]))
    offdiag = np.array(offdiag)
    print(f"  서로 다른 충돌 해석 간 cos: "
          f"min={offdiag.min():.3f} mean={offdiag.mean():.3f} "
          f"max={offdiag.max():.3f}")
    print(f"  (낮을수록 좋음 — 다른 충돌이 키 공간에서 멀리 떨어짐)")

    # ── S2: 실제 recall — probe(trigger) 로 interpretation 키 검색 ──
    print(f"\n{'='*64}\n## S2 — probe(trigger) → interpretation 키 검색\n"
          f"{'='*64}")
    print("  실제 recall 상황: 입력 trigger 임베딩으로 저장된 "
          "interpretation 키를 검색.\n")

    tp_best, hard_best, mid_best, easy_best = [], [], [], []
    rows = []
    for kind, ptext, anchor_idx in PROBES:
        qv = torch.tensor(enc.encode_query(ptext), dtype=torch.float32)
        sims = sorted(
            ((k, _cos(qv, iv)) for k, iv in enumerate(interp_vecs)),
            key=lambda x: -x[1])
        best_k, best_c = sims[0]
        intended_hit = (best_k == anchor_idx)
        rows.append({"kind": kind, "probe": ptext, "best_c": round(best_c, 4),
                     "best_is_intended": intended_hit})
        if kind == "TRUE_POS":
            tp_best.append(best_c)
        elif kind == "HARD_NEG":
            hard_best.append(best_c)
        elif kind == "MID_NEG":
            mid_best.append(best_c)
        elif kind == "EASY_NEG":
            easy_best.append(best_c)

    tp = np.array(tp_best)
    hn = np.array(hard_best)
    print(f"  TRUE_POS best-cos : min={tp.min():.3f} "
          f"mean={tp.mean():.3f} max={tp.max():.3f}")
    print(f"  HARD_NEG best-cos : min={hn.min():.3f} "
          f"mean={hn.mean():.3f} max={hn.max():.3f}")
    if mid_best:
        mn = np.array(mid_best)
        print(f"  MID_NEG  best-cos : mean={mn.mean():.3f}")
    if easy_best:
        en = np.array(easy_best)
        print(f"  EASY_NEG best-cos : mean={en.mean():.3f}")

    # ── 판정 ──
    print(f"\n{'='*64}\n## 판정\n{'='*64}")
    separation = tp.min() - hn.max()  # >0 이면 완전 분리
    margin_mean = tp.mean() - hn.mean()
    print(f"  완전분리 지표  = TP.min - HN.max = "
          f"{tp.min():.3f} - {hn.max():.3f} = {separation:+.4f}")
    print(f"  평균마진       = TP.mean - HN.mean = {margin_mean:+.4f}")

    # 측정 2 와 비교 (trigger 임베딩 결과)
    print(f"\n  [측정 2 (trigger-key) 대조]")
    print(f"    당시: TP cos 0.865~0.890 / HN cos 0.848~0.869 → 겹침(FAIL)")

    if separation > 0:
        verdict = "분리 성공 — 내용 문제"
        phase3 = ("Phase 3 = 소규모. interpretation-key 로 교체.\n"
                  "    recall_conflict 의 검색 키를 trigger_vec →\n"
                  "    interpretation 임베딩으로. 키 한 군데 변경 + xfail GREEN 전환.")
    elif margin_mean > 0.02:
        verdict = "부분 분리 — 내용이 신호 일부 보유, 단 꼬리 겹침"
        phase3 = ("Phase 3 = 중규모. interpretation-key + 상대마진/분포컷\n"
                  "    조합. 측정 2 의 P1/P2 를 interpretation 공간에서 재시험.")
    else:
        verdict = "분리 실패 — 공간 문제 확정 (e5-small 분해능 한계)"
        phase3 = ("Phase 3 = 대규모. (가)(나) 폐기. (다) 구조화 키만 생존.\n"
                  "    LLM 으로 충돌 유형 라벨 추출 → 라벨 임베딩/스파스 키.\n"
                  "    Claude Code 가 말한 '키 한 줄 교체' 는 여기서 반증됨.")
    print(f"\n  ▶ {verdict}")
    print(f"  ▶ {phase3}")

    out = {"S1_interp_offdiag": {"min": float(offdiag.min()),
                                 "mean": float(offdiag.mean()),
                                 "max": float(offdiag.max())},
           "S2": {"tp": tp.tolist(), "hard": hn.tolist(),
                  "separation": float(separation),
                  "margin_mean": float(margin_mean)},
           "verdict": verdict, "rows": rows}
    Path("docs/03-analysis").mkdir(parents=True, exist_ok=True)
    with open("docs/03-analysis/conflict_recall_keyspace_raw.json",
              "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n# raw → docs/03-analysis/conflict_recall_keyspace_raw.json")


if __name__ == "__main__":
    main()
