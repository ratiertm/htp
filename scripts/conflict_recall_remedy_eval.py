"""
처방 비교 — relative-margin vs distribution-cut.

이전 측정 결과:
  - 현재 시스템 (절대 L2 < 0.6): 거짓 양성 12/12 (100%)
  - query-prefix 적용 시:        거짓 양성 10/12 (83%) — HARD_NEG 6/6 여전 FP
  - cos↔L2 완전 단조 (corr -0.997) → 지표 불일치는 원인 아님
  - 진짜 문제: e5-small 에서 TRUE_POS cos(0.905~0.945) 와
               HARD_NEG cos(0.901~0.929) 분포가 겹침 → 절대컷 분리 불가

두 처방을 같은 anchor/probe 에 적용:
  P1 RELATIVE MARGIN : top1 - top2 cos gap >= m 일 때만 HIT
                       ("압도적 단일 후보" 일 때만 신뢰)
  P2 DISTRIBUTION CUT: 무관(NEG) cos 분포의 상위 percentile 로 절대컷 재산출
                       (데이터로 threshold 교정)
  P3 둘의 결합 (margin AND dist-cut)

합격선:
  - HARD_NEG FP <= 1/6  (경계 거부 가능)
  - TRUE_POS TP >= 2/2  (재현율 유지)
  둘 다 만족 못 하면 그 처방 탈락. 전부 탈락 시 결론:
  "e5-small + trigger-key recall 은 이 task 에 부적합" (= 설계 재검토 트리거).

query-prefix 는 이전 측정에서 개선 확인됨 → 본 측정은 prefix ON 고정.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path
import tempfile

import numpy as np
import torch
import torch.nn.functional as F

from htp.knowledge.embedding import EmbeddingBridge
from htp.memory.memory_system import MemorySystem

# 이전 스크립트와 동일 데이터셋 재사용
from scripts.conflict_recall_fp_eval import ANCHORS, PROBES


def _vecs(enc, blob_q):
    qv = torch.tensor(blob_q, dtype=torch.float32)
    return qv


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def build():
    enc = EmbeddingBridge()
    tmp = tempfile.mkdtemp(prefix="htp-rx-")
    mem = MemorySystem(memory_dir=Path(tmp) / "mem")
    anchor_vecs = []
    anchor_ids = []
    for trig, interp in ANCHORS:
        tv = torch.tensor(enc.encode(trig), dtype=torch.float32)
        ep_id = mem.save_conflict(trigger_vec=tv, new_text=trig,
                                  partner_texts=["d"], interpretation=interp,
                                  conflict_score=0.15)
        anchor_vecs.append(tv)
        anchor_ids.append(ep_id)
    return enc, anchor_vecs, anchor_ids


def rank(enc, anchor_vecs, probe_text):
    """probe → 전 anchor 와 cos, 내림차순 [(idx, cos), ...]."""
    qv = torch.tensor(enc.encode_query(probe_text), dtype=torch.float32)  # prefix ON
    sims = [(i, _cos(qv, av)) for i, av in enumerate(anchor_vecs)]
    sims.sort(key=lambda x: -x[1])
    return sims


def calibrate_dist_cut(enc, anchor_vecs):
    """NEG probe 들의 best-cos 분포 → 그 분포 95th percentile 를 절대컷으로.
    (무관 쌍이 이보다 가까우면 거의 안 나온다는 통계적 컷)"""
    neg_best = []
    for kind, ptext, _ in PROBES:
        if not kind.endswith("NEG"):
            continue
        sims = rank(enc, anchor_vecs, ptext)
        neg_best.append(sims[0][1])
    neg_best = np.array(neg_best)
    # NEG 가 이 값을 넘는 일이 5% 뿐이도록 → 95th pct
    cut = float(np.percentile(neg_best, 95))
    return cut, neg_best


def evaluate():
    enc, anchor_vecs, anchor_ids = build()

    dist_cut, neg_best = calibrate_dist_cut(enc, anchor_vecs)
    print(f"# encoder: {enc._model_name} dim={enc.dim}")
    print(f"# NEG best-cos 분포: min={neg_best.min():.3f} "
          f"mean={neg_best.mean():.3f} max={neg_best.max():.3f}")
    print(f"# → 분포컷(P2) = NEG 95th pct = {dist_cut:.4f}")
    print(f"#   (현재 시스템 절대컷 ≈ cos 0.82 와 비교)\n")

    MARGIN = 0.05  # P1: top1-top2 cos gap 최소치 (탐색 가능)

    results = []
    for kind, ptext, anchor_idx in PROBES:
        sims = rank(enc, anchor_vecs, ptext)
        (i1, c1), (i2, c2) = sims[0], sims[1]
        gap = c1 - c2

        p1_hit = gap >= MARGIN                       # relative margin
        p2_hit = c1 >= dist_cut                       # distribution cut
        p3_hit = p1_hit and p2_hit                    # 결합
        baseline_hit = c1 >= 0.82                     # 현재 시스템 근사 (L2<0.6)

        results.append({
            "kind": kind, "probe": ptext,
            "best_anchor": i1, "intended": anchor_idx,
            "c1": round(c1, 4), "c2": round(c2, 4), "gap": round(gap, 4),
            "baseline_hit": baseline_hit,
            "P1_margin_hit": p1_hit,
            "P2_distcut_hit": p2_hit,
            "P3_both_hit": p3_hit,
        })

    def score(key):
        hard_fp = sum(1 for r in results
                      if r["kind"] == "HARD_NEG" and r[key])
        all_neg_fp = sum(1 for r in results
                         if r["kind"].endswith("NEG") and r[key])
        tp = sum(1 for r in results
                 if r["kind"] == "TRUE_POS" and r[key])
        return hard_fp, all_neg_fp, tp

    print(f"{'처방':32} {'HARD_NEG FP':>12} {'전체NEG FP':>11} "
          f"{'TRUE_POS TP':>12} {'판정':>8}")
    print("-" * 80)
    table = []
    for key, name in [("baseline_hit",   "Baseline (현재: 절대 cos≥0.82)"),
                      ("P1_margin_hit",  f"P1 상대마진 (gap≥{MARGIN})"),
                      ("P2_distcut_hit", f"P2 분포컷 (cos≥{dist_cut:.3f})"),
                      ("P3_both_hit",    "P3 결합 (P1 AND P2)")]:
        h, a, t = score(key)
        # 합격: HARD_NEG FP<=1 AND TRUE_POS TP>=2
        verdict = "PASS" if (h <= 1 and t >= 2) else "FAIL"
        print(f"{name:32} {h:>7}/6     {a:>6}/12    "
              f"{t:>7}/2      {verdict:>8}")
        table.append({"name": name, "hard_fp": h, "all_neg_fp": a,
                       "tp": t, "verdict": verdict})

    print("\n## 케이스별 상세 (cos1=best, cos2=2nd, gap=분리도)")
    for kind in ("TRUE_POS", "HARD_NEG", "MID_NEG", "EASY_NEG"):
        print(f"\n  [{kind}]")
        for r in results:
            if r["kind"] != kind:
                continue
            tag = []
            if r["kind"] == "TRUE_POS":
                tag.append("P1✓" if r["P1_margin_hit"] else "P1✗")
                tag.append("P2✓" if r["P2_distcut_hit"] else "P2✗")
            else:
                tag.append("P1" + ("⚠FP" if r["P1_margin_hit"] else "ok "))
                tag.append("P2" + ("⚠FP" if r["P2_distcut_hit"] else "ok "))
            print(f"    {' '.join(tag)}  {r['probe'][:40]:40} "
                  f"c1={r['c1']:.3f} c2={r['c2']:.3f} gap={r['gap']:+.3f}")

    out = {"dist_cut": dist_cut, "margin": MARGIN,
           "neg_best_stats": {"min": float(neg_best.min()),
                              "mean": float(neg_best.mean()),
                              "max": float(neg_best.max())},
           "summary": table, "rows": results}
    Path("docs/03-analysis").mkdir(parents=True, exist_ok=True)
    with open("docs/03-analysis/conflict_recall_remedy_raw.json",
              "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)
    print("\n# raw → docs/03-analysis/conflict_recall_remedy_raw.json")
    return table


if __name__ == "__main__":
    evaluate()
