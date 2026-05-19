"""
Phase 2.7 — out-of-sample 일반화 검정.

지시서: docs/05-measure/claude_code_측정지시서_phase2.7_oos_generalization.md

Phase 2.6 PASS 의 두 약점 보정:
  A. 표본 작음 (TP=2, FP=1/6 합격선 경계)
  B. 0.621 컷 in-sample 과적합 (같은 NEG 의 95th pct 로 산출)

프로토콜:
  1. PROBES_V2 의 split=calib (NEG 절반) 만 cos → 95th pct 로 컷 산출
  2. 컷 고정
  3. split=eval (NEG 나머지 절반 + 모든 TRUE_POS) 만 적용해 성능 측정

판정 (지시서 §1.M3):
  | eval 재현율 ≥80% AND FP ≤20% AND 격차 ≤15%p | 중   |
  | 재현율/FP 통과 단 격차 >15%p                  | 중-대 |
  | 재현율 <80% OR FP >20%                         | 대   |

remedy_eval.py 의 MARGIN=0.05 / percentile=95 / score 형태 무변경.
split 라우팅만 추가.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
import tempfile

import numpy as np
import torch
import torch.nn.functional as F

from htp.knowledge.embedding import EmbeddingBridge
from scripts.conflict_recall_fp_eval import ANCHORS, PROBES_V2


MARGIN = 0.05            # remedy_eval 과 동일 (무변경)
PERCENTILE = 95          # remedy_eval 과 동일 (무변경)


def _cos(a, b):
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def build(model_name=None):
    enc = EmbeddingBridge() if model_name is None \
        else EmbeddingBridge(model_name=model_name)
    anchor_vecs = [torch.tensor(enc.encode(t), dtype=torch.float32)
                    for t, _ in ANCHORS]
    return enc, anchor_vecs


def rank(enc, anchor_vecs, probe_text):
    """probe → 전 anchor 와 cos, 내림차순 [(idx, cos), ...]."""
    qv = torch.tensor(enc.encode_query(probe_text), dtype=torch.float32)
    sims = [(i, _cos(qv, av)) for i, av in enumerate(anchor_vecs)]
    sims.sort(key=lambda x: -x[1])
    return sims


def evaluate(model_name=None):
    enc, anchor_vecs = build(model_name)
    print(f"# encoder: {enc._model_name} dim={enc.dim}")

    # 1. calib NEG → cos → 95th pct 컷
    calib_neg_best = []
    for kind, ptext, _, split in PROBES_V2:
        if split != "calib" or not kind.endswith("NEG"):
            continue
        sims = rank(enc, anchor_vecs, ptext)
        calib_neg_best.append(sims[0][1])
    calib_neg_best = np.array(calib_neg_best)
    dist_cut = float(np.percentile(calib_neg_best, PERCENTILE))
    print(f"# calib NEG ({len(calib_neg_best)}건) best-cos: "
          f"min={calib_neg_best.min():.3f} mean={calib_neg_best.mean():.3f} "
          f"max={calib_neg_best.max():.3f}")
    print(f"# → dist_cut (calib {PERCENTILE}th pct) = {dist_cut:.4f}\n")

    # 2. calib 자체 성능 (참고)
    calib_rows = [r for r in PROBES_V2 if r[3] == "calib"]
    calib_results = []
    for kind, ptext, anchor_idx, _ in calib_rows:
        sims = rank(enc, anchor_vecs, ptext)
        (i1, c1), (i2, c2) = sims[0], sims[1]
        gap = c1 - c2
        hit = (gap >= MARGIN) and (c1 >= dist_cut)   # P3 결합 (Phase 2.6 PASS 처방)
        calib_results.append({"kind": kind, "split": "calib", "probe": ptext,
                              "c1": round(c1, 4), "gap": round(gap, 4),
                              "hit": hit})

    # 3. eval 성능 (out-of-sample)
    eval_rows = [r for r in PROBES_V2 if r[3] == "eval"]
    eval_results = []
    for kind, ptext, anchor_idx, _ in eval_rows:
        sims = rank(enc, anchor_vecs, ptext)
        (i1, c1), (i2, c2) = sims[0], sims[1]
        gap = c1 - c2
        hit = (gap >= MARGIN) and (c1 >= dist_cut)   # 같은 P3 처방, eval 데이터에만 적용
        eval_results.append({"kind": kind, "split": "eval", "probe": ptext,
                             "c1": round(c1, 4), "gap": round(gap, 4),
                             "hit": hit, "intended": anchor_idx,
                             "best_anchor": i1})

    def metrics(rows, set_name):
        tp_total = sum(1 for r in rows if r["kind"] == "TRUE_POS")
        tp_hit = sum(1 for r in rows if r["kind"] == "TRUE_POS" and r["hit"])
        hn_total = sum(1 for r in rows if r["kind"] == "HARD_NEG")
        hn_fp = sum(1 for r in rows if r["kind"] == "HARD_NEG" and r["hit"])
        mid_total = sum(1 for r in rows if r["kind"] == "MID_NEG")
        mid_fp = sum(1 for r in rows if r["kind"] == "MID_NEG" and r["hit"])
        easy_total = sum(1 for r in rows if r["kind"] == "EASY_NEG")
        easy_fp = sum(1 for r in rows if r["kind"] == "EASY_NEG" and r["hit"])
        return {
            "set": set_name,
            "tp_recall":   (tp_hit / tp_total) if tp_total else None,
            "tp":          f"{tp_hit}/{tp_total}",
            "hard_fp_rate":(hn_fp / hn_total) if hn_total else 0.0,
            "hard_fp":     f"{hn_fp}/{hn_total}",
            "mid_fp":      f"{mid_fp}/{mid_total}",
            "easy_fp":     f"{easy_fp}/{easy_total}",
        }

    m_calib = metrics(calib_results, "calib")
    m_eval  = metrics(eval_results, "eval")

    # 4. 판정 (지시서 §1.M3)
    eval_recall = m_eval["tp_recall"] or 0.0
    eval_fp_rate = m_eval["hard_fp_rate"]
    calib_fp_rate = m_calib["hard_fp_rate"]
    # 일반화 격차 — calib FP rate 와 eval FP rate 차이 (재현율은 calib 에 TP 없음)
    fp_gap = eval_fp_rate - calib_fp_rate

    if eval_recall >= 0.80 and eval_fp_rate <= 0.20 and abs(fp_gap) <= 0.15:
        verdict = "중"
        explain = "0.621류 컷이 일반화됨. bge-m3 교체 + 고정컷 확정."
    elif eval_recall >= 0.80 and eval_fp_rate <= 0.20:
        verdict = "중-대"
        explain = "신호 있으나 컷 데이터의존. 운영 중 주기적 재calibrate."
    else:
        verdict = "대"
        explain = "in-sample PASS 가 과적합이었음. 비-임베딩 키 신설."

    print(f"## calib (참고 — 컷 산출용)")
    print(f"  TP recall: {m_calib['tp']} ({m_calib['tp_recall']})  "
          f"HARD_NEG FP: {m_calib['hard_fp']} ({m_calib['hard_fp_rate']:.0%})  "
          f"MID FP: {m_calib['mid_fp']}  EASY FP: {m_calib['easy_fp']}")
    print(f"\n## eval (out-of-sample, 합격 기준)")
    print(f"  TP recall: {m_eval['tp']} ({eval_recall:.0%})  "
          f"HARD_NEG FP: {m_eval['hard_fp']} ({eval_fp_rate:.0%})  "
          f"MID FP: {m_eval['mid_fp']}  EASY FP: {m_eval['easy_fp']}")
    print(f"\n## 일반화 격차 (eval FP - calib FP) = {fp_gap:+.3f}")
    print(f"\n▶ 판정: {verdict}")
    print(f"  근거: eval 재현율 {eval_recall:.0%} (>=80% ?), "
          f"FP rate {eval_fp_rate:.0%} (<=20% ?), "
          f"격차 {abs(fp_gap):.0%} (<=15%p ?)")
    print(f"  → {explain}")

    out = {
        "model": enc._model_name, "dim": enc.dim,
        "dist_cut": dist_cut, "MARGIN": MARGIN, "PERCENTILE": PERCENTILE,
        "calib": m_calib, "eval": m_eval,
        "fp_gap": fp_gap, "verdict": verdict, "explain": explain,
        "calib_rows": calib_results, "eval_rows": eval_results,
    }
    Path("docs/03-analysis").mkdir(parents=True, exist_ok=True)
    return out


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else None
    result = evaluate(model)
    # raw JSON 출력
    name = (model or "default").replace("/", "_")
    p = Path(f"docs/03-analysis/conflict_recall_oos_raw.{name}.json")
    with p.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n# raw → {p}")
