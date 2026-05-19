"""
거짓 양성(False Positive) recall 측정 — htp-conflict-memory 외부 리뷰 후속.

목적 (외부 리뷰 §5 + 코드 리딩 발견 2건 검증):
  1. threshold=0.6 에서 *무관·경계* 충돌이 잘못 recall HIT 하는 비율 측정
  2. [발견 A] 후보 선택(cosine) vs 채택 컷(L2 norm) 지표 불일치 정량화
  3. [발견 B] encode() passage-prefix vs encode_query() query-prefix
     누락이 recall 품질에 주는 영향 A/B 분리

핵심 설계 (이전 합의):
  - "완전 무관"(eviction ↔ 윤리) 은 너무 쉬움 → threshold 0.9 여도 통과.
    진짜 위험은 "같은 도메인·다른 메커니즘" 경계 쌍.
  - 따라서 probe 를 3 난이도로 구성: HARD_NEG(경계) / MID_NEG / EASY_NEG.

실행:
  source .venv/bin/activate
  python scripts/conflict_recall_fp_eval.py
"""
from __future__ import annotations

import json
import struct
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from htp.knowledge.embedding import EmbeddingBridge
from htp.memory.memory_system import MemorySystem


# ── 1. 저장될 "기준 충돌 해석" (anchor) ────────────────
# 실제 시스템이 Episode 로 들고 있을 법한 충돌 해석 6건.
# 각 anchor 는 (트리거 텍스트, 해석 본문) — 트리거가 recall key.
ANCHORS = [
    ("Redis LRU 캐시 eviction 전략 메모리 축출",
     "categorical conflict: 결정론적 인프라 캐시 정책 vs 확률적 생물 기억 — "
     "scope(local) x mechanism(deterministic) 2축 분해"),
    ("Transformer self attention 전역 가중 메커니즘",
     "scope(global vs local) x temporal(parallel vs serial) 2-axis. "
     "attention 은 전역 동시 가중, 생물 주의는 국소 순차"),
    ("거래소 order book 매수매도 호가 유동성",
     "complementarity: 시장 미시구조의 가격 발견 vs 정보 집계 — "
     "두 layer 가 trade-off 아니라 보완 관계"),
    ("PostgreSQL WAL replication 강한 일관성",
     "strong vs eventual consistency 는 CAP 축의 양 끝 — "
     "partition tolerance 고정 시 latency-consistency dimension trade-off"),
    ("도파민 reward prediction error 시냅스 강화",
     "temporal difference learning 과 동형 — RL 의 TD error 가 "
     "생물 신경조절 메커니즘과 mechanism mapping"),
    ("CNN 합성곱 지역 수용장 가중치 공유",
     "지역 수용장 = 시각피질 simple cell 의 inductive bias 와 구조 동형. "
     "translation invariance 가 양쪽 공통 framing"),
]

# ── 2. Probe (recall 쿼리) — 3 난이도 ──────────────────
# HARD_NEG: anchor 와 *같은 도메인·다른 메커니즘*. threshold 진짜 시험대.
#           정답: recall MISS 여야 함 (다른 충돌이므로). HIT 하면 FALSE POSITIVE.
# MID_NEG : 인접 도메인. MISS 기대.
# EASY_NEG: 완전 무관. MISS 당연 (sanity check — 여기서 HIT 면 심각).
# TRUE_POS: anchor 와 의미상 같은 충돌의 다른 표현. HIT 기대 (재현율 sanity).
PROBES = [
    # --- HARD_NEG : 같은 도메인, 다른 메커니즘 (FP 주 위험원) ---
    ("HARD_NEG", "Redis cluster 샤딩 hash slot 재분배 리밸런싱",          0),  # vs anchor0: 둘다 Redis·인프라, eviction≠샤딩
    ("HARD_NEG", "Transformer KV cache 메모리 추론 최적화",               1),  # vs anchor1: 둘다 Transformer, attention≠KV cache
    ("HARD_NEG", "거래소 settlement T+2 청산 결제 리스크",                2),  # vs anchor2: 둘다 거래소, order book≠settlement
    ("HARD_NEG", "PostgreSQL VACUUM 디스크 공간 회수 autovacuum",         3),  # vs anchor3: 둘다 PG, replication≠vacuum
    ("HARD_NEG", "GABA 억제성 시냅스 lateral inhibition",                 4),  # vs anchor4: 둘다 신경, dopamine≠GABA
    ("HARD_NEG", "RNN 순환 게이트 LSTM forget gate 시계열",               5),  # vs anchor5: 둘다 신경망, CNN≠RNN
    # --- MID_NEG : 인접 도메인 ---
    ("MID_NEG",  "Kafka consumer group partition rebalance",             0),  # 인프라지만 Redis 아님
    ("MID_NEG",  "RLHF 인간 피드백 보상 모델 정렬",                       1),  # AI지만 attention 아님
    ("MID_NEG",  "옵션 implied volatility 변동성 스마일",                 2),  # 금융이지만 order book 아님
    ("MID_NEG",  "Raft 합의 알고리즘 leader election",                    3),  # 분산이지만 PG WAL 아님
    # --- EASY_NEG : 완전 무관 (sanity) ---
    ("EASY_NEG", "중세 고딕 성당 부벽 구조 하중 분산",                    0),
    ("EASY_NEG", "김치 발효 유산균 pH 변화 숙성",                         3),
    # --- TRUE_POS : 같은 충돌의 다른 표현 (재현율 sanity) ---
    ("TRUE_POS", "메모리 캐시에서 오래된 항목 제거하는 LRU 축출 정책",    0),  # ≈ anchor0
    ("TRUE_POS", "셀프 어텐션이 모든 토큰 쌍을 동시에 가중하는 전역 연산", 1),  # ≈ anchor1
]


def _l2_from_bytes(blob: bytes, qv: torch.Tensor) -> "tuple[float,float]":
    """저장된 state_vec bytes vs query → (cosine, L2 norm). 둘 다 반환."""
    n = len(blob) // 4
    prev = torch.tensor(struct.unpack(f"{n}f", blob), dtype=torch.float32)
    if prev.shape != qv.shape:
        return float("nan"), float("nan")
    cos = F.cosine_similarity(qv.unsqueeze(0), prev.unsqueeze(0)).item()
    l2  = float((qv - prev).norm())
    return cos, l2


def run(use_query_prefix: bool, enc: EmbeddingBridge, label: str):
    """한 prefix 모드로 전체 probe recall. rows 반환."""
    tmp = tempfile.mkdtemp(prefix=f"htp-fp-{label}-")
    mem = MemorySystem(memory_dir=Path(tmp) / "mem")
    thr = mem.CONFLICT_RECALL_MISMATCH_THRESHOLD

    # anchor 저장 — 저장은 항상 passage(encode), 시스템 현 동작과 동일
    anchor_ids = []
    for trig, interp in ANCHORS:
        tv = torch.tensor(enc.encode(trig), dtype=torch.float32)
        ep_id = mem.save_conflict(
            trigger_vec=tv, new_text=trig, partner_texts=["dummy"],
            interpretation=interp, conflict_score=0.15,
        )
        anchor_ids.append(ep_id)

    rows = []
    for kind, probe_text, anchor_idx in PROBES:
        # 검색 벡터 — A/B: encode() vs encode_query()
        if use_query_prefix:
            qvec = enc.encode_query(probe_text)
        else:
            qvec = enc.encode(probe_text)
        qv = torch.tensor(qvec, dtype=torch.float32)

        results = mem.recall_conflict(qv, top_k=3)
        if not results:
            rows.append({"kind": kind, "probe": probe_text,
                         "hit": False, "reason": "no_candidate"})
            continue

        best_ep, best_q = results[0]
        cos, l2 = _l2_from_bytes(best_ep.state_vec, qv)
        # 시스템 실제 게이트: mismatch(L2) >= thr → None
        hit = (not np.isnan(l2)) and (l2 < thr)

        # best 후보가 "의도한 anchor" 인가? (정렬 정확도 부수 측정)
        intended_id = anchor_ids[anchor_idx]
        matched_intended = (best_ep.episode_id == intended_id)

        rows.append({
            "kind": kind, "probe": probe_text,
            "hit": hit,
            "cos_to_best": round(cos, 4),
            "l2_to_best":  round(l2, 4),
            "threshold":   thr,
            "best_matches_intended_anchor": matched_intended,
        })
    return rows, thr


def summarize(rows, label):
    print(f"\n{'='*64}\n## 모드: {label}\n{'='*64}")
    by_kind: dict = {}
    for r in rows:
        by_kind.setdefault(r["kind"], []).append(r)

    # 거짓 양성: NEG 류인데 hit=True
    fp = sum(1 for r in rows
             if r["kind"].endswith("NEG") and r.get("hit"))
    neg_total = sum(1 for r in rows if r["kind"].endswith("NEG"))
    tp = sum(1 for r in rows if r["kind"] == "TRUE_POS" and r.get("hit"))
    tp_total = sum(1 for r in rows if r["kind"] == "TRUE_POS")

    print(f"\n  거짓 양성(FP): {fp}/{neg_total}  "
          f"(NEG 인데 잘못 HIT)   ← 낮을수록 좋음")
    print(f"  참 양성(TP) : {tp}/{tp_total}  "
          f"(TRUE_POS 정상 HIT)   ← 높을수록 좋음")

    for kind in ("HARD_NEG", "MID_NEG", "EASY_NEG", "TRUE_POS"):
        items = by_kind.get(kind, [])
        if not items:
            continue
        print(f"\n  [{kind}]")
        for r in items:
            if "cos_to_best" not in r:
                print(f"    {'·':>3} {r['probe'][:42]:42} (no candidate)")
                continue
            flag = "⚠FP" if (kind.endswith('NEG') and r['hit']) else \
                   ("✓TP" if (kind == 'TRUE_POS' and r['hit']) else "  ok")
            print(f"    {flag} {r['probe'][:42]:42} "
                  f"cos={r['cos_to_best']:+.3f} L2={r['l2_to_best']:.3f} "
                  f"hit={r['hit']!s:5} "
                  f"{'intended' if r['best_matches_intended_anchor'] else 'OTHER-anchor'}")
    return fp, neg_total, tp, tp_total


def main():
    print("# 거짓 양성 recall 측정 — htp-conflict-memory")
    print("# anchor 6 / probe %d (HARD/MID/EASY NEG + TRUE_POS)" % len(PROBES))
    enc = EmbeddingBridge()
    print(f"# encoder: {enc._model_name} dim={enc.dim}")

    out = {}
    for use_q, label in [(False, "A_passage_prefix(현재시스템)"),
                         (True,  "B_query_prefix(발견B 수정안)")]:
        rows, thr = run(use_q, enc, label[:12])
        fp, nt, tp, tpt = summarize(rows, label)
        out[label] = {"rows": rows, "fp": fp, "neg": nt,
                      "tp": tp, "tp_total": tpt, "threshold": thr}

    # ── 발견 A: 지표 불일치 정량화 ────────────────────
    print(f"\n{'='*64}\n## 발견 A — cosine(선택) vs L2(컷) 불일치\n{'='*64}")
    print("  후보는 cosine 최대로 뽑고, 채택은 L2<thr 로 컷.")
    print("  두 지표가 단조 일치하면 정규화된 것 — 어긋나면 구조 결함.")
    for label, d in out.items():
        pairs = [(r["cos_to_best"], r["l2_to_best"])
                 for r in d["rows"] if "cos_to_best" in r
                 and not np.isnan(r["l2_to_best"])]
        if len(pairs) < 3:
            continue
        cs = np.array([p[0] for p in pairs])
        ls = np.array([p[1] for p in pairs])
        # cosine↑ 이면 L2↓ 여야 정상 → 상관계수 음수 기대
        corr = float(np.corrcoef(cs, ls)[0, 1])
        # 정규화 시 L2^2 = 2(1-cos) → L2 = sqrt(2(1-cos)) 이론선과 잔차
        theo = np.sqrt(np.clip(2 * (1 - cs), 0, None))
        resid = float(np.mean(np.abs(ls - theo)))
        print(f"\n  [{label}]")
        print(f"    corr(cos, L2) = {corr:+.3f}  "
              f"(정규화면 강한 음수 ≈ -0.98 기대)")
        print(f"    |L2 - sqrt(2(1-cos))| 평균잔차 = {resid:.4f}  "
              f"(0 이면 완전 정규화)")

    Path("docs/03-analysis").mkdir(parents=True, exist_ok=True)
    with open("docs/03-analysis/conflict_recall_fp_raw.json",
              "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)
    print("\n# raw → docs/03-analysis/conflict_recall_fp_raw.json")


if __name__ == "__main__":
    main()
