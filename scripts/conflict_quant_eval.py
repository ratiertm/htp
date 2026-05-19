"""
양적 검증 — 50건 이질 ingest 에서 LLM interpretation hit rate 측정.

목적:
  - 어떤 종류의 충돌에서 LLM 이 의미 있는 통찰을 내는가?
  - confidence threshold 를 어디 잡아야 B (Memory 연결) 에서 저장 가치 있는
    interpretation 만 추릴 수 있는가?

방법:
  - 4 도메인 (뇌과학/AI/인프라/금융) × 7 entries 적재
  - cross-domain ingest 50건 (의도적 이질)
  - 각 시도에서 coherence/conflict/escalate/interpretation/latency 기록
  - raw 결과 docs/03-analysis/conflict_quant_raw.jsonl 저장
  - 요약 마크다운 자동 생성

실행:
  source .venv/bin/activate
  python scripts/conflict_quant_eval.py
"""
from __future__ import annotations

import json
import time
import tempfile
from pathlib import Path

from htp.knowledge          import KnowledgeLoop, KnowledgeStore
from htp.knowledge.embedding import EmbeddingBridge
from htp.knowledge.conflict_prompt import SYSTEM_PROMPT
from htp.llm                import LLMRegion, ClaudeCliNode


# ── 데이터셋 ───────────────────────────────────────────

BASELINE = {
    "뇌과학": [
        "해마 CA3 의 패턴 완성 메커니즘 시냅스 recurrent",
        "시냅스 가소성과 헵의 학습 법칙",
        "감마 진동과 시간적 바인딩",
        "시상의 게이팅과 의식 통합",
        "SWR 수면 중 기억 공고화",
        "PFC working memory 와 top-down 조절",
        "신피질 미니컬럼 6층 구조",
    ],
    "AI": [
        "Transformer self attention 메커니즘 encoder",
        "RLHF 인간 피드백 강화 학습",
        "MoE sparse experts gating",
        "RAG 검색 증강 생성 retrieval pipeline",
        "Diffusion 모델 latent score function",
        "Contrastive learning embedding distance",
        "Beam search decoding sequence generation",
    ],
    "인프라": [
        "Redis LRU 캐시 eviction 전략",
        "Kubernetes pod scheduler resource quota",
        "Nginx 로드밸런서 round robin",
        "CDN edge caching TTL fastly",
        "gRPC service mesh istio",
        "PostgreSQL WAL replication consistency",
        "Kafka partition consumer offset",
    ],
    "금융": [
        "거래소 order book 매수매도 호가",
        "시장조성자 bid-ask spread 유동성 공급",
        "포트폴리오 효율적 변경 리밸런싱",
        "옵션 implied volatility 변동성",
        "마진콜 청산 강제매도",
        "high frequency trading 지연 차익거래",
        "ETF 추종오차 NAV 괴리율",
    ],
}

CROSS_INGEST = [
    # (text, intended_target_source)  — 의도적 cross-domain
    # 뇌-AI (10)
    ("주의 메커니즘은 국소적이며 한 번에 한 영역만 순차 처리",     "뇌과학"),
    ("뇌의 reward learning 은 dopamine 신호로 시냅스를 강화",      "뇌과학"),
    ("attention 은 모든 입력 쌍을 동시에 가중평균하는 전역 연산",   "AI"),
    ("self supervised learning 은 데이터의 구조에서 신호를 추출",   "AI"),
    ("뉴런 발화율 sparse coding population vector",               "뇌과학"),
    ("contrastive 학습 anchor positive negative triplet loss",     "AI"),
    ("뇌의 sequence learning 은 striatum 의 순차 패턴 강화",      "뇌과학"),
    ("autoregressive generation next token prediction",            "AI"),
    ("hippocampal replay 는 깨어있을 때 일어나는 시뮬레이션",      "뇌과학"),
    ("world model latent dynamics rollout planning",               "AI"),

    # 뇌-인프라 (10)
    ("synaptic pruning 미세아교세포가 약한 시냅스 제거",          "뇌과학"),
    ("axonal myelination 으로 신호 전도 속도 증가",                "뇌과학"),
    ("Redis LRU policy 가 오래된 키를 캐시에서 제거",              "인프라"),
    ("로드밸런싱 round robin server 분산 처리",                    "인프라"),
    ("lateral inhibition 인접 뉴런 활성 억제 winner takes all",    "뇌과학"),
    ("CDN cache invalidation purge TTL expire",                    "인프라"),
    ("뇌의 critical period 발달 단계에서만 특정 학습 가능",        "뇌과학"),
    ("blue green deployment zero downtime release",                "인프라"),
    ("neurogenesis 성체 해마 치상회 새 뉴런 생성",                 "뇌과학"),
    ("auto scaling group instance launch terminate",              "인프라"),

    # AI-금융 (10)
    ("Q-learning agent state action reward maximization",          "AI"),
    ("market making algorithm bid ask spread profit",              "금융"),
    ("multi armed bandit exploration exploitation",                "AI"),
    ("statistical arbitrage pair trading mean reversion",         "금융"),
    ("PPO policy gradient clip ratio update",                     "AI"),
    ("portfolio optimization Markowitz mean variance",            "금융"),
    ("VAE variational autoencoder latent posterior",              "AI"),
    ("Monte Carlo VaR risk simulation tail event",                "금융"),
    ("ensemble bagging boosting variance reduction",              "AI"),
    ("factor model alpha beta risk premium",                       "금융"),

    # 인프라-금융 (10)
    ("eventual consistency CAP theorem partition tolerance",       "인프라"),
    ("settlement T+2 clearing house counterparty",                "금융"),
    ("write ahead log durability crash recovery",                  "인프라"),
    ("limit order book matching engine price priority",           "금융"),
    ("circuit breaker pattern resilience fault tolerance",         "인프라"),
    ("market circuit breaker price limit halt",                    "금융"),
    ("idempotent operation retry deduplication",                   "인프라"),
    ("trade reconciliation matching settlement break",            "금융"),
    ("rate limiting token bucket leaky bucket",                    "인프라"),
    ("liquidity provider market depth slippage",                  "금융"),

    # 뇌-금융 (10) — extra
    ("dopamine reward prediction error temporal difference",       "뇌과학"),
    ("loss aversion 행동경제학 prospect theory",                   "금융"),
    ("amygdala fear learning conditioning",                        "뇌과학"),
    ("market panic flash crash herding behavior",                  "금융"),
    ("PFC executive function inhibition control",                  "뇌과학"),
    ("trader discipline risk management drawdown",                "금융"),
    ("habit formation basal ganglia procedural learning",          "뇌과학"),
    ("algorithmic trading rule based execution",                  "금융"),
    ("sleep deprivation cognitive impairment decision",            "뇌과학"),
    ("overnight risk position sizing leverage",                    "금융"),
]
# 총 50건


def main():
    tmp = tempfile.mkdtemp(prefix="htp-quant-eval-")
    print(f"# sandbox: {tmp}")
    raw_path = Path("docs/03-analysis/conflict_quant_raw.jsonl")
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    cli = ClaudeCliNode(
        name="quant_interpreter",
        system=SYSTEM_PROMPT,
        timeout_sec=120.0,
    )
    interpreter = LLMRegion(
        region_name="quant_interpreter",
        specialty="reasoning",
        llm_node=cli,
    )

    store = KnowledgeStore(Path(tmp) / "log.jsonl")
    loop = KnowledgeLoop(
        encoder=EmbeddingBridge(),
        store=store,
        conflict_interpreter=interpreter,
        coherence_thresholds=(0.10, 0.12),
        max_interpretations=100,   # cap 충분히
    )
    print(f"# encoder: EmbeddingBridge dim={loop.encoder.dim}")
    print(f"# thresholds: (conflict={loop.coherence_conflict_threshold}, "
          f"escalation={loop.coherence_escalation_threshold})")
    print()

    # ── baseline 적재 ─────────────────────────────────
    print("## Baseline 적재 (각 도메인 7건)")
    t0 = time.perf_counter()
    for src, texts in BASELINE.items():
        for t in texts:
            loop.ingest(t, source=src)
    print(f"  done: {len(loop._cache)} entries in {time.perf_counter()-t0:.1f}s")
    print(f"  baseline interpretations: {loop._interpretations_count}")
    print()

    # ── cross-domain ingest 50건 ─────────────────────
    print("## Cross-domain ingest 50건 + raw 데이터 수집")
    rows = []
    with raw_path.open("w", encoding="utf-8") as f:
        for i, (text, source) in enumerate(CROSS_INGEST, 1):
            t_call = time.perf_counter()
            r = loop.ingest(text, source=source)
            elapsed_ms = (time.perf_counter() - t_call) * 1000

            ci = r.coherence_info or {}
            row = {
                "idx":           i,
                "text":          text,
                "source":        source,
                "coherence":     ci.get("coherence"),
                "conflict":      ci.get("conflict"),
                "escalate":      ci.get("escalate"),
                "interpretation": r.entry.interpretation,
                "elapsed_ms":    round(elapsed_ms, 1),
                "interp_len":    len(r.entry.interpretation or ""),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            rows.append(row)

            mark = "💡" if r.entry.interpretation else "  "
            esc  = "T" if ci.get("escalate") else "F"
            print(f"  [{i:2}/50] {mark} esc={esc} conf={ci.get('conflict', 0):.3f} "
                  f"ms={elapsed_ms:>6.0f} | {text[:50]}...")

    # ── 요약 통계 ─────────────────────────────────────
    print()
    print("## 요약 통계")
    n_total       = len(rows)
    n_escalate    = sum(1 for r in rows if r["escalate"])
    n_interp      = sum(1 for r in rows if r["interpretation"])
    avg_conflict  = sum(r["conflict"] for r in rows if r["conflict"]) / n_total
    avg_ms_interp = (sum(r["elapsed_ms"] for r in rows if r["interpretation"])
                     / max(n_interp, 1))

    print(f"  total           : {n_total}")
    print(f"  escalate=True   : {n_escalate} ({n_escalate/n_total:.0%})")
    print(f"  interpretations : {n_interp} ({n_interp/n_total:.0%})")
    print(f"  avg conflict    : {avg_conflict:.3f}")
    print(f"  avg interp ms   : {avg_ms_interp:.0f}")
    print(f"  total elapsed   : {(time.perf_counter() - t0):.1f}s")

    print(f"\n  raw: {raw_path}")


if __name__ == "__main__":
    main()
