"""
Phase 1 회귀 — 허브 형성 + PageRank.

검증:
  - 반복 패턴 입력 시 fire_count/가중치 합이 해당 노드에 집중
  - PageRank top_hubs가 정상 반환 (합≈1, 정렬됨)
  - Oja's Rule로 W가 [0, 1]에 유지됨
"""
from __future__ import annotations

import pytest
import torch


@pytest.mark.regression
def test_oja_rule_keeps_weights_bounded(simple_runtime):
    """Oja's Rule — W가 [0, 1] 범위 내 유지되고 폭주하지 않아야."""
    rt, nodes = simple_runtime

    for _ in range(50):
        rt.run("success deploy", entry=nodes["parse"])

    W = rt.wm.W
    assert W.min().item() >= 0.0, f"가중치 음수: {W.min().item()}"
    assert W.max().item() <= 1.0, f"가중치 폭주: {W.max().item()}"


@pytest.mark.regression
def test_pagerank_sums_to_one(simple_runtime):
    """PageRank 결과는 확률 분포 — 합 ≈ 1."""
    rt, nodes = simple_runtime

    # 빌드를 강제하기 위해 1회 실행
    rt.run("success", entry=nodes["parse"])

    pr = rt.hfe.pagerank()
    assert torch.isfinite(pr).all(), "PageRank NaN/Inf 발생"
    assert abs(pr.sum().item() - 1.0) < 1e-3, f"PageRank 합 != 1: {pr.sum().item()}"


@pytest.mark.regression
def test_top_hubs_returns_sorted_descending(simple_runtime):
    """top_hubs는 PageRank 기준 내림차순 정렬이어야."""
    rt, nodes = simple_runtime

    for _ in range(20):
        rt.run("success deploy", entry=nodes["parse"])

    top = rt.hfe.top_hubs(3)
    assert len(top) == 3
    scores = [s for _, s in top]
    assert scores == sorted(scores, reverse=True), f"정렬 깨짐: {scores}"


@pytest.mark.regression
def test_repeated_pattern_creates_uneven_centrality(simple_runtime):
    """
    동일 패턴 반복 입력 시 PageRank 분포가 균등을 벗어나 특정 노드에 집중되어야.

    Stage 2-A2 (PageRank 기반) 이후: 5-노드 소규모 체인에서는 teleportation 지배로
    pr*N 이 절대 허브 threshold(2.5)에 미치기 어렵다. 대신 **상대적 중심성**을 검증.
    """
    rt, nodes = simple_runtime

    for _ in range(60):
        rt.run("success deployed ok", entry=nodes["parse"])

    pr = rt.hfe.pagerank()
    pr_max = pr.max().item()
    pr_min = pr.min().item()
    pr_uniform = 1.0 / rt.wm.n

    # 학습 후 최고 / 최저 PR의 상대차가 15%+ 벌어져야 (학습 신호 존재)
    assert pr_max - pr_min > 0.03, \
        f"PageRank 분포가 거의 균등함 (max={pr_max:.3f}, min={pr_min:.3f}, uniform={pr_uniform:.3f})"
    # 최대 노드는 uniform 대비 10%+ 중앙성 우위
    assert pr_max > pr_uniform * 1.1, \
        f"최대 노드의 중심성이 uniform 대비 미약함 ({pr_max:.3f} vs {pr_uniform * 1.1:.3f})"
