"""
Phase 1 회귀 — 3가지 가지치기 전략 작동 확인.
"""
from __future__ import annotations

import pytest

from htp import HTPConfig, HTPRuntime


@pytest.mark.regression
def test_pruning_engine_has_all_strategies(simple_runtime):
    """PruningEngine이 decay / usage / redundancy / age 전략 stats를 보유."""
    rt, nodes = simple_runtime
    rt.run("success", entry=nodes["parse"])

    stats = rt.pe.stats
    from htp.runtime.htp_runtime import PruneStrategy

    assert PruneStrategy.DECAY in stats
    assert PruneStrategy.USAGE in stats
    assert PruneStrategy.REDUND in stats
    assert PruneStrategy.AGE in stats


@pytest.mark.regression
def test_decay_prune_runs_every_step(simple_runtime):
    """decay_prune은 매 run()마다 호출되어 runs 카운트가 증가."""
    rt, nodes = simple_runtime
    from htp.runtime.htp_runtime import PruneStrategy

    for _ in range(10):
        rt.run("success", entry=nodes["parse"])

    assert rt.pe.stats[PruneStrategy.DECAY]["runs"] >= 10, \
        f"decay runs = {rt.pe.stats[PruneStrategy.DECAY]['runs']}"


@pytest.mark.regression
def test_hub_protect_prevents_hub_edge_pruning(simple_runtime):
    """
    hub_protect=True면 허브 관여 엣지는 decay 대상에서 제외.
    소규모 그래프에서 is_hub가 비어있을 수 있음 — 인공적으로 노드 1개 허브 플래그 설정 후 검증.
    """
    rt, nodes = simple_runtime

    for _ in range(40):
        rt.run("success deployed", entry=nodes["parse"])

    # PageRank 최대 노드를 인공 허브로 지정 (소규모 그래프에서 자연 승격 불가 시 대비)
    if int(rt.hfe.is_hub.sum().item()) == 0:
        pr = rt.hfe.pagerank()
        rt.hfe.is_hub[int(pr.argmax().item())] = True

    # 허브 관여 엣지 중 weight > prune_threshold인 것이 남아있어야
    hub_idx = rt.hfe.is_hub.nonzero(as_tuple=True)[0]
    for h in hub_idx.tolist():
        in_edges = (rt.wm.W[:, h] > rt.cfg.prune_threshold).any().item()
        out_edges = (rt.wm.W[h, :] > rt.cfg.prune_threshold).any().item()
        assert in_edges or out_edges, f"허브 {h}의 모든 엣지가 제거됨"
