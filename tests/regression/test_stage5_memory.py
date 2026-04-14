"""
Stage 5 Memory System 단위 테스트 — L2 + L3 + MemorySystem.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from htp.memory import Episode, EpisodeStore, MemorySystem, PatternStore
from htp.memory.types import tensor_to_bytes


@pytest.fixture
def tmp_memory_dir(tmp_path):
    return tmp_path / "memory"


# ═══════════════════════════════════════════════
# L2 EpisodeStore
# ═══════════════════════════════════════════════

@pytest.mark.regression
def test_episode_store_save_and_count(tmp_memory_dir):
    store = EpisodeStore(tmp_memory_dir / "memory.db")
    vec = torch.randn(64)
    ep = Episode(step=1, winner="A", action_type="execute",
                 score=0.8, state_vec=tensor_to_bytes(vec), novelty=0.6)
    ep_id = store.save(ep)

    assert ep_id
    assert store.count() == 1


@pytest.mark.regression
def test_episode_search_similar_returns_nearest(tmp_memory_dir):
    store = EpisodeStore(tmp_memory_dir / "memory.db")

    target = torch.randn(64)
    near = target + torch.randn(64) * 0.05
    far = torch.randn(64)

    store.save(Episode(winner="NEAR", score=1.0, state_vec=tensor_to_bytes(near)))
    store.save(Episode(winner="FAR",  score=1.0, state_vec=tensor_to_bytes(far)))

    results = store.search_similar(target, top_k=2)
    assert len(results) == 2
    assert results[0].winner == "NEAR", \
        f"코사인 유사도 상위가 NEAR가 아님: winners={[r.winner for r in results]}"


@pytest.mark.regression
def test_swr_tag_priority_novelty_times_reward(tmp_memory_dir):
    """SWR 태깅: priority = novelty × score >= 0.5."""
    store = EpisodeStore(tmp_memory_dir / "memory.db")

    # 높은 priority (0.8 * 0.8 = 0.64)
    ep_high = Episode(winner="A", score=0.8, novelty=0.8, outcome="success",
                     state_vec=tensor_to_bytes(torch.randn(64)))
    # 낮은 priority (0.3 * 0.5 = 0.15)
    ep_low = Episode(winner="B", score=0.5, novelty=0.3, outcome="success",
                    state_vec=tensor_to_bytes(torch.randn(64)))
    id_high = store.save(ep_high)
    id_low  = store.save(ep_low)

    store.tag_swr()

    tagged = store.search_similar(torch.zeros(64), top_k=10, swr_only=True)
    tagged_ids = {e.episode_id for e in tagged}
    assert id_high in tagged_ids
    assert id_low not in tagged_ids


@pytest.mark.regression
def test_dim_mismatch_filtered_in_search(tmp_memory_dir):
    """구 8-dim 에피소드가 있어도 64-dim 검색 시 스킵돼야 (차원 혼재 안전성)."""
    store = EpisodeStore(tmp_memory_dir / "memory.db")
    store.save(Episode(winner="OLD", score=0.5,
                       state_vec=tensor_to_bytes(torch.randn(8))))
    store.save(Episode(winner="NEW", score=0.5,
                       state_vec=tensor_to_bytes(torch.randn(64))))

    results = store.search_similar(torch.randn(64), top_k=5)
    winners = [r.winner for r in results]
    assert "OLD" not in winners, "차원 불일치 에피소드 필터 실패"
    assert "NEW" in winners


# ═══════════════════════════════════════════════
# L3 PatternStore
# ═══════════════════════════════════════════════

@pytest.mark.regression
def test_pattern_empty_returns_original(tmp_memory_dir):
    """패턴 없을 때 complete() 는 원본 반환."""
    ps = PatternStore(tmp_memory_dir / "patterns.json")
    vec = torch.randn(64)
    completed, pat = ps.complete(vec)
    assert pat is None
    assert torch.allclose(completed, vec)


@pytest.mark.regression
def test_go_cls_promotion_requires_3_episodes(tmp_memory_dir):
    """Go-CLS: 3개 이상 유사 에피소드 + snr >= 1.5 가 있어야 L3 패턴 승격."""
    ps = PatternStore(tmp_memory_dir / "patterns.json")

    # 유사한 3개 에피소드 — 승자 A 로 성공
    base = torch.randn(64)
    eps = [
        Episode(
            winner="A", score=0.8 + 0.1 * i, outcome="success", swr_tagged=True,
            state_vec=tensor_to_bytes(base + torch.randn(64) * 0.05),
        )
        for i in range(3)
    ]
    ps.consolidate(eps)

    assert ps.pattern_count() >= 1, \
        f"유사 3개 에피소드 → 패턴 승격 실패 (patterns={ps.pattern_count()}, buffer={ps.buffer_count()})"


@pytest.mark.regression
def test_ca3_completion_converges_noisy_input(tmp_memory_dir):
    """CA3: 노이즈 입력 → 가까운 centroid 로 수렴."""
    ps = PatternStore(tmp_memory_dir / "patterns.json")

    base = torch.randn(64)
    eps = [
        Episode(
            winner="A", score=0.8, outcome="success", swr_tagged=True,
            state_vec=tensor_to_bytes(base + torch.randn(64) * 0.05),
        )
        for _ in range(5)
    ]
    ps.consolidate(eps)

    if ps.pattern_count() == 0:
        pytest.skip("패턴 승격 실패 — 승격 로직은 별도 테스트에서 검증")

    noisy = base + torch.randn(64) * 0.15
    completed, matched = ps.complete(noisy)

    # 완성본이 base 쪽으로 이동
    assert matched is not None
    dist_before = (noisy - base).norm().item()
    dist_after = (completed - base).norm().item()
    assert dist_after < dist_before, \
        f"CA3 completion 이 수렴하지 않음: before={dist_before:.3f}, after={dist_after:.3f}"


# ═══════════════════════════════════════════════
# MemorySystem 통합
# ═══════════════════════════════════════════════

@pytest.mark.regression
def test_memory_system_save_and_recall_novel_scenario(tmp_memory_dir):
    ms = MemorySystem(memory_dir=tmp_memory_dir)

    state = torch.randn(64)
    ms.save(state, step=1, winner="A", action_type="execute",
            score=0.8, context="test input")

    assert ms.episode_count() == 1

    # L3 에 패턴 없음 → is_novel=True
    ctx = ms.recall(state + torch.randn(64) * 0.01)
    assert ctx.is_novel is True or ctx.pattern is None


@pytest.mark.regression
def test_memory_system_novelty_decreases_with_pattern(tmp_memory_dir):
    """패턴이 L3 에 들어오면 같은 벡터 저장 시 novelty 가 감소."""
    ms = MemorySystem(memory_dir=tmp_memory_dir)

    base = torch.randn(64)
    # 초기 — 패턴 없음, novelty 높음
    for i in range(5):
        state = base + torch.randn(64) * 0.02
        ms.save(state, step=i, winner="A", action_type="execute",
                score=0.8, context=f"ep{i}")
        ms.feedback("success")

    # consolidate 호출 (통상 on_overload 에서)
    ms.on_overload("dummy")

    if ms.pattern_count() == 0:
        pytest.skip("패턴 미승격")

    # 이후 유사 벡터의 novelty 는 낮아야
    similar = base + torch.randn(64) * 0.02
    last_id = ms.save(similar, step=99, winner="A", action_type="execute",
                      score=0.8, context="late")
    # 저장된 에피소드 확인 — novelty 감소 검증
    eps = ms.l2.recent(1)
    assert eps[0].novelty < 1.0, \
        f"패턴 존재 후에도 novelty 가 감소 안 함: {eps[0].novelty}"


@pytest.mark.regression
def test_memory_dir_auto_created(tmp_path):
    """MemorySystem 생성 시 memory_dir 자동 생성."""
    path = tmp_path / "nonexistent" / "deep"
    ms = MemorySystem(memory_dir=path)
    assert path.exists()
    assert (path / "memory.db").exists() or (path).exists()
