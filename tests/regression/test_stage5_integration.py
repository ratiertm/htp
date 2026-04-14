"""
Stage 5 통합 시나리오 — BrainRuntime 과 MemorySystem 의 end-to-end 검증.

시나리오:
  2 Region × 여러 step 실행 — 에피소드 저장, recall, CUSUM 오버로드 트리거 흐름.
"""
from __future__ import annotations

import pytest

from htp import BrainRuntime, HTPConfig, RegionRuntime, tag, terminal


@pytest.fixture
def two_region_brain_memory(tmp_path):
    brain = BrainRuntime(memory_dir=tmp_path / "mem")

    lang = RegionRuntime("language", "text_processing",
                         config=HTPConfig(threshold=0.35))
    mem = RegionRuntime("memory", "cache_storage",
                        config=HTPConfig(threshold=0.35))

    @lang.node
    @terminal
    @tag("parse", "text")
    def parse_text(data):
        return {"parsed": True}

    @mem.node
    @terminal
    @tag("cache", "store")
    def cache_store(data):
        return {"cached": True}

    brain.add_region("language", lang)
    brain.add_region("memory", mem)
    return brain


@pytest.mark.regression
def test_episodes_accumulate_across_steps(two_region_brain_memory):
    brain = two_region_brain_memory

    for i in range(10):
        brain.run({"label": "text", "i": i})

    # 매 step 마다 1개 저장 → 10개
    assert brain.memory.episode_count() == 10, \
        f"에피소드 누적 실패: {brain.memory.episode_count()}"


@pytest.mark.regression
def test_last_state_vec_persists_for_recall(two_region_brain_memory):
    """첫 스텝 후 _last_state_vec 이 보존되어 다음 스텝 recall 이 가능."""
    brain = two_region_brain_memory

    assert brain._last_state_vec is None
    brain.run({"label": "text"})
    assert brain._last_state_vec is not None
    assert brain._last_state_vec.shape[0] == 64


@pytest.mark.regression
def test_memory_hint_injection_without_pattern_is_noop(two_region_brain_memory):
    """L3 패턴 미형성 상태 (첫 몇 스텝) — recall 은 호출되지만 추천 없음."""
    brain = two_region_brain_memory

    # 2 step 실행: step 2 에서 recall 호출되지만 L3 비어있음
    brain.run({"label": "text"})
    brain.run({"label": "text"})

    # is_novel=True 이므로 top-down hint 가 memory 로 치우치지 않음 — 정상 동작
    # 여기서는 단순히 예외 없이 동작 + 에피소드 2 개 쌓였는지 확인
    assert brain.memory.episode_count() == 2


@pytest.mark.regression
def test_feedback_records_outcome(two_region_brain_memory):
    """brain.feedback() 호출이 마지막 에피소드의 outcome 을 업데이트."""
    brain = two_region_brain_memory

    brain.run({"label": "text"})
    brain.feedback("success")

    ep = brain.memory.l2.recent(1)[0]
    assert ep.outcome == "success", f"outcome 미반영: {ep.outcome}"


@pytest.mark.regression
def test_disable_memory_flag(tmp_path):
    """enable_memory=False 일 때 memory=None, 기존 동작 유지."""
    from htp.thalamus.region_signal import Action

    brain = BrainRuntime(memory_dir=tmp_path, enable_memory=False)

    lang = RegionRuntime("language", "text_processing",
                         config=HTPConfig(threshold=0.35))

    @lang.node
    @terminal
    @tag("text")
    def parse_text(data): return data

    brain.add_region("language", lang)
    action = brain.run({"label": "text"})

    assert brain.memory is None
    assert isinstance(action, Action)


@pytest.mark.regression
def test_cusum_overload_triggers_consolidation(two_region_brain_memory):
    """
    인공적으로 region._cusum_S 를 초과시키면 on_overload 가 호출되어 reset.
    """
    brain = two_region_brain_memory
    brain.run({"label": "text"})  # 빌드 + 1 step

    # 인공 과부하 유도
    lang = brain.regions["language"]
    lang._cusum_S = lang._cusum_h + 1.0   # threshold 초과

    # 다음 step 실행 시 on_overload 호출 → cusum reset
    brain.run({"label": "text"})
    assert lang._cusum_S == 0.0, \
        f"overload 후 CUSUM reset 안 됨: {lang._cusum_S}"


@pytest.mark.regression
def test_score_extraction_from_action_reason():
    """_extract_score 가 action.reason 에서 score 파싱."""
    from htp.runtime.brain_runtime import BrainRuntime
    from htp.thalamus.region_signal import Action

    a1 = Action(type="execute", winner="x", reason="score=0.723 (cos=0.5 goal=1.0)")
    assert BrainRuntime._extract_score(a1) == pytest.approx(0.723)

    a2 = Action(type="inhibit", winner="x", reason="no score here")
    assert BrainRuntime._extract_score(a2) == 0.5  # fallback default
