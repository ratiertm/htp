"""공통 pytest fixtures."""
from __future__ import annotations

import random

import pytest
import torch


@pytest.fixture(autouse=True)
def deterministic():
    """모든 테스트에 시드 고정. 허브 형성·PageRank 수렴의 결정론적 재현을 위함."""
    torch.manual_seed(42)
    random.seed(42)
    yield


@pytest.fixture
def simple_runtime():
    """
    Phase 1 데모 시나리오의 parse→classify→{to_cache|to_alert}→log_result 런타임.
    htp_runtime.demo()와 같은 위상, 테스트에서 재사용.
    """
    from htp import HTPConfig, HTPRuntime, tag, terminal

    rt = HTPRuntime(HTPConfig(
        hub_pr_threshold=2.5,
        hebbian_lr=0.13,
        decay_rate=0.005,
        prune_threshold=0.02,
        usage_window=15,
        usage_min=0.05,
        redundancy_cos=0.95,
        threshold=0.35,
        hub_protect=True,
        age_threshold=50,
    ))

    @rt.node
    def parse(data):
        text = str(data).lower().strip()
        return {"text": text, "len": len(text)}

    @rt.node
    def classify(data):
        text = data.get("text", "") if isinstance(data, dict) else str(data)
        err = ["error", "fail", "bug", "timeout", "fatal", "oom"]
        ok = ["success", "ok", "done", "deployed", "completed"]
        if any(w in text for w in err):
            return {**data, "label": "error"}
        if any(w in text for w in ok):
            return {**data, "label": "success"}
        return {**data, "label": "neutral"}

    @rt.node
    @tag("success", "ok", "done", "deployed", "completed", "cache")
    def to_cache(data):
        return {**data, "cached": True}

    @rt.node
    @tag("error", "fail", "bug", "timeout", "fatal", "oom")
    def to_alert(data):
        return {**data, "alerted": True}

    @rt.node
    @terminal
    @tag("success", "error", "neutral", "cached", "alerted")
    def log_result(data):
        return {**data, "logged": True}

    rt.connect(parse, classify, weight=0.6)
    rt.connect(classify, to_cache, weight=0.5)
    rt.connect(classify, to_alert, weight=0.5)
    rt.connect(to_cache, log_result, weight=0.5)
    rt.connect(to_alert, log_result, weight=0.5)

    return rt, {
        "parse": parse, "classify": classify,
        "to_cache": to_cache, "to_alert": to_alert, "log_result": log_result,
    }


@pytest.fixture
def two_region_brain(tmp_path):
    """
    Phase 2-3 회귀용 최소 BrainRuntime — 2 Region.
    각 Region은 한 개 terminal node만 가진다.
    memory_dir 은 tmp_path 로 격리.
    """
    from htp import BrainRuntime, HTPConfig, RegionRuntime, tag, terminal

    brain = BrainRuntime(memory_dir=tmp_path / "brain_mem")

    lang = RegionRuntime("language", "text_processing",
                         config=HTPConfig(threshold=0.35))
    mem = RegionRuntime("memory", "cache_storage",
                        config=HTPConfig(threshold=0.35))

    @lang.node
    @terminal
    @tag("parse", "text")
    def parse_text(data):
        return {"parsed": True, "data": data}

    @mem.node
    @terminal
    @tag("cache", "store")
    def cache_store(data):
        return {"cached": True, "data": data}

    brain.add_region("language", lang)
    brain.add_region("memory", mem)
    return brain
