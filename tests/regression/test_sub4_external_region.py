"""sub-4 Stage 4 — ExternalRegion 추상 검증.

Design Ref: docs/02-design/features/htp-thalamus-car.sub-4.design.md §3-1
Plan SC: FR-16 (ExternalRegion 추상)
"""
from __future__ import annotations

import asyncio
import pytest
import torch

from htp.runtime.external_region import ExternalRegion
from htp.thalamus.region_signal  import RegionSignal


def test_external_region_abstract_cannot_instantiate():
    """ExternalRegion 직접 인스턴스화 불가 (ABC)."""
    with pytest.raises(TypeError):
        ExternalRegion()


def test_external_region_subclass_must_implement_run():
    """run 미구현 하위 클래스는 인스턴스화 불가."""
    class Incomplete(ExternalRegion):
        # collect_signal 만 구현, run 누락
        def collect_signal(self):
            return RegionSignal(
                region_id="x", hub_strength=0.0, fire_rate=0.0,
                top_hubs=[], overload=False,
                output_vec=torch.zeros(1), precision=1.0,
            )
    with pytest.raises(TypeError):
        Incomplete()


def test_external_region_subclass_must_implement_collect_signal():
    """collect_signal 미구현 하위 클래스는 인스턴스화 불가."""
    class Incomplete(ExternalRegion):
        def run(self, data):
            return data
    with pytest.raises(TypeError):
        Incomplete()


def test_external_region_default_arun_wraps_sync():
    """default arun 은 sync run 을 await — 하위 클래스 override 없을 때."""
    class Echo(ExternalRegion):
        region_name = "echo"; specialty = "test"; step = 0
        def run(self, data):
            self.step += 1
            return {"echo": data}
        def collect_signal(self):
            return RegionSignal(
                region_id=self.region_name, hub_strength=0.0,
                fire_rate=0.0, top_hubs=[], overload=False,
                output_vec=torch.zeros(1), precision=1.0,
            )

    e = Echo()
    out = asyncio.run(e.arun("hi"))
    assert out == {"echo": "hi"}
    assert e.step == 1


def test_external_region_apply_suppression_default_noop():
    """apply_suppression default 는 no-op (예외 없이 통과)."""
    class Min(ExternalRegion):
        region_name = "m"; specialty = "t"; step = 0
        def run(self, data): return data
        def collect_signal(self):
            return RegionSignal(
                region_id="m", hub_strength=0.0, fire_rate=0.0,
                top_hubs=[], overload=False,
                output_vec=torch.zeros(1), precision=1.0,
            )
    m = Min()
    m.apply_suppression(0.5)   # 예외 없으면 PASS
