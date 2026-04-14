"""
Stage 2-A3 — CoreCells homeostatic term 회귀 테스트.

검증:
  - 과흥분 Region (fire_rate > target) 의 theta_bias 가 + 방향으로 이동 (억제)
  - 저활성 Region (fire_rate < target) 의 theta_bias 가 - 방향으로 이동 (강화)
  - fire_rates=None 이면 기존 Hebbian 동작 유지 (하위 호환)
"""
from __future__ import annotations

import pytest

from htp import CoreCells


@pytest.mark.regression
def test_homeostatic_suppresses_overactive_region():
    """과흥분 Region은 theta_bias가 + 방향으로 이동해야."""
    cc = CoreCells(eta_heb=0.0, eta_hom=0.1, target_rate=0.1)

    for _ in range(20):
        cc.update(
            winner_id="A",
            all_ids=["A", "B"],
            fire_rates={"A": 0.5, "B": 0.1},  # A는 과흥분, B는 target
        )

    assert cc._theta_bias["A"] > 0.0, \
        f"과흥분 A의 theta_bias가 +방향 아님: {cc._theta_bias['A']:.3f}"
    # B는 target과 일치 → 거의 0
    assert abs(cc._theta_bias["B"]) < 0.01, \
        f"target 일치 B의 theta_bias가 0 부근이어야: {cc._theta_bias['B']:.3f}"


@pytest.mark.regression
def test_homeostatic_amplifies_underactive_region():
    """저활성 Region은 theta_bias가 - 방향으로 이동해야."""
    cc = CoreCells(eta_heb=0.0, eta_hom=0.1, target_rate=0.1)

    for _ in range(20):
        cc.update(
            winner_id="A",
            all_ids=["A", "B"],
            fire_rates={"A": 0.02, "B": 0.1},  # A 저활성
        )

    assert cc._theta_bias["A"] < 0.0, \
        f"저활성 A의 theta_bias가 -방향 아님: {cc._theta_bias['A']:.3f}"


@pytest.mark.regression
def test_hebbian_and_homeostatic_coexist():
    """
    Hebbian(승자 θ↓)과 Homeostatic(과흥분 θ↑)이 공존하며 상쇄 가능.
    승자 로테이션 + 과흥분 조건에서 두 term이 반대 방향 기여.
    """
    # 20 step 정도에서 두 term이 서로 상쇄되어 clamp 에 닿지 않는 균형 상태 검증
    cc = CoreCells(eta_heb=0.05, eta_hom=0.05, target_rate=0.1)

    for i in range(20):
        winner = "A" if i % 2 == 0 else "B"
        cc.update(
            winner_id=winner,
            all_ids=["A", "B"],
            fire_rates={"A": 0.3, "B": 0.1},   # A 는 살짝 과흥분
        )

    # clamp 미도달
    assert -0.2 < cc._theta_bias["A"] < 0.2, \
        f"A theta_bias clamp 도달: {cc._theta_bias['A']:.3f}"

    # Homeostatic 이 Hebbian 하강을 상쇄 → Hebbian-only 대비 덜 음수
    cc_heb_only = CoreCells(eta_heb=0.05, eta_hom=0.0)
    for i in range(20):
        winner = "A" if i % 2 == 0 else "B"
        cc_heb_only.update(winner_id=winner, all_ids=["A", "B"])
    assert cc._theta_bias["A"] > cc_heb_only._theta_bias["A"], \
        f"Homeostatic 이 Hebbian 하강을 상쇄하지 않음: coexist={cc._theta_bias['A']:.3f}, heb_only={cc_heb_only._theta_bias['A']:.3f}"


@pytest.mark.regression
def test_fire_rates_none_backward_compat():
    """fire_rates=None 이면 기존 Hebbian-only 동작 유지."""
    cc = CoreCells(eta_heb=0.05, eta_hom=0.02)

    for _ in range(10):
        cc.update(winner_id="A", all_ids=["A", "B"])  # fire_rates 생략

    # 승자 A만 theta_bias가 - 방향 (Hebbian만 작동)
    assert cc._theta_bias["A"] < 0.0
    assert cc._theta_bias["B"] == 0.0  # 패자 win_history=0 → 변화 없음
