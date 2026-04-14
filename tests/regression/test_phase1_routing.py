"""
Phase 1 회귀 — 시맨틱 라우팅 정확도.

CLAUDE.md 주장: "12/12 라우팅 정확도 (success → to_cache, error → to_alert)"
"""
from __future__ import annotations

import pytest


@pytest.mark.regression
def test_success_data_routes_to_cache(simple_runtime):
    rt, nodes = simple_runtime

    result = rt.run("deployment success ok", entry=nodes["parse"])

    path_names = [n.name for n in result.route_path]
    assert "to_cache" in path_names, f"success → to_cache 라우팅 실패: path={path_names}"
    assert "to_alert" not in path_names, f"success인데 to_alert 호출됨: path={path_names}"


@pytest.mark.regression
def test_error_data_routes_to_alert(simple_runtime):
    rt, nodes = simple_runtime

    result = rt.run("critical error timeout bug", entry=nodes["parse"])

    path_names = [n.name for n in result.route_path]
    assert "to_alert" in path_names, f"error → to_alert 라우팅 실패: path={path_names}"
    assert "to_cache" not in path_names, f"error인데 to_cache 호출됨: path={path_names}"


@pytest.mark.regression
def test_routing_accuracy_12_over_12(simple_runtime):
    """6 success + 6 error = 12개 전부 올바른 terminal로 라우팅되어야."""
    rt, nodes = simple_runtime

    success_inputs = [
        "success deployed", "done ok", "completed successfully",
        "build success", "tests ok", "deploy completed",
    ]
    error_inputs = [
        "error timeout", "fatal bug", "build fail",
        "oom error", "test fail", "critical timeout",
    ]

    correct = 0
    for text in success_inputs:
        path = [n.name for n in rt.run(text, entry=nodes["parse"]).route_path]
        if "to_cache" in path and "to_alert" not in path:
            correct += 1
    for text in error_inputs:
        path = [n.name for n in rt.run(text, entry=nodes["parse"]).route_path]
        if "to_alert" in path and "to_cache" not in path:
            correct += 1

    assert correct == 12, f"12/12 기대, 실제 {correct}/12"


@pytest.mark.regression
def test_terminal_log_result_always_reached(simple_runtime):
    """terminal 노드는 정상 경로에서 항상 발화해야 한다."""
    rt, nodes = simple_runtime

    for text in ["success deploy", "error bug"]:
        result = rt.run(text, entry=nodes["parse"])
        assert "log_result" in [n.name for n in result.route_path], \
            f"terminal log_result 미발화: input={text!r}"
