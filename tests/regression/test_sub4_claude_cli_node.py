"""ClaudeCliNode 단위 검증 — subprocess 호출은 mock 으로.

실 `claude -p` 호출은 별도 시연 (시간/비용 비결정적).
"""
from __future__ import annotations

from unittest.mock import patch

import asyncio
import subprocess

from htp.llm.claude_cli_node import ClaudeCliNode
from htp.llm.llm_region      import LLMRegion


def test_claude_cli_node_interface_matches_llm_node():
    """LLMNode 와 동일한 attribute 보유 — LLMRegion drop-in 호환."""
    n = ClaudeCliNode(name="test", system="sys", tags={"t"})
    assert hasattr(n, "name")
    assert hasattr(n, "model")
    assert hasattr(n, "system")
    assert hasattr(n, "tags")
    assert hasattr(n, "_token_log")
    assert callable(n.run)
    assert callable(n.arun)
    assert callable(n.cost_report)


def test_build_prompt_includes_system():
    """system + user data 가 prompt 에 모두 포함."""
    n = ClaudeCliNode(name="t", system="You are X.")
    p = n._build_prompt({"q": "hello"})
    assert "You are X." in p
    assert "q:" in p
    assert "hello" in p


def test_build_prompt_no_system():
    """system 미설정 시 user 만 prompt."""
    n = ClaudeCliNode(name="t", system="")
    p = n._build_prompt("plain")
    assert p == "plain"


def test_env_no_api_key_removes_anthropic_key():
    """ANTHROPIC_API_KEY 가 env 에서 제거됨 (OAuth 강제)."""
    import os
    os.environ.setdefault("ANTHROPIC_API_KEY", "test-value")
    n = ClaudeCliNode(name="t")
    env = n._env_no_api_key()
    assert "ANTHROPIC_API_KEY" not in env


def test_run_mocked_subprocess_success():
    """subprocess 성공 시 stdout parse → dict 반환 + token_log 갱신."""
    n = ClaudeCliNode(name="t")
    fake_result = subprocess.CompletedProcess(
        args=["claude"], returncode=0,
        stdout="interpretation text here", stderr="",
    )
    with patch("subprocess.run", return_value=fake_result):
        result = n.run("anything")
    assert result["text"] == "interpretation text here"
    assert result["label"] == "claude_cli_response"
    assert n.call_count == 1
    assert len(n._token_log) == 1


def test_run_mocked_subprocess_failure_returns_error_dict():
    """subprocess returncode != 0 시 error dict."""
    n = ClaudeCliNode(name="t")
    fake_result = subprocess.CompletedProcess(
        args=["claude"], returncode=1,
        stdout="", stderr="auth failed",
    )
    with patch("subprocess.run", return_value=fake_result):
        result = n.run("anything")
    assert result["label"] == "error"
    assert "rc=1" in result["text"]


def test_run_mocked_subprocess_timeout():
    """timeout 발생 시 timeout dict + call_count 증가."""
    n = ClaudeCliNode(name="t", timeout_sec=0.001)
    with patch("subprocess.run",
                side_effect=subprocess.TimeoutExpired(cmd="claude", timeout=0.001)):
        result = n.run("anything")
    assert result["label"] == "timeout"
    assert n.call_count == 1


def test_run_command_not_found():
    """claude CLI 없으면 error label."""
    n = ClaudeCliNode(name="t")
    with patch("subprocess.run", side_effect=FileNotFoundError("no claude")):
        result = n.run("anything")
    assert result["label"] == "error"
    assert "PATH" in result["text"]


def test_parse_response_json():
    """stdout 이 JSON 이면 그대로 dict 반환."""
    n = ClaudeCliNode(name="t")
    r = n._parse_response('{"interpretation": "hello", "hypothesis": "world"}')
    assert r["interpretation"] == "hello"
    assert r["hypothesis"] == "world"


def test_arun_works_via_executor():
    """arun 도 동작 (async wrapping)."""
    n = ClaudeCliNode(name="t")
    fake_result = subprocess.CompletedProcess(
        args=["claude"], returncode=0,
        stdout="async ok", stderr="",
    )
    with patch("subprocess.run", return_value=fake_result):
        result = asyncio.run(n.arun("anything"))
    assert result["text"] == "async ok"


# ── LLMRegion drop-in 호환 ──────────────────────────

def test_llm_region_accepts_claude_cli_node():
    """LLMRegion 의 llm_node 인자로 ClaudeCliNode 전달 가능."""
    cli = ClaudeCliNode(name="custom_cli")
    region = LLMRegion(
        region_name="r",
        specialty="reasoning",
        llm_node=cli,
    )
    assert region._llm_node is cli


def test_llm_region_default_unchanged_without_llm_node():
    """llm_node 미지정 시 기존 use_mock 분기 그대로."""
    region = LLMRegion(
        region_name="r",
        specialty="reasoning",
        use_mock=True,
    )
    from htp.llm.llm_node import MockLLMNode
    assert isinstance(region._llm_node, MockLLMNode)
