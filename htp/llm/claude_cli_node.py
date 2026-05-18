"""
ClaudeCliNode — `claude -p` subprocess 를 통한 LLM 호출 (OAuth headless).

Anthropic API key 직접 사용 대신 Claude Code CLI 의 OAuth session 활용.
LLMNode 와 동일 interface 만족 — LLMRegion 에 drop-in 으로 교체 가능.

장점:
  - ANTHROPIC_API_KEY 환경변수 불필요 (OAuth 사용)
  - 사용자의 Claude Code 구독 한도 활용
  - 별도 결제 / 키 관리 부담 없음

단점:
  - subprocess 호출이라 latency 가 직접 API 보다 큼 (~수 초)
  - 토큰 사용량 / 비용 정확한 측정 불가 (stderr 미파싱)
  - claude CLI 가 PATH 에 있어야 함

호출:
  $ echo "<prompt>" | env -u ANTHROPIC_API_KEY claude -p --output-format text

DAG: subprocess + os 외 의존 없음. anthropic 패키지 불필요.
"""
from __future__ import annotations

import os
import subprocess
import time
from typing import Any


class ClaudeCliNode:
    """`claude -p` subprocess wrapper. LLMNode interface 만족.

    LLMNode 의 시그니처 (name / model / system / tags / run / arun /
    _token_log / cost_report) 와 호환되어 LLMRegion 의 `_llm_node` 자리에
    drop-in 가능.

    Parameters
    ----------
    name        : 식별자
    model       : (정보용만 — claude CLI 가 내부 default 모델 사용)
    system      : system prompt (prompt 앞에 prepend)
    tags        : set — LLMRegion 호환용
    timeout_sec : subprocess 최대 대기 시간
    extra_args  : `claude -p` 에 추가로 넘길 인자 (예: ["--output-format", "json"])
    """

    def __init__(
        self,
        name:        str,
        model:       str           = "claude-cli-default",
        system:      str           = "",
        tags:        "set | None"  = None,
        timeout_sec: float         = 60.0,
        extra_args:  "list[str] | None" = None,
    ):
        self.name        = name
        self.model       = model
        self.system      = system
        self.tags        = tags or set()
        self.timeout_sec = timeout_sec
        self.extra_args  = list(extra_args or ["--output-format", "text"])

        # 추적 — LLMNode 와 호환
        self.call_count: int = 0
        self.total_cost: float = 0.0
        self.total_ms:   float = 0.0
        # _token_log 는 LLMNode 와 다르게 비용 정보 없음 (CLI 가 안 줌).
        # 형태만 호환: [{"in": None, "out": None, "cost": 0, "ms": ...}]
        self._token_log: "list[dict]" = []

    # ── 내부 헬퍼 ────────────────────────────────────

    def _build_prompt(self, data: Any) -> str:
        """system prompt + user data 를 합쳐 하나의 prompt 로 구성.

        claude -p 는 단일 prompt 만 받으므로 system 을 prepend.
        """
        user_str = (
            "\n".join(f"{k}: {v}" for k, v in data.items())
            if isinstance(data, dict)
            else str(data)
        )
        if self.system:
            return f"{self.system}\n\n---\n\n{user_str}"
        return user_str

    def _env_no_api_key(self) -> "dict[str, str]":
        """ANTHROPIC_API_KEY 제거된 env — OAuth session 강제."""
        env = dict(os.environ)
        env.pop("ANTHROPIC_API_KEY", None)
        return env

    def _track(self, elapsed_ms: float) -> None:
        self.call_count += 1
        self.total_ms   += elapsed_ms
        self._token_log = (self._token_log + [{
            "in":   None,
            "out":  None,
            "cost": 0.0,
            "ms":   elapsed_ms,
        }])[-10:]

    def _parse_response(self, stdout: str) -> dict:
        """stdout 을 LLMNode 와 호환되는 dict 로 변환."""
        text = stdout.strip()
        # JSON 시도 (LLMRegion 이 dict.get("interpretation") 등 사용)
        if text.startswith("{") and text.endswith("}"):
            try:
                import json
                return json.loads(text)
            except Exception:
                pass
        return {"text": text, "label": "claude_cli_response"}

    # ── 공개 interface ─────────────────────────────────

    def run(self, data: Any) -> dict:
        """동기 claude -p subprocess."""
        prompt = self._build_prompt(data)
        cmd = ["claude", "-p", *self.extra_args]
        t0 = time.perf_counter()
        try:
            proc = subprocess.run(
                cmd,
                input=prompt,
                capture_output=True,
                text=True,
                env=self._env_no_api_key(),
                timeout=self.timeout_sec,
            )
        except subprocess.TimeoutExpired:
            elapsed = (time.perf_counter() - t0) * 1000
            self._track(elapsed)
            return {"text": "(claude cli timeout)", "label": "timeout"}
        except FileNotFoundError:
            return {"text": "(claude CLI not in PATH)", "label": "error"}

        elapsed = (time.perf_counter() - t0) * 1000
        self._track(elapsed)
        if proc.returncode != 0:
            return {
                "text":  f"(claude cli rc={proc.returncode}: {proc.stderr[:120]})",
                "label": "error",
            }
        return self._parse_response(proc.stdout)

    async def arun(self, data: Any) -> dict:
        """비동기 — asyncio executor 로 sync run 감쌈.

        진짜 async subprocess 가 필요하면 asyncio.create_subprocess_exec 로
        구현 가능. 현 시점은 시연 목적이라 단순 wrap.
        """
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.run, data)

    def cost_report(self) -> str:
        avg_ms = self.total_ms / max(self.call_count, 1)
        return (
            f"    ClaudeCliNode({self.name})  model={self.model}\n"
            f"      calls={self.call_count}  "
            f"total_cost=N/A (OAuth)  "
            f"avg_ms={avg_ms:.1f}"
        )


__all__ = ["ClaudeCliNode"]
