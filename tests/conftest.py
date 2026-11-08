import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from pydantic import JsonValue

from copatch_rl.contracts import (
    Action,
    Message,
    ModelClient,
    ModelResponse,
    Role,
    RunConfig,
    ToolName,
    TraceContext,
)


@dataclass
class FakeModelClient:
    """Return hand-authored Main actions without invoking a provider or executing them."""

    actions: list[dict[str, JsonValue]]

    def generate(
        self,
        role: Role,
        messages: Sequence[Message],
        tools: Sequence[ToolName],
        config: RunConfig,
        trace_ctx: TraceContext,
    ) -> ModelResponse:
        raw = self.actions[trace_ctx.main_step]
        action = Action.model_validate({**raw, "main_step": trace_ctx.main_step})
        if role != Role.MAIN or action.tool_name not in tools:
            raise ValueError("fixture action was not offered to this role")
        return ModelResponse(
            requested_model="mock",
            returned_model="mock",
            model_revision="fixture-v1",
            raw_tool_call=raw,
            action=action,
            stop_reason="tool_call",
            elapsed_ms=0,
        )


@pytest.fixture
def bootstrap_cases() -> dict[str, Any]:
    fixture = Path(__file__).parent / "fixtures" / "bootstrap_cases.json"
    return json.loads(fixture.read_text(encoding="utf-8"))


@pytest.fixture
def fake_model_client(bootstrap_cases: dict[str, Any]) -> ModelClient:
    return FakeModelClient(bootstrap_cases["actions"])
