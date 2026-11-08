from typing import Any

import pytest
from pydantic import ValidationError

from copatch_rl.contracts import (
    CONTRACT_VERSION,
    Action,
    EpisodeRecord,
    ErrorCode,
    GraderResult,
    MainGeneration,
    Message,
    ModelClient,
    Observation,
    PublicTaskSpec,
    Role,
    RunConfig,
    TaskSpec,
    ToolName,
    TraceContext,
)


def test_task_public_view_excludes_host_fields(bootstrap_cases: dict[str, Any]) -> None:
    task = TaskSpec.model_validate(bootstrap_cases["task"])
    view = task.public_view()

    assert task.schema_version == CONTRACT_VERSION
    assert type(view) is PublicTaskSpec
    assert set(view.model_dump()) == {
        "task_id",
        "description",
        "entrypoint",
        "allowed_files",
        "public_suite_id",
        "budget",
    }
    assert task.private_suite_id not in view.model_dump_json()
    assert task.initial_snapshot not in view.model_dump_json()
    assert task.private_suite_id not in repr(task)
    assert view.budget == task.budget
    assert view.budget is not task.budget
    with pytest.raises(ValidationError):
        PublicTaskSpec.model_validate(task)


def test_fake_model_client_produces_legal_action_and_observation(
    bootstrap_cases: dict[str, Any], fake_model_client: ModelClient
) -> None:
    task = TaskSpec.model_validate(bootstrap_cases["task"]).public_view()
    config = RunConfig(mode="local", policy_version="mock-v1", seed=42)
    trace = TraceContext(
        episode_id="ep-1", session_id="main-1", main_step=0, policy_version=config.policy_version
    )
    response = fake_model_client.generate(
        Role.MAIN,
        [Message(role="user", content=task.model_dump_json())],
        [ToolName.READ_FILE],
        config,
        trace,
    )
    observation = Observation.model_validate(bootstrap_cases["observation"])

    assert response.action == Action(
        tool_name=ToolName.READ_FILE, arguments={"path": "solution.py"}, main_step=0
    )
    assert Action.argument_schema(ToolName.READ_FILE)["required"] == ["path"]
    assert response.requested_model == response.returned_model == "mock"
    assert response.main_generation is None
    assert observation.status == "ok"
    assert observation.remaining_steps == task.budget.max_main_steps - 1
    assert observation.data["path"] == response.action.arguments["path"]


@pytest.mark.parametrize(("main_step", "message"), [(1, "Field required"), (2, "run_shell")])
def test_fake_model_client_rejects_malformed_and_unknown_tools(
    fake_model_client: ModelClient, main_step: int, message: str
) -> None:
    config = RunConfig(mode="local", policy_version="mock-v1", seed=42)
    trace = TraceContext(
        episode_id="ep-1",
        session_id="main-1",
        main_step=main_step,
        policy_version=config.policy_version,
    )
    with pytest.raises(ValidationError, match=message):
        fake_model_client.generate(Role.MAIN, [], [ToolName.READ_FILE], config, trace)


@pytest.mark.parametrize("error_code", [None, ErrorCode.SUBMITTED])
def test_error_observation_requires_a_failure_code(error_code: ErrorCode | None) -> None:
    with pytest.raises(ValidationError):
        Observation(
            status="error",
            summary="",
            code_hash="fixture-hash",
            remaining_steps=0,
            elapsed_ms=0,
            error_code=error_code,
        )


def test_submission_and_grading_are_separate() -> None:
    grade = GraderResult(
        task_id="fixture", submitted_snapshot_id="snapshot", status=ErrorCode.TEST_FAILED
    )
    episode = EpisodeRecord(
        task_id="fixture",
        run_id="run-1",
        episode_id="ep-1",
        root_session_id="main-1",
        policy_version="mock-v1",
        final_snapshot_id="snapshot",
        ended_by=ErrorCode.SUBMITTED,
        grade=grade,
        reward=0.0,
    )

    assert episode.ended_by == ErrorCode.SUBMITTED
    assert episode.grade is not None
    assert episode.grade.status == ErrorCode.TEST_FAILED
    with pytest.raises(ValidationError):
        GraderResult.model_validate({**grade.model_dump(), "status": ErrorCode.SUBMITTED})

    mismatched = episode.model_dump()
    mismatched["final_snapshot_id"] = "another-snapshot"
    with pytest.raises(ValidationError, match="grade must match"):
        EpisodeRecord.model_validate(mismatched)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"old_logprobs": (-1.0,)}, "must have the same length"),
        ({"loss_mask": (True,)}, "must have the same length"),
        ({"old_logprobs": (0.1, -1.0)}, "less than or equal to 0"),
    ],
)
def test_main_generation_rejects_invalid_sampling_data(
    changes: dict[str, Any], message: str
) -> None:
    generation = MainGeneration(
        main_step=0,
        input_ids=(1,),
        generated_ids=(2, 3),
        old_logprobs=(-1.0, -1.0),
        loss_mask=(True, False),
        generation_params={"temperature": 0.7},
    )
    with pytest.raises(ValidationError, match=message):
        MainGeneration.model_validate({**generation.model_dump(), **changes})
