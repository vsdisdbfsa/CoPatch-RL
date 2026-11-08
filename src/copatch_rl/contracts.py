"""Versioned data contracts shared by the Harness, execution and training modules."""

from collections.abc import Sequence
from enum import StrEnum
from typing import Annotated, Literal, Protocol, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    JsonValue,
    NonNegativeInt,
    PositiveInt,
    StringConstraints,
    model_validator,
)

CONTRACT_VERSION = 1
NonEmpty = Annotated[str, StringConstraints(min_length=1)]
LogProbability = Annotated[FiniteFloat, Field(le=0)]


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Role(StrEnum):
    MAIN = "main"
    PLANNER = "planner"
    CODER = "coder"
    REVIEWER = "reviewer"


class ErrorCode(StrEnum):
    INVALID_ACTION = "INVALID_ACTION"
    PERMISSION_DENIED = "PERMISSION_DENIED"
    MODEL_ERROR = "MODEL_ERROR"
    API_RATE_LIMIT = "API_RATE_LIMIT"
    TOOL_ERROR = "TOOL_ERROR"
    TEST_FAILED = "TEST_FAILED"
    INVALID_CODE = "INVALID_CODE"
    TIMEOUT = "TIMEOUT"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    INFRA_ERROR = "INFRA_ERROR"
    SUBMITTED = "SUBMITTED"


class Budget(ContractModel):
    max_main_steps: PositiveInt
    max_sub_calls: NonNegativeInt
    max_public_tests: NonNegativeInt
    max_code_revisions: NonNegativeInt
    max_generated_tokens: PositiveInt
    wall_time_seconds: PositiveInt


class _TaskFields(ContractModel):
    task_id: NonEmpty
    description: NonEmpty
    entrypoint: NonEmpty
    allowed_files: tuple[NonEmpty, ...]
    public_suite_id: NonEmpty
    budget: Budget


class PublicTaskSpec(_TaskFields):
    """Only the task fields allowed in model input."""


class TaskSpec(_TaskFields):
    """Host task metadata. Use public_view() when building a model context."""

    schema_version: Literal[1] = CONTRACT_VERSION
    split: Literal["train", "dev", "test", "smoke"]
    task_family: NonEmpty
    difficulty: NonEmpty
    kind: NonEmpty
    source_revision: NonEmpty
    initial_snapshot: NonEmpty = Field(repr=False)
    private_suite_id: NonEmpty = Field(repr=False)

    def public_view(self) -> PublicTaskSpec:
        public_fields = set(PublicTaskSpec.model_fields)
        return PublicTaskSpec.model_validate(self.model_dump(include=public_fields))


class ToolName(StrEnum):
    CALL_PLANNER = "call_planner"
    CALL_CODER = "call_coder"
    CALL_REVIEWER = "call_reviewer"
    READ_FILE = "read_file"
    APPLY_PATCH = "apply_patch"
    RUN_PUBLIC_TESTS = "run_public_tests"
    SUBMIT = "submit"


class _PlannerArguments(ContractModel):
    goal: NonEmpty
    task_context: str = ""


class _CoderArguments(ContractModel):
    instruction: NonEmpty
    files: list[NonEmpty]
    feedback: str = ""


class _ReviewerArguments(ContractModel):
    code_hash: NonEmpty
    focus: NonEmpty


class _FileArguments(ContractModel):
    path: NonEmpty


class _PatchArguments(_FileArguments):
    patch: NonEmpty


class _TestArguments(ContractModel):
    test_suite_id: NonEmpty


class _SubmitArguments(ContractModel):
    reason: NonEmpty


_ARGUMENT_MODELS: dict[ToolName, type[ContractModel]] = {
    ToolName.CALL_PLANNER: _PlannerArguments,
    ToolName.CALL_CODER: _CoderArguments,
    ToolName.CALL_REVIEWER: _ReviewerArguments,
    ToolName.READ_FILE: _FileArguments,
    ToolName.APPLY_PATCH: _PatchArguments,
    ToolName.RUN_PUBLIC_TESTS: _TestArguments,
    ToolName.SUBMIT: _SubmitArguments,
}


class Action(ContractModel):
    """A validated tool request. The dispatcher enforces scope and permissions."""

    tool_name: ToolName
    arguments: dict[str, JsonValue]
    main_step: NonNegativeInt

    @model_validator(mode="after")
    def check_arguments(self) -> Self:
        _ARGUMENT_MODELS[self.tool_name].model_validate(self.arguments)
        return self

    @staticmethod
    def argument_schema(tool_name: ToolName) -> dict[str, JsonValue]:
        return _ARGUMENT_MODELS[tool_name].model_json_schema()


class Observation(ContractModel):
    """Public feedback produced by a tool or a delegated role."""

    status: Literal["ok", "error"]
    summary: str
    code_hash: NonEmpty
    remaining_steps: NonNegativeInt
    elapsed_ms: NonNegativeInt
    data: dict[str, JsonValue] = Field(default_factory=dict)
    error_code: ErrorCode | None = None

    @model_validator(mode="after")
    def check_status(self) -> Self:
        if (self.status == "error") != (self.error_code is not None):
            raise ValueError("an error code is required exactly when status is error")
        if self.error_code == ErrorCode.SUBMITTED:
            raise ValueError("submission is a terminal status, not an error")
        return self


class RunConfig(ContractModel):
    mode: Literal["local", "hybrid"]
    policy_version: NonEmpty
    seed: NonNegativeInt


class Message(ContractModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str


class TraceContext(ContractModel):
    episode_id: NonEmpty
    session_id: NonEmpty
    main_step: NonNegativeInt
    policy_version: NonEmpty


class TokenUsage(ContractModel):
    input_tokens: NonNegativeInt
    output_tokens: NonNegativeInt


class MainGeneration(ContractModel):
    """Exact token sequence and aligned sampling data for a single Main turn."""

    main_step: NonNegativeInt
    input_ids: tuple[NonNegativeInt, ...] = Field(min_length=1)
    generated_ids: tuple[NonNegativeInt, ...] = Field(min_length=1)
    old_logprobs: tuple[LogProbability, ...]
    loss_mask: tuple[bool, ...]
    generation_params: dict[str, JsonValue]

    @model_validator(mode="after")
    def check_token_alignment(self) -> Self:
        generated_count = len(self.generated_ids)
        if len(self.old_logprobs) != generated_count or len(self.loss_mask) != generated_count:
            raise ValueError("generated_ids, old_logprobs and loss_mask must have the same length")
        return self


class ModelResponse(ContractModel):
    requested_model: NonEmpty
    returned_model: NonEmpty
    model_revision: NonEmpty | None = None
    text: str = ""
    raw_tool_call: JsonValue = None
    action: Action | None = None
    stop_reason: NonEmpty
    usage: TokenUsage | None = None
    elapsed_ms: NonNegativeInt
    main_generation: MainGeneration | None = None


class ModelClient(Protocol):
    """Provider adapters receive public messages and return normalized output."""

    def generate(
        self,
        role: Role,
        messages: Sequence[Message],
        tools: Sequence[ToolName],
        config: RunConfig,
        trace_ctx: TraceContext,
    ) -> ModelResponse: ...


class GraderResult(ContractModel):
    """Terminal grading result. Only host evaluation and training consume it."""

    task_id: NonEmpty
    submitted_snapshot_id: NonEmpty
    status: Literal[
        "SUCCESS",
        ErrorCode.TEST_FAILED,
        ErrorCode.INVALID_CODE,
        ErrorCode.TIMEOUT,
        ErrorCode.INFRA_ERROR,
    ]


class EpisodeRecord(ContractModel):
    """Host-only record. A submission may still fail independent grading."""

    schema_version: Literal[1] = CONTRACT_VERSION
    task_id: NonEmpty
    run_id: NonEmpty
    episode_id: NonEmpty
    root_session_id: NonEmpty
    group_id: NonEmpty | None = None
    policy_version: NonEmpty
    model_revisions: dict[Role, NonEmpty] = Field(default_factory=dict)
    event_ids: tuple[NonEmpty, ...] = ()
    final_snapshot_id: NonEmpty | None = None
    ended_by: ErrorCode
    grade: GraderResult | None = None
    reward: FiniteFloat | None = None
    cost_usd: Annotated[FiniteFloat, Field(ge=0)] | None = None

    @model_validator(mode="after")
    def check_grade_identity(self) -> Self:
        if self.grade is not None and (
            self.grade.task_id != self.task_id
            or self.grade.submitted_snapshot_id != self.final_snapshot_id
        ):
            raise ValueError("grade must match the episode task and final snapshot")
        return self


class Trajectory(ContractModel):
    """Training-side rollout bound to its initial state and old policy."""

    episode: EpisodeRecord
    initial_snapshot: NonEmpty
    env_seed: NonNegativeInt
    main_generations: tuple[MainGeneration, ...]
    observations: tuple[Observation, ...]

    @model_validator(mode="after")
    def check_group(self) -> Self:
        if self.episode.group_id is None:
            raise ValueError("a training trajectory must belong to a rollout group")
        return self
