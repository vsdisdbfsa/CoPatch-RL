"""Internal experiment declarations; no Agent Loop, execution, or config writes."""

from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, StringConstraints, model_validator

from copatch_rl.contracts import Budget

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelSpec(Record):
    model_revision: Text
    prompt_hash: Text
    temperature: Annotated[FiniteFloat, Field(ge=0)]
    output_limit: Annotated[int, Field(strict=True, gt=0)]
    log_version: Text


class TaskEntry(Record):
    task_id: Text
    snapshot_hash: Text
    public_suite_hash: Text
    private_suite_hash: Text
    allowed_files: tuple[Text, ...] = Field(min_length=1)
    image_digest: Text


class Entry(Record):
    owner: Literal["B"]
    reference: Text
    kind: Literal["single", "fixed", "dynamic"]
    availability: Literal["mock", "implemented"]
    evidence: Text | None

    @model_validator(mode="after")
    def check_evidence(self) -> Self:
        if self.availability == "implemented" and self.evidence is None:
            raise ValueError("implemented B entry requires verification evidence")
        return self


class Condition(Record):
    group: Literal["A", "B", "C", "D"]
    layer: Literal["function", "repo"]
    mode: Literal["Local", "Hybrid"]
    offline: bool = Field(strict=True)
    main_base: Text | None
    tokenizer: Text | None
    main_lora: Text | None
    sft_origin: Text | None
    sub_models: dict[str, ModelSpec]
    tools: tuple[Text, ...] = Field(min_length=1)
    tool_schema_hash: Text
    budget: Budget
    tasks: tuple[TaskEntry, ...] = Field(min_length=1)
    seeds: tuple[Annotated[int, Field(strict=True, ge=0)], ...] = Field(min_length=1)
    harness_version: Text
    test_interface_version: Text
    error_rules_hash: Text
    termination_hash: Text
    grader_version: Text
    statistics_hash: Text
    decoding_hash: Text
    entry: Entry

    @model_validator(mode="after")
    def check_condition(self) -> Self:
        expected = "single" if self.group == "A" else "fixed" if self.group == "B" else "dynamic"
        if self.entry.kind != expected:
            raise ValueError("B entry kind does not match condition group")
        roles = {"coder"} if self.group == "A" else {"planner", "coder", "reviewer"}
        if set(self.sub_models) != roles:
            raise ValueError("exact frozen role configurations required")
        main = (self.main_base, self.tokenizer, self.main_lora, self.sft_origin)
        if self.group in {"C", "D"} and any(value is None for value in main):
            raise ValueError("dynamic conditions require Main versions and SFT origin")
        if self.group in {"A", "B"} and any(value is not None for value in main):
            raise ValueError("single/fixed conditions do not use a Main model")
        if self.group == "C" and self.main_lora != self.sft_origin:
            raise ValueError("C must use the common SFT origin")
        if len(set(self.tools)) != len(self.tools) or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("duplicate tools or seeds")
        if self.offline and "ask_user" in self.tools:
            raise ValueError("offline conditions cannot register ask_user")
        if len({task.task_id for task in self.tasks}) != len(self.tasks):
            raise ValueError("duplicate task IDs")
        if self.group == "A" and any(tool.startswith("call_") for tool in self.tools):
            raise ValueError("single Coder condition has no delegation network")
        return self


def validate_pair(c: Condition, d: Condition) -> None:
    """Compare every declared field except group label and Main LoRA identity."""
    # Revalidate to catch changes to nested dictionaries/objects after construction.
    c = Condition.model_validate(c.model_dump())
    d = Condition.model_validate(d.model_dump())
    if (c.group, d.group) != ("C", "D"):
        raise ValueError("expected C and D conditions")
    left = c.model_dump(exclude={"group", "main_lora"})
    right = d.model_dump(exclude={"group", "main_lora"})
    mismatches = sorted(key for key in left if left[key] != right[key])
    if mismatches:
        raise ValueError("C/D controls differ: " + ", ".join(mismatches))
