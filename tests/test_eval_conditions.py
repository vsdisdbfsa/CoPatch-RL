from copy import deepcopy

import pytest

from copatch_rl.evaluation.conditions import Condition, validate_pair


def fixture(group="C"):
    model = dict(
        model_revision="synthetic-model-v1",
        prompt_hash="h-prompt",
        temperature=0.0,
        output_limit=128,
        log_version="log-v1",
    )
    return dict(
        group=group,
        layer="function",
        mode="Local",
        offline=True,
        main_base="base-v1",
        tokenizer="tokenizer-v1",
        main_lora="sft-hash",
        sft_origin="sft-hash",
        sub_models={role: deepcopy(model) for role in ("planner", "coder", "reviewer")},
        tools=["read_file", "apply_patch", "run_public_tests", "submit"],
        tool_schema_hash="schema-hash",
        budget=dict(
            max_main_steps=8,
            max_sub_calls=4,
            max_public_tests=2,
            max_code_revisions=2,
            max_generated_tokens=6000,
            wall_time_seconds=180,
        ),
        tasks=[
            dict(
                task_id="t",
                snapshot_hash="snap",
                public_suite_hash="public",
                private_suite_hash="private",
                allowed_files=["solution.py"],
                image_digest="image-digest",
            )
        ],
        seeds=[42],
        harness_version="h-v1",
        test_interface_version="test-v1",
        error_rules_hash="errors",
        termination_hash="termination",
        grader_version="grader-v1",
        statistics_hash="stats",
        decoding_hash="decode",
        entry=dict(
            owner="B",
            reference="harness:dynamic",
            kind="dynamic",
            availability="mock",
            evidence=None,
        ),
    )


@pytest.mark.parametrize(
    "field",
    [
        "main_base",
        "tokenizer",
        "budget",
        "tool_schema_hash",
        "harness_version",
        "grader_version",
        "tasks",
        "seeds",
        "decoding_hash",
    ],
)
def test_missing_rejected(field):
    data = fixture()
    del data[field]
    with pytest.raises(ValueError):
        Condition.model_validate(data)


@pytest.mark.parametrize(
    "field", ["tokenizer", "harness_version", "error_rules_hash", "statistics_hash"]
)
def test_pair_mismatch(field):
    c = Condition.model_validate(fixture())
    data = fixture("D")
    data["main_lora"] = "grpo-hash"
    data[field] = "different"
    with pytest.raises(ValueError, match=field):
        validate_pair(c, Condition.model_validate(data))


def test_budget_mismatch():
    c = Condition.model_validate(fixture())
    data = fixture("D")
    data["budget"]["max_main_steps"] = 9
    with pytest.raises(ValueError, match="budget"):
        validate_pair(c, Condition.model_validate(data))


def test_ask_user_offline():
    data = fixture()
    data["tools"].append("ask_user")
    with pytest.raises(ValueError):
        Condition.model_validate(data)


def test_mock_fixed_entry():
    data = fixture("B")
    data["main_base"] = data["tokenizer"] = data["main_lora"] = data["sft_origin"] = None
    data["entry"].update(kind="fixed", reference="harness:fixed")
    assert Condition.model_validate(data).entry.availability == "mock"


def test_pair_only_lora_changes():
    data = fixture("D")
    data["main_lora"] = "grpo-hash"
    validate_pair(Condition.model_validate(fixture()), Condition.model_validate(data))


@pytest.mark.parametrize("group", ["A", "B", "C", "D"])
@pytest.mark.parametrize("layer,mode", [("function", "Local"), ("repo", "Hybrid")])
def test_condition_variants(group, layer, mode):
    data = fixture(group)
    data.update(layer=layer, mode=mode)
    if group in {"A", "B"}:
        for field in ("main_base", "tokenizer", "main_lora", "sft_origin"):
            data[field] = None
        data["entry"]["kind"] = "single" if group == "A" else "fixed"
    if group == "A":
        data["sub_models"] = {"coder": data["sub_models"]["coder"]}
    assert Condition.model_validate(data).group == group


@pytest.mark.parametrize(
    "fault", ["empty_budget", "empty_version", "missing_sub", "missing_evidence", "empty_tasks"]
)
def test_invalid_nested_fields(fault):
    data = fixture()
    if fault == "empty_budget":
        data["budget"] = {}
    elif fault == "empty_version":
        data["tokenizer"] = " "
    elif fault == "missing_sub":
        del data["sub_models"]["reviewer"]["prompt_hash"]
    elif fault == "missing_evidence":
        data["entry"]["availability"] = "implemented"
    else:
        data["tasks"] = []
    with pytest.raises(ValueError):
        Condition.model_validate(data)


def test_nested_control_mismatch():
    c = Condition.model_validate(fixture())
    data = fixture("D")
    data["tasks"][0]["private_suite_hash"] = "changed"
    with pytest.raises(ValueError, match="tasks"):
        validate_pair(c, Condition.model_validate(data))
