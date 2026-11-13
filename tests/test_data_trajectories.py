from copy import deepcopy

import pytest

from copatch_rl.data.trajectories import validate


def fixture():
    base = dict(
        synthetic=True,
        episode_id="e",
        task_id="t",
        workspace_id="w",
        group_id="g",
        policy_version="p",
        initial_hash="h0",
        role="main",
    )
    return [
        dict(base, event_id="a", kind="action", sequence=0, timestamp=1, code_hash="h0"),
        dict(
            base,
            event_id="o",
            kind="observation",
            sequence=1,
            timestamp=2,
            code_hash="h1",
            action_id="a",
        ),
    ]


def history():
    return {("e", 0): "h0", ("e", 1): "h1"}


@pytest.mark.parametrize(
    "fault,code",
    [
        ("missing", "UNPAIRED_ACTION"),
        ("order", "EVENT_ORDER"),
        ("time", "OBSERVATION_BEFORE_ACTION"),
        ("episode", "PAIR_IDENTITY"),
        ("hash", "STALE_HASH"),
        ("label", "INVALID_SFT_LABEL"),
    ],
)
def test_failures(fault, code):
    events = fixture()
    labels = ("a",)
    if fault == "missing":
        events.pop()
    elif fault == "order":
        events.reverse()
    elif fault == "time":
        events[1]["timestamp"] = 0
    elif fault == "episode":
        events[1]["episode_id"] = "other"
    elif fault == "hash":
        events[1]["code_hash"] = "h0"
    else:
        labels = ("o",)
    assert code in validate(events, history(), mode="sft", label_ids=labels).errors


def test_missing_sampling_not_filled():
    events = fixture()
    original = deepcopy(events)
    result = validate(events, history(), mode="sft", label_ids=("a",))
    assert result.valid
    assert result.sampling["a"] == "unavailable"
    assert events == original


def group():
    events = []
    hashes = {}
    for i in range(4):
        pair = fixture()
        for e in pair:
            e["episode_id"] = f"e{i}"
            e["workspace_id"] = f"w{i}"
            e["event_id"] += str(i)
        pair[1]["action_id"] = "a" + str(i)
        events.extend(pair)
        hashes.update({(f"e{i}", 0): "h0", (f"e{i}", 1): "h1"})
    return events, hashes


@pytest.mark.parametrize(
    "fault,code",
    [
        ("count", "GROUP_SIZE"),
        ("workspace", "WORKSPACE_REUSED"),
        ("task", "GROUP_IDENTITY"),
        ("policy", "GROUP_IDENTITY"),
    ],
)
def test_grpo_failures(fault, code):
    events, hashes = group()
    if fault == "count":
        events = events[:-2]
    elif fault == "workspace":
        for e in events[-2:]:
            e["workspace_id"] = "w0"
    else:
        for e in events[-2:]:
            e["task_id" if fault == "task" else "policy_version"] = "different"
    assert code in validate(events, hashes, mode="grpo").errors


def test_valid_group_synthetic():
    events, hashes = group()
    result = validate(events, hashes, mode="grpo")
    assert result.valid and result.synthetic
    assert all(value == "unavailable" for value in result.sampling.values())


def test_stale_feedback_and_future_context():
    events = fixture()
    next_action = dict(
        events[0],
        event_id="a2",
        sequence=2,
        timestamp=3,
        code_hash="h2",
        visible_observation_ids=["o"],
    )
    next_obs = dict(
        events[1], event_id="o2", action_id="a2", sequence=3, timestamp=4, code_hash="h2"
    )
    hashes = {**history(), ("e", 2): "h2", ("e", 3): "h2"}
    assert "STALE_FEEDBACK" in validate(events + [next_action, next_obs], hashes, mode="sft").errors
    events[0]["visible_observation_ids"] = ["o"]
    assert "FUTURE_FEEDBACK" in validate(events, history(), mode="sft").errors


def test_subagent_label_rejected():
    events = fixture()
    events[0]["role"] = "coder"
    assert "INVALID_SFT_LABEL" in validate(events, history(), mode="sft", label_ids=["a"]).errors


@pytest.mark.parametrize("missing", ["input_ids", "generated_ids", "old_logprobs", "loss_mask"])
def test_partial_sampling_unavailable(missing):
    events = fixture()
    events[0].update(input_ids=[1], generated_ids=[2], old_logprobs=[-0.5], loss_mask=[True])
    del events[0][missing]
    before = deepcopy(events)
    result = validate(events, history(), mode="sft")
    assert result.sampling["a"] == "unavailable"
    assert events == before


def test_duplicate_pair_rejected():
    events = fixture()
    duplicate = dict(events[1], event_id="o2", sequence=2, timestamp=3)
    assert (
        "DUPLICATE_OBSERVATION"
        in validate(events + [duplicate], {**history(), ("e", 2): "h1"}, mode="sft").errors
    )
