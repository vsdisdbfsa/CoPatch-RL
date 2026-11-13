import json
import os
from dataclasses import replace

import pytest

from copatch_rl.contracts import Budget, TaskSpec
from copatch_rl.data.validate import Assets, validate
from copatch_rl.execution.repository_tests import RepositorySuite
from copatch_rl.execution.sandbox import ExecutionResult, _docker
from copatch_rl.execution.workspace import Snapshot, _hash

NODES = ("public_tests/test_quality.py::test_fix", "public_tests/test_quality.py::test_keep")


def inputs():
    budget = Budget(
        max_main_steps=8,
        max_sub_calls=4,
        max_public_tests=2,
        max_code_revisions=2,
        max_generated_tokens=6000,
        wall_time_seconds=180,
    )
    spec = TaskSpec(
        task_id="t",
        split="train",
        task_family="f",
        difficulty="easy",
        kind="repo_repair",
        source_revision="synthetic",
        description="Handle positive inputs.",
        entrypoint="solution:answer",
        allowed_files=("solution.py",),
        initial_snapshot="bad",
        public_suite_id="pub",
        private_suite_id="private",
        budget=budget,
    )
    row = {
        "task_id": "t",
        "task": spec.model_dump(),
        "license": "synthetic",
        "audit_status": "approved",
        "raw": {"patch": "SECRET_PATCH"},
        "source": "swe-smith",
        "revision": "synthetic",
    }
    assignment = {"task_id": "t", "split": "train"}
    files = (
        (
            "public_tests/test_quality.py",
            b"from solution import answer\n"
            b"def test_fix(): assert answer(1)==2\n"
            b"def test_keep(): assert answer(0)==1",
        ),
    )
    suite = RepositorySuite("private", "t", files, _hash(files), NODES, "sha256:" + "0" * 64)
    snapshots = {}
    for name, source in [
        ("bad", b"def answer(x): return 1"),
        ("ref", b"def answer(x): return x+1"),
        ("wrong", b"def answer(x): return 0"),
    ]:
        entries = (("solution.py", source),)
        snapshots[name] = Snapshot(name, entries, _hash(entries))
    assets = Assets(
        suite,
        snapshots,
        "ref",
        ("wrong",),
        NODES[:1],
        NODES[1:],
        "linux/amd64",
        "sha256:" + "1" * 64,
        "quality-v1",
        True,
        True,
    )
    return row, assignment, assets


def fake(snapshot, *args):
    text = dict(snapshot.files)["solution.py"]
    fail_fix = b"return 1" in text or b"return 0" in text
    fail_keep = b"return 0" in text
    reports = []
    for node, fail in zip(NODES, (fail_fix, fail_keep)):
        reports.extend(
            [
                [node, p, "failed" if fail and p == "call" else "passed", False]
                for p in ("setup", "call", "teardown")
            ]
        )
    report = {
        "collected": list(NODES),
        "reports": reports,
        "collection_errors": 0,
        "exit_code": int(fail_fix or fail_keep),
    }
    return ExecutionResult(
        "EXITED",
        0,
        json.dumps(report).encode(),
        False,
        snapshot.code_hash,
        "sha256:" + "0" * 64,
        "linux/amd64",
    )


@pytest.mark.parametrize(
    "fault,reason",
    [
        ("empty", "EMPTY_SUITE"),
        ("missing", "MISSING_CASE"),
        ("hash", "HASH_MISMATCH"),
        ("snapshot", "SNAPSHOT_UNAVAILABLE"),
        ("leak", "DESCRIPTION_LEAK"),
        ("defect", "DEFECT_NOT_FAILING"),
        ("reference", "REFERENCE_FAILED"),
        ("regression", "P2P_REGRESSION"),
        ("platform", "PLATFORM_MISMATCH"),
    ],
)
def test_rejections(fault, reason):
    row, assignment, a = inputs()
    if fault == "empty":
        a = replace(a, suite=replace(a.suite, nodeids=()))
    elif fault == "missing":
        a = replace(a, f2p=("public_tests/test_quality.py::missing",))
    elif fault == "hash":
        a = replace(a, suite=replace(a.suite, tests_hash="bad"))
    elif fault == "snapshot":
        a = replace(a, snapshots={})
    elif fault == "leak":
        row["task"]["description"] = "SECRET_PATCH"
    elif fault == "defect":
        a.snapshots["bad"] = replace(a.snapshots["ref"], snapshot_id="bad")
    elif fault == "reference":
        a.snapshots["ref"] = replace(a.snapshots["bad"], snapshot_id="ref")
    elif fault == "regression":
        entries = (("solution.py", b"def answer(x): return 0"),)
        a.snapshots["ref"] = Snapshot("ref", entries, _hash(entries))
    else:
        a = replace(a, platform="windows/amd64")
    result = validate(row, assignment, a, runner=fake)
    assert result.decision == "quarantine"
    assert reason in result.reasons


@pytest.mark.parametrize(
    "fault,code",
    [
        ("schema", "INVALID_SCHEMA"),
        ("license", "LICENSE_MISSING"),
        ("split", "SPLIT_CONFLICT"),
        ("deps", "DEPENDENCY_METADATA_INVALID"),
        ("trust", "UNTRUSTED_QA_SOURCE"),
    ],
)
def test_metadata(fault, code):
    row, assignment, a = inputs()
    if fault == "schema":
        row["task"]["schema_version"] = 2
    elif fault == "license":
        row["license"] = ""
    elif fault == "split":
        assignment["split"] = None
    elif fault == "deps":
        a = replace(a, dependency_hash="")
    else:
        a = replace(a, trusted_implementations=False)
    assert code in validate(row, assignment, a, runner=fake).reasons


@pytest.mark.skipif(os.environ.get("COPATCH_DOCKER_TESTS") != "1", reason="real Docker opt-in")
@pytest.mark.parametrize(
    "fault,code",
    [
        ("good", None),
        ("defect", "DEFECT_NOT_FAILING"),
        ("reference", "REFERENCE_FAILED"),
        ("regression", "P2P_REGRESSION"),
        ("wrong", "COUNTEREXAMPLE_NOT_CAUGHT"),
    ],
)
def test_real_quality(fault, code):
    row, assignment, a = inputs()
    image = json.loads(_docker("image", "inspect", "copatch-public-pytest:dev"))[0]["Id"]
    a = replace(a, suite=replace(a.suite, image_id=image))
    if fault == "defect":
        a.snapshots["bad"] = replace(a.snapshots["ref"], snapshot_id="bad")
    elif fault == "reference":
        a.snapshots["ref"] = replace(a.snapshots["bad"], snapshot_id="ref")
    elif fault == "regression":
        a.snapshots["ref"] = replace(a.snapshots["wrong"], snapshot_id="ref")
    elif fault == "wrong":
        a.snapshots["wrong"] = replace(a.snapshots["ref"], snapshot_id="wrong")
    result = validate(row, assignment, a)
    if code is None:
        assert result.decision == "accept", result
        assert len(result.stages) == 3
    else:
        assert result.decision == "quarantine"
        assert code in result.reasons, result
    assert not _docker("ps", "-aq", "--filter", "name=copatch-c03-").strip()
