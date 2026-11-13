"""C-14: end-to-end C chain over the hand-written dev example."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from copatch_rl.execution.fixtures import load_suite, load_task, snapshot_files
from copatch_rl.execution.grader import Case, Frozen, Grader, Task
from copatch_rl.execution.public_tests import Binding, Executor
from copatch_rl.execution.sandbox import Limits
from copatch_rl.execution.workspace import Workspace

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "dev" / "count_positive"
IMAGE = "python:3.11-slim@sha256:07d4824036af339fa1b6ef769cc99d9c10a141769538b52b3b597c84f7a7b535"
FIXED = "def count_positive(values):\n    return sum(1 for value in values if value > 0)\n"


def workspace(tmp_path: Path) -> Workspace:
    task = load_task(EXAMPLE / "task.json")
    return Workspace(
        root=tmp_path / "episode",
        workspace_id="w1",
        initial_root=EXAMPLE / "snapshot",
        readable_files=task.allowed_files,
        allowed_files=task.allowed_files,
    )


def test_example_task_and_suite_load():
    task = load_task(EXAMPLE / "task.json")
    suite = load_suite(EXAMPLE / "public_suite.json")

    assert task.task_id == suite.task_id
    assert suite.source_path in task.allowed_files
    assert [case.case_id for case in suite.cases] == ["mixed", "empty"]


def test_public_view_hides_host_only_fields():
    payload = load_task(EXAMPLE / "task.json").public_view().model_dump_json()

    assert "private_suite_id" not in payload
    assert "private/dev_count_positive_001" not in payload
    assert "initial_snapshot" not in payload


def test_snapshot_layout_matches_allowed_files():
    files = snapshot_files(EXAMPLE / "snapshot")
    assert [name for name, _ in files] == ["solution.py"]


def test_initial_example_has_the_boundary_defect():
    content = (EXAMPLE / "snapshot" / "solution.py").read_text(encoding="utf-8")
    assert "value >= 0" in content, "example must stay defective until fixed"


def test_frozen_workspace_rejects_writes(tmp_path):
    ws = workspace(tmp_path)
    snapshot = ws.submit("s0")

    with pytest.raises(PermissionError):
        ws.write_file("solution.py", FIXED)
    assert snapshot.code_hash
    assert ws.source_hash() == snapshot.code_hash


@pytest.mark.skipif(os.environ.get("COPATCH_DOCKER_TESTS") != "1", reason="real Docker opt-in")
def test_function_loop_defect_fix_submit_and_grade(tmp_path):
    task = load_task(EXAMPLE / "task.json")
    suite = load_suite(EXAMPLE / "public_suite.json")
    ws = workspace(tmp_path)
    executor = Executor(
        {"w": Binding(task.task_id, suite.suite_id, ws)},
        {suite.suite_id: suite},
        IMAGE,
        Limits(seconds=10),
    )

    first = executor.run_public_tests("w", suite.suite_id)
    assert first.status == "TEST_FAILED"
    assert (first.passed, first.failed) == (1, 1)
    assert first.code_hash == ws.source_hash()

    ws.write_file("solution.py", FIXED)
    second = executor.run_public_tests("w", suite.suite_id)
    assert second.status == "SUCCESS"
    assert second.passed == 2
    assert second.code_hash == ws.source_hash()

    snapshot = ws.submit("s1")
    grader = Grader(
        {snapshot.snapshot_id: Frozen(task.task_id, snapshot, snapshot.code_hash)},
        {
            task.task_id: Task(
                suite.source_path,
                suite.function,
                tuple(Case(case.args_json, case.expected_json) for case in suite.cases),
            )
        },
        IMAGE,
        Limits(seconds=10),
    )
    assert grader.score(snapshot.snapshot_id, task.task_id).status == "SUCCESS"
