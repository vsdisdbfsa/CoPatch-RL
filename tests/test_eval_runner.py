"""C-11: resumable, idempotent evaluation runs; no hidden retries."""

from __future__ import annotations

import json

import pytest

from copatch_rl.evaluation.conditions import Condition, TaskEntry
from copatch_rl.evaluation.runner import (
    INFRA_ERROR,
    Journal,
    Outcome,
    Runner,
)


def _spec() -> dict:
    return {
        "model_revision": "rev",
        "prompt_hash": "p",
        "temperature": 0.0,
        "output_limit": 8,
        "log_version": "v",
    }


def _task(task_id: str, snapshot: str = "snap-h") -> dict:
    return {
        "task_id": task_id,
        "snapshot_hash": snapshot,
        "public_suite_hash": "pub",
        "private_suite_hash": "priv",
        "allowed_files": ("solution.py",),
        "image_digest": "sha256:img",
    }


def _condition(
    group: str = "C",
    *,
    tasks: tuple[dict, ...] | None = None,
    seeds: tuple[int, ...] = (7,),
) -> Condition:
    entry = {
        "owner": "B",
        "reference": "fixed-or-dynamic-workflow",
        "kind": "single" if group == "A" else "fixed" if group == "B" else "dynamic",
        "availability": "mock",
        "evidence": None,
    }
    data = {
        "group": group,
        "layer": "function",
        "mode": "Local",
        "offline": True,
        "main_base": "qwen-4b",
        "tokenizer": "tok",
        "main_lora": "sft_main",
        "sft_origin": "sft_main",
        "sub_models": {role: _spec() for role in ("planner", "coder", "reviewer")},
        "tools": (
            "read_file",
            "write_file",
            "run_public_tests",
            "call_planner",
            "call_coder",
            "call_reviewer",
        ),
        "tool_schema_hash": "tools",
        "budget": {
            "max_main_steps": 4,
            "max_sub_calls": 2,
            "max_public_tests": 2,
            "max_code_revisions": 2,
            "max_generated_tokens": 100,
            "wall_time_seconds": 60,
        },
        "tasks": tasks or (_task("t1"), _task("t2")),
        "seeds": seeds,
        "harness_version": "h1",
        "test_interface_version": "i1",
        "error_rules_hash": "e1",
        "termination_hash": "x1",
        "grader_version": "g1",
        "statistics_hash": "s1",
        "decoding_hash": "d1",
        "entry": entry,
    }
    if group == "D":
        data["main_lora"] = "grpo_main"
    return Condition.model_validate(data)


class Fake:
    """Deterministic harness + grader; statuses are consumed in order."""

    def __init__(self, statuses: list[str] | None = None, *, real: bool = False) -> None:
        self._statuses = list(statuses or [])
        self._real = real
        self.calls: list[tuple[str, int]] = []

    def harness(self, condition: Condition, task: TaskEntry, seed: int) -> Outcome:
        self.calls.append((task.task_id, seed))
        return Outcome(
            submitted_snapshot_id=f"snap-{task.task_id}-{seed}",
            initial_hash=task.snapshot_hash,
            terminated_by="submit",
            cost=0.25,
            real=self._real,
        )

    def grader(self, snapshot_id: str, task_id: str) -> str:
        return self._statuses.pop(0) if self._statuses else "SUCCESS"


def _runner(tmp_path, fake: Fake, **kwargs) -> Runner:
    return Runner(
        harness=fake.harness,
        grader=fake.grader,
        journal=Journal(tmp_path / "journal.jsonl"),
        **kwargs,
    )


def test_runs_each_task_and_seed(tmp_path):
    fake = Fake()
    condition = _condition(seeds=(7, 8))
    report = _runner(tmp_path, fake).run(condition, "run-1")

    assert len(fake.calls) == 4
    assert report.attempts == 4
    assert report.totals["SUCCESS"] == 4


def test_resume_does_not_duplicate(tmp_path):
    fake = Fake()
    condition = _condition()
    journal = Journal(tmp_path / "journal.jsonl")
    first = Runner(harness=fake.harness, grader=fake.grader, journal=journal)

    first.run(condition, "run-1")
    calls = len(fake.calls)
    second = first.run(condition, "run-1")

    assert len(fake.calls) == calls, "resume must not re-execute completed attempts"
    assert second.attempts == 2
    assert len(journal.rows()) == 2


def test_truncated_tail_is_tolerated(tmp_path):
    fake = Fake()
    condition = _condition()
    path = tmp_path / "journal.jsonl"
    _runner(tmp_path, fake).run(condition, "run-1")
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"run_id":"run-1","condition":"C"')  # interrupted append

    journal = Journal(path)
    assert len(journal.rows()) == 2
    report = Runner(harness=fake.harness, grader=fake.grader, journal=journal).run(
        condition, "run-1"
    )
    assert report.attempts == 2


def test_corrupt_middle_line_raises(tmp_path):
    path = tmp_path / "journal.jsonl"
    path.write_text('{"garbage":1}\n{"garbage":2}\n', encoding="utf-8")
    with pytest.raises(ValueError):
        Journal(path)


def test_unknown_grader_status_rejected(tmp_path):
    fake = Fake(["MAYBE"])
    with pytest.raises(ValueError):
        _runner(tmp_path, fake).run(_condition(), "run-1")


def test_infra_retried_when_configured(tmp_path):
    fake = Fake([INFRA_ERROR, "SUCCESS"])
    condition = _condition(tasks=(_task("t1"),), seeds=(1,))
    report = _runner(tmp_path, fake, max_attempts=2).run(condition, "run-1")

    assert report.attempts == 1  # counted once, latest attempt wins
    assert report.totals["SUCCESS"] == 1
    assert report.infra_error == 0
    rows = Journal(tmp_path / "journal.jsonl").rows()
    assert [row.status for row in rows] == [INFRA_ERROR, "SUCCESS"]


def test_infra_not_retried_by_default(tmp_path):
    fake = Fake([INFRA_ERROR])
    condition = _condition(tasks=(_task("t1"),), seeds=(1,))
    report = _runner(tmp_path, fake).run(condition, "run-1")

    assert report.attempts == 1
    assert report.infra_error == 1
    assert report.totals[INFRA_ERROR] == 1
    assert len(fake.calls) == 1


def test_report_counts_latest_attempt_only(tmp_path):
    fake = Fake([INFRA_ERROR, INFRA_ERROR, "TEST_FAILED"])
    condition = _condition(tasks=(_task("t1"),), seeds=(1,))
    report = _runner(tmp_path, fake, max_attempts=3).run(condition, "run-1")

    assert report.attempts == 1
    assert report.totals["TEST_FAILED"] == 1
    assert len(Journal(tmp_path / "journal.jsonl").rows()) == 3


def test_mock_rows_are_counted(tmp_path):
    report = _runner(tmp_path, Fake(real=False)).run(_condition(), "run-1")
    assert report.mock_rows == report.attempts


def test_cost_sum_over_counted_rows(tmp_path):
    report = _runner(tmp_path, Fake()).run(_condition(seeds=(1, 2)), "run-1")
    assert report.attempts == 4
    assert report.cost == pytest.approx(1.0)


def test_manifest_mismatch_rejected(tmp_path):
    condition = _condition()
    other = tuple(
        TaskEntry.model_validate(_task(name, snapshot))
        for name, snapshot in (("t1", "different"), ("t2", "snap-h"))
    )
    with pytest.raises(ValueError):
        _runner(tmp_path, Fake()).run(condition, "run-1", tasks=other)


def test_harness_failure_writes_nothing(tmp_path):
    condition = _condition()

    def boom(*args):
        raise RuntimeError("harness crashed")

    runner = Runner(
        harness=boom,
        grader=lambda snapshot, task: "SUCCESS",
        journal=Journal(tmp_path / "journal.jsonl"),
    )
    with pytest.raises(RuntimeError):
        runner.run(condition, "run-1")
    assert Journal(tmp_path / "journal.jsonl").rows() == []


def test_report_scoped_by_run_id(tmp_path):
    fake = Fake()
    condition = _condition(tasks=(_task("t1"),), seeds=(1,))
    runner = _runner(tmp_path, fake)
    runner.run(condition, "run-1")
    runner.run(condition, "run-2")

    assert runner.report("run-1", "C").attempts == 1
    assert runner.report("run-2", "C").attempts == 1
    assert runner.report("run-3", "C").attempts == 0


def test_rows_sorted_by_task_and_seed(tmp_path):
    fake = Fake()
    condition = _condition(tasks=(_task("t2"), _task("t1")), seeds=(9, 3))
    report = _runner(tmp_path, fake).run(condition, "run-1")

    pairs = [(row.attempt.task_id, row.attempt.seed) for row in report.rows]
    assert pairs == [("t1", 3), ("t1", 9), ("t2", 3), ("t2", 9)]


def test_journal_creates_parent_directory(tmp_path):
    journal = Journal(tmp_path / "nested" / "deep" / "journal.jsonl")
    Runner(
        harness=Fake().harness,
        grader=Fake().grader,
        journal=journal,
    ).run(_condition(tasks=(_task("t1"),)), "run-1")
    assert (tmp_path / "nested" / "deep" / "journal.jsonl").exists()


def test_real_and_mock_rows_mixed(tmp_path):
    real = Fake(real=True)
    mock = Fake(real=False)
    condition = _condition(tasks=(_task("t1"),), seeds=(1,))
    journal = Journal(tmp_path / "journal.jsonl")
    Runner(harness=real.harness, grader=real.grader, journal=journal).run(condition, "run-1")
    Runner(harness=mock.harness, grader=mock.grader, journal=journal).run(condition, "run-2")

    assert (
        Runner(harness=real.harness, grader=real.grader, journal=journal)
        .report("run-1", "C")
        .mock_rows
        == 0
    )
    assert (
        Runner(harness=mock.harness, grader=mock.grader, journal=journal)
        .report("run-2", "C")
        .mock_rows
        == 1
    )


def test_rows_survive_restart(tmp_path):
    fake = Fake()
    condition = _condition(seeds=(1, 2))
    path = tmp_path / "journal.jsonl"
    Runner(harness=fake.harness, grader=fake.grader, journal=Journal(path)).run(condition, "run-1")
    reloaded = Journal(path)
    assert len(reloaded.rows()) == 4
    assert {row.status for row in reloaded.rows()} == {"SUCCESS"}
    json.loads(reloaded.rows()[0].dump())  # rows stay JSON-serializable
