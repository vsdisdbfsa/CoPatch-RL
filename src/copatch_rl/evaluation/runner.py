"""Resumable evaluation batches over frozen conditions.

One row per attempt is appended to a JSONL journal; a resumed run skips
attempt keys that already exist, so a crash never double-counts a task.
Infrastructure failures are recorded as-is and retried only under an
explicit policy, never rewritten into a model failure.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from copatch_rl.contracts import ErrorCode
from copatch_rl.evaluation.conditions import Condition, TaskEntry
from copatch_rl.evaluation.metrics import GRADER_STATUSES, NOT_SUBMITTED, STATUSES, TaskRecord

INFRA_ERROR = "INFRA_ERROR"
_HOST_FAILURES = {
    ErrorCode.INFRA_ERROR,
    ErrorCode.API_RATE_LIMIT,
    ErrorCode.MODEL_ERROR,
    ErrorCode.TOOL_ERROR,
}

Key = tuple[str, str, str, int, int]


@dataclass(frozen=True)
class Attempt:
    run_id: str
    condition: str
    task_id: str
    seed: int
    attempt: int

    @property
    def key(self) -> Key:
        return (self.run_id, self.condition, self.task_id, self.seed, self.attempt)


@dataclass(frozen=True)
class Outcome:
    """What the injected harness reports after one episode."""

    submitted_snapshot_id: str | None
    initial_hash: str
    terminated_by: str
    cost: float | None
    real: bool
    episode_id: str | None = None

    def ungraded_status(self) -> str | None:
        """Return the terminal status for an unsubmitted episode."""
        if self.terminated_by in (ErrorCode.SUBMITTED, "submit"):
            if not isinstance(self.submitted_snapshot_id, str) or not self.submitted_snapshot_id:
                raise ValueError("submitted outcome requires a snapshot ID")
            return None
        if self.submitted_snapshot_id is not None:
            raise ValueError("unsubmitted outcome cannot contain a submitted snapshot")
        reason = ErrorCode(self.terminated_by)
        if reason in _HOST_FAILURES:
            return INFRA_ERROR
        if reason in (ErrorCode.TIMEOUT, ErrorCode.INVALID_CODE):
            return reason.value
        return NOT_SUBMITTED


@dataclass(frozen=True)
class Row:
    attempt: Attempt
    submitted_snapshot_id: str | None
    initial_hash: str
    terminated_by: str
    status: str
    cost: float | None
    real: bool
    episode_id: str | None = None

    @property
    def key(self) -> Key:
        return self.attempt.key

    def task_record(
        self,
        family: str,
        *,
        initial_pass: bool | None = None,
        public_pass: bool | None = None,
    ) -> TaskRecord:
        """Project a journaled attempt into the report's task/seed schema."""
        return TaskRecord(
            task_id=self.attempt.task_id,
            family=family,
            seed=self.attempt.seed,
            status=self.status,
            initial_pass=initial_pass,
            final_pass=None if self.status == INFRA_ERROR else self.status == "SUCCESS",
            cost=self.cost,
            public_pass=public_pass,
            terminated_by=self.terminated_by,
            episode_id=self.episode_id,
        )

    def dump(self) -> str:
        return json.dumps(
            {
                "run_id": self.attempt.run_id,
                "condition": self.attempt.condition,
                "task_id": self.attempt.task_id,
                "seed": self.attempt.seed,
                "attempt": self.attempt.attempt,
                "submitted_snapshot_id": self.submitted_snapshot_id,
                "initial_hash": self.initial_hash,
                "terminated_by": self.terminated_by,
                "status": self.status,
                "cost": self.cost,
                "real": self.real,
                **({"episode_id": self.episode_id} if self.episode_id is not None else {}),
            },
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )


def _cost(value: int | float | None) -> float | None:
    if value is not None and (
        type(value) not in (int, float) or not math.isfinite(value) or value < 0
    ):
        raise ValueError("cost must be a finite nonnegative amount or null")
    return None if value is None else float(value)


def _parse(line: str) -> Row:
    data = json.loads(line)
    if not isinstance(data, dict):
        raise ValueError("journal row must be an object")
    attempt = Attempt(
        run_id=str(data["run_id"]),
        condition=str(data["condition"]),
        task_id=str(data["task_id"]),
        seed=int(data["seed"]),
        attempt=int(data["attempt"]),
    )
    if data["status"] not in STATUSES:
        raise ValueError("unknown journal status")
    snapshot_id = data["submitted_snapshot_id"]
    if snapshot_id is not None and (not isinstance(snapshot_id, str) or not snapshot_id):
        raise ValueError("invalid submitted snapshot ID")
    cost = data["cost"]
    if isinstance(cost, str):
        cost = float(cost)
    return Row(
        attempt=attempt,
        submitted_snapshot_id=snapshot_id,
        initial_hash=str(data["initial_hash"]),
        terminated_by=str(data["terminated_by"]),
        status=str(data["status"]),
        cost=_cost(cost),
        real=bool(data["real"]),
        episode_id=data.get("episode_id"),
    )


class Journal:
    """Append-only attempt log. Only a truncated final line may be ignored."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._rows: list[Row] = []
        self._by_key: dict[Key, Row] = {}
        if self.path.exists():
            self._load()

    def _load(self) -> None:
        lines = [
            line for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()
        ]
        for index, line in enumerate(lines):
            try:
                row = _parse(line)
            except (ValueError, KeyError, TypeError) as exc:
                if index == len(lines) - 1:
                    return  # interrupted append; the tail is not a record
                raise ValueError("corrupt journal line") from exc
            if row.key in self._by_key:
                raise ValueError("duplicate attempt in journal")
            self._rows.append(row)
            self._by_key[row.key] = row

    def rows(self) -> list[Row]:
        return list(self._rows)

    def get(self, key: Key) -> Row | None:
        return self._by_key.get(key)

    def append(self, row: Row) -> None:
        if row.key in self._by_key:
            raise ValueError("attempt already recorded")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(row.dump() + "\n")
        self._rows.append(row)
        self._by_key[row.key] = row


@dataclass(frozen=True)
class Report:
    rows: tuple[Row, ...]
    totals: dict[str, int]
    infra_error: int
    mock_rows: int
    cost: float | None

    @property
    def unknown_cost_rows(self) -> int:
        return sum(row.cost is None for row in self.rows)

    @property
    def attempts(self) -> int:
        return len(self.rows)


class Runner:
    """Drive one frozen condition; every attempt is journaled before returning."""

    def __init__(
        self,
        *,
        harness: Callable[[Condition, TaskEntry, int], Outcome],
        grader: Callable[[str, str], str],
        journal: Journal,
        max_attempts: int = 1,
        retry_on: Iterable[str] = (INFRA_ERROR,),
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        self._harness = harness
        self._grader = grader
        self._journal = journal
        self._max_attempts = max_attempts
        self._retry_on = frozenset(retry_on)

    def _run_once(
        self, condition: Condition, task: TaskEntry, seed: int, number: int, run_id: str
    ) -> Row:
        outcome = self._harness(condition, task, seed)
        if not isinstance(outcome, Outcome):
            raise TypeError("harness must return an Outcome")
        if outcome.initial_hash != task.snapshot_hash or type(outcome.real) is not bool:
            raise ValueError(
                "harness outcome does not match the frozen task or provenance contract"
            )
        cost = _cost(outcome.cost)
        status = outcome.ungraded_status()
        if outcome.submitted_snapshot_id is not None:
            status = self._grader(outcome.submitted_snapshot_id, task.task_id)
            if status not in GRADER_STATUSES:
                raise ValueError("unknown grader status: " + str(status))
        assert status is not None
        return Row(
            attempt=Attempt(run_id, condition.group, task.task_id, seed, number),
            submitted_snapshot_id=outcome.submitted_snapshot_id,
            initial_hash=outcome.initial_hash,
            terminated_by=outcome.terminated_by,
            status=status,
            cost=cost,
            real=outcome.real,
            episode_id=outcome.episode_id,
        )

    def run(
        self,
        condition: Condition,
        run_id: str,
        *,
        tasks: Sequence[TaskEntry] | None = None,
    ) -> Report:
        chosen = tuple(condition.tasks) if tasks is None else tuple(tasks)
        if [(t.task_id, t.snapshot_hash) for t in chosen] != [
            (t.task_id, t.snapshot_hash) for t in condition.tasks
        ]:
            raise ValueError("task manifest does not match the frozen condition")

        for task in chosen:
            for seed in condition.seeds:
                for number in range(self._max_attempts):
                    key = (run_id, condition.group, task.task_id, seed, number)
                    row = self._journal.get(key)
                    if row is None:
                        row = self._run_once(condition, task, seed, number, run_id)
                        self._journal.append(row)
                    if row.status not in self._retry_on:
                        break
        return self.report(run_id, condition.group)

    def report(self, run_id: str, condition: str) -> Report:
        latest: dict[tuple[str, int], Row] = {}
        for row in self._journal.rows():
            if row.attempt.run_id != run_id or row.attempt.condition != condition:
                continue
            pair = (row.attempt.task_id, row.attempt.seed)
            current = latest.get(pair)
            if current is None or row.attempt.attempt > current.attempt.attempt:
                latest[pair] = row
        rows = tuple(sorted(latest.values(), key=lambda r: (r.attempt.task_id, r.attempt.seed)))
        totals = Counter(row.status for row in rows)
        # The latest row per task/seed drives this report as before. All
        # billed or unknown retry costs remain in the append-only journal.
        costs = tuple(row.cost for row in rows if row.cost is not None)
        return Report(
            rows=rows,
            totals={status: totals.get(status, 0) for status in STATUSES},
            infra_error=totals.get(INFRA_ERROR, 0),
            mock_rows=sum(1 for row in rows if not row.real),
            cost=round(sum(costs), 6) if len(costs) == len(rows) else None,
        )
