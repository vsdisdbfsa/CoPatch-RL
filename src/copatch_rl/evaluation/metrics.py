"""Correctness, cost and paired statistics with explicit denominators.

Every ratio is None (NA) when its denominator is empty. Infrastructure
failures are reported separately and never folded into model failures.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass

INFRA = "INFRA_ERROR"
INVALID = frozenset({"INVALID_CODE", "TIMEOUT"})
STATUSES = ("SUCCESS", "TEST_FAILED", "INVALID_CODE", "TIMEOUT", "INFRA_ERROR")


@dataclass(frozen=True)
class TaskRecord:
    """One evaluated task/seed. `None` means "not measured", never zero."""

    task_id: str
    family: str
    seed: int
    status: str
    initial_pass: bool | None
    final_pass: bool | None
    cost: float | None = None
    public_pass: bool | None = None

    @property
    def key(self) -> tuple[str, int]:
        return (self.task_id, self.seed)


@dataclass(frozen=True)
class Summary:
    tasks: int
    evaluated: int
    success: int
    infra: int
    final_all_pass: float | None
    initial_pass: float | None
    fix_rate: float | None
    regression_rate: float | None
    invalid_timeout: float | None
    public_private_gap: float | None
    cost_per_attempt: float | None
    cost_per_success: float | None


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _rate(records: Sequence[TaskRecord], attr: str) -> tuple[int, int]:
    values = [getattr(record, attr) for record in records]
    known = [value for value in values if value is not None]
    return sum(1 for value in known if value), len(known)


def summarize(records: Sequence[TaskRecord]) -> Summary:
    rows = list(records)
    keys = [row.key for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate task/seed record")
    unknown = [row.status for row in rows if row.status not in STATUSES]
    if unknown:
        raise ValueError("unknown status: " + unknown[0])

    evaluated = [row for row in rows if row.final_pass is not None]
    success = sum(1 for row in evaluated if row.final_pass)
    infra = sum(1 for row in rows if row.status == INFRA)
    invalid = sum(1 for row in rows if row.status in INVALID)

    initial_known = [row for row in rows if row.initial_pass is not None]
    fixed = [
        row for row in initial_known if row.initial_pass is False and row.final_pass is not None
    ]
    regressed = [
        row for row in initial_known if row.initial_pass is True and row.final_pass is not None
    ]

    paired = [row for row in evaluated if row.public_pass is not None]
    public_rate = _ratio(sum(1 for row in paired if row.public_pass), len(paired))
    private_rate = _ratio(sum(1 for row in paired if row.final_pass), len(paired))
    gap = None if public_rate is None or private_rate is None else public_rate - private_rate

    costs = [row.cost for row in rows]
    complete_costs = all(cost is not None for cost in costs)
    total_cost = sum(cost for cost in costs if cost is not None)

    return Summary(
        tasks=len(rows),
        evaluated=len(evaluated),
        success=success,
        infra=infra,
        final_all_pass=_ratio(success, len(evaluated)),
        initial_pass=_ratio(
            sum(1 for row in initial_known if row.initial_pass), len(initial_known)
        ),
        fix_rate=_ratio(sum(1 for row in fixed if row.final_pass), len(fixed)),
        regression_rate=_ratio(sum(1 for row in regressed if not row.final_pass), len(regressed)),
        invalid_timeout=_ratio(invalid, len(rows)),
        public_private_gap=gap,
        cost_per_attempt=round(total_cost / len(rows), 6) if complete_costs and rows else None,
        cost_per_success=(round(total_cost / success, 6) if complete_costs and success else None),
    )


def by_group(records: Sequence[TaskRecord], key: Callable[[TaskRecord], str]) -> dict[str, Summary]:
    groups: dict[str, list[TaskRecord]] = {}
    for record in records:
        groups.setdefault(key(record), []).append(record)
    return {name: summarize(rows) for name, rows in sorted(groups.items())}


def paired_family_bootstrap(
    left: Sequence[TaskRecord],
    right: Sequence[TaskRecord],
    *,
    draws: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Family-level paired bootstrap of `right - left` all-pass rate.

    Families are resampled with replacement; tasks inside a family are not
    treated as independent samples.
    """
    if draws < 1:
        raise ValueError("draws must be positive")
    lmap = {row.key: row for row in left}
    rmap = {row.key: row for row in right}
    if set(lmap) != set(rmap):
        raise ValueError("paired records required")

    families: dict[str, list[tuple[str, int]]] = {}
    for key, row in lmap.items():
        families.setdefault(row.family, []).append(key)

    diffs: list[float] = []
    for keys in families.values():
        lrate = _ratio(
            sum(1 for key in keys if lmap[key].final_pass),
            sum(1 for key in keys if lmap[key].final_pass is not None),
        )
        rrate = _ratio(
            sum(1 for key in keys if rmap[key].final_pass),
            sum(1 for key in keys if rmap[key].final_pass is not None),
        )
        if lrate is None or rrate is None:
            continue
        diffs.append(rrate - lrate)
    if not diffs:
        raise ValueError("no comparable family")

    observed = sum(diffs) / len(diffs)
    rng = random.Random(seed)
    samples = sorted(sum(rng.choices(diffs, k=len(diffs))) / len(diffs) for _ in range(draws))
    low = samples[int(draws * alpha / 2)]
    high = samples[min(draws - 1, int(draws * (1 - alpha / 2)))]
    return round(observed, 6), round(low, 6), round(high, 6)
