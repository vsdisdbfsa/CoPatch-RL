"""C-12: metrics denominators, NA policy, cost and paired bootstrap."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from copatch_rl.evaluation.metrics import (
    Summary,
    TaskRecord,
    by_group,
    paired_family_bootstrap,
    summarize,
)


def row(
    task_id: str,
    *,
    family: str = "f1",
    seed: int = 1,
    status: str = "SUCCESS",
    initial: bool | None = True,
    final: bool | None = True,
    cost: float | None = 2.0,
    public: bool | None = None,
) -> TaskRecord:
    return TaskRecord(
        task_id=task_id,
        family=family,
        seed=seed,
        status=status,
        initial_pass=initial,
        final_pass=final,
        cost=cost,
        public_pass=public,
    )


def test_hand_computed_ratios():
    # 初版 [T,T,F,F]，最终 [T,F,T,F]：四个正确性比率都是 50%
    records = [
        row("1", initial=True, final=True),
        row("2", initial=True, final=False),
        row("3", initial=False, final=True),
        row("4", initial=False, final=False),
    ]
    summary = summarize(records)

    assert summary.tasks == 4
    assert summary.final_all_pass == pytest.approx(0.5)
    assert summary.initial_pass == pytest.approx(0.5)
    assert summary.fix_rate == pytest.approx(0.5)  # 初版错 2 个，修好 1 个
    assert summary.regression_rate == pytest.approx(0.5)  # 初版对 2 个，改坏 1 个


def test_cost_per_attempt_and_success():
    # 费用 8、四题、两成功
    records = [
        row("1", cost=2.0),
        row("2", cost=2.0),
        row("3", cost=2.0, final=False),
        row("4", cost=2.0, final=False),
    ]
    summary = summarize(records)
    assert summary.cost_per_attempt == pytest.approx(2.0)
    assert summary.cost_per_success == pytest.approx(4.0)


def test_zero_success_cost_is_na():
    summary = summarize([row("1", final=False)])
    assert summary.cost_per_success is None


def test_missing_initial_is_na():
    summary = summarize([row("1", initial=None, final=True)])
    assert summary.initial_pass is None
    assert summary.fix_rate is None
    assert summary.regression_rate is None


def test_missing_cost_is_na_not_zero():
    summary = summarize([row("1", cost=None), row("2", cost=2.0)])
    assert summary.cost_per_attempt is None
    assert summary.cost_per_success is None


def test_duplicate_record_rejected():
    with pytest.raises(ValueError):
        summarize([row("1"), row("1")])


def test_unknown_status_rejected():
    with pytest.raises(ValueError):
        summarize([row("1", status="MAYBE")])


def test_infra_excluded_from_denominator():
    records = [
        row("1", final=True),
        row("2", status="INFRA_ERROR", final=None),
    ]
    summary = summarize(records)
    assert summary.tasks == 2
    assert summary.evaluated == 1
    assert summary.infra == 1
    assert summary.final_all_pass == pytest.approx(1.0)


def test_invalid_timeout_rate():
    records = [
        row("1", status="INVALID_CODE", final=False),
        row("2", status="TIMEOUT", final=False),
        row("3", final=True),
        row("4", final=True),
    ]
    assert summarize(records).invalid_timeout == pytest.approx(0.5)


def test_public_private_gap():
    records = [
        row("1", public=True, final=True),
        row("2", public=True, final=False),
        row("3", public=True, final=True),
        row("4", public=False, final=False),
    ]
    summary = summarize(records)
    assert summary.public_private_gap == pytest.approx(0.25)


def test_by_group_splits_on_family():
    records = [
        row("1", family="a"),
        row("2", family="a", final=False),
        row("3", family="b"),
    ]
    groups = by_group(records, lambda record: record.family)
    assert set(groups) == {"a", "b"}
    assert groups["a"].final_all_pass == pytest.approx(0.5)
    assert groups["b"].final_all_pass == pytest.approx(1.0)


def test_empty_records_is_na():
    summary = summarize([])
    assert summary.tasks == 0
    assert summary.final_all_pass is None
    assert summary.cost_per_attempt is None


def test_bootstrap_identical_groups():
    left = [row("1", family="a"), row("2", family="b")]
    right = [row("1", family="a"), row("2", family="b")]
    observed, low, high = paired_family_bootstrap(left, right, draws=200, seed=1)
    assert observed == 0.0
    assert low == 0.0 and high == 0.0


def test_bootstrap_detects_improvement():
    left = [row("1", family="a", final=False), row("2", family="b", final=False)]
    right = [row("1", family="a", final=True), row("2", family="b", final=True)]
    observed, low, high = paired_family_bootstrap(left, right, draws=200, seed=1)
    assert observed == pytest.approx(1.0)
    assert 0 < low <= high


def test_bootstrap_requires_paired_keys():
    with pytest.raises(ValueError):
        paired_family_bootstrap([row("1")], [row("2")], draws=10)


def test_bootstrap_is_deterministic():
    left = [
        row("1", family="a", final=False),
        row("2", family="b", final=True),
        row("3", family="c", final=False),
    ]
    right = [
        row("1", family="a", final=True),
        row("2", family="b", final=True),
        row("3", family="c", final=False),
    ]
    first = paired_family_bootstrap(left, right, draws=100, seed=7)
    second = paired_family_bootstrap(left, right, draws=100, seed=7)
    assert first == second


def test_summary_is_immutable():
    summary = summarize([row("1")])
    with pytest.raises(FrozenInstanceError):
        summary.tasks = 99  # type: ignore[misc]
    assert isinstance(summary, Summary)
