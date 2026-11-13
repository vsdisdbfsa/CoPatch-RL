"""C-13: six deterministic artifacts, reconciliation, mock banner, redaction."""

from __future__ import annotations

import csv
import json

import pytest

from copatch_rl.evaluation.metrics import TaskRecord, summarize
from copatch_rl.evaluation.reports import write_report


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


RECORDS = (
    row("t1", family="alpha", final=True),
    row("t2", family="alpha", final=False, status="TEST_FAILED"),
    row("t3", family="beta", final=None, status="INFRA_ERROR"),
)


def report_dir(tmp_path, records=RECORDS, mock=False, protocol=None):
    return write_report(
        tmp_path / "report",
        run_id="run-1",
        condition="C",
        protocol=protocol or {"budget": {"max_main_steps": 8}},
        records=records,
        mock=mock,
    )


def test_writes_six_files(tmp_path):
    paths = report_dir(tmp_path)
    assert set(paths) == {
        "per_task",
        "summary",
        "splits",
        "cost",
        "failures",
        "protocol",
    }
    for name, path in paths.items():
        assert path.exists(), name


def test_per_task_matches_records(tmp_path):
    paths = report_dir(tmp_path)
    lines = paths["per_task"].read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(RECORDS)
    first = json.loads(lines[0])
    assert set(first) == {
        "task_id",
        "family",
        "seed",
        "status",
        "initial_pass",
        "final_pass",
        "cost",
        "public_pass",
    }


def test_summary_matches_metrics(tmp_path):
    paths = report_dir(tmp_path)
    payload = json.loads(paths["summary"].read_text(encoding="utf-8"))
    expected = summarize(RECORDS)

    assert payload["run_id"] == "run-1"
    assert payload["condition"] == "C"
    assert payload["mock"] is False
    assert payload["summary"]["tasks"] == expected.tasks
    assert payload["summary"]["final_all_pass"] == pytest.approx(expected.final_all_pass)
    assert payload["summary"]["infra"] == 1
    assert set(payload["splits"]) == {"alpha", "beta"}


def test_splits_csv(tmp_path):
    paths = report_dir(tmp_path)
    with paths["splits"].open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))

    assert rows[0][0] == "family"
    body = {line[0]: line for line in rows[1:]}
    assert set(body) == {"alpha", "beta"}
    assert body["alpha"][2] == "2"  # evaluated
    assert body["alpha"][5] == "0.500000"


def test_cost_csv_totals_reconcile(tmp_path):
    paths = report_dir(tmp_path)
    with paths["cost"].open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == len(RECORDS)
    total = sum(float(item["cost"]) for item in rows if item["cost"])
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))["summary"]
    assert total == pytest.approx(summary["cost_per_attempt"] * summary["tasks"])


def test_missing_cost_is_empty_not_zero(tmp_path):
    paths = report_dir(tmp_path, records=(row("t1", cost=None),))
    with paths["cost"].open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["cost"] == ""


def test_failures_lists_only_non_success(tmp_path):
    paths = report_dir(tmp_path)
    text = paths["failures"].read_text(encoding="utf-8")
    assert "t2" in text and "TEST_FAILED" in text
    assert "t3" in text and "INFRA_ERROR" in text
    assert "| t1 |" not in text


def test_mock_banner_is_visible(tmp_path):
    paths = report_dir(tmp_path, mock=True)
    assert json.loads(paths["summary"].read_text(encoding="utf-8"))["mock"] is True
    assert "MOCK DATA" in paths["failures"].read_text(encoding="utf-8")


def test_protocol_uses_allowlist_not_blacklist(tmp_path):
    # 白名单：名字里不含 private/secret/patch 的键也必须被丢弃
    protocol = {
        "budget": {"max_main_steps": 8, "max_sub_calls": 2},
        "seeds": (1, 2),
        "private_suite_hash": "abc",
        "answer": "42",
        "gold_patch": "diff --git ...",
        "reference": "def f(): ...",
    }
    paths = report_dir(tmp_path, protocol=protocol)
    saved = json.loads(paths["protocol"].read_text(encoding="utf-8"))

    assert saved["protocol"] == {
        "budget": {"max_main_steps": 8, "max_sub_calls": 2},
        "seeds": [1, 2],
    }
    assert saved["protocol_dropped_keys"] == [
        "answer",
        "gold_patch",
        "private_suite_hash",
        "reference",
    ]


def test_allowed_key_with_secret_value_is_redacted(tmp_path):
    paths = report_dir(tmp_path, protocol={"dataset": "hf://x?token=abc"})
    saved = json.loads(paths["protocol"].read_text(encoding="utf-8"))
    assert saved["protocol"]["dataset"] == "[redacted]"


def test_output_is_deterministic(tmp_path):
    first = report_dir(tmp_path)
    snapshot = {name: path.read_bytes() for name, path in first.items()}
    second = report_dir(tmp_path)
    for name, path in second.items():
        assert path.read_bytes() == snapshot[name], name


def test_empty_report_is_valid(tmp_path):
    paths = report_dir(tmp_path, records=())
    payload = json.loads(paths["summary"].read_text(encoding="utf-8"))
    assert payload["summary"]["tasks"] == 0
    assert payload["summary"]["final_all_pass"] is None
    assert "no failures" in paths["failures"].read_text(encoding="utf-8")


def test_creates_missing_directory(tmp_path):
    paths = write_report(
        tmp_path / "a" / "b" / "c",
        run_id="run-1",
        condition="C",
        protocol={},
        records=(row("t1"),),
    )
    assert paths["summary"].exists()


def test_non_ascii_family_preserved(tmp_path):
    paths = report_dir(tmp_path, records=(row("任务一", family="仓库甲"),))
    text = paths["per_task"].read_text(encoding="utf-8")
    assert "仓库甲" in text
