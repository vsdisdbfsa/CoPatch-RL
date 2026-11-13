"""Deterministic, reconcilable evaluation reports.

Six artifacts are written per run: per_task.jsonl, summary.json, splits.csv,
cost.csv, failures.md and protocol.json. No timestamps and no private grading
material are written, so two runs over the same input are byte-identical.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path

from copatch_rl.evaluation.metrics import TaskRecord, by_group, summarize

ROW_FIELDS = (
    "task_id",
    "family",
    "seed",
    "status",
    "initial_pass",
    "final_pass",
    "cost",
    "public_pass",
)

PROTOCOL_KEYS = frozenset(
    {
        "run_id",
        "condition",
        "protocol_version",
        "commit",
        "git_commit",
        "seeds",
        "layers",
        "modes",
        "models",
        "main",
        "main_base",
        "main_lora",
        "tokenizer",
        "budget",
        "tools",
        "tool_schema_hash",
        "harness_version",
        "grader_version",
        "test_interface_version",
        "error_rules_hash",
        "termination_hash",
        "statistics_hash",
        "decoding_hash",
        "dataset",
        "dataset_hash",
        "manifest_hash",
        "image_digest",
        "model_revision",
        "prompt_hash",
        "temperature",
        "output_limit",
        "log_version",
        "max_main_steps",
        "max_sub_calls",
        "max_public_tests",
        "max_code_revisions",
        "max_generated_tokens",
        "wall_time_seconds",
    }
)

# Defense in depth: allowed keys can still hold free text.
REDACT = ("private", "secret", "password", "token", "api_key")


def _scrub(value: object, dropped: set[str]) -> object:
    """Allowlist-only copy of the protocol; unknown keys are dropped by name."""
    if isinstance(value, Mapping):
        clean: dict[str, object] = {}
        for key, item in value.items():
            name = str(key)
            if name.lower() in PROTOCOL_KEYS:
                clean[name] = _scrub(item, dropped)
            else:
                dropped.add(name)
        return clean
    if isinstance(value, (list, tuple)):
        return [_scrub(item, dropped) for item in value]
    if isinstance(value, str) and any(word in value.lower() for word in REDACT):
        return "[redacted]"
    return value


def _row(record: TaskRecord) -> dict[str, object]:
    return {name: getattr(record, name) for name in ROW_FIELDS}


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, header: Sequence[str], rows: Sequence[Sequence[object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def write_report(
    directory: Path | str,
    *,
    run_id: str,
    condition: str,
    protocol: Mapping[str, object],
    records: Sequence[TaskRecord],
    mock: bool = False,
) -> dict[str, Path]:
    """Write the six artifacts; return their paths."""
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)

    ordered = sorted(records, key=lambda record: (record.task_id, record.seed))
    summary = summarize(ordered)
    groups = by_group(ordered, lambda record: record.family)

    per_task = target / "per_task.jsonl"
    with per_task.open("w", encoding="utf-8") as handle:
        for record in ordered:
            handle.write(json.dumps(_row(record), sort_keys=True, ensure_ascii=False) + "\n")

    summary_path = target / "summary.json"
    _write_json(
        summary_path,
        {
            "run_id": run_id,
            "condition": condition,
            "mock": bool(mock),
            "tasks": summary.tasks,
            "summary": asdict(summary),
            "splits": {name: asdict(item) for name, item in groups.items()},
        },
    )

    splits_path = target / "splits.csv"
    _write_csv(
        splits_path,
        (
            "family",
            "tasks",
            "evaluated",
            "success",
            "infra",
            "all_pass",
            "cost_per_attempt",
        ),
        [
            (
                name,
                item.tasks,
                item.evaluated,
                item.success,
                item.infra,
                "" if item.final_all_pass is None else f"{item.final_all_pass:.6f}",
                "" if item.cost_per_attempt is None else f"{item.cost_per_attempt:.6f}",
            )
            for name, item in groups.items()
        ],
    )

    cost_path = target / "cost.csv"
    _write_csv(
        cost_path,
        ("task_id", "family", "seed", "cost"),
        [
            (
                record.task_id,
                record.family,
                record.seed,
                "" if record.cost is None else f"{record.cost:.6f}",
            )
            for record in ordered
        ],
    )

    failures_path = target / "failures.md"
    failed = [record for record in ordered if record.status != "SUCCESS"]
    lines = ["# Failures", ""]
    if mock:
        lines = [
            "> **MOCK DATA** — 本次报告含 Mock 结果，不得作为正式实验结论。",
            "",
        ] + lines
    lines.append(f"run: `{run_id}` · condition: `{condition}`")
    lines.append("")
    lines.append(f"- total: {summary.tasks}")
    lines.append(f"- evaluated: {summary.evaluated}")
    lines.append(f"- infra_error: {summary.infra}")
    lines.append(f"- failed rows: {len(failed)}")
    lines.append("")
    if failed:
        lines.append("| task_id | family | seed | status |")
        lines.append("| --- | --- | --- | --- |")
        for record in failed:
            lines.append(
                f"| {record.task_id} | {record.family} | {record.seed} | {record.status} |"
            )
    else:
        lines.append("_no failures_")
    failures_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    protocol_path = target / "protocol.json"
    dropped: set[str] = set()
    _write_json(
        protocol_path,
        {
            "run_id": run_id,
            "condition": condition,
            "protocol": _scrub(dict(protocol), dropped),
            "protocol_dropped_keys": sorted(dropped),
        },
    )

    return {
        "per_task": per_task,
        "summary": summary_path,
        "splits": splits_path,
        "cost": cost_path,
        "failures": failures_path,
        "protocol": protocol_path,
    }
