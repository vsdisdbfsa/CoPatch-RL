"""Load hand-written example tasks from `examples/dev` (no execution)."""

from __future__ import annotations

import json
from pathlib import Path

from copatch_rl.contracts import TaskSpec
from copatch_rl.execution.public_tests import PublicCase, PublicSuite


def load_task(path: Path | str) -> TaskSpec:
    """Parse and validate a TaskSpec JSON file."""
    return TaskSpec.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))


def load_suite(path: Path | str) -> PublicSuite:
    """Parse a public suite JSON file into the executor's in-memory form."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    cases = tuple(
        PublicCase(case["case_id"], case["args_json"], case["expected_json"])
        for case in data["cases"]
    )
    if not cases:
        raise ValueError("suite needs at least one case")
    return PublicSuite(
        data["suite_id"], data["task_id"], data["source_path"], data["function"], cases
    )


def snapshot_files(directory: Path | str) -> tuple[tuple[str, bytes], ...]:
    """Read an example snapshot directory into sorted (path, bytes) pairs."""
    root = Path(directory)
    files = tuple(
        (path.relative_to(root).as_posix(), path.read_bytes())
        for path in sorted(root.rglob("*"))
        if path.is_file()
    )
    if not files:
        raise ValueError("snapshot directory is empty")
    return files
