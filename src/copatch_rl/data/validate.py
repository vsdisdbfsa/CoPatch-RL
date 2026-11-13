"""Trusted data QA only. Never route model-generated candidates through this API."""

import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from copatch_rl.contracts import TaskSpec
from copatch_rl.execution.repository_tests import RepositorySuite, parse_report, prepare
from copatch_rl.execution.sandbox import ExecutionResult, Limits, run_python
from copatch_rl.execution.workspace import Snapshot, _hash, _manifest


@dataclass(frozen=True)
class Assets:
    suite: RepositorySuite
    snapshots: Mapping[str, Snapshot]
    reference_id: str
    wrong_ids: tuple[str, ...]
    f2p: tuple[str, ...]
    p2p: tuple[str, ...]
    platform: str
    dependency_hash: str
    runner_version: str
    trusted_implementations: bool
    description_reviewed: bool


@dataclass(frozen=True)
class Validation:
    decision: str
    reasons: tuple[str, ...]
    stages: tuple[tuple[str, str], ...] = ()


def validate(
    record: Mapping,
    assignment: Mapping,
    assets: Assets,
    *,
    runner: Callable[..., ExecutionResult] = run_python,
    limits: Limits = Limits(),
) -> Validation:
    """Consume C-06 host record plus one C-07 assignment and reviewed QA assets.

    All source variants and tests must be host-authored/reviewed, never candidates.
    Suite contents stay in this trusted QA container, not any agent workspace.
    """

    def reject(code: str) -> Validation:
        return Validation("quarantine", (code,))

    try:
        task = TaskSpec.model_validate(record["task"])
    except (KeyError, ValueError, TypeError):
        return reject("INVALID_SCHEMA")
    if record.get("task_id") != task.task_id or assignment.get("task_id") != task.task_id:
        return reject("TASK_MISMATCH")
    if assignment.get("split") not in {"train", "dev", "test"} or task.split != assignment["split"]:
        return reject("SPLIT_CONFLICT")
    if not isinstance(record.get("license"), str) or not record["license"].strip():
        return reject("LICENSE_MISSING")
    if record.get("audit_status") != "approved":
        return reject("AUDIT_PENDING")
    if record.get("source") == "swe-gym":
        return reject("AUDIT_PENDING")
    if not assets.trusted_implementations:
        return reject("UNTRUSTED_QA_SOURCE")
    if not assets.description_reviewed:
        return reject("DESCRIPTION_UNREVIEWED")
    if assets.platform != "linux/amd64":
        return reject("PLATFORM_MISMATCH")
    if (
        not re.fullmatch(r"sha256:[0-9a-f]{64}", assets.dependency_hash)
        or not isinstance(assets.runner_version, str)
        or not assets.runner_version.strip()
    ):
        return reject("DEPENDENCY_METADATA_INVALID")
    suite = assets.suite
    if (
        suite.task_id != task.task_id
        or suite.suite_id != task.private_suite_id
        or not task.public_suite_id
    ):
        return reject("SUITE_REFERENCE_MISMATCH")
    if not suite.nodeids or not suite.tests:
        return reject("EMPTY_SUITE")
    nodes = assets.f2p + assets.p2p
    if (
        not assets.f2p
        or len(set(nodes)) != len(nodes)
        or set(nodes) != set(suite.nodeids)
        or len(set(suite.nodeids)) != len(suite.nodeids)
    ):
        return reject("MISSING_CASE")
    if _hash(suite.tests) != suite.tests_hash:
        return reject("HASH_MISMATCH")
    ids = (task.initial_snapshot, assets.reference_id, *assets.wrong_ids)
    if not assets.wrong_ids:
        return reject("COUNTEREXAMPLE_MISSING")
    try:
        allowed = _manifest(task.allowed_files)
        if not allowed:
            return reject("ALLOWED_FILES_INVALID")
        for identifier in ids:
            snapshot = assets.snapshots[identifier]
            if snapshot.snapshot_id != identifier:
                return reject("SNAPSHOT_UNAVAILABLE")
            if snapshot.code_hash != _hash(snapshot.files):
                return reject("HASH_MISMATCH")
            names = _manifest(n for n, _ in snapshot.files)
            if not allowed <= names:
                return reject("ALLOWED_FILES_INVALID")
            prepare(snapshot, suite)
    except KeyError:
        return reject("SNAPSHOT_UNAVAILABLE")
    except (ValueError, TypeError):
        return reject("ASSET_METADATA_INVALID")
    secret = []
    raw = record.get("raw", {})
    for key in ("code", "patch", "test_patch", "test"):
        if isinstance(raw.get(key), str):
            secret.append(raw[key])
    for key in ("test_list", "challenge_test_list"):
        if isinstance(raw.get(key), list):
            secret.extend(raw[key])
    secret.extend(data.decode("utf-8", errors="replace") for _, data in suite.tests)
    secret.extend(
        data.decode("utf-8", errors="replace")
        for _, data in assets.snapshots[assets.reference_id].files
    )
    description = task.description
    if re.search(
        r"diff --git|@@ |```|reference (?:answer|implementation)|参考(?:答案|实现)",
        description,
        re.I,
    ) or any(isinstance(s, str) and s.strip() and s.strip() in description for s in secret):
        return reject("DESCRIPTION_LEAK")
    stages = []
    for index, identifier in enumerate(ids):
        stage = "defect" if index == 0 else "reference" if index == 1 else "counterexample"
        try:
            execution = prepare(assets.snapshots[identifier], suite)
            result = runner(execution, "__copatch_repo__.py", suite.image_id, limits)
            if result.platform != assets.platform:
                return reject("PLATFORM_MISMATCH")
            if result.image_id != suite.image_id or result.code_hash != execution.code_hash:
                return reject("EXECUTION_IDENTITY_MISMATCH")
            if result.status != "EXITED" or result.exit_code != 0 or result.truncated:
                return reject(
                    "DYNAMIC_TIMEOUT" if result.status == "TIMEOUT" else "DYNAMIC_INFRA_ERROR"
                )
            state, _, _, _ = parse_report(result.output, suite.nodeids)
            report = json.loads(result.output)
            phases = {node: {} for node in suite.nodeids}
            for node, when, outcome, xfail in report["reports"]:
                phases[node][when] = (outcome, xfail)
            if state == "INFRA_ERROR":
                return reject("TEST_INFRA_ERROR")

            def passed(node: str) -> bool:
                return phases[node] == {p: ("passed", False) for p in ("setup", "call", "teardown")}

            def failed(node: str) -> bool:
                return (
                    phases[node].get("setup") == ("passed", False)
                    and phases[node].get("call") == ("failed", False)
                    and phases[node].get("teardown") == ("passed", False)
                )

            if index == 0:
                if not all(passed(n) for n in assets.p2p):
                    return reject("P2P_BASELINE_FAILED")
                if not all(failed(n) for n in assets.f2p):
                    return reject("DEFECT_NOT_FAILING")
            elif index == 1:
                if not all(passed(n) for n in assets.p2p):
                    return reject("P2P_REGRESSION")
                if not all(passed(n) for n in suite.nodeids):
                    return reject("REFERENCE_FAILED")
            elif not any(failed(n) for n in suite.nodeids):
                return reject("COUNTEREXAMPLE_NOT_CAUGHT")
            stages.append((stage, state))
        except (OSError, ValueError, TypeError, KeyError, RuntimeError, UnicodeError):
            return reject("DYNAMIC_INFRA_ERROR")
    return Validation("accept", (), tuple(stages))
