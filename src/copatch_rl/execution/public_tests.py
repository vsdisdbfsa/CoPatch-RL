"""Internal public function testing; host compares values, never printed verdicts.

Only audited public suites belong here. This is not a private grader or a pytest
report adapter. Registry mutation and workspace writes require host serialization.
"""

import ast
import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from copatch_rl.execution.repository_tests import RepositorySuite, parse_report, prepare
from copatch_rl.execution.sandbox import ExecutionResult, Limits, run_python
from copatch_rl.execution.workspace import Snapshot, Workspace, _hash, _relative


@dataclass(frozen=True)
class PublicCase:
    case_id: str
    args_json: str
    expected_json: str


@dataclass(frozen=True)
class PublicSuite:
    suite_id: str
    task_id: str
    source_path: str
    function: str
    cases: tuple[PublicCase, ...]


@dataclass(frozen=True)
class Binding:
    task_id: str
    suite_id: str
    workspace: Workspace


@dataclass(frozen=True)
class PublicResult:
    """C-internal proposal; not an approved cross-module Contracts model."""

    status: str
    code_hash: str | None
    total: int
    passed: int
    failed: int
    not_run: int
    summary: str
    exit_codes: tuple[int | None, ...] = ()
    diagnostic_only: bool = False


def _json(text: str) -> object:
    def reject(value: str) -> None:
        raise ValueError("non-finite JSON")

    value = json.loads(text, parse_constant=reject)
    json.dumps(value, allow_nan=False)
    return value


def _equal(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, list):
        return len(left) == len(right) and all(_equal(a, b) for a, b in zip(left, right))
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_equal(left[k], right[k]) for k in left)
    return left == right


# Candidate imports happen only in a child in Docker. Supervisor never imports it.
# Child stdout/stderr are discarded. A separate pipe carries untrusted values only.
_WORKER = """import contextlib, importlib.util, json, sys, os
serialize = json.dumps
write_result = os.write
result_fd = int(sys.argv[1])
with contextlib.redirect_stdout(sys.stderr):
    sys.path.insert(0, '/source')
    spec = importlib.util.spec_from_file_location('_candidate', SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    value = getattr(module, FUNCTION)(*ARGS)
payload = serialize(value, allow_nan=False).encode()
while payload:
    payload = payload[write_result(result_fd, payload):]
"""


def _driver(suite: PublicSuite, case: PublicCase) -> bytes:
    worker = (
        "SOURCE="
        + repr("/source/" + suite.source_path)
        + "\nFUNCTION="
        + repr(suite.function)
        + "\nimport json\nARGS=json.loads("
        + repr(case.args_json)
        + ")\n"
        + _WORKER
    )
    # The host-owned driver compiles before launching, and never echoes stderr.
    return (
        "import subprocess, sys, os\n"
        "try:\n    compile(open("
        + repr("/source/" + suite.source_path)
        + ", 'rb').read(), 'candidate', 'exec')\n"
        "except SyntaxError:\n    sys.exit(20)\n"
        "read_fd, write_fd = os.pipe()\n"
        "p=subprocess.Popen([sys.executable,'-I','-B','-c',"
        + repr(worker)
        + ",str(write_fd)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, "
        "pass_fds=(write_fd,))\n"
        "os.close(write_fd)\n"
        "while chunk := os.read(read_fd, 4096):\n"
        "    sys.stdout.buffer.write(chunk)\n    sys.stdout.buffer.flush()\n"
        "os.close(read_fd)\n"
        "p.wait()\n"
        "if p.returncode != 0:\n    sys.exit(21)\n"
    ).encode()


class Executor:
    def __init__(
        self,
        bindings: Mapping[str, Binding],
        suites: Mapping[str, PublicSuite | RepositorySuite],
        image: str,
        limits: Limits = Limits(),
        *,
        runner: Callable[..., ExecutionResult] = run_python,
    ) -> None:
        self._bindings = dict(bindings)
        self._suites = dict(suites)
        self._image = image
        self._limits = limits
        self._runner = runner

    def run_public_tests(self, workspace_id: str, suite_id: str) -> PublicResult:
        binding = self._bindings.get(workspace_id)
        if binding is None:
            return PublicResult("PERMISSION_DENIED", None, 0, 0, 0, 0, "Unknown workspace")
        try:
            snapshot = binding.workspace.snapshot("public-test")
        except PermissionError:
            return PublicResult("PERMISSION_DENIED", None, 0, 0, 0, 0, "Workspace access denied")
        except (OSError, ValueError):
            return PublicResult("INFRA_ERROR", None, 0, 0, 0, 0, "Workspace unavailable")
        code_hash = snapshot.code_hash
        suite = self._suites.get(suite_id)
        if (
            suite is None
            or binding.suite_id != suite_id
            or suite.suite_id != suite_id
            or suite.task_id != binding.task_id
        ):
            return PublicResult(
                "PERMISSION_DENIED", code_hash, 0, 0, 0, 0, "Suite association denied"
            )
        if isinstance(suite, RepositorySuite):
            return self._run_repository(binding, snapshot, suite)
        try:
            _relative(suite.source_path)
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", suite.function):
                raise ValueError("invalid function")
            if not suite.cases or len({c.case_id for c in suite.cases}) != len(suite.cases):
                raise ValueError("empty or duplicate cases")
            for case in suite.cases:
                if not case.case_id or not isinstance(_json(case.args_json), list):
                    raise ValueError("invalid arguments")
                _json(case.expected_json)
            files = dict(snapshot.files)
            if suite.source_path not in files or "__copatch_public__.py" in files:
                raise ValueError("missing entrypoint or reserved driver collision")
        except (ValueError, TypeError):
            return PublicResult("INFRA_ERROR", code_hash, 0, 0, 0, 0, "Invalid public suite")
        # Static parsing does not execute candidate code on the host.
        try:
            ast.parse(files[suite.source_path])
        except (SyntaxError, ValueError):
            return PublicResult(
                "INVALID_CODE",
                code_hash,
                len(suite.cases),
                0,
                0,
                len(suite.cases),
                "Candidate syntax invalid",
            )
        passed = failed = 0
        failures = []
        exits = []
        status = "SUCCESS"
        for index, case in enumerate(suite.cases, 1):
            augmented = tuple(
                sorted((*snapshot.files, ("__copatch_public__.py", _driver(suite, case))))
            )
            execution = Snapshot("public-execution", augmented, _hash(augmented))
            try:
                result = self._runner(execution, "__copatch_public__.py", self._image, self._limits)
            except (OSError, RuntimeError, ValueError):
                status = "INFRA_ERROR"
                break
            exits.append(result.exit_code)
            if result.code_hash != execution.code_hash:
                status = "INFRA_ERROR"
                break
            if result.status != "EXITED":
                status = {
                    "TIMEOUT": "TIMEOUT",
                    "OUTPUT_LIMIT": "TEST_FAILED",
                    "RESOURCE_LIMIT": "TEST_FAILED",
                }.get(result.status, "INFRA_ERROR")
                break
            if result.exit_code == 20:
                status = "INVALID_CODE"
                break
            if result.exit_code == 21:
                failed += 1
                failures.append(index)
                continue
            if result.exit_code != 0 or result.truncated:
                status = "INFRA_ERROR"
                break
            try:
                value = _json(result.output.decode("utf-8"))
                matches = _equal(value, _json(case.expected_json))
            except (UnicodeError, ValueError, RecursionError):
                status = "INFRA_ERROR"
                break
            if matches:
                passed += 1
            else:
                failed += 1
                failures.append(index)
        try:
            stale = binding.workspace.source_hash() != code_hash
        except (OSError, ValueError):
            stale = True
        if stale:
            status = "INFRA_ERROR"
        elif status == "SUCCESS" and failed:
            status = "TEST_FAILED"
        remaining = len(suite.cases) - passed - failed
        summary = (
            "Stale source; rerun public tests"
            if stale
            else f"{status}: {passed} passed, {failed} failed, {remaining} not completed"
        )
        if failures:
            summary += f"; failing case positions: {failures[:5]}"
        return PublicResult(
            status, code_hash, len(suite.cases), passed, failed, remaining, summary, tuple(exits)
        )

    def _run_repository(
        self, binding: Binding, snapshot: Snapshot, suite: RepositorySuite
    ) -> PublicResult:
        total = len(suite.nodeids)
        passed = failed = 0
        incomplete = total
        status = "INFRA_ERROR"
        exits = ()
        try:
            execution = prepare(snapshot, suite)
            result = self._runner(execution, "__copatch_repo__.py", suite.image_id, self._limits)
            exits = (result.exit_code,)
            if result.code_hash != execution.code_hash or result.image_id != suite.image_id:
                raise ValueError("execution identity mismatch")
            if result.status == "TIMEOUT":
                status = "TIMEOUT"
            elif result.status in {"RESOURCE_LIMIT", "OUTPUT_LIMIT"}:
                status = "TEST_FAILED"
            elif result.status == "EXITED" and result.exit_code == 0 and not result.truncated:
                status, passed, failed, incomplete = parse_report(result.output, suite.nodeids)
            if binding.workspace.source_hash() != snapshot.code_hash:
                status = "INFRA_ERROR"
        except (OSError, RuntimeError, ValueError, TypeError, KeyError, RecursionError):
            status = "INFRA_ERROR"
        # Native pytest imports candidate code into its reporting process. Never
        # promote its claimed all-pass into trusted SUCCESS without isolation.
        if status == "SUCCESS":
            status = "INFRA_ERROR"
            summary = "Diagnostic all-pass; in-process report integrity is not established"
        else:
            summary = (
                f"Diagnostic {status}: {passed} passed, {failed} failed, {incomplete} incomplete"
            )
        return PublicResult(
            status,
            snapshot.code_hash,
            total,
            passed,
            failed,
            incomplete,
            summary,
            exits,
            diagnostic_only=True,
        )
