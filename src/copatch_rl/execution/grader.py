"""Host-only grading of registered frozen snapshots; never model observations."""

import ast
import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from copatch_rl.contracts import GraderResult
from copatch_rl.execution.sandbox import ExecutionResult, Limits, run_python
from copatch_rl.execution.workspace import Snapshot, _hash, _manifest, _relative


@dataclass(frozen=True)
class Frozen:
    task_id: str
    snapshot: Snapshot
    registered_hash: str
    submitted: bool = True


@dataclass(frozen=True)
class Case:
    args_json: str
    expected_json: str


@dataclass(frozen=True)
class Task:
    source_path: str
    function: str
    cases: tuple[Case, ...]
    f2p: tuple[str, ...] = ()
    p2p: tuple[str, ...] = ()


@dataclass(frozen=True)
class RepositoryDiagnostic:
    collected: tuple[str, ...]
    outcomes: tuple[tuple[str, str], ...]
    diagnostic_only: bool = True
    collection_failed: bool = False


@dataclass(frozen=True)
class Detail:
    passed: int = 0
    failed: int = 0
    diagnostic_only: bool = False
    reason: str = ""


def _json(text: str) -> object:
    value = json.loads(text)
    json.dumps(value, allow_nan=False)
    return value


def _driver(task: Task, case: Case) -> bytes:
    # Only this input, never expected values, case lists, or private test scripts.
    worker = (
        "import os,sys,json,importlib.util\n"
        "write=os.write; dump=json.dumps; fd=int(sys.argv[1])\n"
        'sys.path.insert(0,"/source")\n'
        's=importlib.util.spec_from_file_location("candidate",'
        + repr("/source/" + task.source_path)
        + ")\n"
        "m=importlib.util.module_from_spec(s); s.loader.exec_module(m)\n"
        "v=getattr(m," + repr(task.function) + ")(*json.loads(" + repr(case.args_json) + "))\n"
        "b=dump(v,allow_nan=False).encode()\n"
        "while b: b=b[write(fd,b):]\n"
    )
    return (
        "import os,sys,subprocess\nr,w=os.pipe()\n"
        'p=subprocess.Popen([sys.executable,"-I","-B","-c",' + repr(worker) + ",str(w)],"
        "pass_fds=(w,),stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)\n"
        "os.close(w)\nwhile b:=os.read(r,4096):\n"
        " sys.stdout.buffer.write(b); sys.stdout.buffer.flush()\n"
        "os.close(r)\np.wait()\nsys.exit(0 if p.returncode==0 else 21)\n"
    ).encode()


class Grader:
    def __init__(
        self,
        snapshot_store: Mapping[str, Frozen],
        tasks: Mapping[str, Task],
        image: str,
        limits: Limits = Limits(),
        *,
        runner: Callable[..., ExecutionResult] = run_python,
        repository_runner: Callable[..., RepositoryDiagnostic] | None = None,
    ) -> None:
        self._store = snapshot_store
        self._tasks = dict(tasks)
        self._image = image
        self._limits = limits
        self._runner = runner
        self._repository_runner = repository_runner
        self.details: dict[tuple[str, str], Detail] = {}

    def score(self, submitted_snapshot_id: str, task_id: str) -> GraderResult:
        key = (submitted_snapshot_id, task_id)
        self.details[key] = Detail(reason="unavailable")
        status = "INFRA_ERROR"
        try:
            frozen = self._store[submitted_snapshot_id]
            snapshot = frozen.snapshot
            task = self._tasks[task_id]
            if (
                not frozen.submitted
                or frozen.task_id != task_id
                or snapshot.snapshot_id != submitted_snapshot_id
                or _hash(snapshot.files) != snapshot.code_hash
                or snapshot.code_hash != frozen.registered_hash
            ):
                raise ValueError("snapshot registration mismatch")
            _manifest(name for name, _ in snapshot.files)
            if snapshot.files != tuple(sorted(snapshot.files)):
                raise ValueError("noncanonical snapshot")
            if task.f2p or task.p2p:
                status = self._repository(snapshot, task, key)
            else:
                status = self._function(snapshot, task, key)
        except (
            KeyError,
            OSError,
            ValueError,
            TypeError,
            RuntimeError,
            RecursionError,
            UnicodeError,
        ):
            status = "INFRA_ERROR"
        return GraderResult(
            task_id=task_id, submitted_snapshot_id=submitted_snapshot_id, status=status
        )

    def _repository(self, snapshot: Snapshot, task: Task, key: tuple[str, str]) -> str:
        nodes = task.f2p + task.p2p
        self.details[key] = Detail(diagnostic_only=True, reason="Diagnostic unavailable")
        if not task.f2p or len(set(nodes)) != len(nodes) or self._repository_runner is None:
            return "INFRA_ERROR"
        # Adapter receives only predeclared upstream/public IDs, never private assets.
        report = self._repository_runner(snapshot, nodes)
        pairs = dict(report.outcomes)
        if (
            report.collection_failed
            or len(report.collected) != len(nodes)
            or set(report.collected) != set(nodes)
            or len(pairs) != len(report.outcomes)
            or set(pairs) != set(nodes)
        ):
            return "INFRA_ERROR"
        passed = sum(value == "passed" for value in pairs.values())
        self.details[key] = Detail(
            passed, len(nodes) - passed, True, "Diagnostic F2P/P2P; no trusted SUCCESS"
        )
        return "INFRA_ERROR"

    def _function(self, snapshot: Snapshot, task: Task, key: tuple[str, str]) -> str:
        if not task.cases or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", task.function):
            return "INFRA_ERROR"
        _relative(task.source_path)
        files = dict(snapshot.files)
        if "__copatch_grade__.py" in files:
            return "INFRA_ERROR"
        for case in task.cases:
            if not isinstance(_json(case.args_json), list):
                return "INFRA_ERROR"
            _json(case.expected_json)
        try:
            ast.parse(files[task.source_path])
        except SyntaxError:
            return "INVALID_CODE"
        passed = failed = 0
        for case in task.cases:
            entries = tuple(
                sorted((*snapshot.files, ("__copatch_grade__.py", _driver(task, case))))
            )
            execution = Snapshot("grade-input", entries, _hash(entries))
            result = self._runner(execution, "__copatch_grade__.py", self._image, self._limits)
            if result.code_hash != execution.code_hash:
                return "INFRA_ERROR"
            if result.status == "TIMEOUT":
                return "TIMEOUT"
            if result.status in {"RESOURCE_LIMIT", "OUTPUT_LIMIT"}:
                return "TEST_FAILED"
            if result.status != "EXITED" or result.truncated:
                return "INFRA_ERROR"
            if result.exit_code == 21:
                failed += 1
            elif result.exit_code == 0:
                actual = _json(result.output.decode("utf-8"))
                expected = _json(case.expected_json)
                if type(actual) is type(expected) and actual == expected:
                    passed += 1
                else:
                    failed += 1
            else:
                return "INFRA_ERROR"
        self.details[key] = Detail(passed, failed)
        return "TEST_FAILED" if failed else "SUCCESS"
