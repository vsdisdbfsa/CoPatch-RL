import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

from copatch_rl.execution.grader import Case, Frozen, Grader, RepositoryDiagnostic, Task
from copatch_rl.execution.sandbox import ExecutionResult, Limits, _docker, run_python
from copatch_rl.execution.workspace import Snapshot, _hash


def frozen(source=b"def answer(x): return x + 1"):
    files = (("solution.py", source),)
    return Frozen("t", Snapshot("s", files, _hash(files)), _hash(files))


def task():
    return Task("solution.py", "answer", (Case("[1]", "2"),))


def runner(status="EXITED", output=b"2"):
    def run(snapshot, *args):
        return ExecutionResult(status, 0, output, False, snapshot.code_hash, "mock", "mock")

    return run


def test_wrong_hash():
    item = replace(frozen(), registered_hash="bad")
    assert (
        Grader({"s": item}, {"t": task()}, "mock", runner=runner()).score("s", "t").status
        == "INFRA_ERROR"
    )


def test_empty_suite():
    assert (
        Grader({"s": frozen()}, {"t": replace(task(), cases=())}, "mock").score("s", "t").status
        == "INFRA_ERROR"
    )


def test_private_leakage():
    cases = json.loads(
        (Path(__file__).parent / "fixtures/private/grader_cases.json").read_text(
            encoding="utf-8-sig"
        )
    )
    t = replace(
        task(), cases=tuple(Case(json.dumps(c["args"]), json.dumps(c["expected"])) for c in cases)
    )
    calls = []

    def inspect(snapshot, *args):
        content = b"\n".join(data for _, data in snapshot.files)
        assert b"HOST_ONLY_EXPECTED" not in content
        assert not (b"first_input" in content and b"second_input" in content)
        calls.append(1)
        return runner(output=b"null")(snapshot)

    assert (
        Grader({"s": frozen()}, {"t": t}, "mock", runner=inspect).score("s", "t").status
        == "TEST_FAILED"
    )
    assert calls


def test_regression():
    t = Task("", "", (), f2p=("fix",), p2p=("keep",))
    report = RepositoryDiagnostic(("fix", "keep"), (("fix", "passed"), ("keep", "failed")))
    g = Grader({"s": frozen()}, {"t": t}, "mock", repository_runner=lambda *args: report)
    assert g.score("s", "t").status == "INFRA_ERROR"
    assert g.details[("s", "t")].failed == 1


def test_timeout():
    assert (
        Grader({"s": frozen()}, {"t": task()}, "mock", runner=runner("TIMEOUT"))
        .score("s", "t")
        .status
        == "TIMEOUT"
    )


def test_infrastructure():
    assert (
        Grader({"s": frozen()}, {"t": task()}, "mock", runner=runner("INFRA_ERROR"))
        .score("s", "t")
        .status
        == "INFRA_ERROR"
    )


@pytest.mark.parametrize("outcome", ["passed", "failed", "skipped", "xfail", "xpass"])
def test_repository_diagnostic(outcome):
    t = Task("", "", (), f2p=("fix",), p2p=("keep",))
    report = RepositoryDiagnostic(("fix", "keep"), (("fix", outcome), ("keep", "passed")))
    g = Grader({"s": frozen()}, {"t": t}, "mock", repository_runner=lambda *args: report)
    assert g.score("s", "t").status == "INFRA_ERROR"
    assert g.details[("s", "t")].diagnostic_only


def test_wrong_task_and_unfrozen():
    for item in (replace(frozen(), task_id="other"), replace(frozen(), submitted=False)):
        assert Grader({"s": item}, {"t": task()}, "mock").score("s", "t").status == "INFRA_ERROR"


@pytest.mark.skipif(os.environ.get("COPATCH_DOCKER_TESTS") != "1", reason="real Docker opt-in")
@pytest.mark.parametrize(
    "source,status",
    [
        (b"def answer(x): return x+1", "SUCCESS"),
        (b"def answer(x): return 0", "TEST_FAILED"),
        (b"def answer(", "INVALID_CODE"),
        (b"def answer(x):\n while True: pass", "TIMEOUT"),
        (b"import os\ndef answer(x): os._exit(0)", "INFRA_ERROR"),
        (
            b"import pathlib\n"
            b"def answer(x):\n"
            b' assert not any(("HOST_" + "ONLY_EXPECTED") in p.read_text() for p in '
            b'pathlib.Path("/source").rglob("*.py"))\n'
            b' assert not pathlib.Path("/source/private").exists()\n'
            b" return x+1",
            "SUCCESS",
        ),
    ],
)
def test_real_function(source, status):
    g = Grader({"s": frozen(source)}, {"t": task()}, "python:3.11-slim", Limits(seconds=2))
    assert g.score("s", "t").status == status
    assert not _docker("ps", "-aq", "--filter", "name=copatch-c03-").strip()


@pytest.mark.skipif(os.environ.get("COPATCH_DOCKER_TESTS") != "1", reason="real Docker opt-in")
def test_real_repository_diagnostic():
    from copatch_rl.execution.repository_tests import RepositorySuite, parse_report, prepare

    image = json.loads(_docker("image", "inspect", "copatch-public-pytest:dev"))[0]["Id"]
    nodes = ("public_tests/test_upstream.py::test_fix", "public_tests/test_upstream.py::test_keep")
    files = (
        (
            "public_tests/test_upstream.py",
            b"from solution import answer\n"
            b"def test_fix(): assert answer(1)==2\n"
            b"def test_keep(): assert answer(0)==1",
        ),
    )
    suite = RepositorySuite("upstream", "t", files, _hash(files), nodes, image)

    def diagnostic(snapshot, requested):
        assert requested == nodes
        execution = prepare(snapshot, suite)
        result = run_python(execution, "__copatch_repo__.py", image)
        assert result.status == "EXITED" and result.exit_code == 0
        status, passed, failed, _ = parse_report(result.output, nodes)
        assert (status, passed, failed) == ("SUCCESS", 2, 0)
        return RepositoryDiagnostic(nodes, tuple((n, "passed") for n in nodes))

    g = Grader(
        {"s": frozen()},
        {"t": Task("", "", (), f2p=nodes[:1], p2p=nodes[1:])},
        image,
        repository_runner=diagnostic,
    )
    assert g.score("s", "t").status == "INFRA_ERROR"
    assert g.details[("s", "t")].passed == 2
