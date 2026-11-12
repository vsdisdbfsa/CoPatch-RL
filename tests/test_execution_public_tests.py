"""Public evaluator unit tests and explicitly enabled real Docker integration."""

import os
from dataclasses import replace

import pytest

from copatch_rl.execution.public_tests import Binding, Executor, PublicCase, PublicSuite
from copatch_rl.execution.sandbox import ExecutionResult, Limits
from copatch_rl.execution.workspace import Workspace


@pytest.fixture
def setup(tmp_path):
    initial = tmp_path / "initial"
    initial.mkdir()
    (initial / "solution.py").write_text("def answer(x): return x + 1\n")
    workspace = Workspace(tmp_path / "work", "w", initial, ["solution.py"], ["solution.py"])
    suite = PublicSuite("s", "t", "solution.py", "answer", (PublicCase("one", "[1]", "2"),))
    return workspace, suite


def fake(output=b"2", status="EXITED", exit_code=0, wrong_hash=False):
    def run(snapshot, *args):
        return ExecutionResult(
            status,
            exit_code,
            output,
            False,
            "wrong" if wrong_hash else snapshot.code_hash,
            "mock",
            "mock",
        )

    return run


def executor(setup, runner=None):
    w, s = setup
    kwargs = {} if runner is None else {"runner": runner}
    return Executor(
        {"w": Binding("t", "s", w)}, {"s": s}, "python:3.11-slim", Limits(seconds=2), **kwargs
    )


@pytest.mark.parametrize(
    "output,status",
    [
        (b"2", "SUCCESS"),
        (b"3", "TEST_FAILED"),
        (b'{"passed": 100}', "TEST_FAILED"),
        (b"100 passed", "INFRA_ERROR"),
        (b"", "INFRA_ERROR"),
        (b"NaN", "INFRA_ERROR"),
        (b"2\n2", "INFRA_ERROR"),
    ],
)
def test_host_comparison(setup, output, status):
    result = executor(setup, fake(output)).run_public_tests("w", "s")
    assert result.status == status
    assert result.code_hash == setup[0].source_hash()
    assert result.total == result.passed + result.failed + result.not_run
    assert len(result.summary) < 200


@pytest.mark.parametrize(
    "status,expected",
    [
        ("TIMEOUT", "TIMEOUT"),
        ("INFRA_ERROR", "INFRA_ERROR"),
        ("RESOURCE_LIMIT", "TEST_FAILED"),
        ("OUTPUT_LIMIT", "TEST_FAILED"),
        ("CANCELLED", "INFRA_ERROR"),
    ],
)
def test_execution_errors(setup, status, expected):
    assert executor(setup, fake(status=status)).run_public_tests("w", "s").status == expected


def test_associations(setup):
    e = executor(setup, fake())
    assert e.run_public_tests("bad", "s").status == "PERMISSION_DENIED"
    assert e.run_public_tests("w", "bad").status == "PERMISSION_DENIED"
    w, s = setup
    assert (
        executor((w, replace(s, task_id="other")), fake()).run_public_tests("w", "s").status
        == "PERMISSION_DENIED"
    )


@pytest.mark.parametrize(
    "cases",
    [
        (),
        (PublicCase("a", "bad", "2"),),
        (PublicCase("a", "[]", "NaN"),),
        (PublicCase("a", "[]", "1"), PublicCase("a", "[]", "1")),
    ],
)
def test_invalid_suites(setup, cases):
    w, s = setup
    assert (
        executor((w, replace(s, cases=cases)), fake()).run_public_tests("w", "s").status
        == "INFRA_ERROR"
    )


def test_invalid_code(setup):
    setup[0].write_file("solution.py", "def broken(")
    assert executor(setup, fake()).run_public_tests("w", "s").status == "INVALID_CODE"


def test_old_hash(setup):
    assert executor(setup, fake(wrong_hash=True)).run_public_tests("w", "s").status == "INFRA_ERROR"
    before = setup[0].source_hash()

    def change(snapshot, *args):
        setup[0].write_file("solution.py", "def answer(x): return 999")
        return fake()(snapshot)

    result = executor(setup, change).run_public_tests("w", "s")
    assert result.status == "INFRA_ERROR"
    assert result.code_hash == before
    assert "Stale" in result.summary


def test_runner_exception(setup):
    def broken(*args):
        raise RuntimeError("private host message must not leak")

    result = executor(setup, broken).run_public_tests("w", "s")
    assert result.status == "INFRA_ERROR"
    assert "private" not in result.summary


@pytest.mark.parametrize(
    "code,expected", [(20, "INVALID_CODE"), (21, "TEST_FAILED"), (125, "INFRA_ERROR")]
)
def test_exit_states(setup, code, expected):
    assert executor(setup, fake(exit_code=code)).run_public_tests("w", "s").status == expected


@pytest.mark.skipif(os.environ.get("COPATCH_DOCKER_TESTS") != "1", reason="real Docker opt-in")
@pytest.mark.parametrize(
    "source,status",
    [
        ("def answer(x): return x + 1", "SUCCESS"),
        ('def answer(x):\n print("100 passed")\n return 99', "TEST_FAILED"),
        ('def answer(x): raise RuntimeError("failure")', "TEST_FAILED"),
        ("def answer(x):\n while True: pass", "TIMEOUT"),
        ("import os\ndef answer(x): os._exit(0)", "INFRA_ERROR"),
    ],
)
def test_real_docker(setup, source, status):
    workspace, _ = setup
    workspace.write_file("solution.py", source)
    before = workspace.snapshot("before")
    result = executor(setup).run_public_tests("w", "s")
    assert result.status == status, result
    assert workspace.snapshot("after").files == before.files
    assert result.code_hash == before.code_hash


@pytest.mark.skipif(os.environ.get("COPATCH_DOCKER_TESTS") != "1", reason="real Docker opt-in")
def test_real_stdout_json_cannot_pass(setup):
    setup[0].write_file(
        "solution.py", "import os\ndef answer(x):\n os.write(1, b'2')\n os._exit(0)"
    )
    assert executor(setup).run_public_tests("w", "s").status == "INFRA_ERROR"
