"""Mock-only input and infrastructure failure checks."""

import subprocess

import pytest

from copatch_rl.execution import sandbox
from copatch_rl.execution.workspace import Snapshot, _hash


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), True])
def test_limits(value):
    with pytest.raises(ValueError):
        sandbox.Limits(seconds=value)


def test_bad_snapshot():
    with pytest.raises(ValueError):
        sandbox.run_python(Snapshot("x", (("../x", b""),), "bad"), "../x", "unused")


def test_missing_docker(monkeypatch):
    def missing(*args):
        raise FileNotFoundError("mock")

    monkeypatch.setattr(sandbox, "_docker", missing)
    files = (("main.py", b""),)
    result = sandbox.run_python(Snapshot("x", files, _hash(files)), "main.py", "mock")
    assert result.status == "INFRA_ERROR"


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_create_failure_cleanup(monkeypatch, cleanup_fails):
    calls = []

    def fake(*args):
        calls.append(args[0])
        if args[0] == "image":
            return '[{"Id":"sha256:mock","Os":"linux","Architecture":"amd64","Config":{}}]'
        if args[0] == "create" or cleanup_fails:
            raise subprocess.CalledProcessError(1, "mock")
        return ""

    monkeypatch.setattr(sandbox, "_docker", fake)
    files = (("main.py", b""),)
    if cleanup_fails:
        with pytest.raises(RuntimeError, match="cleanup unconfirmed"):
            sandbox.run_python(Snapshot("x", files, _hash(files)), "main.py", "mock")
    else:
        assert (
            sandbox.run_python(Snapshot("x", files, _hash(files)), "main.py", "mock").status
            == "INFRA_ERROR"
        )
    assert calls == ["image", "create", "rm"]


@pytest.mark.parametrize("mode", ["start_failure", "timeout", "cancel", "cleanup_failure"])
def test_mock_lifecycle(monkeypatch, mode):
    import io
    import threading
    from unittest.mock import Mock

    calls = []

    def fake(*args):
        calls.append(args)
        if args[0] == "image":
            return '[{"Id":"sha256:mock","Os":"linux","Architecture":"amd64","Config":{}}]'
        if args[0] == "rm" and mode == "cleanup_failure":
            raise subprocess.CalledProcessError(1, "mock")
        return ""

    proc = Mock()
    proc.stdout = io.BytesIO(b"")
    proc.poll.return_value = None

    def launch(*args, **kwargs):
        if mode == "start_failure":
            raise OSError("mock start failure")
        return proc

    monkeypatch.setattr(sandbox, "_docker", fake)
    monkeypatch.setattr(sandbox.subprocess, "Popen", launch)
    cancel = threading.Event()
    if mode in {"cancel", "cleanup_failure"}:
        cancel.set()
    files = (("main.py", b""),)
    args = (Snapshot("x", files, _hash(files)), "main.py", "mock", sandbox.Limits(seconds=0.01))
    if mode == "cleanup_failure":
        with pytest.raises(RuntimeError, match="cleanup unconfirmed"):
            sandbox.run_python(*args, cancel=cancel)
        proc.wait.assert_called()
    else:
        result = sandbox.run_python(*args, cancel=cancel)
        assert (
            result.status
            == {"start_failure": "INFRA_ERROR", "timeout": "TIMEOUT", "cancel": "CANCELLED"}[mode]
        )
    assert calls[-1][0] == "rm"
    create = next(call for call in calls if call[0] == "create")
    assert "--network=none" in create
    assert "--cap-drop=ALL" in create
    assert "--read-only" in create
