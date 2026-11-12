"""Native pytest diagnostic adapter: fail closed on report trust."""

import json
import os
from dataclasses import replace

import pytest

from copatch_rl.execution.public_tests import Binding, Executor
from copatch_rl.execution.repository_tests import RepositorySuite, parse_report, prepare
from copatch_rl.execution.sandbox import Limits, _docker
from copatch_rl.execution.workspace import Snapshot, Workspace, _hash

NODE = "public_tests/test_math.py::test_value"


def report(outcome="passed", phase="call", xfail=False):
    rows = [
        [NODE, p, outcome if p == phase else "passed", xfail if p == phase else False]
        for p in ("setup", "call", "teardown")
    ]
    return {
        "collected": [NODE],
        "reports": rows,
        "collection_errors": 0,
        "exit_code": 1 if outcome == "failed" else 0,
    }


@pytest.mark.parametrize(
    "outcome,phase,xfail,status",
    [
        ("passed", "call", False, "SUCCESS"),
        ("failed", "call", False, "TEST_FAILED"),
        ("skipped", "call", False, "TEST_FAILED"),
        ("skipped", "call", True, "TEST_FAILED"),
        ("passed", "call", True, "TEST_FAILED"),
        ("failed", "setup", False, "INFRA_ERROR"),
        ("failed", "teardown", False, "INFRA_ERROR"),
    ],
)
def test_phases(outcome, phase, xfail, status):
    assert parse_report(json.dumps(report(outcome, phase, xfail)).encode(), (NODE,))[0] == status


@pytest.mark.parametrize("change", ["empty", "duplicate", "missing", "collection", "bad_json"])
def test_bad_reports(change):
    value = report()
    if change == "empty":
        value["collected"] = []
    elif change == "duplicate":
        value["reports"].append(value["reports"][0])
    elif change == "missing":
        value["collected"] = ["other"]
    elif change == "collection":
        value["collection_errors"] = 1
    with pytest.raises(ValueError):
        parse_report(b"100 passed" if change == "bad_json" else json.dumps(value).encode(), (NODE,))


def suite(image="sha256:" + "0" * 64, text="def test_value(): pass"):
    files = (("public_tests/test_math.py", text.encode()),)
    return RepositorySuite("s", "t", files, _hash(files), (NODE,), image)


@pytest.mark.parametrize("change", ["hash", "empty", "duplicate", "config", "collision"])
def test_manifest(change):
    s = suite()
    files = (("maths.py", b""),)
    if change == "hash":
        s = replace(s, tests_hash="bad")
    elif change == "empty":
        s = replace(s, nodeids=())
    elif change == "duplicate":
        s = replace(s, nodeids=(NODE, NODE))
    elif change == "config":
        files = (("conftest.py", b""),)
    else:
        files = s.tests
    with pytest.raises(ValueError):
        prepare(Snapshot("x", files, _hash(files)), s)


@pytest.mark.skipif(os.environ.get("COPATCH_DOCKER_TESTS") != "1", reason="real Docker opt-in")
@pytest.mark.parametrize(
    "text,source,status",
    [
        (
            "from maths import answer\ndef test_value(): assert answer()==2",
            "from helper import value\ndef answer(): return value",
            "INFRA_ERROR",
        ),
        (
            "from maths import answer\ndef test_value(): assert answer()==2",
            "def answer(): return 9",
            "TEST_FAILED",
        ),
        ("import pytest\n@pytest.mark.skip\ndef test_value(): pass", "", "TEST_FAILED"),
        ("import pytest\n@pytest.mark.xfail\ndef test_value(): assert False", "", "TEST_FAILED"),
        ("import pytest\n@pytest.mark.xfail\ndef test_value(): pass", "", "TEST_FAILED"),
        ("def setup_function(): raise RuntimeError()\ndef test_value(): pass", "", "INFRA_ERROR"),
        (
            "def teardown_function(): raise RuntimeError()\ndef test_value(): pass",
            "",
            "INFRA_ERROR",
        ),
        ("import nonexistent_module\ndef test_value(): pass", "", "INFRA_ERROR"),
        ("def test_value():\n while True: pass", "", "TIMEOUT"),
        ("import os\ndef test_value(): os._exit(0)", "", "INFRA_ERROR"),
        ('def test_value():\n print("100 passed")\n assert False', "", "TEST_FAILED"),
        ('def test_value():\n open(__file__, "w").write("bad")', "", "TEST_FAILED"),
    ],
)
def test_real_repository(tmp_path, text, source, status):
    initial = tmp_path / "initial"
    initial.mkdir()
    (initial / "maths.py").write_text(source)
    (initial / "helper.py").write_text("value=2")
    w = Workspace(tmp_path / "work", "w", initial, ["maths.py", "helper.py"], ["maths.py"])
    before = w.snapshot("before")
    image = json.loads(_docker("image", "inspect", "copatch-public-pytest:dev"))[0]["Id"]
    s = suite(image, text)
    e = Executor({"w": Binding("t", "s", w)}, {"s": s}, image, Limits(seconds=3))
    result = e.run_public_tests("w", "s")
    assert result.status == status, result
    if "from helper" in source:
        assert result.passed == 1, result
        assert "all-pass" in result.summary
    assert result.diagnostic_only
    assert result.code_hash == before.code_hash
    assert w.snapshot("after").files == before.files
    assert not _docker("ps", "-aq", "--filter", "name=copatch-c03-").strip()
