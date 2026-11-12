"""Public repository diagnostics. Native pytest is NOT an adversarial trust boundary."""

import json
import re
from dataclasses import dataclass

from copatch_rl.execution.workspace import Snapshot, _hash, _manifest


@dataclass(frozen=True)
class RepositorySuite:
    suite_id: str
    task_id: str
    tests: tuple[tuple[str, bytes], ...]
    tests_hash: str
    nodeids: tuple[str, ...]
    image_id: str


def prepare(snapshot: Snapshot, suite: RepositorySuite) -> Snapshot:
    names = _manifest(name for name, _ in suite.tests)
    if not names or _hash(suite.tests) != suite.tests_hash:
        raise ValueError("bad test manifest")
    if tuple(sorted(suite.tests)) != suite.tests:
        raise ValueError("unsorted test manifest")
    if not suite.nodeids or len(set(suite.nodeids)) != len(suite.nodeids):
        raise ValueError("empty or duplicate node IDs")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", suite.image_id):
        raise ValueError("immutable image ID required")
    for name in names:
        if not name.startswith("public_tests/") or not name.endswith(".py"):
            raise ValueError("tests must be Python files in public_tests")
        if name.rsplit("/", 1)[-1] == "conftest.py":
            raise ValueError("conftest requires a separate reviewed adapter")
    for node in suite.nodeids:
        if node.split("::")[0] not in names or "::" not in node:
            raise ValueError("unregistered node ID")
    for name, _ in snapshot.files:
        if name.startswith(("public_tests/", "__copatch_")):
            raise ValueError("reserved path collision")
        if name.rsplit("/", 1)[-1] in {
            "conftest.py",
            "pytest.ini",
            "pyproject.toml",
            "setup.cfg",
            "tox.ini",
        }:
            raise ValueError("candidate test configuration forbidden")
    driver = _DRIVER.replace("EXPECTED_NODES", repr(list(suite.nodeids)))
    files = tuple(sorted((*snapshot.files, *suite.tests, ("__copatch_repo__.py", driver.encode()))))
    return Snapshot("repository-public", files, _hash(files))


_DRIVER = r"""import os, sys, json
os.environ['PYTEST_DISABLE_PLUGIN_AUTOLOAD'] = '1'
os.environ.pop('PYTEST_ADDOPTS', None)
os.environ.pop('PYTEST_PLUGINS', None)
import pytest
# Load framework before adding the candidate import root.
sys.path.insert(0, '/source')
output = os.dup(1)
null = os.open(os.devnull, os.O_WRONLY)
os.dup2(null, 1)
os.dup2(null, 2)
write = os.write
encode = json.dumps
class Recorder:
    def __init__(self):
        self.collected = []
        self.reports = []
        self.collection_errors = 0
    def pytest_collection_finish(self, session):
        self.collected = [item.nodeid for item in session.items]
    def pytest_collectreport(self, report):
        if report.failed:
            self.collection_errors += 1
    def pytest_runtest_logreport(self, report):
        self.reports.append([report.nodeid, report.when, report.outcome,
                             hasattr(report, 'wasxfail')])
r = Recorder()
code = pytest.main(['-c', '/dev/null', '--rootdir=/source', '--noconftest',
                    '-p', 'no:cacheprovider', '--import-mode=importlib',
                    *EXPECTED_NODES], plugins=[r])
payload = encode({'collected': r.collected, 'reports': r.reports,
                  'collection_errors': r.collection_errors, 'exit_code': int(code)}).encode()
while payload:
    payload = payload[write(output, payload):]
"""


def parse_report(data: bytes, nodes: tuple[str, ...]) -> tuple[str, int, int, int]:
    report = json.loads(data)
    if not isinstance(report, dict) or set(report) != {
        "collected",
        "reports",
        "collection_errors",
        "exit_code",
    }:
        raise ValueError("invalid report schema")
    collected = report["collected"]
    if (
        report["collection_errors"] != 0
        or not isinstance(collected, list)
        or len(collected) != len(nodes)
        or set(collected) != set(nodes)
    ):
        raise ValueError("collection mismatch")
    phases = {node: {} for node in nodes}
    for row in report["reports"]:
        if not isinstance(row, list) or len(row) != 4:
            raise ValueError("invalid phase")
        node, when, outcome, xfail = row
        if (
            node not in phases
            or when not in {"setup", "call", "teardown"}
            or outcome not in {"passed", "failed", "skipped"}
            or type(xfail) is not bool
            or when in phases[node]
        ):
            raise ValueError("unknown or duplicate phase")
        phases[node][when] = (outcome, xfail)
    passed = failed = incomplete = 0
    infrastructure = False
    for values in phases.values():
        if values == {phase: ("passed", False) for phase in ("setup", "call", "teardown")}:
            passed += 1
        elif any(values.get(phase, (None,))[0] == "failed" for phase in ("setup", "teardown")):
            incomplete += 1
            infrastructure = True
        elif values.get("call", (None,))[0] == "failed":
            failed += 1
        else:
            incomplete += 1
    code = report["exit_code"]
    if type(code) is not int or code not in {0, 1}:
        infrastructure = True
    if code == 1 and passed == len(nodes):
        infrastructure = True
    status = (
        "INFRA_ERROR" if infrastructure else "TEST_FAILED" if failed or incomplete else "SUCCESS"
    )
    return status, passed, failed, incomplete
