"""Real Docker tests: explicitly enable with COPATCH_DOCKER_TESTS=1."""

import os
import threading

import pytest

from copatch_rl.execution.sandbox import Limits, _docker, run_python
from copatch_rl.execution.workspace import Snapshot, _hash

pytestmark = pytest.mark.skipif(
    os.environ.get("COPATCH_DOCKER_TESTS") != "1", reason="real Docker opt-in"
)


def execute(source, limits=Limits(), **kwargs):
    files = (("main.py", source.encode()),)
    snapshot = Snapshot("real", files, _hash(files))
    result = run_python(snapshot, "main.py", "python:3.11-slim", limits, **kwargs)
    assert snapshot.files == files
    assert result.code_hash == _hash(files)
    assert result.image_id.startswith("sha256:")
    assert not _docker("ps", "-aq", "--filter", "name=copatch-c03-").strip()
    return result


@pytest.mark.parametrize(
    "source,code",
    [("print('hello')", 0), ("this is invalid syntax", 1), ("raise RuntimeError('fixture')", 1)],
)
def test_exit(source, code):
    result = execute(source)
    assert result.status == "EXITED"
    assert result.exit_code == code


def test_security():
    result = execute("""import os, socket, pathlib, subprocess
assert os.getuid() != 0
for path in ['/source/main.py', '/etc/copatch-test', '/source/../../etc/copatch-test']:
    try:
        pathlib.Path(path).write_text('bad')
    except OSError:
        pass
    else:
        raise AssertionError(path)
os.symlink('/etc/copatch-test', '/tmp/link')
try:
    pathlib.Path('/tmp/link').write_text('bad')
except OSError:
    pass
else:
    raise AssertionError('symlink write')
assert not pathlib.Path('/var/run/docker.sock').exists()
assert not pathlib.Path('/source/.git').exists()
s = socket.socket()
s.settimeout(1)
assert s.connect_ex(('1.1.1.1', 443)) != 0
assert 'CapEff:\\t0000000000000000' in pathlib.Path('/proc/self/status').read_text()
assert 'NoNewPrivs:\\t1' in pathlib.Path('/proc/self/status').read_text()
subprocess.Popen(['python', '-c', 'import time; time.sleep(60)'])
print('security checks passed')
""")
    assert result.exit_code == 0, result.output


def test_timeout():
    assert execute("while True: pass", Limits(seconds=1)).status == "TIMEOUT"


def test_output():
    result = execute("while True: print('x'*4096, flush=True)", Limits(output_bytes=8192))
    assert result.status == "OUTPUT_LIMIT"
    assert len(result.output) == 8192


def test_cancel():
    cancel = threading.Event()
    cancel.set()
    assert execute("while True: pass", cancel=cancel).status == "CANCELLED"


def test_memory():
    result = execute("x=bytearray(512*1024*1024)", Limits(memory_mb=64))
    assert result.status == "RESOURCE_LIMIT", result


def test_resource_configuration():
    result = execute("""from pathlib import Path
assert int(Path('/sys/fs/cgroup/pids.max').read_text()) == 32
assert int(Path('/sys/fs/cgroup/memory.max').read_text()) == 128*1024*1024
quota, period = map(int, Path('/sys/fs/cgroup/cpu.max').read_text().split())
assert quota / period == 1
""")
    assert result.exit_code == 0, result.output


def test_pid_exhaustion():
    result = execute("""import subprocess
children = []
try:
    for _ in range(100):
        children.append(subprocess.Popen(['python', '-c', 'import time; time.sleep(30)']))
except OSError:
    print('pid limit enforced')
else:
    raise AssertionError('unlimited processes')
finally:
    for child in children:
        child.kill()
        child.wait()
""")
    assert result.exit_code == 0, result.output
    assert b"pid limit enforced" in result.output


def test_import_side_effect_blocked():
    result = execute("""import sys, pathlib
pathlib.Path('/tmp/evil.py').write_text("open('/source/main.py', 'w').write('changed')")
sys.path.insert(0, '/tmp')
try:
    import evil
except OSError:
    print('import write denied')
else:
    raise AssertionError('source changed')
""")
    assert result.exit_code == 0, result.output
    assert b"import write denied" in result.output
