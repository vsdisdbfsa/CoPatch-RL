"""Host-only Docker execution of audited public snapshots, never task grading."""

import json
import math
import re
import subprocess
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from copatch_rl.execution.workspace import Snapshot, _hash, _manifest


@dataclass(frozen=True)
class Limits:
    seconds: float = 10
    cpus: float = 1
    memory_mb: int = 128
    pids: int = 32
    output_bytes: int = 65536

    def __post_init__(self) -> None:
        for value in (self.seconds, self.cpus, self.memory_mb, self.pids, self.output_bytes):
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError("limits must be positive and finite")
        for value in (self.memory_mb, self.pids, self.output_bytes):
            if not isinstance(value, int):
                raise ValueError("integer limit required")


@dataclass(frozen=True)
class ExecutionResult:
    status: str
    exit_code: int | None
    output: bytes
    truncated: bool
    code_hash: str
    image_id: str
    platform: str


def _docker(*args: str) -> str:
    result = subprocess.run(
        ["docker", *args], capture_output=True, timeout=30, check=True, shell=False
    )
    return result.stdout.decode("utf-8")


def run_python(
    snapshot: Snapshot,
    entrypoint: str,
    image: str,
    limits: Limits = Limits(),
    *,
    cancel: threading.Event | None = None,
) -> ExecutionResult:
    """Execute one Python file. Inputs and image selection belong to trusted host.

    No private tests may be included. Source is read-only; scratch is bounded tmpfs.
    Output is untrusted bytes, never proof of passing tests. Cleanup failure raises.
    """
    if not isinstance(image, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/:@-]*", image):
        raise ValueError("invalid trusted image reference")
    names = _manifest(name for name, _ in snapshot.files)
    if entrypoint not in names or _hash(snapshot.files) != snapshot.code_hash:
        raise ValueError("invalid entrypoint or snapshot hash")
    if tuple(sorted(snapshot.files)) != snapshot.files:
        raise ValueError("snapshot must be sorted")
    name = "copatch-c03-" + uuid.uuid4().hex
    image_id = platform = ""
    output = bytearray()
    truncated = threading.Event()
    status, code = "INFRA_ERROR", None
    proc = None
    created = False
    with tempfile.TemporaryDirectory(prefix="copatch-exec-") as directory:
        root = Path(directory)
        root.chmod(0o755)
        for path, content in snapshot.files:
            target = root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            target.chmod(0o444)
            for parent in target.parents:
                if parent == root:
                    break
                parent.chmod(0o755)
        try:
            info = json.loads(_docker("image", "inspect", image))[0]
            image_id = info["Id"]
            platform = info["Os"] + "/" + info["Architecture"]
            if info["Os"] != "linux" or info["Config"].get("Volumes"):
                raise ValueError("Linux image without declared volumes required")
            # Mark before create: even a timed-out CLI may have created a container.
            created = True
            _docker(
                "create",
                "--name",
                name,
                "--pull=never",
                "--network=none",
                "--user=65534:65534",
                "--cap-drop=ALL",
                "--security-opt=no-new-privileges",
                "--read-only",
                "--ipc=private",
                "--stop-timeout=1",
                "--cpus",
                str(limits.cpus),
                "--memory",
                f"{limits.memory_mb}m",
                "--memory-swap",
                f"{limits.memory_mb}m",
                "--pids-limit",
                str(limits.pids),
                "--ulimit",
                "nofile=64:64",
                "--log-driver=none",
                "--mount",
                f"type=bind,source={root},target=/source,readonly",
                "--tmpfs",
                "/tmp:rw,noexec,nosuid,nodev,size=16m,mode=1777",
                "--workdir=/source",
                "--entrypoint=python",
                image_id,
                "-I",
                "-B",
                "/source/" + entrypoint,
            )
            proc = subprocess.Popen(
                ["docker", "start", "--attach", name],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                shell=False,
            )

            def drain() -> None:
                assert proc is not None and proc.stdout is not None
                while chunk := proc.stdout.read(4096):
                    remaining = limits.output_bytes - len(output)
                    output.extend(chunk[:remaining])
                    if len(chunk) > remaining:
                        truncated.set()

            reader = threading.Thread(target=drain, daemon=True)
            reader.start()
            deadline = time.monotonic() + limits.seconds
            while (
                proc.poll() is None
                and time.monotonic() < deadline
                and not truncated.is_set()
                and not (cancel and cancel.is_set())
            ):
                time.sleep(0.01)
            if cancel and cancel.is_set():
                status = "CANCELLED"
            elif truncated.is_set():
                status = "OUTPUT_LIMIT"
            elif proc.poll() is None:
                status = "TIMEOUT"
            else:
                state = json.loads(_docker("inspect", "--format", "{{json .State}}", name))
                code = state["ExitCode"]
                status = "RESOURCE_LIMIT" if state["OOMKilled"] else "EXITED"
                if state.get("Error") or state["Running"] or proc.returncode != code:
                    status = "INFRA_ERROR"
        except (OSError, subprocess.SubprocessError, json.JSONDecodeError, KeyError):
            status = "INFRA_ERROR"
        finally:
            cleanup_error = None
            if created:
                # Successful removal is required before releasing the execution copy.
                try:
                    _docker("rm", "--force", name)
                except (OSError, subprocess.SubprocessError) as exc:
                    cleanup_error = exc
            if proc is not None:
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)
                reader.join(timeout=5)
                if reader.is_alive():
                    raise RuntimeError("output reader did not stop")
                proc.stdout.close()
            if cleanup_error is not None:
                raise RuntimeError("container cleanup unconfirmed: " + name) from cleanup_error
        if truncated.is_set() and status == "EXITED":
            status = "OUTPUT_LIMIT"
        return ExecutionResult(
            status, code, bytes(output), truncated.is_set(), snapshot.code_hash, image_id, platform
        )
