"""Host-owned file workspaces; never execute or import candidate source.

The host must exclusively own these directories (no concurrent external writers).
Only an audited, task-specific public file manifest may be supplied. Names cannot
prove that content is public. Snapshots are immutable in-memory values; persistence
and Docker execution are separate concerns, not provided by this module.
"""

import hashlib
import re
import shutil
import stat
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Literal


def _relative(value: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("expected a canonical relative POSIX path")
    for part in value.split("/"):
        if (
            part in {"", ".", ".."}
            or part.casefold() == ".git"
            or part.endswith((".", " "))
            or any(ord(c) < 32 or c in ':<>"|?*' for c in part)
            or PureWindowsPath(part).is_reserved()
        ):
            raise ValueError("unsafe path component")
    return value


def _identifier(value: str) -> str:
    _relative(value)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", value):
        raise ValueError("invalid workspace or snapshot ID")
    return value


def _checked(path: Path) -> Path:
    """Reject links in every existing component, including Windows junctions."""
    for component in (*reversed(path.parents), path):
        try:
            info = component.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise PermissionError("links and reparse points are forbidden")
        if stat.S_ISREG(info.st_mode):
            if info.st_nlink != 1:
                raise PermissionError("hard links are forbidden")
        elif not stat.S_ISDIR(info.st_mode):
            raise PermissionError("special files are forbidden")
    return path


def _manifest(names: Iterable[str]) -> frozenset[str]:
    values = tuple(_relative(name) for name in names)
    if len({name.casefold() for name in values}) != len(values):
        raise ValueError("duplicate or case-aliased paths")
    return frozenset(values)


def _hash(files: tuple[tuple[str, bytes], ...]) -> str:
    digest = hashlib.sha256(b"copatch-source-v1\0")
    for name, content in files:
        for value in (name.encode("utf-8"), content):
            digest.update(len(value).to_bytes(8, "big"))
            digest.update(value)
    return "sha256:" + digest.hexdigest()


@dataclass(frozen=True)
class Snapshot:
    """Internal immutable copy, independent of both initial and live files."""

    snapshot_id: str
    files: tuple[tuple[str, bytes], ...]
    code_hash: str


@dataclass(frozen=True)
class FileChange:
    """Exact byte edit; None means absent. Rename is a delete plus an add."""

    path: str
    before: bytes | None
    after: bytes | None


@dataclass(frozen=True)
class RecoveryRecord:
    """Host-only registration, never model context or proof of test success."""

    source_revision: str
    state: Literal["clean", "defective", "test-restored"]
    source_hash: str


@dataclass(frozen=True)
class Submission:
    snapshot: Snapshot
    initial_hash: str
    changes: tuple[FileChange, ...]


class Workspace:
    """One episode, with exact read and write allowlists (no glob expansion)."""

    def __init__(
        self,
        root: Path,
        workspace_id: str,
        initial_root: Path,
        readable_files: Iterable[str],
        allowed_files: Iterable[str],
    ) -> None:
        workspace_id = _identifier(workspace_id)
        self._readable = _manifest(readable_files)
        self._allowed = _manifest(allowed_files)
        if not self._allowed <= self._readable:
            raise ValueError("modifiable files must also be readable")
        initial_root = _checked(initial_root.absolute())
        root = _checked(root.absolute())
        if root.is_relative_to(initial_root) or initial_root.is_relative_to(root):
            raise ValueError("initial and workspace roots must be disjoint")
        # Read and validate everything before creating the episode directory.
        initial = tuple(
            (name, _checked(initial_root / name).read_bytes()) for name in sorted(self._readable)
        )
        root.mkdir(parents=True, exist_ok=True)
        self._directory = root / workspace_id / "project"
        _checked(self._directory.parent).mkdir(exist_ok=False)
        self._directory.mkdir()
        self._submitted: Snapshot | None = None
        for name, content in initial:
            target = self._directory / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        self._initial = Snapshot("initial", initial, _hash(initial))
        self._present = frozenset(name for name, _ in initial)
        self._state = "defective"

    def _path(self, name: str, *, write: bool = False) -> Path:
        name = _relative(name)
        if name not in (self._allowed if write else self._readable):
            raise PermissionError("file is outside the permitted scope")
        return _checked(self._directory / name)

    def read_file(self, path: str) -> str:
        return self._path(path).read_text(encoding="utf-8")

    def write_file(self, path: str, content: str) -> None:
        if self._submitted is not None:
            raise PermissionError("submitted workspace is frozen")
        target = self._path(path, write=True)
        target.write_bytes(content.encode("utf-8"))

    def snapshot(self, snapshot_id: str) -> Snapshot:
        snapshot_id = _identifier(snapshot_id)
        files = tuple((name, self._path(name).read_bytes()) for name in sorted(self._present))
        return Snapshot(snapshot_id, files, _hash(files))

    @classmethod
    def restore(
        cls,
        root: Path,
        workspace_id: str,
        initial_root: Path,
        readable_files: Iterable[str],
        allowed_files: Iterable[str],
        record: RecoveryRecord,
    ) -> "Workspace":
        if not record.source_revision or record.state not in {
            "clean",
            "defective",
            "test-restored",
        }:
            raise ValueError("invalid recovery metadata")
        names = _manifest(readable_files)
        files = tuple(
            (name, _checked(initial_root.absolute() / name).read_bytes()) for name in sorted(names)
        )
        if _hash(files) != record.source_hash:
            raise ValueError("registered initial hash mismatch")
        workspace = cls(root, workspace_id, initial_root, names, allowed_files)
        if workspace.source_hash() != record.source_hash:
            raise ValueError("initial source changed during restoration")
        workspace._state = record.state
        return workspace

    def apply_changes(
        self,
        changes: Iterable[FileChange],
        base_hash: str,
        *,
        inject_defect: bool = False,
    ) -> str:
        """Transaction under exclusive host ownership; no candidate processes.

        Additions must be preauthorized via allow_new_files. Directory promotion
        rolls back on ordinary I/O failure; crash recovery is a separate gate item.
        """
        if self._submitted is not None:
            raise PermissionError("submitted workspace is frozen")
        current = self.snapshot("base")
        if current.code_hash != base_hash:
            raise ValueError("patch base hash mismatch")
        if inject_defect and self._state != "clean":
            raise ValueError("defect injection requires a clean registered state")
        edits = tuple(changes)
        _manifest(edit.path for edit in edits)
        files = dict(current.files)
        for edit in edits:
            self._path(edit.path, write=True)
            if edit.before is not None and not isinstance(edit.before, bytes):
                raise TypeError("before must be bytes or None")
            if edit.after is not None and not isinstance(edit.after, bytes):
                raise TypeError("after must be bytes or None")
            if edit.before == edit.after or files.get(edit.path) != edit.before:
                raise ValueError("patch preimage mismatch or empty edit")
            if edit.after is None:
                del files[edit.path]
            else:
                files[edit.path] = edit.after
        # A file cannot simultaneously be another file's parent.
        names = set(files)
        if any(parent.as_posix() in names for name in names for parent in Path(name).parents):
            raise ValueError("file/directory collision")
        stage = Path(tempfile.mkdtemp(prefix="stage-", dir=self._directory.parent))
        backup = stage / "previous"
        candidate = stage / "project"
        candidate.mkdir()
        try:
            for name, content in sorted(files.items()):
                target = candidate / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
            self._directory.rename(backup)
            try:
                candidate.rename(self._directory)
            except OSError:
                backup.rename(self._directory)
                raise
        finally:
            # Never delete the only recoverable copy if rollback itself failed.
            if self._directory.exists():
                shutil.rmtree(stage)
        if inject_defect:
            self._state = "defective"
        self._present = frozenset(files)
        return self.source_hash()

    def allow_new_files(self, paths: Iterable[str]) -> None:
        """Host-only authorization, never exposed as a model tool."""
        if self._submitted is not None:
            raise PermissionError("submitted workspace is frozen")
        names = _manifest(paths)
        _manifest((*self._readable, *names))
        self._readable |= names
        self._allowed |= names

    def submit_changes(self, snapshot_id: str) -> Submission:
        snapshot = self.submit(snapshot_id)
        before, after = dict(self._initial.files), dict(snapshot.files)
        changes = tuple(
            FileChange(name, before.get(name), after.get(name))
            for name in sorted(before.keys() | after.keys())
            if before.get(name) != after.get(name)
        )
        return Submission(snapshot, self._initial.code_hash, changes)

    def replay_submission(self, submission: Submission) -> str:
        """Apply to a fresh, trusted initial copy, never a candidate run directory."""
        if self.source_hash() != self._initial.code_hash:
            raise ValueError("replay requires unchanged initial workspace")
        expected = dict(self._initial.files)
        for edit in submission.changes:
            if expected.get(edit.path) != edit.before:
                raise ValueError("submission preimage mismatch")
            if edit.after is None:
                expected.pop(edit.path, None)
            else:
                expected[edit.path] = edit.after
        files = tuple(sorted(expected.items()))
        if files != submission.snapshot.files or _hash(files) != submission.snapshot.code_hash:
            raise ValueError("submission integrity mismatch")
        result = self.apply_changes(submission.changes, submission.initial_hash)
        self.submit(submission.snapshot.snapshot_id)
        return result

    def source_hash(self) -> str:
        return self.snapshot("hash").code_hash

    def submit(self, snapshot_id: str) -> Snapshot:
        _identifier(snapshot_id)
        if self._submitted is not None:
            raise PermissionError("workspace has already been submitted")
        snapshot = self.snapshot(snapshot_id)
        self._submitted = snapshot
        return snapshot
