"""Handwritten source text only: no candidate execution and no model calls."""

import os
from pathlib import Path

import pytest

from copatch_rl.execution.workspace import Workspace


@pytest.fixture
def initial(tmp_path: Path) -> Path:
    source = tmp_path / "initial"
    source.mkdir()
    (source / "main.py").write_bytes(b"value = 1\n")
    (source / "README.md").write_text("public context", encoding="utf-8")
    for name in (".git", "reference", "private", "other_task"):
        (source / name).mkdir()
        (source / name / "secret.txt").write_text("excluded fixture", encoding="utf-8")
    return source


def make(initial: Path, episode: str = "episode") -> Workspace:
    return Workspace(
        initial.parent / "work", episode, initial, ["main.py", "README.md"], ["main.py"]
    )


def test_copy_isolation_snapshot_and_submit(initial: Path) -> None:
    first, second = make(initial), make(initial, "second")
    project = initial.parent / "work/episode/project"
    assert sorted(p.name for p in project.iterdir()) == ["README.md", "main.py"]
    before = first.snapshot("before")
    assert before.code_hash == second.source_hash()
    first.write_file("main.py", "value = 2\n")
    assert first.source_hash() != before.code_hash
    assert second.read_file("main.py") == "value = 1\n"
    assert (initial / "main.py").read_text() == "value = 1\n"
    assert dict(before.files)["main.py"] == b"value = 1\n"
    submitted = first.submit("submitted")
    assert dict(submitted.files)["main.py"] == b"value = 2\n"
    with pytest.raises(PermissionError):
        first.write_file("main.py", "changed")
    with pytest.raises(PermissionError):
        first.submit("again")
    # Even an out-of-band host edit cannot mutate the submitted byte snapshot.
    (project / "main.py").write_text("host change")
    assert dict(submitted.files)["main.py"] == b"value = 2\n"


@pytest.mark.parametrize(
    "path",
    [
        "../x",
        "/x",
        "C:/x",
        "x\\y",
        "x//y",
        "./x",
        "a/../x",
        "x:stream",
        "NUL",
        "x.",
        "x ",
        ".git/a",
    ],
)
def test_invalid_paths(initial: Path, path: str) -> None:
    workspace = make(initial)
    with pytest.raises(ValueError):
        workspace.read_file(path)
    with pytest.raises(ValueError):
        workspace.write_file(path, "text")


@pytest.mark.parametrize("identifier", ["../escape", "/abs", "a/b", "NUL", "", "a.b"])
def test_invalid_ids(initial: Path, identifier: str) -> None:
    with pytest.raises(ValueError):
        make(initial, identifier)
    with pytest.raises(ValueError):
        make(initial).snapshot(identifier)


def test_permissions_and_collision(initial: Path) -> None:
    workspace = make(initial)
    assert workspace.read_file("README.md") == "public context"
    for name in ("README.md", "private/secret.txt", "new.py"):
        with pytest.raises(PermissionError):
            workspace.write_file(name, "bad")
    with pytest.raises(PermissionError):
        workspace.read_file("private/secret.txt")
    with pytest.raises(FileExistsError):
        make(initial)
    with pytest.raises(ValueError):
        Workspace(initial.parent / "work", "bad", initial, ["main.py"], ["new.py"])
    with pytest.raises(ValueError):
        Workspace(initial.parent / "work", "bad", initial, ["main.py", "MAIN.py"], [])


def test_hash_order_and_bytes(initial: Path) -> None:
    first = make(initial)
    other = Workspace(
        initial.parent / "work", "other", initial, ["README.md", "main.py"], ["main.py"]
    )
    assert first.source_hash() == other.source_hash()
    other.write_file("main.py", "value = 1\r\n")
    assert first.source_hash() != other.source_hash()


@pytest.mark.parametrize("directory", [False, True])
def test_symlink_escape(initial: Path, directory: bool) -> None:
    workspace = make(initial)
    project = initial.parent / "work/episode/project"
    target = project if directory else project / "main.py"
    if directory:
        target.rename(project.with_name("saved"))
    else:
        target.unlink()
    try:
        target.symlink_to(
            initial if directory else initial / "main.py", target_is_directory=directory
        )
    except OSError as exc:
        if os.name == "nt" and exc.winerror == 1314:
            pytest.skip("Windows symlink privilege unavailable")
        raise
    for operation in (
        lambda: workspace.read_file("main.py"),
        lambda: workspace.write_file("main.py", "bad"),
        lambda: workspace.submit("snapshot"),
    ):
        with pytest.raises(PermissionError):
            operation()


def test_initial_hardlink_rejected(initial: Path) -> None:
    os.link(initial / "main.py", initial / "alias.py")
    with pytest.raises(PermissionError):
        make(initial)
    assert not (initial.parent / "work").exists()


def test_nested_copy_and_failed_initialization(initial: Path) -> None:
    (initial / "pkg").mkdir()
    (initial / "pkg/module.py").write_bytes(b"# handwritten fixture\n")
    workspace = Workspace(
        initial.parent / "work", "nested", initial, ["pkg/module.py"], ["pkg/module.py"]
    )
    assert workspace.read_file("pkg/module.py") == "# handwritten fixture\n"
    workspace.write_file("pkg/module.py", "# changed\n")
    assert (initial / "pkg/module.py").read_bytes() == b"# handwritten fixture\n"
    with pytest.raises(FileNotFoundError):
        Workspace(initial.parent / "work", "missing", initial, ["absent.py"], [])
    assert not (initial.parent / "work/missing").exists()
    with pytest.raises(ValueError):
        Workspace(initial / "work", "overlap", initial, ["main.py"], [])


def test_live_hardlink_rejected(initial: Path) -> None:
    workspace = make(initial)
    candidate = initial.parent / "work/episode/project/main.py"
    candidate.unlink()
    os.link(initial / "main.py", candidate)
    with pytest.raises(PermissionError):
        workspace.write_file("main.py", "bad")
    with pytest.raises(PermissionError):
        workspace.snapshot("bad")
    assert (initial / "main.py").read_bytes() == b"value = 1\n"
