"""C-02 filesystem tests. Source strings are never executed."""

from pathlib import Path

import pytest

from copatch_rl.execution.workspace import FileChange, RecoveryRecord, Workspace


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    source.mkdir()
    for name in ("a.py", "b.py", "test_a.py", "pyproject.toml"):
        (source / name).write_bytes(b"original\n")
    return source


def create(repo: Path, name: str = "one") -> Workspace:
    return Workspace(
        repo.parent / "runs",
        name,
        repo,
        ["a.py", "b.py", "test_a.py", "pyproject.toml"],
        ["a.py", "b.py"],
    )


def test_four_copies_and_replay(repo: Path) -> None:
    workspaces = [create(repo, f"run{i}") for i in range(4)]
    original = workspaces[0].source_hash()
    assert all(w.source_hash() == original for w in workspaces)
    workspaces[0].apply_changes([FileChange("a.py", b"original\n", b"fixed\n")], original)
    submission = workspaces[0].submit_changes("final")
    assert all(w.source_hash() == original for w in workspaces[1:])
    assert workspaces[1].replay_submission(submission) == submission.snapshot.code_hash
    with pytest.raises(PermissionError):
        workspaces[1].apply_changes([], submission.snapshot.code_hash)
    assert submission.initial_hash == original
    assert (repo / "a.py").read_bytes() == b"original\n"


@pytest.mark.parametrize("state", ["clean", "defective", "test-restored"])
def test_registered_state_and_injection(repo: Path, state: str) -> None:
    baseline = create(repo).source_hash()
    record = RecoveryRecord("handwritten-v1", state, baseline)
    workspace = Workspace.restore(
        repo.parent / "runs",
        "restored",
        repo,
        ["a.py", "b.py", "test_a.py", "pyproject.toml"],
        ["a.py", "b.py"],
        record,
    )
    edits = [FileChange("a.py", b"original\n", b"defect\n")]
    if state == "clean":
        workspace.apply_changes(edits, baseline, inject_defect=True)
    with pytest.raises(ValueError):
        workspace.apply_changes(edits, workspace.source_hash(), inject_defect=True)
    with pytest.raises(ValueError):
        Workspace.restore(repo.parent / "runs", "bad", repo, ["a.py"], ["a.py"], record)


@pytest.mark.parametrize("path", ["test_a.py", "pyproject.toml", "../escape", "C:/escape"])
def test_rejected_paths_leave_no_edits(repo: Path, path: str) -> None:
    workspace = create(repo)
    before = workspace.source_hash()
    with pytest.raises((ValueError, PermissionError)):
        workspace.apply_changes(
            [FileChange("a.py", b"original\n", b"new"), FileChange(path, b"original\n", b"bad")],
            before,
        )
    assert workspace.source_hash() == before


def test_add_delete_rename_and_preimages(repo: Path) -> None:
    workspace = create(repo)
    before = workspace.source_hash()
    with pytest.raises(ValueError):
        workspace.apply_changes([FileChange("a.py", b"original\n", b"new")], "wrong")
    with pytest.raises(ValueError):
        workspace.apply_changes(
            [FileChange("a.py", b"original\n", b"new"), FileChange("b.py", b"wrong", None)], before
        )
    assert workspace.source_hash() == before
    workspace.allow_new_files(["nested/renamed.py"])
    workspace.apply_changes(
        [
            FileChange("a.py", b"original\n", None),
            FileChange("nested/renamed.py", None, b"original\n"),
        ],
        before,
    )
    assert workspace.read_file("nested/renamed.py") == "original\n"
    with pytest.raises(FileNotFoundError):
        workspace.read_file("a.py")
    submission = workspace.submit_changes("rename")
    fresh = create(repo, "fresh")
    fresh.allow_new_files(["nested/renamed.py"])
    assert fresh.replay_submission(submission) == submission.snapshot.code_hash


def test_promotion_failure_rolls_back(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace = create(repo)
    before = workspace.source_hash()
    rename = Path.rename

    def fail_promotion(path: Path, target: Path) -> Path:
        if path.name == "project" and path.parent.name.startswith("stage-"):
            raise OSError("synthetic promotion failure")
        return rename(path, target)

    monkeypatch.setattr(Path, "rename", fail_promotion)
    with pytest.raises(OSError, match="synthetic"):
        workspace.apply_changes([FileChange("a.py", b"original\n", b"new")], before)
    assert workspace.source_hash() == before
    assert not list((repo.parent / "runs/one").glob("stage-*"))


def test_missing_file_is_not_an_authorized_deletion(repo: Path) -> None:
    workspace = create(repo)
    (repo.parent / "runs/one/project/a.py").unlink()
    with pytest.raises(FileNotFoundError):
        workspace.snapshot("missing")
