"""Deterministic host-side grouping of C-06 private manifests; no sampling or deletion."""

import ast
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations


@dataclass(frozen=True)
class SplitPolicy:
    train: int = 7
    dev: int = 1
    test: int = 2

    def __post_init__(self) -> None:
        weights = (self.train, self.dev, self.test)
        if any(type(weight) is not int or weight < 0 for weight in weights):
            raise ValueError("split weights must be nonnegative integers")
        if sum(weights) == 0:
            raise ValueError("split weights must have a positive sum")


class FrozenConflict(ValueError):
    def __init__(self, report: dict) -> None:
        super().__init__("frozen manifest conflicts require explicit review")
        self.report = report


def _canonical(value: object) -> str:
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    )


def _repo(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("repository identity required")
    name = value.strip().lower().removeprefix("https://").removeprefix("http://")
    name = name.removeprefix("github.com/").removesuffix("/").removesuffix(".git")
    return name


class _Union:
    def __init__(self, values: Sequence[str]) -> None:
        self.parent = {v: v for v in values}

    def root(self, value: str) -> str:
        while self.parent[value] != value:
            value = self.parent[value]
        return value

    def join(self, a: str, b: str) -> None:
        a, b = self.root(a), self.root(b)
        self.parent[max(a, b)] = min(a, b)


def _features(code: str) -> tuple[str, str]:
    if not isinstance(code, str) or not code.strip():
        return "", ""
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, TypeError):
        return "", ""
    signatures = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            signatures.append(
                (
                    len(node.args.posonlyargs),
                    len(node.args.args),
                    len(node.args.kwonlyargs),
                    bool(node.args.vararg),
                    bool(node.args.kwarg),
                )
            )
            node.name = "function"
        elif isinstance(node, ast.Name):
            node.id = "name"
        elif isinstance(node, ast.arg):
            node.arg = "arg"
    return ast.dump(tree, include_attributes=False), _canonical(
        sorted(signatures)
    ) if signatures else ""


def dedupe(
    manifest: Sequence[Mapping] | bytes,
    *,
    aliases: Mapping[str, str] | None = None,
    image_aliases: Mapping[str, str] | None = None,
    frozen: bool = False,
    policy: SplitPolicy = SplitPolicy(),
) -> dict:
    """Consume C-06 private_index, not its six-field public provenance manifest.

    Fork/rename and image equivalences must be supplied by a reviewed host registry.
    Unrelated repositories are never inferred to be identical from spelling alone.
    Existing train/dev/test assignments are constraints; smoke is unassigned.
    """
    raw = json.loads(manifest) if isinstance(manifest, bytes) else list(manifest)
    rows = sorted(raw, key=_canonical)
    aliases = {_repo(k): _repo(v) for k, v in (aliases or {}).items()}
    images = {k: _repo(v) for k, v in (image_aliases or {}).items()}
    repo_names = set(aliases) | set(aliases.values()) | set(images.values())
    prepared = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("manifest row required")
        for key in ("task_id", "source", "original_task_id"):
            if not isinstance(row.get(key), str) or not row[key]:
                raise ValueError("missing " + key)
        task = row.get("task", {})
        split = task.get("split")
        if split not in {"train", "dev", "test", "smoke"}:
            raise ValueError("invalid split")
        source = row["source"]
        if source not in {"mbpp", "mbppplus", "swe-smith", "swe-gym"}:
            raise ValueError("unknown source")
        data = row.get("raw")
        if not isinstance(data, Mapping):
            raise ValueError("private raw manifest required")
        repo = None
        if source.startswith("swe-"):
            repo = _repo(data.get("repo"))
            repo_names.add(repo)
        prepared.append((row, task, split, data, repo))
    identities = _Union(sorted(repo_names))
    for a, b in sorted(aliases.items()):
        identities.join(a, b)
    for _, _, _, data, repo in prepared:
        image = data.get("image_name", data.get("image_digest"))
        if repo and image in images:
            identities.join(repo, images[image])
    keys = [str(i) for i in range(len(prepared))]
    groups = _Union(keys)
    entries = []
    duplicates = []
    candidates = []
    for i, (row, task, split, data, repo) in enumerate(prepared):
        # Hash task content, not provenance hash (which contains ID/revision/split).
        content = {
            k: data[k]
            for k in (
                "text",
                "prompt",
                "code",
                "test_list",
                "test",
                "test_imports",
                "test_setup_code",
                "challenge_test_list",
                "problem_statement",
                "patch",
                "test_patch",
                "FAIL_TO_PASS",
                "PASS_TO_PASS",
            )
            if k in data
        }
        if not content:
            raise ValueError("missing task content")
        family = task.get("problem_family", task.get("task_family"))
        if not isinstance(family, str) or not family:
            raise ValueError("problem family required")
        entries.append(
            dict(
                task_id=row["task_id"],
                source=row["source"],
                original=row["original_task_id"],
                split=split,
                repo_family=identities.root(repo) if repo else None,
                problem_family=family,
                content_hash=hashlib.sha256(_canonical(content).encode()).hexdigest(),
                features=_features(data.get("code", "")),
                template=re.sub(
                    r"\d+",
                    "#",
                    " ".join(
                        str(data.get("text", data.get("prompt", data.get("problem_statement", ""))))
                        .lower()
                        .split()
                    ),
                ),
            )
        )
    for i, j in combinations(range(len(entries)), 2):
        a, b = entries[i], entries[j]

        def namespace(x: dict) -> str:
            return "mbpp" if x["source"] in {"mbpp", "mbppplus"} else x["source"]

        reasons = []
        if namespace(a) == namespace(b) and a["original"] == b["original"]:
            reasons.append("source_id")
        if a["content_hash"] == b["content_hash"]:
            reasons.append("content_hash")
        if a["task_id"] == b["task_id"]:
            reasons.append("task_id")
        pair = dict(left=i, right=j, task_ids=[a["task_id"], b["task_id"]])
        if reasons:
            duplicates.append(dict(pair, reasons=reasons, decision="duplicate"))
            groups.join(str(i), str(j))
        else:
            near = []
            if a["features"][0] and a["features"][0] == b["features"][0]:
                near.append("normalized_ast")
            if a["features"][1] and a["features"][1] == b["features"][1]:
                near.append("signature")
            if a["template"] and a["template"] == b["template"]:
                near.append("description_template")
            if near:
                candidates.append(
                    dict(pair, reasons=near, decision="candidate", review_status="pending")
                )
        if (a["repo_family"] and a["repo_family"] == b["repo_family"]) or a["problem_family"] == b[
            "problem_family"
        ]:
            groups.join(str(i), str(j))
    components = {}
    for i in range(len(entries)):
        components.setdefault(groups.root(str(i)), []).append(i)
    assignments = []
    conflicts = []
    for members in sorted(components.values()):
        existing = sorted({entries[i]["split"] for i in members if entries[i]["split"] != "smoke"})
        split = None
        if len(existing) > 1:
            conflicts.append(
                dict(
                    members=members,
                    task_ids=[entries[i]["task_id"] for i in members],
                    splits=existing,
                    reason="linked_family_cross_split",
                )
            )
        elif existing:
            split = existing[0]
        else:
            digest = hashlib.sha256(
                _canonical(sorted({entries[i]["task_id"] for i in members})).encode()
            ).digest()
            bucket = int.from_bytes(digest[:8], "big") % (policy.train + policy.dev + policy.test)
            split = (
                "train"
                if bucket < policy.train
                else "dev"
                if bucket < policy.train + policy.dev
                else "test"
            )
        for i in members:
            assignments.append(
                dict(
                    index=i,
                    task_id=entries[i]["task_id"],
                    split=split,
                    original_split=entries[i]["split"],
                    repo_family=entries[i]["repo_family"],
                    problem_family=entries[i]["problem_family"],
                )
            )
    report = dict(
        version=1,
        duplicates=duplicates,
        candidates=candidates,
        assignments=sorted(assignments, key=lambda x: x["index"]),
        conflicts=conflicts,
        split_policy=(
            f"preserve assigned; SHA256 family bucket "
            f"{policy.train}/{policy.dev}/{policy.test} for smoke only"
        ),
    )
    if frozen and conflicts:
        raise FrozenConflict(report)
    return report
