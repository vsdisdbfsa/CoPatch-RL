"""Offline source adapters. Metadata and revision allowlists belong to trusted host."""

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from copatch_rl.contracts import Budget, TaskSpec


@dataclass(frozen=True)
class Ingested:
    tasks: tuple[TaskSpec, ...]
    public_manifest: bytes
    private_index: bytes


def _bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _text(row: Mapping, name: str) -> str:
    value = row.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError("missing or invalid " + name)
    return value


def _strings(value: object) -> list[str]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError as exc:
            raise ValueError("invalid list") from exc
    if not isinstance(value, list) or any(not isinstance(x, str) or not x.strip() for x in value):
        raise ValueError("expected string list")
    return value


def ingest(
    source: str,
    rows: Sequence[Mapping],
    *,
    revision: str,
    license_id: str,
    known_revisions: Mapping[str, set[str]],
    metadata: Mapping[str, Mapping],
    formal: bool = False,
) -> Ingested:
    if source not in {"mbpp", "mbppplus", "swe-smith", "swe-gym"}:
        raise ValueError("unknown source")
    if not isinstance(revision, str) or revision not in known_revisions.get(source, set()):
        raise ValueError("unknown revision")
    if not isinstance(license_id, str) or not license_id.strip():
        raise ValueError("license required")
    if source == "swe-gym" and formal:
        raise ValueError("SWE-Gym pending review cannot enter formal manifest")
    records = []
    seen = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("row must be a mapping")
        function = source in {"mbpp", "mbppplus"}
        raw_id = row.get("task_id" if function else "instance_id")
        if function:
            if isinstance(raw_id, bool) or not isinstance(raw_id, (int, str)):
                raise ValueError("invalid original task ID")
            match = re.fullmatch(r"(?:Mbpp/|MBPP/)?([0-9]+)", str(raw_id))
            if match is None:
                raise ValueError("unresolved MBPP original ID")
            original = str(int(match[1]))
        else:
            original = _text(row, "instance_id")
        if original in seen:
            raise ValueError("duplicate task_id")
        seen.add(original)
        host = metadata.get(str(raw_id))
        if not isinstance(host, Mapping):
            raise ValueError("trusted task metadata required")
        description = _text(
            row,
            "text"
            if source == "mbpp"
            else "prompt"
            if source == "mbppplus"
            else "problem_statement",
        )
        direction = None
        if function:
            _text(row, "code")
            _strings(row.get("test_list"))
        else:
            _text(row, "repo")
            _text(row, "patch")
            _strings(row.get("FAIL_TO_PASS"))
            _strings(row.get("PASS_TO_PASS"))
            direction = "inject_defect" if source == "swe-smith" else "repair"
            if row.get("patch_direction", direction) != direction:
                raise ValueError("patch direction conflicts with source")
        if host.get("description_reviewed") is not True:
            raise ValueError("description requires trusted review")
        if re.search(
            r"diff --git|@@ |```|reference (?:answer|implementation)|参考(?:答案|实现)|答案[:：]",
            description,
            re.I,
        ):
            raise ValueError("description needs leakage review")
        secrets = [row.get("code"), row.get("patch"), row.get("test_patch")]
        secrets += row.get("test_list", []) if isinstance(row.get("test_list"), list) else []
        if any(isinstance(v, str) and v.strip() and v.strip() in description for v in secrets):
            raise ValueError("answer or test in description")
        audit = "pending" if source == "swe-gym" else host.get("audit_status", "pending")
        if audit not in {"approved", "pending", "rejected"}:
            raise ValueError("invalid audit status")
        if formal and audit != "approved":
            raise ValueError("formal rows require approval")
        commit = row.get("base_commit")
        if commit is not None and (not isinstance(commit, str) or not commit.strip()):
            raise ValueError("invalid commit")
        for field in (
            "image_name",
            "image_digest",
            "test_patch",
            "test_setup_code",
            "test",
            "hints_text",
        ):
            if field in row and not isinstance(row[field], str):
                raise ValueError("invalid " + field)
        allowed = _strings(host.get("allowed_files"))
        if not allowed or any(
            p.startswith(("/", "\\"))
            or ":" in p
            or "\\" in p
            or any(x in {"", "..", ".", ".git"} for x in p.split("/"))
            for p in allowed
        ):
            raise ValueError("invalid allowed files")
        try:
            budget = Budget.model_validate(host["budget"])
        except (KeyError, ValueError, TypeError) as exc:
            raise ValueError("host budget required") from exc
        spec = TaskSpec(
            task_id=source + ":" + original,
            split=_text(host, "split"),
            task_family=_text(host, "task_family"),
            difficulty=_text(host, "difficulty"),
            kind="function_impl" if function else "repo_repair",
            source_revision=revision,
            description=description,
            entrypoint=_text(host, "entrypoint"),
            allowed_files=tuple(allowed),
            initial_snapshot=_text(host, "initial_snapshot"),
            public_suite_id=_text(host, "public_suite_id"),
            private_suite_id=_text(host, "private_suite_id"),
            budget=budget,
        )
        public = {
            "task_id": spec.task_id,
            "source": source,
            "revision": revision,
            "license": license_id,
            "split": spec.split,
        }
        # Public hash covers public metadata + contract public view, never secret assets.
        public["hash"] = (
            "sha256:"
            + hashlib.sha256(_bytes([public, spec.public_view().model_dump()])).hexdigest()
        )
        private = {
            "task_id": spec.task_id,
            "original_task_id": original,
            "source": source,
            "revision": revision,
            "license": license_id,
            "audit_status": audit,
            "patch_direction": direction,
            "raw": dict(row),
            "task": spec.model_dump(),
            "recovery": {
                "commit": commit,
                "commit_status": "unresolved" if commit is None else "unverified",
                "image": row.get("image_digest", row.get("image_name")),
            },
            "raw_hash": "sha256:" + hashlib.sha256(_bytes(dict(row))).hexdigest(),
        }
        records.append((spec, public, private))
    records.sort(key=lambda item: item[0].task_id)
    return Ingested(
        tuple(r[0] for r in records),
        _bytes([r[1] for r in records]),
        _bytes([r[2] for r in records]),
    )
