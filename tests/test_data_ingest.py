import json

import pytest

from copatch_rl.data.ingest import ingest


def batch(source="mbpp"):
    if source in {"mbpp", "mbppplus"}:
        rows = [
            {
                "task_id": i,
                "text": "Return the number of items.",
                "prompt": "Return the number of items.",
                "code": "SECRET_REFERENCE",
                "test_list": ["SECRET_TEST"],
            }
            for i in (11, 12)
        ]
    else:
        rows = [
            {
                "instance_id": f"issue-{i}",
                "repo": "demo/repo",
                "problem_statement": "Handle empty input.",
                "patch": "SECRET_PATCH",
                "FAIL_TO_PASS": ["test_fix"],
                "PASS_TO_PASS": ["test_keep"],
                **(
                    {"image_name": "private-image"}
                    if source == "swe-smith"
                    else {"base_commit": "abc"}
                ),
            }
            for i in (1, 2)
        ]
    host = {
        str(r.get("task_id", r.get("instance_id"))): dict(
            split="test",
            task_family="family",
            difficulty="easy",
            entrypoint="solution:answer",
            allowed_files=["solution.py"],
            initial_snapshot="host-snapshot",
            public_suite_id="public-suite",
            private_suite_id="private-suite",
            budget=dict(
                max_main_steps=8,
                max_sub_calls=4,
                max_public_tests=2,
                max_code_revisions=2,
                max_generated_tokens=6000,
                wall_time_seconds=180,
            ),
            description_reviewed=True,
            audit_status="approved",
        )
        for r in rows
    }
    return rows, host


def convert(source="mbpp", **kwargs):
    rows, host = batch(source)
    return ingest(
        source,
        rows,
        revision="synthetic-v1",
        license_id="synthetic-license",
        known_revisions={source: {"synthetic-v1"}},
        metadata=host,
        **kwargs,
    )


@pytest.mark.parametrize("fault", ["missing", "type", "empty", "answer", "patch", "direction"])
def test_bad_rows(fault):
    rows, host = batch("swe-smith")
    if fault == "missing":
        del rows[0]["patch"]
    elif fault == "type":
        rows[0]["PASS_TO_PASS"] = 42
    elif fault == "empty":
        rows[0]["problem_statement"] = " "
    elif fault == "answer":
        rows[0]["problem_statement"] = "Reference answer: SECRET_PATCH"
    elif fault == "patch":
        rows[0]["problem_statement"] = "diff --git a/x b/x"
    else:
        rows[0]["patch_direction"] = "repair"
    with pytest.raises(ValueError):
        ingest(
            "swe-smith",
            rows,
            revision="synthetic-v1",
            license_id="MIT",
            known_revisions={"swe-smith": {"synthetic-v1"}},
            metadata=host,
        )


def test_unknown_revision_and_duplicate():
    rows, host = batch()
    with pytest.raises(ValueError):
        ingest("mbpp", rows, revision="unknown", license_id="x", known_revisions={}, metadata=host)
    with pytest.raises(ValueError):
        ingest(
            "mbpp",
            rows + rows[:1],
            revision="v",
            license_id="x",
            known_revisions={"mbpp": {"v"}},
            metadata=host,
        )


def test_gym_formal_rejected():
    with pytest.raises(ValueError):
        convert("swe-gym", formal=True)


@pytest.mark.parametrize("source", ["mbpp", "mbppplus", "swe-smith", "swe-gym"])
def test_deterministic_and_separated(source):
    result = convert(source)
    assert result == convert(source)
    manifest = json.loads(result.public_manifest)
    assert all(
        set(r) == {"task_id", "source", "revision", "license", "split", "hash"} for r in manifest
    )
    assert b"SECRET" not in result.public_manifest
    private = json.loads(result.private_index)
    assert "SECRET" in result.private_index.decode()
    for spec in result.tasks:
        assert "SECRET" not in spec.public_view().model_dump_json()
    if source == "swe-smith":
        assert private[0]["recovery"]["commit_status"] == "unresolved"
        assert private[0]["recovery"]["commit"] is None
        assert private[0]["patch_direction"] == "inject_defect"
    if source == "swe-gym":
        assert private[0]["audit_status"] == "pending"


def test_mbpp_link():
    a = json.loads(convert("mbpp").private_index)
    b = json.loads(convert("mbppplus").private_index)
    assert a[0]["original_task_id"] == b[0]["original_task_id"] == "11"


@pytest.mark.parametrize(
    "field,value", [("task_id", True), ("code", 7), ("test_list", 9), ("text", "")]
)
def test_mbpp_types(field, value):
    rows, host = batch()
    rows[0][field] = value
    with pytest.raises(ValueError):
        ingest(
            "mbpp",
            rows,
            revision="v",
            license_id="x",
            known_revisions={"mbpp": {"v"}},
            metadata=host,
        )


def test_order_and_secret_hash():
    rows, host = batch()
    kwargs = dict(revision="v", license_id="x", known_revisions={"mbpp": {"v"}}, metadata=host)
    first = ingest("mbpp", rows, **kwargs)
    assert first == ingest("mbpp", list(reversed(rows)), **kwargs)
    rows[0]["code"] = "DIFFERENT_PRIVATE_REFERENCE"
    second = ingest("mbpp", rows, **kwargs)
    assert first.public_manifest == second.public_manifest
    assert first.private_index != second.private_index


def test_prefixed_mbppplus_id():
    rows, host = batch("mbppplus")
    rows[0]["task_id"] = "Mbpp/11"
    host["Mbpp/11"] = host.pop("11")
    result = ingest(
        "mbppplus",
        rows,
        revision="v",
        license_id="x",
        known_revisions={"mbppplus": {"v"}},
        metadata=host,
    )
    assert json.loads(result.private_index)[0]["original_task_id"] == "11"


def test_unreviewed_description():
    rows, host = batch()
    host["11"]["description_reviewed"] = False
    with pytest.raises(ValueError):
        ingest(
            "mbpp",
            rows,
            revision="v",
            license_id="x",
            known_revisions={"mbpp": {"v"}},
            metadata=host,
        )
