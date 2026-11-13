import json

import pytest

from copatch_rl.data.dedupe import FrozenConflict, SplitPolicy, dedupe
from copatch_rl.data.ingest import ingest


def fixtures(source="swe-smith", split="train", repo="Org/Repo"):
    rows = [
        dict(
            instance_id=f"i{i}",
            repo=repo,
            problem_statement=f"Fix case {i}",
            patch=f"patch-{i}",
            FAIL_TO_PASS=["fix"],
            PASS_TO_PASS=["keep"],
        )
        for i in (1, 2)
    ]
    host = {
        r["instance_id"]: dict(
            split=split,
            task_family="problem-" + r["instance_id"],
            difficulty="easy",
            entrypoint="pkg:fn",
            allowed_files=["pkg.py"],
            initial_snapshot="initial",
            public_suite_id="pub",
            private_suite_id="private",
            description_reviewed=True,
            audit_status="approved",
            budget=dict(
                max_main_steps=8,
                max_sub_calls=4,
                max_public_tests=2,
                max_code_revisions=2,
                max_generated_tokens=6000,
                wall_time_seconds=180,
            ),
        )
        for r in rows
    }
    return json.loads(
        ingest(
            source,
            rows,
            revision="synthetic",
            license_id="synthetic",
            known_revisions={source: {"synthetic"}},
            metadata=host,
        ).private_index
    )


def function(i, code, split="train"):
    return dict(
        task_id=f"mbpp:{i}",
        source="mbpp",
        original_task_id=str(i),
        raw={"code": code},
        task={"split": split, "task_family": f"problem-{i}"},
    )


def test_exact_and_renamed():
    rows = [
        function(1, "def a(x): return x+1"),
        function(2, "def a(x): return x+1"),
        function(3, "def b(y): return y+1"),
    ]
    result = dedupe(rows)
    assert result["duplicates"]
    assert result["candidates"]
    assert all(c["review_status"] == "pending" for c in result["candidates"])
    assert len(result["assignments"]) == 3


def test_similar_not_equivalent():
    result = dedupe([function(1, "def f(x): return x+1"), function(2, "def g(y): return y-1")])
    assert not result["duplicates"]
    assert result["candidates"]


def test_cross_source_repo_and_fork():
    rows = fixtures() + fixtures("swe-gym", repo="fork/renamed")
    result = dedupe(rows, aliases={"fork/renamed": "org/repo"})
    assert len({x["repo_family"] for x in result["assignments"]}) == 1


def test_leak_and_frozen():
    rows = fixtures() + fixtures("swe-gym", "test", repo="ORG/REPO")
    assert dedupe(rows)["conflicts"]
    with pytest.raises(FrozenConflict) as exc:
        dedupe(rows, frozen=True)
    assert exc.value.report["conflicts"]


def test_order_stable():
    rows = fixtures() + fixtures("swe-gym", repo="org/repo")
    assert dedupe(rows) == dedupe(list(reversed(rows)))


def test_source_id_and_problem_family():
    a, b = function(1, "x=1"), function(2, "x=2", "test")
    b["source"] = "mbppplus"
    b["original_task_id"] = "1"
    result = dedupe([a, b])
    assert result["duplicates"] and result["conflicts"]
    b["original_task_id"] = "2"
    b["task"]["task_family"] = a["task"]["task_family"]
    assert dedupe([a, b])["conflicts"]


def test_image_identity_and_transitive_conflict():
    rows = fixtures() + fixtures("swe-gym", "dev", repo="other/repo")
    for row in rows[2:]:
        row["raw"]["image_name"] = "registry/custom:tag"
    result = dedupe(rows, image_aliases={"registry/custom:tag": "org/repo"})
    assert result["conflicts"]
    assert all(a["split"] is None for a in result["assignments"])
    assert len(result["assignments"]) == len(rows)


def test_unassigned_order_stable():
    rows = [function(i, f"value={i}", "smoke") for i in range(5)]
    a = dedupe(rows)
    assert a == dedupe(rows[::-1])
    assert all(r["split"] in {"train", "dev", "test"} for r in a["assignments"])


def test_template_review_only():
    a, b = function(1, "x=1"), function(2, "x=2")
    a["raw"]["text"] = "Return 10 items"
    b["raw"]["text"] = "Return 20 items"
    result = dedupe([a, b])
    assert not result["duplicates"]
    assert "description_template" in result["candidates"][0]["reasons"]


def test_exact_input_duplicate_retained():
    row = function(1, "x=1")
    result = dedupe([row, row])
    assert len(result["assignments"]) == 2
    assert "source_id" in result["duplicates"][0]["reasons"]


def test_frozen_valid_unchanged():
    rows = fixtures()
    result = dedupe(rows, frozen=True)
    assert not result["conflicts"]
    assert all(r["split"] == "train" for r in result["assignments"])


def test_test_only_split_policy():
    rows = [function(i, f"value={i}", "smoke") for i in range(10)]
    result = dedupe(rows, policy=SplitPolicy(train=0, dev=0, test=1))
    assert all(row["split"] == "test" for row in result["assignments"])
    assert "0/0/1" in result["split_policy"]
