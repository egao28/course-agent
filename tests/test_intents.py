"""Tests for eval/intents.py: the gold-intent sidecar's validator and its meta stamp (#102).

The rows here are placeholders (every test id gets ``course_fact``) written to ``tmp_path``. They
exercise the checks and say nothing about what any question's intent is. The real labels are a
person's, and the last test only holds the committed file, once it exists, to the same checks.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from eval import intents as it
from eval.scripts.review import DATASET_PATH


@pytest.fixture(scope="module")
def test_half() -> tuple[str, ...]:
    return it.labelled_half()


@pytest.fixture(scope="module")
def a_calib_id() -> str:
    return next(i for i, half in it.all_ids().items() if half == "calib")


def write(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def placeholder(ids) -> list[dict]:
    return [{"id": i, "gold_intents": ["course_fact"], "note": ""} for i in ids]


@pytest.fixture
def sidecar(tmp_path: Path, test_half) -> Path:
    return write(tmp_path / "v1_intents.jsonl", placeholder(test_half))


def errors(violations) -> list[str]:
    return [f"{v.field}: {v.message}" for v in violations if v.level == "error"]


def test_the_test_half_is_the_63_the_issue_counts(test_half):
    assert len(test_half) == 63 and len(set(test_half)) == 63


def test_a_complete_sidecar_passes_before_it_is_stamped(sidecar):
    assert errors(it.validate(sidecar, with_meta=False)) == []


def test_an_unstamped_sidecar_fails_the_full_check(sidecar):
    [problem] = errors(it.validate(sidecar))
    assert "--stamp --labeller" in problem


def test_stamp_writes_a_meta_the_check_then_accepts(sidecar):
    path, violations = it.stamp("eve", sidecar)
    assert path == sidecar.with_suffix(".meta.json") and errors(violations) == []
    meta = json.loads(path.read_text(encoding="utf-8"))
    assert meta["labeller"] == "eve"
    assert meta["rows"] == 63
    assert meta["rows_with_label"]["course_fact"] == 63
    assert meta["rows_with_more_than_one_label"] == 0
    assert meta["dataset_sha256"].startswith("2a931822")
    assert meta["predictions_in_tree"] == ["signals_test_37164723.jsonl"]
    assert errors(it.validate(sidecar)) == []


def test_an_edit_after_stamping_fails(sidecar, test_half):
    it.stamp("eve", sidecar)
    rows = placeholder(test_half)
    rows[0]["gold_intents"] = ["prereq_chain"]
    write(sidecar, rows)
    [problem] = errors(it.validate(sidecar))
    assert problem.startswith("sha256:") and "re-stamp" in problem


def test_an_amended_intent_section_fails(sidecar, tmp_path: Path):
    guidelines = shutil.copy(it.GUIDELINES_PATH, tmp_path / "GUIDELINES.md")
    it.stamp("eve", sidecar, guidelines=guidelines)
    assert errors(it.validate(sidecar, guidelines=guidelines)) == []
    text = guidelines.read_text(encoding="utf-8")
    guidelines.write_text(text.replace("## Intent\n", "## Intent\n\nA new rule.\n"), "utf-8")
    [problem] = errors(it.validate(sidecar, guidelines=guidelines))
    assert problem.startswith("guidelines_intent_sha256:") and "#106" in problem


def test_an_amendment_elsewhere_in_the_guidelines_does_not(sidecar, tmp_path: Path):
    guidelines = shutil.copy(it.GUIDELINES_PATH, tmp_path / "GUIDELINES.md")
    it.stamp("eve", sidecar, guidelines=guidelines)
    with guidelines.open("a", encoding="utf-8") as f:
        f.write("\nA Day 3 amendment.\n")
    assert errors(it.validate(sidecar, guidelines=guidelines)) == []


def test_a_missing_test_id_is_an_error_not_a_short_count(tmp_path: Path, test_half):
    path = write(tmp_path / "s.jsonl", placeholder(test_half[1:]))
    [problem] = errors(it.validate(path, with_meta=False))
    assert f"1 test id not labelled ({test_half[0]})" in problem


def test_a_calib_id_an_unknown_id_and_a_repeat_are_each_named(
    tmp_path: Path, test_half, a_calib_id
):
    rows = placeholder(test_half) + placeholder([a_calib_id, "v1-9999", test_half[0]])
    found = errors(it.validate(write(tmp_path / "s.jsonl", rows), with_meta=False))
    assert found == [
        "id: is a calib row; only the test half is labelled",
        "id: is not a v1 id",
        "id: already labelled on line 1",
    ]


@pytest.mark.parametrize(
    ("labels", "note", "expected"),
    [
        (["course_facts"], "", "gold_intents.0"),
        ([], "", "gold_intents"),
        (["course_fact", "course_fact"], "", "gold_intents: repeats course_fact"),
        (["course_fact", "prereq_chain"], " ", "note: is empty, but the set has 2 labels"),
    ],
)
def test_a_bad_label_set_is_an_error(tmp_path: Path, test_half, labels, note, expected):
    rows = placeholder(test_half)
    rows[0] = {"id": test_half[0], "gold_intents": labels, "note": note}
    [problem] = errors(it.validate(write(tmp_path / "s.jsonl", rows), with_meta=False))
    assert problem.startswith(expected)


def test_a_row_without_a_note_field_is_an_error(tmp_path: Path, test_half):
    rows = placeholder(test_half)
    del rows[0]["note"]
    [problem] = errors(it.validate(write(tmp_path / "s.jsonl", rows), with_meta=False))
    assert problem.startswith("note:")


def test_two_labels_with_a_note_pass(tmp_path: Path, test_half):
    rows = placeholder(test_half)
    rows[0].update(gold_intents=["course_fact", "prereq_chain"], note="the chunk prints it")
    assert it.validate(write(tmp_path / "s.jsonl", rows), with_meta=False) == []


def test_three_labels_warn_and_do_not_fail(tmp_path: Path, test_half):
    rows = placeholder(test_half)
    rows[0].update(gold_intents=list(it.INTENTS[:3]), note="three readings")
    [warning] = it.validate(write(tmp_path / "s.jsonl", rows), with_meta=False)
    assert warning.level == "warning" and "unclear" in warning.message


def test_stamp_refuses_a_failing_sidecar_and_writes_nothing(tmp_path: Path, test_half):
    path = write(tmp_path / "s.jsonl", placeholder(test_half[:-1]))
    written, violations = it.stamp("eve", path)
    assert written is None and errors(violations)
    assert not path.with_suffix(".meta.json").exists()


def test_stamp_needs_a_labeller(sidecar):
    with pytest.raises(ValueError, match="labeller"):
        it.stamp("  ", sidecar)


def test_an_empty_labeller_in_the_meta_fails(sidecar):
    path, _ = it.stamp("eve", sidecar)
    meta = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps({**meta, "labeller": ""}), encoding="utf-8")
    [problem] = errors(it.validate(sidecar))
    assert problem.startswith("labeller:")


def test_an_empty_file_is_not_an_empty_pass(tmp_path: Path):
    path = tmp_path / "s.jsonl"
    path.write_text("\n", encoding="utf-8")
    assert errors(it.validate(path, with_meta=False)) == ["None: holds no rows"]


def test_the_command_exits_non_zero_on_errors_and_zero_once_stamped(sidecar, capsys):
    assert it.main(["--sidecar", str(sidecar)]) == 1
    assert it.main(["--sidecar", str(sidecar), "--stamp", "--labeller", "eve"]) == 0
    assert "wrote" in capsys.readouterr().out
    assert it.main(["--sidecar", str(sidecar)]) == 0


def test_the_committed_sidecar_once_there_passes_and_is_stamped():
    """Until a person commits the labels there is nothing to hold; a meta alone is an error."""
    meta = it.SIDECAR_PATH.with_suffix(".meta.json")
    if not it.SIDECAR_PATH.exists():
        assert not meta.exists(), "a meta with no labels beside it"
        return
    assert errors(it.validate(it.SIDECAR_PATH, DATASET_PATH)) == []
