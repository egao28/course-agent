"""Tests for eval/validate.py: every violation, one message each, and a non-zero exit.

Two fixtures carry most of it. ``tests/fixtures/eval_good.jsonl`` is four rows, one per bucket,
with gold ids that are really in the catalog; ``tests/fixtures/eval_bad.jsonl`` is nine rows with
**one problem each**, so "prints every violation rather than the first" is a count and not a
judgement call. A row added to either fixture without a matching expectation below fails
:func:`test_the_bad_fixture_reports_one_message_per_problem`, which is the point of writing them
that way.

The corpus these run against is the real one -- 4,502 ids -- because the check worth testing is
that a hand-typed ``2026-sp:MATH 241`` fails against the ids that actually exist. A synthetic
five-id corpus would test the loop and not the dataset.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from config import REPO_ROOT, rel_to_repo
from eval import validate
from eval.validate import (
    Violation,
    article,
    corpus_doc_ids,
    main,
    plural,
    render,
    suggest,
    validate_file,
)
from ingest.chunk import CHUNKS_PATH, doc_id, load_chunks, load_courses

FIXTURES = Path(__file__).parent / "fixtures"
GOOD = FIXTURES / "eval_good.jsonl"
BAD = FIXTURES / "eval_bad.jsonl"
DAY2 = REPO_ROOT / "eval" / "datasets" / "day2_queries.jsonl"
#: #62's deliverable. Absent until it lands, which is why the test over it skips rather than
#: fails -- the check itself is the thing #54 owes, and it is wired up here from the start.
V1 = REPO_ROOT / "eval" / "datasets" / "v1.jsonl"

#: What ``eval_bad.jsonl`` is supposed to say, line by line: the field and a phrase from the
#: message. Nine rows, nine errors -- no row contributes two and none is silently fine.
EXPECTED_BAD = [
    (1, "gold_doc_ids.0", "did you mean '2026-sp:MATH-241'?"),
    (2, "gold_doc_ids.0", "is not in the corpus"),
    (3, "gold_doc_ids", "an unanswerable row carries no gold ids at all"),
    (4, "question", "repeats line 1"),
    (5, "gold_doc_ids", "a multihop row needs at least 2"),
    (6, "gold_doc_ids.1", "repeats '2026-sp:CS-173' from position 0"),
    (7, "should_abstain", "is True, but every fact row is False"),
    (8, "id", "already used on line 7"),
    (9, "difficulty", "Extra inputs are not permitted"),
]


@pytest.fixture
def eval_set():
    if not V1.exists():
        pytest.skip(f"{rel_to_repo(V1)} is #62's deliverable and is not in the tree yet")
    return V1


def errors(violations):
    return [v for v in violations if v.level == "error"]


def warnings(violations):
    return [v for v in violations if v.level == "warning"]


def write(tmp_path, *lines):
    path = tmp_path / "rows.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def as_row(**overrides):
    row = {
        "id": "v1-0001",
        "bucket": "fact",
        "question": "How many credits is MATH 241?",
        "gold_answer": "4 credit hours.",
        "gold_doc_ids": ["2026-sp:MATH-241"],
        "should_abstain": False,
    }
    return json.dumps({**row, **overrides})


# --- the corpus the gold ids are checked against --------------------------------------------------


def test_the_corpus_is_the_whole_catalog():
    known = corpus_doc_ids()
    assert len(known) == 4502
    assert "2026-sp:MATH-241" in known
    assert "2026-sp:MATH 241" not in known


@pytest.mark.skipif(not CHUNKS_PATH.exists(), reason="chunks.jsonl has not been built here")
def test_the_two_ways_of_getting_the_ids_agree():
    """The fallback is only honest if it produces the same set; both come from ``doc_id``."""
    assert {c["doc_id"] for c in load_chunks()} == {doc_id(c) for c in load_courses()}


def test_a_missing_chunks_file_falls_back_to_the_committed_json(tmp_path, monkeypatch):
    monkeypatch.setattr(validate, "CHUNKS_PATH", tmp_path / "never-built.jsonl")
    validate.clear_cache()
    try:
        assert len(corpus_doc_ids()) == 4502
    finally:
        validate.clear_cache()


def test_a_corpus_named_outright_is_not_second_guessed(tmp_path):
    with pytest.raises(FileNotFoundError):
        corpus_doc_ids(tmp_path / "nope.jsonl")


@pytest.mark.parametrize(
    ("typed", "meant"),
    [
        ("2026-sp:MATH 241", "2026-sp:MATH-241"),
        ("MATH 241", "2026-sp:MATH-241"),
        ("math 241", "2026-sp:MATH-241"),
        ("MATH-241", "2026-sp:MATH-241"),
    ],
)
def test_suggest_repairs_the_spellings_a_person_actually_types(typed, meant):
    assert suggest(typed, corpus_doc_ids()) == meant


@pytest.mark.parametrize("hopeless", ["2026-sp:CS-999", "", "quantum origami"])
def test_suggest_stays_quiet_rather_than_guessing_a_course(hopeless):
    assert suggest(hopeless, corpus_doc_ids()) is None


# --- the two fixtures -----------------------------------------------------------------------------


def test_the_good_fixture_has_nothing_wrong_with_it():
    assert errors(validate_file(GOOD)) == []


def test_a_short_file_only_warns_so_verification_can_delete_a_bad_row():
    found = warnings(validate_file(GOOD))
    assert len(found) == 5
    assert all(v.line is None for v in found)
    assert "4 rows in total, not 120" in found[-1].message


def test_the_bad_fixture_reports_one_message_per_problem():
    found = errors(validate_file(BAD))
    assert [(v.line, v.field) for v in found] == [(line, f) for line, f, _ in EXPECTED_BAD]
    for violation, (_, _, phrase) in zip(found, EXPECTED_BAD, strict=True):
        assert phrase in violation.message


def test_the_bad_fixture_is_reported_in_file_order():
    found = errors(validate_file(BAD))
    assert [v.line for v in found] == sorted(v.line for v in found)


def test_every_reported_row_names_the_id_the_reader_has_to_go_and_find():
    for violation in errors(validate_file(BAD)):
        assert violation.item_id is not None


# --- one check at a time --------------------------------------------------------------------------


def test_a_dangling_id_fails_rather_than_warns(tmp_path):
    path = write(tmp_path, as_row(gold_doc_ids=["2026-sp:CS-999"]))
    assert len(errors(validate_file(path))) == 1


def test_an_ambiguous_row_may_carry_a_gold_id_or_none(tmp_path):
    clarify = {
        "bucket": "ambiguous",
        "gold_answer": "Which course do you mean?",
        "should_abstain": True,
    }
    empty = write(tmp_path, as_row(**clarify, gold_doc_ids=[]))
    assert errors(validate_file(empty)) == []
    one = write(tmp_path, as_row(**clarify, gold_doc_ids=["2026-sp:MATH-241"]))
    assert errors(validate_file(one)) == []


def test_a_multihop_row_needs_two_different_documents(tmp_path):
    same = write(
        tmp_path,
        as_row(bucket="multihop", gold_doc_ids=["2026-sp:MATH-241", "2026-sp:MATH-241"]),
    )
    assert len(errors(validate_file(same))) == 1
    two = write(
        tmp_path, as_row(bucket="multihop", gold_doc_ids=["2026-sp:MATH-241", "2026-sp:CS-374"])
    )
    assert errors(validate_file(two)) == []


def test_a_row_may_be_wrong_in_more_than_one_way(tmp_path):
    path = write(
        tmp_path,
        as_row(bucket="unanswerable", should_abstain=False, gold_doc_ids=["2026-sp:CS-999"]),
    )
    fields = [v.field for v in errors(validate_file(path))]
    assert fields == ["gold_doc_ids", "gold_doc_ids.0", "should_abstain"]


def test_a_line_that_is_not_json(tmp_path):
    path = write(tmp_path, as_row(), "{not json")
    found = errors(validate_file(path))
    assert len(found) == 1
    assert found[0].line == 2
    assert "not JSON" in found[0].message


def test_a_line_that_is_json_but_not_a_row(tmp_path):
    path = write(tmp_path, "[1, 2, 3]")
    found = errors(validate_file(path))
    assert len(found) == 1
    assert "not an object, but a list" in found[0].message


def test_blank_lines_are_not_rows(tmp_path):
    path = write(tmp_path, as_row(), "", "   ")
    assert errors(validate_file(path)) == []


def test_an_empty_file_is_not_an_empty_pass(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")
    found = validate_file(path)
    assert [v.message for v in found] == ["holds no rows"]
    assert errors(found)


def test_a_file_that_is_not_there_says_so(tmp_path):
    with pytest.raises(FileNotFoundError):
        validate_file(tmp_path / "nowhere.jsonl")


# --- the command ----------------------------------------------------------------------------------


def test_main_exits_zero_on_a_good_file(capsys):
    assert main([str(GOOD)]) == 0
    assert "ok -- 0 error(s), 5 warning(s)" in capsys.readouterr().out


def test_main_exits_non_zero_and_prints_everything(capsys):
    assert main([str(BAD)]) == 1
    out = capsys.readouterr().out
    assert "FAILED -- 9 error(s)" in out
    for _, _, phrase in EXPECTED_BAD:
        assert phrase in out


def test_main_takes_a_corpus_of_its_own(capsys, tmp_path):
    """``--chunks`` swaps the catalog, which is what a re-export would need."""
    corpus = tmp_path / "chunks.jsonl"
    row = {"doc_id": "2026-sp:MATH-241", "text": "MATH 241", "metadata": {}}
    corpus.write_text(json.dumps(row) + "\n", encoding="utf-8")
    assert main([str(GOOD), "--chunks", str(corpus)]) == 1
    assert "'2026-sp:CS-374' is not in the corpus" in capsys.readouterr().out


def test_the_messages_count_and_agree_in_english():
    """A validator read by a person all afternoon should not say "1 row(s)" to them."""
    assert (plural(1, "row"), plural(0, "row"), plural(2, "id")) == ("1 row", "0 rows", "2 ids")
    assert (article("fact"), article("unanswerable"), article("ambiguous")) == ("a", "an", "an")


def test_render_points_at_the_line_an_editor_would_open():
    line = render(V1, [Violation(12, "v1-0012", "id", "wrong")])
    assert line == "eval/datasets/v1.jsonl:12  v1-0012 id  wrong"


def test_render_leaves_the_line_off_a_whole_file_finding():
    line = render(V1, [Violation(None, None, None, "holds no rows", "error")])
    assert line == "eval/datasets/v1.jsonl  holds no rows"


# --- the split (#61) ------------------------------------------------------------------------------


def test_an_unfrozen_file_says_nothing_about_splits(tmp_path):
    """Every file in the tree today is unfrozen, and that is not a finding."""
    path = write(tmp_path, as_row(), as_row(id="v1-0002", question="How many credits is CS 225?"))
    assert [v for v in validate_file(path) if v.field == "split"] == []


def test_a_half_frozen_file_is_one_error_naming_the_rows_that_have_no_half(tmp_path):
    path = write(
        tmp_path,
        as_row(split="calib"),
        as_row(id="v1-0002", question="How many credits is CS 225?"),
    )
    found = [v for v in errors(validate_file(path)) if v.field == "split"]
    assert len(found) == 1
    assert "1 row carry a split and 1 do not (v1-0002)" in found[0].message
    assert found[0].line is None  # a property of the file, not of a row


def test_a_frozen_file_passes(tmp_path):
    path = write(
        tmp_path,
        as_row(split="calib"),
        as_row(id="v1-0002", question="How many credits is CS 225?", split="test"),
    )
    assert [v for v in errors(validate_file(path)) if v.field == "split"] == []


def test_a_half_that_is_not_one_of_the_two_is_a_field_error(tmp_path):
    path = write(tmp_path, as_row(split="cal"))
    assert any(v.field == "split" for v in errors(validate_file(path)))


def test_a_lopsided_bucket_warns_without_failing(tmp_path):
    """The one stratification property that needs no seed. `--verify` is what checks the rest."""
    rows = [
        as_row(id=f"v1-{n:04d}", question=f"How many credits is TEST {n}?", split="calib")
        for n in range(1, 4)
    ]
    path = write(tmp_path, *rows)
    found = [v for v in warnings(validate_file(path)) if v.field == "fact"]
    assert any("3 calib / 0 test" in v.message for v in found)
    assert not errors(validate_file(path))


def test_a_bucket_one_row_apart_is_not_lopsided(tmp_path):
    rows = [
        as_row(id=f"v1-{n:04d}", question=f"How many credits is TEST {n}?", split=half)
        for n, half in ((1, "calib"), (2, "calib"), (3, "test"))
    ]
    path = write(tmp_path, *rows)
    assert not [v for v in warnings(validate_file(path)) if "calib /" in v.message]


def test_the_extra_field_message_names_the_seven_columns_that_exist(tmp_path):
    path = write(tmp_path, as_row(difficulty="hard"))
    message = next(v.message for v in errors(validate_file(path)) if v.field == "difficulty")
    assert "the row has six fields, and `split` once #61 freezes it" in message


# --- the datasets in the repo ---------------------------------------------------------------------


def test_the_day2_queries_are_a_different_artifact_and_stay_one():
    """#54 says so outright: v1 is new, and the smoke set is not retrofitted into its shape."""
    assert errors(validate_file(DAY2)), (
        f"{rel_to_repo(DAY2)} now validates as a v1 eval set, which means one of the two shapes "
        "moved; they are different artifacts with different jobs"
    )


def test_the_committed_eval_set_validates(eval_set):
    """CI's version of "re-run the validator": an edit to v1.jsonl cannot land unchecked."""
    assert errors(validate_file(eval_set)) == []
