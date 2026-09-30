"""Tests for eval/schema.py: six fields, four buckets, and an id nothing may re-mint.

The model is shape only -- whether a ``multihop`` row really carries two *different* ids is
``tests/test_eval_validate.py`` -- so what is pinned here is the contract a drafting run (#56)
writes against: the field set, the id format, and the table every later check reads out of.

The bucket table is transcribed from #54 rather than imported, on purpose. A test that reads
``RULES`` to check ``RULES`` would pass through any edit; these numbers are the plan, and a change
to them should have to change this file too.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agent.state import EXITS
from eval.schema import (
    ANNOTATED_FIELDS,
    BUCKETS,
    ID_RE,
    MAX_N,
    RULES,
    TARGET_TOTAL,
    VERSION,
    EvalItem,
    item_id,
    normalize_question,
    row_json,
    rule,
)

ROW = {
    "id": "v1-0001",
    "bucket": "fact",
    "question": "How many credits is MATH 241?",
    "gold_answer": "4 credit hours.",
    "gold_doc_ids": ["2026-sp:MATH-241"],
    "should_abstain": False,
}


def row(**overrides: object) -> dict[str, object]:
    """``ROW`` with fields replaced, so each test says only what it is about."""
    return {**ROW, **overrides}


# --- the six fields -------------------------------------------------------------------------------


def test_the_six_fields_in_the_order_the_issue_writes_them():
    assert ANNOTATED_FIELDS == (
        "id",
        "bucket",
        "question",
        "gold_answer",
        "gold_doc_ids",
        "should_abstain",
    )


def test_the_model_is_the_six_annotated_fields_and_the_one_assigned_one():
    """#61 adds ``split``, and it goes last so a frozen row still reads six-then-one."""
    assert tuple(EvalItem.model_fields) == (*ANNOTATED_FIELDS, "split")


def test_a_valid_row_round_trips_unchanged():
    assert row_json(EvalItem.model_validate(ROW)) == ROW


def test_an_unfrozen_row_is_written_without_a_split_column():
    """``model_dump`` would write ``"split": null`` into every draft file; row_json does not."""
    item = EvalItem.model_validate(ROW)
    assert item.split is None
    assert "split" not in row_json(item)
    assert row_json(item.model_copy(update={"split": "calib"}))["split"] == "calib"


@pytest.mark.parametrize("field", list(ROW))
def test_every_field_is_required(field):
    missing = {k: v for k, v in ROW.items() if k != field}
    with pytest.raises(ValidationError, match="Field required"):
        EvalItem.model_validate(missing)


def test_an_invented_field_is_refused_rather_than_ignored():
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        EvalItem.model_validate(row(difficulty="easy"))


def test_a_fifth_bucket_is_refused():
    with pytest.raises(ValidationError):
        EvalItem.model_validate(row(bucket="tricky"))


def test_should_abstain_is_a_bool_not_a_string():
    with pytest.raises(ValidationError):
        EvalItem.model_validate(row(should_abstain="yes"))


def test_gold_doc_ids_must_be_a_list():
    with pytest.raises(ValidationError):
        EvalItem.model_validate(row(gold_doc_ids="2026-sp:MATH-241"))


# --- ids ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("good", ["v1-0001", "v1-0120", "v1-9999", "v2-0001", "v10-0001"])
def test_ids_that_are_the_format(good):
    assert ID_RE.match(good)
    assert EvalItem.model_validate(row(id=good)).id == good


@pytest.mark.parametrize(
    "bad", ["v1-1", "v1-00001", "0001", "V1-0001", "v1_0001", " v1-0001", "v1-0001 ", "v0-0001"]
)
def test_ids_that_are_not(bad):
    assert not ID_RE.match(bad)
    with pytest.raises(ValidationError):
        EvalItem.model_validate(row(id=bad))


def test_item_id_zero_pads_to_four():
    assert item_id(1) == "v1-0001"
    assert item_id(7) == "v1-0007"
    assert item_id(120) == "v1-0120"
    assert item_id(MAX_N) == f"{VERSION}-9999"


@pytest.mark.parametrize("n", [0, -1, MAX_N + 1, True, 1.0, "3"])
def test_item_id_refuses_a_number_it_cannot_mint(n):
    with pytest.raises(ValueError, match="eval item numbers"):
        item_id(n)


def test_item_id_takes_a_later_version_but_not_a_made_up_one():
    assert item_id(3, "v2") == "v2-0003"
    with pytest.raises(ValueError, match="version"):
        item_id(3, "2")


# --- the prose fields -----------------------------------------------------------------------------


@pytest.mark.parametrize("field", ["question", "gold_answer"])
@pytest.mark.parametrize("blank", ["", "   ", "\n", "\t "])
def test_a_blank_prose_field_is_refused(field, blank):
    with pytest.raises(ValidationError, match="must not be empty"):
        EvalItem.model_validate(row(**{field: blank}))


def test_prose_is_stripped_because_a_stray_space_is_not_a_different_question():
    item = EvalItem.model_validate(row(question="  How many credits is MATH 241?\n"))
    assert item.question == "How many credits is MATH 241?"


def test_a_gold_id_is_left_exactly_as_typed():
    """The space is the error the validator is hunting; repairing it here would hide it."""
    item = EvalItem.model_validate(row(gold_doc_ids=["2026-sp:MATH 241", " 2026-sp:CS-374"]))
    assert item.gold_doc_ids == ["2026-sp:MATH 241", " 2026-sp:CS-374"]


def test_normalize_question_collapses_whitespace_and_case():
    assert normalize_question("How  many credits\nis MATH 241?") == normalize_question(
        "how many credits is math 241?"
    )
    assert normalize_question("How many credits is MATH 241?") != normalize_question(
        "How many credits is CS 374?"
    )


# --- the bucket table -----------------------------------------------------------------------------


def test_the_table_covers_the_four_buckets_and_nothing_else():
    assert tuple(RULES) == BUCKETS
    assert all(RULES[bucket].bucket == bucket for bucket in BUCKETS)


def test_the_counts_are_the_plan():
    assert {bucket: RULES[bucket].target_n for bucket in BUCKETS} == {
        "fact": 50,
        "multihop": 25,
        "unanswerable": 25,
        "ambiguous": 20,
    }
    assert TARGET_TOTAL == 120


def test_should_abstain_is_fixed_by_the_bucket():
    assert {bucket: RULES[bucket].should_abstain for bucket in BUCKETS} == {
        "fact": False,
        "multihop": False,
        "unanswerable": True,
        "ambiguous": True,
    }


def test_the_gold_id_bounds():
    bounds = {b: (RULES[b].min_gold_ids, RULES[b].max_gold_ids) for b in BUCKETS}
    assert bounds == {
        "fact": (1, None),
        "multihop": (2, None),
        "unanswerable": (0, 0),
        "ambiguous": (0, None),
    }


def test_every_expected_exit_is_one_the_graph_can_take():
    for bucket in BUCKETS:
        assert RULES[bucket].exits, f"{bucket} expects no exit at all"
        assert set(RULES[bucket].exits) <= set(EXITS)


def test_the_two_abstaining_buckets_do_not_expect_the_same_exit():
    """One ends at 'this catalog does not have that', the other at 'which course do you mean?'."""
    assert RULES["ambiguous"].exits == ("refuse_handoff",)
    assert "out_of_scope" in RULES["unanswerable"].exits


def test_rule_names_the_four_when_asked_for_a_fifth():
    assert rule("fact") is RULES["fact"]
    with pytest.raises(ValueError, match="unknown bucket"):
        rule("tricky")
