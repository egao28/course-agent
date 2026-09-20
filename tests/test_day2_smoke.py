"""Tests for eval/scripts/day2_smoke.py: the dataset, the run, and what the report has to say.

Two halves. The first is about ``eval/datasets/day2_queries.jsonl`` itself -- it is a tracked
artifact with a job (cover the five intents, force the three exits), and a row that quietly
stopped doing that job would weaken every report made from it. The second runs the runner over a
three-row dataset of its own with a fake dispatch node, so the report's structure is tested with
no BM25 pickle on disk and no model anywhere: what is asserted there is the report's *shape* --
the exit counts, the disagreements, the provenance of every answer -- and not the numbers of one
particular corpus.

Nothing here writes into ``report/``. :func:`~eval.scripts.day2_smoke.render_report` returns a
string, and only ``main`` puts it on disk.
"""

from __future__ import annotations

import json

import pytest

from agent.state import EXITS, INTENTS, Evidence
from config import Config
from eval.scripts import day2_smoke as smoke

TERM = "2026-sp"
#: A synthetic config, not the live one -- but a fuller one than most tests need: the report's
#: header names the index its retrieval rows read, so ``chunk`` and ``bm25`` have to be there for
#: ``bm25_key`` to exist. The numbers are round; the live thresholds are asserted in
#: ``tests/test_check_evidence.py``.
CFG = Config(
    {
        "data": {"term": TERM},
        "chunk": {"strategy": "one_course_one_chunk", "template_version": "v1"},
        "bm25": {"k1": 1.5, "b": 0.75, "tokenizer": "simple_v1", "code_boost": 2},
        "llm": {"provider": "openai", "model": "gpt-4o-mini", "temperature": 0, "max_tokens": 800},
        "agent": {"retrieval_mode": "bm25"},
        "evidence": {
            "method": "top_score",
            "thresholds": {"bm25": 15.0, "dense": 0.4, "hybrid": 0.025},
        },
    }
)
LOOKUP = Evidence(
    doc_id="CS 374", text="CS 374 — Algorithms\n4 hours.", score=None, source="get_course"
)


# --- the shipped dataset --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def shipped():
    return smoke.load_queries()


def test_the_set_is_ten_handwritten_queries(shipped):
    assert len(shipped) == 10


def test_every_intent_is_covered(shipped):
    # The dispatch table has a row per intent, and a set that never exercises one is a row nothing
    # in the report stands behind.
    assert {row.expected_intent for row in shipped} == set(INTENTS)


def test_every_exit_is_expected_by_at_least_one_row(shipped):
    # Day 2's acceptance is that all three fire. A set that expects only two could never show it.
    assert {row.expected_exit for row in shipped} == set(EXITS)


def test_no_query_appears_twice(shipped):
    queries = [row.query for row in shipped]
    assert len(set(queries)) == len(queries)


def test_every_row_says_why_it_is_in_the_set(shipped):
    # The note is what a reader of the report gets instead of having to reconstruct the intent of
    # a query from the query.
    assert all(len(row.note.split()) >= 8 for row in shipped)


# --- loading it -----------------------------------------------------------------------------------


def row(**changes):
    base = {
        "query": "How many credits is CS 374?",
        "expected_intent": "course_fact",
        "expected_exit": "answer",
        "note": "a lookup",
    }
    return {**base, **changes}


def write(tmp_path, rows):
    path = tmp_path / "queries.jsonl"
    path.write_text("\n".join(json.dumps(one) for one in rows) + "\n", encoding="utf-8")
    return path


def test_a_row_is_loaded_whole(tmp_path):
    loaded = smoke.load_queries(write(tmp_path, [row()]))
    expected = smoke.Query("How many credits is CS 374?", "course_fact", "answer", "a lookup")
    assert loaded == [expected]


def test_blank_lines_are_skipped_and_an_empty_file_is_refused(tmp_path):
    path = tmp_path / "queries.jsonl"
    path.write_text("\n\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no queries"):
        smoke.load_queries(path)


@pytest.mark.parametrize(
    "bad, message",
    [
        ({"expected_intent": "course_facts"}, "five labels"),
        ({"expected_exit": "handoff"}, "three exits"),
        ({"note": "   "}, "non-empty"),
        ({"query": 3}, "non-empty"),
    ],
)
def test_a_row_that_cannot_be_read_names_the_line_and_the_field(tmp_path, bad, message):
    path = write(tmp_path, [row(), row(**bad)])
    with pytest.raises((ValueError, TypeError), match=message) as raised:
        smoke.load_queries(path)
    assert "line 2" in str(raised.value)


def test_a_missing_field_and_an_unknown_one_are_both_refused(tmp_path):
    without = row()
    without.pop("note")
    with pytest.raises(ValueError, match="is missing note"):
        smoke.load_queries(write(tmp_path, [without]))
    with pytest.raises(ValueError, match="which nothing reads"):
        smoke.load_queries(write(tmp_path, [row(expected_answer="4 hours")]))


def test_a_line_that_is_not_json_is_reported_as_that_line(tmp_path):
    path = tmp_path / "queries.jsonl"
    path.write_text(json.dumps(row()) + "\n{not json}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 2 is not JSON"):
        smoke.load_queries(path)


# --- running the set ------------------------------------------------------------------------------


def dispatcher(state):
    """A stand-in for #36: one query gets a lookup block, one gets a tool error, and that is all.

    The real dispatcher needs ``data/courses.json`` and, for a retrieval row, a 6 MB pickle. What
    is under test here is the report, so the evidence is faked and the answer node above it is
    the real one.
    """
    if "999" in state["query"]:
        return {"docs": [], "signals": {"tool_error": "course_not_found", "retrieval_mode": None}}
    return {"docs": [LOOKUP], "signals": {"tool_error": None, "retrieval_mode": None}}


ROWS = [
    smoke.Query("How many credits is CS 374?", "course_fact", "answer", "the lookup row"),
    smoke.Query("How many credits is CS 999?", "course_fact", "refuse_handoff", "not in the term"),
    smoke.Query("Who is the coach?", "out_of_scope", "out_of_scope", "not about courses"),
]


@pytest.fixture(scope="module")
def runs():
    return smoke.run_queries(CFG, ROWS, fake_model=True, nodes={"dispatch": dispatcher})


def test_each_row_takes_the_exit_it_expects(runs):
    assert [run.state["exit"] for run in runs] == ["answer", "refuse_handoff", "out_of_scope"]
    assert all(run.exit_ok and run.intent_ok for run in runs)


def test_the_stand_in_answer_is_verified_against_the_evidence_it_was_given(runs):
    answered = runs[0].state
    assert answered["answer"] == "CS 374 — Algorithms [CS 374]"
    assert answered["signals"]["citations_verified"] == ["CS 374"]
    assert answered["signals"]["citations_unknown"] is None


def test_a_forced_label_leaves_the_raw_signal_empty_on_every_row(runs):
    # What the report's "nothing was classified" caveat is read off.
    assert all(run.state["signals"]["intent_raw"] is None for run in runs)


def test_every_exit_is_counted_including_the_ones_nobody_took():
    assert smoke.exit_counts(["answer", "answer"]) == {
        "answer": 2,
        "refuse_handoff": 0,
        "out_of_scope": 0,
    }


def test_the_acceptance_is_a_check_and_not_a_sentence(runs):
    assert smoke.missing_exits(runs) == []
    assert smoke.missing_exits(runs[:1]) == ["refuse_handoff", "out_of_scope"]


# --- the report -----------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def report(runs):
    return smoke.render_report(CFG, runs, smoke.DATASET, fake_model=True)


def test_the_header_says_what_produced_the_file(report):
    assert f"| config hash | `{CFG.hash}` |" in report
    assert "uv run python -m eval.scripts.day2_smoke --fake-model" in report
    assert "| answers written by | a stand-in, not a model |" in report


def test_a_run_with_no_model_in_it_never_reads_as_one(report):
    assert "**No model was called anywhere in this run.**" in report
    assert "Nothing was classified." in report
    assert report.count("written by the stand-in") == 1  # the one row that answered


def test_the_exit_table_is_the_acceptance(report):
    assert "| `answer` | 1 | 1 |" in report
    assert "| `refuse_handoff` | 1 | 1 |" in report
    assert "| `out_of_scope` | 1 | 1 |" in report
    assert "**All three exits fired.**" in report


def test_every_query_gets_a_section_with_the_command_that_reproduces_it(report):
    for n, row_ in enumerate(ROWS, start=1):
        assert f"### {n}. {row_.query}" in report
        assert f'python -m agent "{row_.query}" --intent {row_.expected_intent}' in report


def test_a_run_that_agrees_with_the_dataset_says_so_rather_than_saying_nothing(report):
    assert "None: every row took the intent and the exit it expects." in report


def test_a_disagreement_is_recorded_with_the_measurement_behind_it():
    # The same three runs, against a dataset row that expected the other exit: the report has to
    # say so in the table, in the section, and in the row's own header.
    expects_wrongly = [
        smoke.Query("How many credits is CS 999?", "course_fact", "answer", "in the set for this")
    ]
    runs = smoke.run_queries(CFG, expects_wrongly, fake_model=True, nodes={"dispatch": dispatcher})
    text = smoke.render_report(CFG, runs, smoke.DATASET, fake_model=True)
    assert "`answer` → `refuse_handoff`" in text
    assert "1 of 1 rows did something other than what the dataset expects" in text
    assert "course_not_found" in text
    assert "why the row is in the set: in the set for this" in text
    assert "✗" in text


def test_the_report_carries_no_local_path_and_no_clock(report):
    # rel_to_repo() everywhere, and nothing that changes between two runs of the same commit.
    from pathlib import Path

    assert str(Path.home()) not in report
    assert "eval/datasets/day2_queries.jsonl" in report


def test_the_same_run_renders_the_same_file(runs):
    first = smoke.render_report(CFG, runs, smoke.DATASET, fake_model=True)
    second = smoke.render_report(CFG, runs, smoke.DATASET, fake_model=True)
    assert first == second and first.endswith("\n")


def test_a_live_report_describes_a_live_run(runs):
    # The caveats are written from the run, not kept as one paragraph: regenerating with credits
    # must not leave the "no model was called" section behind.
    text = smoke.render_report(CFG, runs, smoke.DATASET, fake_model=False)
    assert "**This run called `gpt-4o-mini`**" in text
    assert "No model was called" not in text
    assert "uv run --env-file .env python -m eval.scripts.day2_smoke" in text
