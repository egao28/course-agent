"""Tests for agent/trace.py: what a run says about itself, in the words a person reads.

Every state here is built by hand. The trace is a function of the final state and nothing else --
that is the point of the module -- so a test does not need the graph to produce one, and the
shapes that are awkward to reach through a real run (an unparsed label, a discarded draft) are as
cheap here as the easy ones.
"""

from __future__ import annotations

import pytest

from agent import trace
from agent.state import Evidence, initial_state

LOOKUP = Evidence(doc_id="CS 374", text="CS 374 — Algorithms\n4 hours.", score=None,
                  source="get_course")
CHUNK = Evidence(
    doc_id="2026-sp:IS-557", text="IS 557 — Applied Machine Learning", score=18.2746, source="bm25"
)


def state(**changes):
    """A finished state, with ``signals`` merged rather than replaced."""
    base = initial_state(changes.pop("query", "How many credits is CS 374?"))
    base["signals"].update(changes.pop("signals", {}))
    base.update(changes)
    return base


# --- the intent line ------------------------------------------------------------------------------


def test_a_forced_label_says_no_model_was_asked():
    # #35's invariant: `intent_raw is None` happens on that path and on no other.
    forced = state(intent="course_fact", signals={"intent_raw": None, "intent_unparsed": False})
    assert trace.intent_note(forced) == "forced -- no model was asked"


def test_a_classified_label_does_not_claim_to_be_forced():
    classified = state(
        intent="course_fact", signals={"intent_raw": "course_fact", "intent_unparsed": False}
    )
    assert trace.intent_note(classified) == "classified"


def test_an_unparsed_completion_is_quoted_back():
    refused = state(
        intent="out_of_scope", signals={"intent_raw": "Course_fact.", "intent_unparsed": True}
    )
    note = trace.intent_note(refused)
    assert "Course_fact." in note and "not one of the five labels" in note


# --- where the evidence came from -----------------------------------------------------------------


def test_a_retrieval_row_names_the_mode_and_counts_the_hits():
    retrieved = state(docs=[CHUNK, CHUNK], signals={"retrieval_mode": "bm25"})
    assert trace.gathered(retrieved) == "retrieve(bm25), 2 hits"


def test_a_tool_row_names_the_tool():
    assert trace.gathered(state(docs=[LOOKUP])) == "get_course, 1 block"


def test_a_row_that_produced_nothing_names_the_tool_error():
    empty = state(docs=[], signals={"tool_error": "no_course_code"})
    assert trace.gathered(empty) == "none -- the tool it needed reported no_course_code"


def test_an_out_of_scope_run_says_dispatch_never_ran():
    # evidence_ok is None and the exit is the router's: nothing was gathered because nothing was
    # asked for, which is a different sentence from "the retriever came back empty".
    routed = state(intent="out_of_scope", exit="out_of_scope", answer="Outside what I answer.")
    assert "router's own exit" in trace.gathered(routed)


# --- the gate -------------------------------------------------------------------------------------


def test_a_lookup_passes_without_a_threshold_and_the_line_says_why():
    passed = state(docs=[LOOKUP], evidence_ok=True)
    assert trace.gate_note(passed) == (
        "passed -- an exact catalog lookup carries no score, so no threshold was consulted"
    )


def test_a_scored_comparison_prints_both_numbers_whichever_way_it_went():
    signals = {"retrieval_mode": "bm25", "retrieval_top_score": 11.7914, "evidence_threshold": 15.0}
    refused = state(docs=[CHUNK], evidence_ok=False, signals=signals)
    assert trace.gate_note(refused) == "refused -- 11.7914 < 15.0 (bm25)"
    passed = state(docs=[CHUNK], evidence_ok=True, signals={**signals, "retrieval_top_score": 18.2})
    assert trace.gate_note(passed) == "passed -- 18.2000 >= 15.0 (bm25)"


def test_a_tool_error_is_reported_instead_of_a_comparison():
    refused = state(evidence_ok=False, signals={"tool_error": "course_not_found"})
    assert trace.gate_note(refused).startswith("refused -- course_not_found")


def test_a_gate_that_never_ran_says_so_rather_than_refusing():
    assert trace.gate_note(state()).startswith("not reached")


# --- the evidence table ---------------------------------------------------------------------------


def test_a_tool_result_shows_no_score_rather_than_a_zero():
    rows = trace.evidence_rows(state(docs=[LOOKUP]))
    assert rows == [("1", trace.MISSING, "CS 374", "CS 374 — Algorithms")]


def test_a_scored_block_keeps_four_decimal_places():
    assert trace.evidence_rows(state(docs=[CHUNK]))[0][1] == "18.2746"


def test_a_long_first_line_is_elided_to_the_width_asked_for():
    long = Evidence(doc_id="x", text="y" * 200, score=None, source="get_course")
    assert trace.evidence_rows(state(docs=[long]), 20)[0][3] == "y" * 19 + "…"


def test_the_table_is_empty_when_nothing_was_gathered():
    assert trace.format_evidence_table(state()) == ""


# --- the answer line ------------------------------------------------------------------------------


def test_the_stand_in_is_never_reported_as_a_model():
    # The one line that must not read as "gpt-4o-mini wrote this".
    answered = state(
        exit="answer", answer="CS 374 — Algorithms [CS 374]",
        signals={"citations_verified": ["CS 374"], "citations_unknown": None},
    )
    note = trace.answer_note(answered, trace.STANDIN)
    assert "canned text, no model was called" in note
    assert "gpt-4o-mini" not in note


def test_the_name_this_module_matches_is_the_one_the_stand_in_reports():
    # Matched as a string so `agent.trace` does not import `llm` (and `openai` with it); this is
    # what keeps that safe.
    from llm import StandInChat

    assert StandInChat.model == trace.STANDIN


def test_a_refusal_says_its_text_was_not_generated():
    refused = state(exit="refuse_handoff", answer="I could not answer this ...")
    assert trace.answer_note(refused, "gpt-4o-mini") == (
        "fixed text -- neither refusal exit calls a model"
    )


def test_a_discarded_draft_says_the_model_ran_and_was_overruled():
    discarded = state(
        exit="refuse_handoff", answer="I could not answer this ...",
        signals={"citations_verified": [], "citations_unknown": ["CS 999"]},
    )
    note = trace.answer_note(discarded, "gpt-4o-mini")
    assert "drafted by gpt-4o-mini and discarded" in note
    assert trace.exit_note(discarded) == (
        "draft dropped -- it cited CS 999, which this run never retrieved"
    )


def test_a_draft_that_cited_nothing_is_distinguishable_from_one_that_invented_a_citation():
    uncited = state(
        exit="refuse_handoff", answer="I could not answer this ...",
        signals={"citations_verified": [], "citations_unknown": None},
    )
    assert trace.exit_note(uncited) == "draft dropped -- it cited nothing at all"


# --- the whole trace ------------------------------------------------------------------------------


def test_the_trace_carries_every_part_of_the_run_a_person_debugs_with():
    answered = state(
        query="Which intro machine learning course should I take?",
        intent="compare_recommend",
        docs=[CHUNK],
        evidence_ok=True,
        exit="answer",
        answer="IS 557 — Applied Machine Learning [2026-sp:IS-557]",
        signals={
            "intent_raw": None,
            "intent_unparsed": False,
            "retrieval_mode": "bm25",
            "retrieval_top_score": 18.2746,
            "evidence_threshold": 15.0,
            "citations_verified": ["2026-sp:IS-557"],
            "citations_unknown": None,
        },
    )
    text = trace.format_trace(answered, model=trace.STANDIN)
    for part in (
        "Which intro machine learning course should I take?",  # the query, verbatim
        "compare_recommend",  # the label
        "retrieve(bm25), 1 hit",  # which row of the dispatch table ran
        "18.2746",  # the evidence's score, and the gate's comparison
        "15.0",
        "exit      answer",
        "cited 2026-sp:IS-557",
        "IS 557 — Applied Machine Learning [2026-sp:IS-557]",  # the answer itself
    ):
        assert part in text, part
    assert text.endswith("\n") and not text.endswith("\n\n")


def test_a_trace_renders_for_a_run_that_gathered_nothing():
    # No evidence table, no citations, and nothing raised: the refusing runs are half of #39's set.
    refused = state(
        intent="prereq_chain", evidence_ok=False, exit="refuse_handoff",
        answer="I could not answer this from the 2026-sp course catalog ...",
        signals={"intent_raw": None, "tool_error": "no_course_code"},
    )
    text = trace.format_trace(refused, model="gpt-4o-mini")
    assert "no_course_code" in text
    assert "doc_id" not in text
    assert text.rstrip().endswith("I could not answer this from the 2026-sp course catalog ...")


@pytest.mark.parametrize("model", [None, "gpt-4o-mini", trace.STANDIN])
def test_every_model_argument_renders(model):
    answered = state(
        exit="answer", answer="CS 374 — Algorithms [CS 374]", docs=[LOOKUP], evidence_ok=True,
        signals={"citations_verified": ["CS 374"], "citations_unknown": None},
    )
    assert "CS 374 — Algorithms [CS 374]" in trace.format_trace(answered, model=model)
