"""Tests for agent/state.py: the record shapes, the signal declaration, and the write table.

Nothing here builds a graph -- these are the guards that make the graph's contract checkable, and
they are tested as plain functions so that a failure names the rule rather than a superstep. The
wiring is ``tests/test_graph.py``.

The write table is the part most worth pinning. LangGraph silently drops an update key the state
schema does not have, so the only thing standing between a typo and a run that quietly did nothing
is :func:`agent.state.check_update` -- and the only thing standing between that and rot is this
file.
"""

from __future__ import annotations

from typing import get_args

import pytest

from agent.state import (
    EXITS,
    INTENTS,
    NODE_WRITES,
    SHARED_WRITES,
    SIGNAL_KEYS,
    STATE_KEYS,
    AgentState,
    Evidence,
    Exit,
    Intent,
    check_docs,
    check_signals,
    check_update,
    initial_signals,
    initial_state,
    merge_signals,
    owners_of,
)

CHUNK = Evidence(doc_id="2026-sp:CS-374", text="CS 374 - Algorithms", score=29.0337, source="bm25")
LOOKUP = Evidence(doc_id="CS 374", text="CS 374, 4 hours.", score=None, source="get_course")


# --- the labels -----------------------------------------------------------------------------------


def test_the_five_intents_and_their_literal_cannot_drift():
    # Two declarations of the same five labels -- one for the type checker, one for the router's
    # prompt and #36's dispatch table. get_args is what keeps them one list.
    assert INTENTS == get_args(Intent)
    assert len(set(INTENTS)) == 5


def test_the_three_exits_and_their_literal_cannot_drift():
    assert EXITS == get_args(Exit)
    assert EXITS == ("answer", "refuse_handoff", "out_of_scope")


def test_out_of_scope_is_both_an_intent_and_an_exit():
    # The one label that is also a destination: the router sends it straight to its exit, which is
    # why route_after_intent branches on it by name.
    assert "out_of_scope" in INTENTS
    assert "out_of_scope" in EXITS


# --- evidence -------------------------------------------------------------------------------------


def test_evidence_carries_four_fields_in_order():
    assert tuple(CHUNK) == ("2026-sp:CS-374", "CS 374 - Algorithms", 29.0337, "bm25")


def test_a_tool_result_and_a_retrieval_hit_are_the_same_type():
    # The whole reason Evidence exists: #38 renders one input shape and #37 tells them apart by
    # score, not by class.
    assert type(CHUNK) is type(LOOKUP)
    assert LOOKUP.score is None


def test_check_docs_refuses_anything_that_is_not_evidence():
    from retrieval.types import Hit

    hit = Hit(doc_id="2026-sp:CS-374", score=29.0, rank=1, text="CS 374", metadata={})
    with pytest.raises(TypeError, match="every block must be an agent.state.Evidence"):
        check_docs([hit], "dispatch")


def test_check_docs_refuses_a_bare_string():
    # A string is a Sequence, so the length check would have passed it through one character at a
    # time.
    with pytest.raises(TypeError, match="must be a list of Evidence"):
        check_docs("CS 374", "dispatch")


def test_check_docs_returns_a_list_copy():
    docs = [CHUNK, LOOKUP]
    checked = check_docs(tuple(docs), "dispatch")
    assert checked == docs
    assert isinstance(checked, list)


# --- signals --------------------------------------------------------------------------------------


def test_the_declared_signals_are_the_seven_day_5_reads():
    assert set(SIGNAL_KEYS) == {
        "intent_raw",
        "intent_unparsed",
        "retrieval_mode",
        "retrieval_top_score",
        "evidence_threshold",
        "tool_error",
        "citations_unknown",
    }
    assert len(set(SIGNAL_KEYS)) == len(SIGNAL_KEYS)


def test_every_run_starts_with_the_same_shaped_signal_record():
    # Prefilled with None rather than empty: Day 5 wants a rectangular table, where a measurement
    # nobody took is a None cell and not an absent key.
    signals = initial_signals()
    assert set(signals) == set(SIGNAL_KEYS)
    assert set(signals.values()) == {None}


def test_an_undeclared_signal_is_refused_by_name():
    with pytest.raises(ValueError, match="undeclared signal\\(s\\) 'vibes'"):
        check_signals({"vibes": 0.9}, "check_evidence")


def test_check_signals_refuses_a_non_mapping():
    with pytest.raises(TypeError, match="not a mapping"):
        check_signals(["intent_raw"], "route_intent")


def test_merge_signals_keeps_the_earlier_keys():
    # Without the reducer the gate's update would replace the router's, and intent_raw -- the
    # thing #35 logs precisely so it survives -- would be gone by the time the run ended.
    merged = merge_signals({"intent_raw": "prereq chain!"}, {"evidence_threshold": 0.02})
    assert merged == {"intent_raw": "prereq chain!", "evidence_threshold": 0.02}


def test_merge_signals_lets_a_later_write_correct_an_earlier_one():
    assert merge_signals({"tool_error": "no_course_code"}, {"tool_error": None}) == {
        "tool_error": None
    }


def test_merge_signals_is_idempotent():
    # langgraph 1.2.11 applies the reducer to the same update twice on the final superstep, so an
    # accumulating reducer would double-count there.
    once = merge_signals({"intent_raw": "x"}, {"tool_error": "y"})
    assert merge_signals(once, {"tool_error": "y"}) == once


def test_merge_signals_handles_the_empty_base():
    assert merge_signals(None, None) == {}
    assert merge_signals(None, {"intent_unparsed": True}) == {"intent_unparsed": True}


# --- the initial state ----------------------------------------------------------------------------


def test_the_initial_state_has_every_key_the_schema_declares():
    # Every key present from the start, so a node reading one before its owner ran gets None and
    # not a KeyError.
    state = initial_state("How many credits is CS 374?")
    assert set(state) == STATE_KEYS
    assert state["query"] == "How many credits is CS 374?"
    assert state["docs"] == []
    assert (state["intent"], state["evidence_ok"], state["answer"], state["exit"]) == (
        None,
        None,
        None,
        None,
    )


@pytest.mark.parametrize("query", ["", "   ", "\n\t"])
def test_an_empty_query_is_refused_before_anything_runs(query):
    with pytest.raises(ValueError, match="query is empty"):
        initial_state(query)


def test_state_keys_are_read_off_the_typed_dict():
    assert STATE_KEYS == frozenset(AgentState.__annotations__)


# --- who writes what ------------------------------------------------------------------------------


def test_every_owned_key_is_a_real_state_key():
    owned = set().union(*NODE_WRITES.values()) | set(SHARED_WRITES)
    assert owned <= STATE_KEYS


def test_nobody_in_the_graph_writes_the_query():
    # The question as asked is the one thing a node must not rewrite: #38's prompt quotes it and
    # #39's report files the run under it.
    assert "query" not in set().union(*NODE_WRITES.values())
    assert "nobody in the graph" in owners_of("query")


@pytest.mark.parametrize(
    ("key", "owner"),
    [("intent", "route_intent"), ("docs", "dispatch"), ("evidence_ok", "check_evidence")],
)
def test_the_three_working_keys_have_exactly_one_owner(key, owner):
    assert owners_of(key) == owner


def test_all_three_exits_write_the_same_pair():
    # An exit is an exit however it was reached, which is what lets #38's citation downgrade land
    # on the same refuse_handoff node instead of writing a second refusal message.
    assert NODE_WRITES["answer"] == NODE_WRITES["out_of_scope"] == NODE_WRITES["refuse_handoff"]
    assert NODE_WRITES["answer"] == frozenset({"answer", "exit"})


# --- check_update ---------------------------------------------------------------------------------


def test_a_node_writing_its_own_key_and_signals_is_fine():
    update = check_update("route_intent", {"intent": "course_fact", "signals": {"intent_raw": "x"}})
    assert update == {"intent": "course_fact", "signals": {"intent_raw": "x"}}


def test_an_empty_update_is_fine():
    # A node that decided nothing still returns something: the gate's "nothing to record" is an
    # empty mapping, not None.
    assert check_update("dispatch", {}) == {}


def test_a_node_may_not_write_another_nodes_key():
    with pytest.raises(ValueError, match="dispatch wrote 'intent', which belongs to route_intent"):
        check_update("dispatch", {"intent": "prereq_chain"})


def test_a_key_the_state_does_not_have_is_refused_rather_than_dropped():
    with pytest.raises(ValueError, match="not a key of AgentState"):
        check_update("dispatch", {"documents": []})


def test_an_unknown_node_name_is_refused():
    with pytest.raises(KeyError, match="no node named 'summarize'"):
        check_update("summarize", {})


def test_a_node_that_returns_something_other_than_a_mapping_is_refused():
    with pytest.raises(TypeError, match="returned str"):
        check_update("answer", "CS 374 is 4 hours")


def test_an_invented_intent_label_never_reaches_the_dispatch_table():
    with pytest.raises(ValueError, match="not one of the five labels"):
        check_update("route_intent", {"intent": "prerequisites_maybe"})


def test_a_none_intent_is_refused_too():
    # None means "the router has not run", and it has.
    with pytest.raises(ValueError, match="not one of the five labels"):
        check_update("route_intent", {"intent": None})


def test_an_invented_exit_is_refused():
    with pytest.raises(ValueError, match="the exits are"):
        check_update("answer", {"answer": "...", "exit": "escalate"})


def test_the_gate_must_decide():
    with pytest.raises(TypeError, match="the gate returns True or False"):
        check_update("check_evidence", {"evidence_ok": None})


@pytest.mark.parametrize("answer", ["", "   "])
def test_an_exit_may_not_write_an_empty_answer(answer):
    with pytest.raises(ValueError, match="an empty answer reaches them"):
        check_update("out_of_scope", {"answer": answer, "exit": "out_of_scope"})


def test_check_update_returns_a_copy():
    update = {"docs": [LOOKUP]}
    checked = check_update("dispatch", update)
    assert checked == update
    assert checked is not update
