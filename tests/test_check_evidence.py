"""Tests for agent/nodes/check_evidence.py: the four outcomes, and the two numbers kept from each.

No index and no model. The gate is one comparison -- a number #36 put in ``docs`` against a number
``config.yaml`` declares -- so the evidence here is built by hand, or gathered by the real
dispatcher over a fake ``retrieve`` whose scores are the ones measured on the live BM25 index
(bm25 key ``877a6610``). ``indexes/`` is gitignored, so a checkout has no 6 MB pickle to open, and
what the acceptance is about is which side of the threshold a measured score falls on, not the
ranking that produced it.

:data:`MEASURED` is that measurement, twelve queries deep, and the last section holds it against
the live config's threshold. It is also where the two inversions are pinned: the query whose right
answer scores *below* a query with no answer at all, and the pair of queries that return the same
document 53% apart. A threshold chosen to admit the first would admit the second, which is the
argument for Day 5 and is the reason the number is not called a confidence anywhere.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from agent import graph as ag
from agent.nodes import check_evidence as ce
from agent.nodes import exits
from agent.nodes.check_evidence import (
    METHODS,
    SOURCES,
    check_evidence_node,
    check_method,
    check_number,
    evidence_section,
    judge,
    one_source,
    threshold_for,
    thresholds,
)
from agent.nodes.dispatch import dispatch_node
from agent.state import Evidence, check_update, initial_state
from config import Config, bm25_key, dense_key, load_config
from ingest.chunk import chunk_key
from retrieval import MODES, RetrievalResult
from retrieval.types import Hit

TERM = "2026-sp"
LLM = {"provider": "openai", "model": "gpt-4o-mini", "temperature": 0, "max_tokens": 800}
AGENT = {"retrieval_mode": "bm25"}
#: Round numbers, so a test reads as "above" or "below" rather than as arithmetic. The live
#: thresholds are the measured ones and are asserted in the last section.
EVIDENCE = {"method": "top_score", "thresholds": {"bm25": 10.0, "dense": 0.4, "hybrid": 0.025}}

#: Top-1 BM25 score on the live index, and which side of the live ``bm25: 15.0`` it falls on.
#: Measured, not guessed -- and the point of the table is how little the order has to do with
#: whether the first hit was the right course (the comment on each row says what came first).
MEASURED: tuple[tuple[str, float, bool], ...] = (
    ("How many credits is MATH 241?", 30.9085, True),  # MATH 241, right
    ("CS 374", 29.0337, True),  # CS 374, right
    ("Which course covers organic chemistry?", 18.6496, True),  # CHEM 360, wrong
    ("Which intro machine learning course should I take?", 18.2746, True),  # IS 557
    ("I want a course on Shakespeare", 16.2165, True),  # ENGL 218, right
    ("linear algebra", 14.9652, False),  # MATH 227, right -- refused
    ("What does the data structures course cover?", 13.2029, False),  # ERAM 557, wrong
    ("machine learning", 11.9125, False),  # IS 557 again, 53% lower
    ("What does the algorithms course cover?", 11.7914, False),  # ECON 300, wrong
    ("Is there a course about quantum origami?", 11.0561, False),  # ECE 305, no such course
    ("introduction to statistics", 9.0038, False),  # STAT 400, right -- refused
    ("algorithms", 8.4812, False),  # CS 374, exactly right -- refused
)


@pytest.fixture
def cfg() -> Config:
    """A synthetic config: the term the exits quote, and the three sections the nodes validate."""
    return Config({"data": {"term": TERM}, "llm": LLM, "agent": AGENT, "evidence": EVIDENCE})


@pytest.fixture(autouse=True)
def fresh_cache():
    """No test inherits a graph another one compiled."""
    ag.clear_cache()
    yield
    ag.clear_cache()


def hit(score: float, doc_id: str = "2026-sp:CS-374") -> Evidence:
    """One retrieved block, as #36 converts a :class:`~retrieval.types.Hit`."""
    return Evidence(doc_id=doc_id, text=f"{doc_id} — a course", score=score, source="bm25")


def lookup(doc_id: str = "CS 374") -> Evidence:
    """One tool block: an exact catalog lookup, which carries no score."""
    return Evidence(doc_id=doc_id, text=f"{doc_id} — a course", score=None, source="get_course")


def gate(cfg: Config, docs: list[Evidence], tool_error: str | None = None) -> dict:
    """Run the built node over a state carrying ``docs`` and ``tool_error``."""
    state = initial_state("q")
    state["docs"] = docs
    state["signals"]["tool_error"] = tool_error
    return dict(check_evidence_node(cfg)(state))


class FakeRetrieval:
    """``retrieve`` with the scores already decided, so the gate can be driven through #36."""

    def __init__(self, *scores: float) -> None:
        self.scores = scores

    def __call__(self, query: str, mode: str, k=None, cfg=None) -> RetrievalResult:
        hits = [
            Hit(doc_id=f"2026-sp:X-{i}", score=score, rank=i, text=f"X {i}", metadata={})
            for i, score in enumerate(self.scores, start=1)
        ]
        return RetrievalResult(
            query=query, mode=mode, config_hash="deadbeef", hits=hits, latency_ms=1.0
        )


# --- the config section ---------------------------------------------------------------------------


def test_the_thresholds_come_from_the_config(cfg):
    assert thresholds(cfg) == {"bm25": 10.0, "dense": 0.4, "hybrid": 0.025}


def test_the_module_declares_no_threshold_of_its_own():
    # #37's first acceptance, read off the syntax tree rather than trusted: every float in this
    # module would be a tuning decision outside the config hash, and the prose in the docstring
    # quotes plenty of numbers without any of them being one.
    tree = ast.parse(Path(ce.__file__).read_text(encoding="utf-8"))
    floats = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, float)
    ]
    assert floats == []


def test_a_missing_evidence_section_is_a_sentence_not_a_key_error(cfg):
    bare = Config({"data": {"term": TERM}})
    with pytest.raises(ValueError, match="no `evidence` section"):
        evidence_section(bare)
    with pytest.raises(ValueError, match="no `evidence` section"):
        check_evidence_node(bare)


def test_a_section_that_is_not_a_mapping_says_so():
    with pytest.raises(TypeError, match="`evidence` section is str"):
        evidence_section(Config({"evidence": "top_score"}))


def test_a_missing_method_names_the_ones_there_are(cfg):
    partial = Config({**cfg.to_dict(), "evidence": {"thresholds": EVIDENCE["thresholds"]}})
    with pytest.raises(ValueError, match="has no method"):
        check_method(partial)


def test_the_only_method_today_is_top_score(cfg):
    assert check_method(cfg) == "top_score"
    assert set(METHODS) == {"top_score"}


def test_day_5s_method_is_refused_by_name_until_it_exists(cfg):
    # Setting it early must not produce a threshold run reported as a calibrated one.
    early = Config({**cfg.to_dict(), "evidence": {**EVIDENCE, "method": "conformal"}})
    with pytest.raises(ValueError, match="evidence.method to 'conformal'"):
        check_evidence_node(early)


def test_a_missing_thresholds_table_is_a_sentence(cfg):
    partial = Config({**cfg.to_dict(), "evidence": {"method": "top_score"}})
    with pytest.raises(ValueError, match="has no thresholds"):
        thresholds(partial)


def test_a_thresholds_table_that_is_not_a_mapping_says_so(cfg):
    listed = Config({**cfg.to_dict(), "evidence": {**EVIDENCE, "thresholds": [10.0]}})
    with pytest.raises(TypeError, match="evidence.thresholds is tuple"):
        thresholds(listed)


def test_every_retrieval_mode_needs_a_threshold_even_the_one_this_run_will_not_use(cfg):
    # An ablation moves agent.retrieval_mode; a gate that only failed on the mode it was switched
    # to would fail in the middle of a sweep.
    partial = {"bm25": 10.0, "hybrid": 0.025}
    missing = Config({**cfg.to_dict(), "evidence": {**EVIDENCE, "thresholds": partial}})
    with pytest.raises(ValueError, match="is missing dense"):
        check_evidence_node(missing)


def test_a_threshold_for_something_retrieve_does_not_run_is_refused(cfg):
    typo = {**EVIDENCE["thresholds"], "bm_25": 10.0}
    named = Config({**cfg.to_dict(), "evidence": {**EVIDENCE, "thresholds": typo}})
    with pytest.raises(ValueError, match="names 'bm_25', which retrieve\\(\\) does not run"):
        thresholds(named)


def test_the_sources_a_threshold_may_be_declared_for_are_retrieves_modes():
    assert SOURCES == MODES


@pytest.mark.parametrize("value", ["10", None, True, [10.0]])
def test_a_threshold_that_is_not_a_number_is_refused(cfg, value):
    # `True` explicitly: it is an int in Python, and `bm25: true` would become a threshold of 1.0.
    table = {**EVIDENCE["thresholds"], "bm25": value}
    broken = Config({**cfg.to_dict(), "evidence": {**EVIDENCE, "thresholds": table}})
    with pytest.raises(TypeError, match="evidence.thresholds.bm25 is"):
        thresholds(broken)


def test_a_negative_threshold_is_refused_and_zero_is_not(cfg):
    # 0 is the no-gate ablation: BM25 drops score-0 documents, so everything returned passes.
    table = {**EVIDENCE["thresholds"], "bm25": -1.0}
    below = Config({**cfg.to_dict(), "evidence": {**EVIDENCE, "thresholds": table}})
    with pytest.raises(ValueError, match="a negative threshold passes every document"):
        thresholds(below)
    open_gate = {**EVIDENCE["thresholds"], "bm25": 0}
    no_gate = Config({**cfg.to_dict(), "evidence": {**EVIDENCE, "thresholds": open_gate}})
    assert thresholds(no_gate)["bm25"] == 0.0
    assert gate(no_gate, [hit(0.001)])["evidence_ok"] is True


def test_a_whole_number_threshold_is_recorded_as_a_float(cfg):
    # Day 5 reads a column, so a YAML `15` and a YAML `15.0` have to arrive the same way.
    table = {**EVIDENCE["thresholds"], "bm25": 10}
    whole = Config({**cfg.to_dict(), "evidence": {**EVIDENCE, "thresholds": table}})
    assert gate(whole, [hit(29.0337)])["signals"]["evidence_threshold"] == 10.0


def test_a_scored_block_from_a_source_with_no_threshold_names_the_gap():
    # Unreachable from #36 -- a tool result carries score=None -- so the message is about the
    # config, which is where the answer to "compared against what?" would have to be written.
    with pytest.raises(ValueError, match="evidence from 'get_course' carries a score"):
        threshold_for("get_course", {"bm25": 10.0})


def test_check_number_refuses_a_bool_and_converts_an_int():
    assert check_number(15, "x") == 15.0
    with pytest.raises(TypeError, match="x is True"):
        check_number(True, "x")


# --- the four outcomes ----------------------------------------------------------------------------


def test_a_tool_result_passes_without_consulting_any_threshold(cfg):
    # #37's third acceptance. `score is None` is an exact catalog lookup: get_course("CS 374")
    # returning the record *is* the evidence, and there is no ranking to be unsure about.
    update = gate(cfg, [lookup()])
    assert update["evidence_ok"] is True
    assert update["signals"] == {"retrieval_top_score": None, "evidence_threshold": None}


def test_one_certain_block_carries_a_mixed_list(cfg):
    # Any, not all: the lookup is enough, whatever came with it -- and no threshold is consulted,
    # so the scored blocks beside it are never compared against the wrong scale.
    update = gate(cfg, [lookup(), hit(1.0)])
    assert update["evidence_ok"] is True
    assert update["signals"]["evidence_threshold"] is None


def test_a_named_tool_error_fails_without_a_threshold(cfg):
    # #36 writes no docs with one, and the label is more specific than any number could be.
    update = gate(cfg, [], tool_error="no_course_code")
    assert update["evidence_ok"] is False
    assert update["signals"] == {"retrieval_top_score": None, "evidence_threshold": None}


def test_a_tool_error_outranks_evidence_that_would_have_passed(cfg):
    assert gate(cfg, [lookup()], tool_error="course_not_found")["evidence_ok"] is False


def test_empty_docs_fail(cfg):
    # A retrieval that came back with nothing: BM25 drops score-0 documents, so a query naming no
    # term the corpus holds ("asdfghjkl qwertyuiop") arrives here rather than at a threshold.
    update = gate(cfg, [])
    assert update["evidence_ok"] is False
    assert update["signals"]["retrieval_top_score"] is None


def test_a_high_scoring_retrieval_passes_and_both_numbers_are_kept(cfg):
    update = gate(cfg, [hit(29.0337), hit(11.5, "2026-sp:CS-173")])
    assert update["evidence_ok"] is True
    assert update["signals"] == {"retrieval_top_score": 29.0337, "evidence_threshold": 10.0}


def test_a_weak_retrieval_fails_and_both_numbers_are_still_kept(cfg):
    # The acceptance's other half, and the reason the numbers are recorded on a failing run too:
    # Day 5 calibrates on scores with outcomes, and a gate that dropped the number after comparing
    # it would make Day 5 start by re-running everything.
    update = gate(cfg, [hit(8.4812)])
    assert update["evidence_ok"] is False
    assert update["signals"] == {"retrieval_top_score": 8.4812, "evidence_threshold": 10.0}


def test_the_threshold_itself_passes(cfg):
    # `>=`, so hybrid's 0.025 admits the document both retrievers returned at their worst rank.
    assert gate(cfg, [hit(10.0)])["evidence_ok"] is True


def test_the_best_score_in_the_list_is_what_is_compared(cfg):
    # max, not docs[0]: the retrievers do return their hits in rank order, but a reordered list
    # must not be a silently different verdict.
    for docs in ([hit(29.0337), hit(1.0)], [hit(1.0), hit(29.0337)]):
        assert gate(cfg, docs)["signals"]["retrieval_top_score"] == 29.0337


def test_each_source_is_judged_against_its_own_threshold(cfg):
    # 0.4 would refuse every BM25 score and 10.0 would pass every dense one; the whole reason the
    # table is keyed by source is that neither number means anything on the other scale.
    dense = Evidence(doc_id="2026-sp:CS-374", text="…", score=0.42, source="dense")
    assert gate(cfg, [dense])["evidence_ok"] is True
    assert gate(cfg, [dense._replace(score=0.2)])["evidence_ok"] is False
    assert gate(cfg, [hit(0.42)])["evidence_ok"] is False


def test_evidence_from_two_retrievers_at_once_is_refused_rather_than_maxed(cfg):
    # Not reachable from #36 (a retrieval row stamps one mode on every hit it converts). A row
    # that mixed two needs a gate decision, and a max() over two scales is not one.
    mixed = [hit(29.0337), Evidence(doc_id="2026-sp:CS-173", text="…", score=0.42, source="dense")]
    with pytest.raises(ValueError, match="scored evidence from 2 sources \\(bm25, dense\\)"):
        gate(cfg, mixed)


def test_one_source_returns_the_source_and_the_best_score():
    assert one_source([hit(1.0), hit(29.0337)]) == ("bm25", 29.0337)


def test_a_block_whose_score_is_not_a_number_is_a_sentence_not_a_comparison_error(cfg):
    with pytest.raises(TypeError, match="a bm25 evidence block's score is '29'"):
        gate(cfg, [hit("29")])  # type: ignore[arg-type]


def test_judge_is_a_function_of_what_it_was_given():
    table = {"bm25": 10.0}
    assert judge([hit(29.0337)], None, table) == (True, 29.0337, 10.0)
    assert judge([hit(1.0)], None, table) == (False, 1.0, 10.0)
    assert judge([lookup()], None, table) == (True, None, None)
    assert judge([], None, table) == (False, None, None)
    assert judge([hit(29.0337)], "course_not_found", table) == (False, None, None)


# --- the node -------------------------------------------------------------------------------------


def test_the_node_writes_only_its_own_key_and_the_two_gate_signals(cfg):
    update = gate(cfg, [hit(29.0337)])
    assert set(update) == {"evidence_ok", "signals"}
    assert set(update["signals"]) == {"retrieval_top_score", "evidence_threshold"}
    assert check_update("check_evidence", update) == update


def test_the_verdict_is_a_real_bool(cfg):
    # The state check refuses anything else, and `numpy.bool_` is what a score that arrived as a
    # numpy scalar would produce -- check_number's float() is what keeps it a Python bool.
    for docs in ([hit(29.0337)], [hit(1.0)], [lookup()], []):
        assert type(gate(cfg, docs)["evidence_ok"]) is bool


def test_the_node_records_nothing_about_the_source_the_dispatcher_already_recorded(cfg):
    # `signals["retrieval_mode"]` is #36's and is exactly Evidence.source for a scored block, so a
    # third column here would agree with it on every row by construction.
    assert "retrieval_mode" not in gate(cfg, [hit(29.0337)])["signals"]


def test_the_node_is_named_after_the_state_key_it_owns(cfg):
    assert check_evidence_node(cfg).__name__ == "check_evidence"


def test_the_config_is_read_when_the_node_is_built_not_when_a_query_arrives(cfg):
    # Like the router's `llm` section: a config that cannot gate anything should not survive until
    # the first query of a report run.
    broken = Config({**cfg.to_dict(), "evidence": {**EVIDENCE, "method": "vibes"}})
    with pytest.raises(ValueError, match="evidence.method to 'vibes'"):
        check_evidence_node(broken)


def test_building_the_gate_opens_no_index_and_needs_no_key(cfg, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert gate(cfg, [hit(29.0337)])["evidence_ok"] is True


# --- the node inside the graph --------------------------------------------------------------------


def test_a_passing_gate_reaches_the_answer_node(cfg):
    state = ag.run(
        "How many credits is CS 374?",
        cfg=cfg,
        nodes={
            "route_intent": lambda s: {"intent": "course_fact"},
            "dispatch": lambda s: {"docs": [lookup()]},
            "answer": lambda s: {"answer": "CS 374 is 4 hours [CS 374].", "exit": "answer"},
        },
    )
    assert state["exit"] == "answer"
    assert state["evidence_ok"] is True


def test_a_refusing_gate_ends_the_run_at_the_hand_off_quoting_both_numbers(cfg):
    # #37's second acceptance, end to end: the weak retrieval, the exit it takes, and the sentence
    # a person reads -- which names the score and the threshold so that "no match" can be told
    # from "threshold too high".
    state = ag.run(
        "Is there a course about quantum origami?",
        cfg=cfg,
        nodes={
            "route_intent": lambda s: {"intent": "compare_recommend"},
            "dispatch": lambda s: {
                "docs": [hit(8.4812)],
                "signals": {"retrieval_mode": "bm25"},
            },
        },
    )
    assert state["exit"] == "refuse_handoff"
    assert state["evidence_ok"] is False
    assert (
        "the best bm25 match scored 8.4812, against the bm25 threshold of 10.0" in (state["answer"])
    )
    assert exits.SCORE_REMEDY in state["answer"]


def test_a_tool_error_reaches_the_hand_off_with_its_own_sentence_not_a_score(cfg):
    state = ag.run(
        "What are the prerequisites?",
        cfg=cfg,
        nodes={
            "route_intent": lambda s: {"intent": "prereq_chain"},
            "dispatch": lambda s: {"docs": [], "signals": {"tool_error": "no_course_code"}},
        },
    )
    assert state["exit"] == "refuse_handoff"
    assert exits.REASONS["no_course_code"][0] in state["answer"]
    assert "scored" not in state["answer"]


def test_an_empty_retrieval_reaches_the_hand_off_with_the_empty_sentence(cfg):
    state = ag.run(
        "asdfghjkl qwertyuiop",
        cfg=cfg,
        nodes={
            "route_intent": lambda s: {"intent": "compare_recommend"},
            "dispatch": lambda s: {"docs": [], "signals": {"retrieval_mode": "bm25"}},
        },
    )
    assert state["exit"] == "refuse_handoff"
    assert exits.EMPTY_REASON in state["answer"]


def test_the_real_dispatcher_and_the_real_gate_agree_on_a_measured_score(cfg):
    # The two nodes together, with only `retrieve` faked: #36 converts the hits and stamps the
    # mode, #37 reads the score off the block it made and compares it to the config's own number.
    strong = FakeRetrieval(18.2746, 15.2363)
    state = ag.run(
        "Which intro machine learning course should I take?",
        cfg=cfg,
        nodes={
            "route_intent": lambda s: {"intent": "compare_recommend"},
            "dispatch": dispatch_node(cfg, retrieve=strong),
            "answer": lambda s: {"answer": "IS 557 [2026-sp:X-1].", "exit": "answer"},
        },
    )
    assert state["exit"] == "answer"
    assert state["signals"] == {
        "intent_raw": None,
        "intent_unparsed": None,
        "retrieval_mode": "bm25",
        "tool_error": None,
        "retrieval_top_score": 18.2746,
        "evidence_threshold": 10.0,
        "citations_verified": None,
        "citations_unknown": None,
    }


def test_the_same_pair_of_nodes_over_a_weak_ranking_hands_over(cfg):
    weak = FakeRetrieval(9.0038, 8.4812)
    state = ag.run(
        "What does the algorithms course cover?",
        cfg=cfg,
        nodes={
            "route_intent": lambda s: {"intent": "course_fact"},
            "dispatch": dispatch_node(cfg, retrieve=weak),
        },
    )
    assert state["exit"] == "refuse_handoff"
    assert state["signals"]["retrieval_top_score"] == 9.0038


def test_the_graph_builds_the_real_gate_for_the_live_config():
    assert ag.build_nodes(load_config())["check_evidence"].__name__ == "check_evidence"


# --- the live config ------------------------------------------------------------------------------


def test_the_live_config_carries_the_section_this_node_reads():
    live = load_config()
    assert check_method(live) == "top_score"
    assert thresholds(live) == {"bm25": 15.0, "dense": 0.35, "hybrid": 0.025}


def test_the_evidence_section_renames_no_index():
    # The acceptance #28's `llm:` and #36's `agent:` each stated: a section no index is a function
    # of moves the full hash (the reports get new names) and must move none of the three keys an
    # artifact on disk is named after.
    live = load_config()
    without = Config({k: v for k, v in live.to_dict().items() if k != "evidence"})
    assert live.hash != without.hash
    assert chunk_key(live) == chunk_key(without)
    assert dense_key(live, chunk_key(live)) == dense_key(without, chunk_key(without))
    assert bm25_key(live, chunk_key(live)) == bm25_key(without, chunk_key(without))


def test_the_hybrid_threshold_is_exactly_both_retrievers_agreeing():
    # Written down rather than left to be inferred, because it is not a confidence: with two
    # retrievers, 2/(rrf_k + top_k_each) is the score of a document both of them returned at the
    # worst rank either reports, and one retriever alone cannot reach it however well it ranked.
    live = load_config()
    rrf_k, deep = live["fusion"]["rrf_k"], live["fusion"]["top_k_each"]
    assert thresholds(live)["hybrid"] == 2 / (rrf_k + deep)
    assert 1 / (rrf_k + 1) < thresholds(live)["hybrid"]


@pytest.mark.parametrize(("query", "score", "passes"), MEASURED)
def test_each_measured_query_falls_on_the_side_of_the_threshold_it_was_chosen_for(
    query, score, passes
):
    live = load_config()
    docs = [hit(score)]
    assert judge(docs, None, thresholds(live)).ok is passes


def test_the_right_answer_can_score_below_a_question_with_no_answer():
    # The inversion that says what this number is not. "algorithms" puts CS 374 first -- exactly
    # right -- at 8.48, below "Is there a course about quantum origami?" at 11.06, which puts ECE
    # 305 first and is a course that does not exist. So no threshold on this scale can admit the
    # first without admitting the second, and none of them is a confidence.
    scores = {query: score for query, score, _ in MEASURED}
    assert scores["algorithms"] < scores["Is there a course about quantum origami?"]


def test_a_longer_question_scores_the_same_document_half_again_higher():
    # BM25 sums a contribution per query term, so part of what the threshold measures is how many
    # words the question had: both of these return IS 557 first.
    scores = {query: score for query, score, _ in MEASURED}
    asked = scores["Which intro machine learning course should I take?"]
    bare = scores["machine learning"]
    assert asked / bare > 1.5
    live = thresholds(load_config())
    assert judge([hit(asked)], None, live).ok
    assert not judge([hit(bare)], None, live).ok
