"""Tests for agent/graph.py and the two fixed exits: the wiring, the routing, the three exits.

Every node here is injected. That is not a workaround but the acceptance for #34: the skeleton is
correct if a run reaches each of the three exits and the final state says which one it took,
whatever the nodes themselves do. It is also how each real node was tested before the next one
existed, and how #39's CLI runs at all while the account has no OpenAI credits. The four real nodes
have their own files -- ``tests/test_route_intent.py``, ``tests/test_dispatch.py``,
``tests/test_check_evidence.py`` and ``tests/test_answer.py``; here they are fakes like the rest.

``cfg`` is a synthetic config, not the live ``config.yaml``: the skeleton reads ``data.term`` from
it for the two exit messages, and the ``llm`` / ``agent`` / ``evidence`` sections are what the
router, the dispatcher and the gate each validate when they are built (no key is read, no client is
made and no index is opened -- see the three node modules). One section at the end runs against the
live config, to prove the default ``run(query)`` path resolves it.
"""

from __future__ import annotations

import subprocess
import sys

import pytest
from langgraph.graph import END

from agent import graph as ag
from agent.nodes import exits
from agent.state import Evidence, initial_state
from config import Config

TERM = "2026-sp"
LLM = {"provider": "openai", "model": "gpt-4o-mini", "temperature": 0, "max_tokens": 800}
#: bm25 because it is the mode that runs today; every test here overrides the node anyway.
AGENT = {"retrieval_mode": "bm25"}
#: What #37's gate validates when the graph builds it. Round numbers: the live thresholds are
#: measured and are asserted in ``tests/test_check_evidence.py``.
EVIDENCE = {"method": "top_score", "thresholds": {"bm25": 10.0, "dense": 0.4, "hybrid": 0.025}}
LOOKUP = Evidence(
    doc_id="CS 374", text="CS 374 - Algorithms. 4 hours.", score=None, source="get_course"
)
CHUNK = Evidence(doc_id="2026-sp:CS-374", text="CS 374 - Algorithms", score=29.0337, source="bm25")


@pytest.fixture
def cfg() -> Config:
    return Config({"data": {"term": TERM}, "llm": LLM, "agent": AGENT, "evidence": EVIDENCE})


@pytest.fixture(autouse=True)
def fresh_cache():
    """No test inherits a graph another one compiled."""
    ag.clear_cache()
    yield
    ag.clear_cache()


# --- fake nodes -----------------------------------------------------------------------------------


def router(label: str, raw: str | None = None):
    """A stand-in for #35: the label it was told to produce, and optionally the raw completion."""

    def node(state):
        update: dict = {"intent": label}
        if raw is not None:
            update["signals"] = {"intent_raw": raw, "intent_unparsed": raw != label}
        return update

    return node


def dispatcher(docs: list[Evidence], tool_error: str | None = None, mode: str | None = None):
    """A stand-in for #36: whatever evidence the test wants, in the one record shape."""

    def node(state):
        return {"docs": list(docs), "signals": {"tool_error": tool_error, "retrieval_mode": mode}}

    return node


def gate(ok: bool, score: float | None = None, threshold: float | None = None):
    """A stand-in for the gate: the verdict, and the two numbers the hand-off can quote."""

    def node(state):
        return {
            "evidence_ok": ok,
            "signals": {"retrieval_top_score": score, "evidence_threshold": threshold},
        }

    return node


def writer(text: str, cited: list[str] | None = None):
    """A stand-in for the answer node: the text, and the ids its citation check verified."""
    signals = {"citations_verified": list(cited or []), "citations_unknown": None}
    return lambda state: {"answer": text, "exit": "answer", "signals": signals}


def rejector(unknown: list[str]):
    """The answer node whose citation check rejected the draft: no answer, the hand-off exit."""
    return lambda state: {
        "exit": "refuse_handoff",
        "signals": {"citations_verified": [], "citations_unknown": unknown},
    }


ANSWERING = {
    "route_intent": router("course_fact", "course_fact"),
    "dispatch": dispatcher([LOOKUP]),
    "check_evidence": gate(True),
    "answer": writer("CS 374 is 4 hours [CS 374].", ["CS 374"]),
}


# --- the shape of the graph -----------------------------------------------------------------------


def test_build_graph_compiles_every_node_from_the_registry(cfg):
    # #34's first acceptance, which held before any of the six nodes existed and still does.
    compiled = ag.build_graph(cfg)
    assert sorted(compiled.get_graph().nodes) == [
        "__end__",
        "__start__",
        "answer",
        "check_evidence",
        "dispatch",
        "out_of_scope",
        "refuse_handoff",
        "route_intent",
    ]


def test_the_edges_are_the_ones_the_issue_draws_plus_the_citation_downgrade(cfg):
    edges = {(edge.source, edge.target) for edge in ag.build_graph(cfg).get_graph().edges}
    assert edges == {
        ("__start__", "route_intent"),
        ("route_intent", "dispatch"),
        ("route_intent", "out_of_scope"),
        ("dispatch", "check_evidence"),
        ("check_evidence", "answer"),
        ("check_evidence", "refuse_handoff"),
        # Not in #34's sketch: #38 discards an answer whose citations it could not verify, and the
        # hand-off message is minted in one place whichever node decided to hand over.
        ("answer", "refuse_handoff"),
        ("answer", "__end__"),
        ("out_of_scope", "__end__"),
        ("refuse_handoff", "__end__"),
    }


def test_compiling_opens_no_index_and_needs_no_key(cfg, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert ag.build_graph(cfg) is not None


def test_importing_the_graph_pulls_in_neither_openai_nor_chromadb():
    # The #22 rule, one layer up: the dispatcher imports `retrieval`, which builds its retrievers
    # lazily, so a run that answers from a tool must not pay for chromadb -- and the graph must
    # import in a checkout with no index and no key.
    code = "import agent.graph, sys; print('openai' in sys.modules, 'chromadb' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False False"


# --- the three exits ------------------------------------------------------------------------------


def test_a_grounded_run_exits_by_answer(cfg):
    state = ag.run("How many credits is CS 374?", cfg=cfg, nodes=ANSWERING)
    assert state["exit"] == "answer"
    assert state["answer"] == "CS 374 is 4 hours [CS 374]."
    assert state["docs"] == [LOOKUP]


def test_an_out_of_scope_query_exits_before_any_evidence_is_gathered(cfg):
    # The real dispatcher raises if it is reached with this label, which is the assertion:
    # nothing downstream of the router ran.
    state = ag.run(
        "What's the weather in Champaign?",
        cfg=cfg,
        nodes={
            "route_intent": router("out_of_scope", "out_of_scope"),
        },
    )
    assert state["exit"] == "out_of_scope"
    assert state["docs"] == []
    assert state["evidence_ok"] is None
    assert state["answer"] == exits.OUT_OF_SCOPE_MESSAGE.format(term=TERM)


def test_the_gate_refusing_exits_by_hand_off(cfg):
    state = ag.run(
        "What are the prerequisites?",
        cfg=cfg,
        nodes={
            "route_intent": router("prereq_chain"),
            "dispatch": dispatcher([], tool_error="no_course_code"),
            "check_evidence": gate(False),
        },
    )
    assert state["exit"] == "refuse_handoff"
    assert "What you asked: What are the prerequisites?" in state["answer"]
    assert "does not name a course" in state["answer"]


def test_the_answer_nodes_citation_check_lands_on_the_same_hand_off_exit(cfg):
    state = ag.run(
        "Who teaches CS 374?",
        cfg=cfg,
        nodes={
            **ANSWERING,
            "answer": rejector(["CS 999"]),
        },
    )
    assert state["exit"] == "refuse_handoff"
    assert state["signals"]["citations_unknown"] == ["CS 999"]
    assert "cited CS 999" in state["answer"]


def test_which_exit_a_run_took_is_readable_off_the_final_state(cfg):
    # #39 counts exits; it reads this field and never infers from which keys are set.
    taken = {
        ag.run("q", cfg=cfg, nodes=nodes)["exit"]
        for nodes in (
            ANSWERING,
            {"route_intent": router("out_of_scope")},
            {**ANSWERING, "check_evidence": gate(False)},
        )
    }
    assert taken == {"answer", "out_of_scope", "refuse_handoff"}


def test_every_nodes_signals_survive_to_the_end_of_the_run(cfg):
    # Four nodes each writing one key: without the reducer on `signals`, only the last would be
    # left in the final state.
    state = ag.run(
        "Which intro ML course should I take?",
        cfg=cfg,
        nodes={
            "route_intent": router("compare_recommend", "compare_recommend"),
            "dispatch": dispatcher([CHUNK], mode="bm25"),
            "check_evidence": gate(True, score=29.0337, threshold=10.0),
            "answer": writer("CS 446 [2026-sp:CS-374].", ["2026-sp:CS-374"]),
        },
    )
    assert state["signals"] == {
        "intent_raw": "compare_recommend",
        "intent_unparsed": False,
        "retrieval_mode": "bm25",
        "tool_error": None,
        "retrieval_top_score": 29.0337,
        "evidence_threshold": 10.0,
        # Not written: the fake gate is Day 2's shape, and no confidence gate is in this graph.
        "evidence_method": None,
        "conformal": None,
        "citations_verified": ["2026-sp:CS-374"],
        "citations_unknown": None,
        "confidence": None,
        "confidence_raw": None,
    }


# --- no node reaches into another's state ---------------------------------------------------------


def test_a_dispatch_node_that_writes_the_intent_is_refused(cfg):
    with pytest.raises(ValueError, match="dispatch wrote 'intent', which belongs to route_intent"):
        ag.run(
            "q",
            cfg=cfg,
            nodes={
                "route_intent": router("course_fact"),
                "dispatch": lambda state: {"intent": "prereq_chain", "docs": []},
            },
        )


def test_a_router_that_writes_the_evidence_verdict_is_refused(cfg):
    with pytest.raises(ValueError, match="route_intent wrote 'evidence_ok'"):
        ag.run(
            "q",
            cfg=cfg,
            nodes={
                "route_intent": lambda state: {"intent": "course_fact", "evidence_ok": True},
            },
        )


def test_a_node_that_invents_a_signal_is_refused(cfg):
    with pytest.raises(ValueError, match="undeclared signal"):
        ag.run(
            "q",
            cfg=cfg,
            nodes={
                "route_intent": lambda state: {"intent": "course_fact", "signals": {"vibes": 1}},
            },
        )


def test_a_dispatch_node_that_passes_a_raw_hit_through_is_refused(cfg):
    from retrieval.types import Hit

    hit = Hit(doc_id="2026-sp:CS-374", score=29.0, rank=1, text="CS 374", metadata={})
    with pytest.raises(TypeError, match="every block must be an agent.state.Evidence"):
        ag.run(
            "q",
            cfg=cfg,
            nodes={
                "route_intent": router("course_fact"),
                "dispatch": lambda state: {"docs": [hit]},
            },
        )


def test_guard_names_the_node_it_wraps(cfg):
    guarded = ag.guard("dispatch", dispatcher([LOOKUP]))
    assert guarded.__name__ == "dispatch"
    assert guarded(initial_state("q"))["docs"] == [LOOKUP]


# --- routing --------------------------------------------------------------------------------------


def test_a_router_that_set_no_intent_stops_the_run(cfg):
    with pytest.raises(ValueError, match="did not set `intent`"):
        ag.run("q", cfg=cfg, nodes={"route_intent": lambda state: {}})


def test_a_gate_that_decided_nothing_stops_the_run(cfg):
    with pytest.raises(ValueError, match="did not set `evidence_ok`"):
        ag.run(
            "q",
            cfg=cfg,
            nodes={
                "route_intent": router("course_fact"),
                "dispatch": dispatcher([LOOKUP]),
                "check_evidence": lambda state: {},
            },
        )


def test_an_answer_node_that_claims_no_exit_stops_the_run(cfg):
    with pytest.raises(ValueError, match="the answer node did not set `exit`"):
        ag.run("q", cfg=cfg, nodes={**ANSWERING, "answer": lambda state: {"answer": "..."}})


def test_the_answer_node_cannot_declare_a_query_out_of_scope(cfg):
    # By the time evidence has been gathered, "out of scope" is not a conclusion available -- that
    # exit belongs to the router, before any retrieval.
    with pytest.raises(ValueError, match="that exit is decided by the router"):
        ag.run(
            "q",
            cfg=cfg,
            nodes={
                **ANSWERING,
                "answer": lambda state: {"answer": "no", "exit": "out_of_scope"},
            },
        )


def test_the_router_sends_only_out_of_scope_to_its_own_exit():
    for intent in ("course_fact", "prereq_chain", "degree_requirement", "compare_recommend"):
        assert ag.route_after_intent({"intent": intent}) == "dispatch"
    assert ag.route_after_intent({"intent": "out_of_scope"}) == "out_of_scope"


def test_route_after_answer_ends_the_run_on_a_kept_answer():
    assert ag.route_after_answer({"exit": "answer"}) == END
    assert ag.route_after_answer({"exit": "refuse_handoff"}) == "refuse_handoff"


# --- no node is a stand-in any more ---------------------------------------------------------------


def test_every_node_the_graph_builds_is_the_real_one(cfg):
    # #38 was the last stub. Each built node is named after the node it is, which is what the
    # stub's `stub_answer` name used to give away -- and what a future stand-in would again.
    built = ag.build_nodes(cfg)
    assert {name: node.__name__ for name, node in built.items()} == {
        name: name for name in ag.NODE_BUILDERS if name != "confidence_gate"
    }


def test_the_run_with_no_overrides_at_all_is_a_whole_run(cfg, monkeypatch):
    # Not invoked -- the two model calls would need credits -- but built: nothing in the six is a
    # placeholder that would raise on the way past, which is what #39's CLI depends on.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert len(ag.build_nodes(cfg)) == 6


# --- building the nodes ---------------------------------------------------------------------------


def test_the_registry_holds_the_six_nodes_and_the_conditional_confidence_gate(cfg):
    assert set(ag.NODE_BUILDERS) == {
        "route_intent",
        "dispatch",
        "check_evidence",
        "answer",
        "confidence_gate",
        "out_of_scope",
        "refuse_handoff",
    }
    assert set(ag.CONDITIONAL) == {"confidence_gate"}
    # A top_score config builds six, and has no seventh to override.
    assert "confidence_gate" not in ag.node_names(cfg)


def test_an_override_replaces_one_node_and_leaves_the_other_five(cfg):
    fake = router("course_fact")
    built = ag.build_nodes(cfg, {"route_intent": fake})
    assert built["route_intent"] is fake
    assert len(built) == 6
    assert built["check_evidence"].__name__ == "check_evidence"
    assert built["answer"].__name__ == "answer"


def test_an_overridden_node_is_never_built(cfg, monkeypatch):
    # The router and the answer node validate the `llm` config section when they are built, and
    # each opens a chat client on its first query. Building one only to discard it would make an
    # injected fake depend on a config section it does not use.
    def explode(_cfg):
        raise AssertionError("the real builder ran for a node that was overridden")

    monkeypatch.setitem(ag.NODE_BUILDERS, "route_intent", explode)
    built = ag.build_nodes(cfg, {"route_intent": router("course_fact")})
    assert built["route_intent"].__name__ == "node"


def test_a_typo_in_an_override_is_refused_rather_than_ignored(cfg):
    # Silently ignoring it would be a fake that never ran and a test that passed anyway.
    with pytest.raises(KeyError, match="no node\\(s\\) named 'rout_intent'"):
        ag.build_nodes(cfg, {"rout_intent": router("course_fact")})


# --- the cache ------------------------------------------------------------------------------------


def test_the_graph_is_compiled_once_per_config(cfg):
    assert ag.get_graph(cfg) is ag.get_graph(cfg)
    other = Config({"data": {"term": "2026-fa"}, "llm": LLM, "agent": AGENT, "evidence": EVIDENCE})
    assert ag.get_graph(other) is not ag.get_graph(cfg)


def test_clear_cache_drops_it(cfg):
    first = ag.get_graph(cfg)
    ag.clear_cache()
    assert ag.get_graph(cfg) is not first


def test_a_graph_built_with_injected_nodes_is_never_cached(cfg):
    ag.run("q", cfg=cfg, nodes=ANSWERING)
    assert ag._GRAPHS == {}


# --- run() ----------------------------------------------------------------------------------------


@pytest.mark.parametrize("query", ["", "   "])
def test_run_refuses_an_empty_query_before_it_builds_anything(query, cfg):
    with pytest.raises(ValueError, match="query is empty"):
        ag.run(query, cfg=cfg)


def test_run_accepts_a_plain_mapping_as_config():
    state = ag.run("q", cfg={"data": {"term": "2026-sp"}, "llm": LLM}, nodes=ANSWERING)
    assert state["exit"] == "answer"


def test_run_defaults_to_the_live_config():
    # The one place the real config.yaml is read here: `data.term` is what the exit messages name.
    state = ag.run(
        "What's the weather in Champaign?", nodes={"route_intent": router("out_of_scope")}
    )
    assert state["exit"] == "out_of_scope"
    assert "2026-sp" in state["answer"]


def test_a_run_that_ends_with_no_exit_is_refused():
    with pytest.raises(ValueError, match="finished with no `exit` set"):
        ag.check_final({"answer": "CS 374 is 4 hours"})


def test_an_exit_with_nothing_to_show_is_refused():
    with pytest.raises(ValueError, match="no answer text to show"):
        ag.check_final({"exit": "answer", "answer": None})


# --- the two fixed exits --------------------------------------------------------------------------


def test_the_out_of_scope_message_is_fixed_and_names_the_catalog(cfg):
    message = exits.out_of_scope_node(cfg)(initial_state("q"))["answer"]
    assert TERM in message
    assert "outside what I can answer" in message


def test_the_hand_off_message_says_what_was_asked_and_where_it_stopped(cfg):
    state = initial_state("What are the prerequisites?")
    state["signals"]["tool_error"] = "no_course_code"
    answer = exits.refuse_handoff_node(cfg)(state)["answer"]
    assert "What you asked: What are the prerequisites?" in answer
    assert "Where I stopped:" in answer and "What would get it answered:" in answer
    assert "advisor" in answer


@pytest.mark.parametrize(
    ("signals", "docs", "expected"),
    [
        ({"tool_error": "no_course_code"}, [], "does not name a course"),
        ({"tool_error": "corpus_unreadable"}, [], "reported 'corpus_unreadable'"),
        ({"citations_unknown": ["CS 999"]}, [CHUNK], "cited CS 999"),
        # `[]` is "the answer node ran and its draft cited nothing"; `None` is "it never ran".
        ({"citations_verified": []}, [CHUNK], "pointed at none of the evidence"),
        ({"citations_verified": ["CS 374"]}, [CHUNK], "was not enough to answer from"),
        ({"retrieval_top_score": 3.1, "evidence_threshold": 10.0}, [CHUNK], "scored 3.1000"),
        ({}, [], "nothing in the catalog came back"),
        ({}, [CHUNK], "was not enough to answer from"),
    ],
)
def test_the_hand_off_reason_is_read_off_the_signals(signals, docs, expected):
    state = initial_state("q")
    state["signals"].update(signals)
    state["docs"] = docs
    reason, remedy = exits.handoff_reason(state, TERM)
    assert expected in reason
    assert remedy


def test_a_named_tool_error_outranks_a_score(cfg):
    # Ordered by how specific the run can be about itself, not by which node wrote last.
    state = initial_state("q")
    state["signals"].update(
        {"tool_error": "no_course_code", "retrieval_top_score": 1.0, "evidence_threshold": 10.0}
    )
    reason, _ = exits.handoff_reason(state, TERM)
    assert "does not name a course" in reason


def test_the_score_sentence_quotes_both_numbers_and_the_source():
    state = initial_state("q")
    state["signals"].update(
        {"retrieval_top_score": 0.0164, "evidence_threshold": 0.025, "retrieval_mode": "hybrid"}
    )
    reason, _ = exits.handoff_reason(state, TERM)
    assert "hybrid" in reason and "0.0164" in reason and "0.025" in reason


def test_neither_exit_message_needs_a_model_or_a_key(cfg, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    state = initial_state("q")
    assert exits.out_of_scope_node(cfg)(state)["exit"] == "out_of_scope"
    assert exits.refuse_handoff_node(cfg)(state)["exit"] == "refuse_handoff"
