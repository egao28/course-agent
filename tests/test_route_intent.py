"""Tests for agent/nodes/route_intent.py: the five labels, and what happens to everything else.

No network, no API key and no credits anywhere here -- the point of #28's injectable client. Two
fakes do all the work: :class:`CannedChat` returns one reply whatever it is asked, and
:class:`KeyedChat` is keyed on the query, which is how the ten handwritten queries of #39 are
classified without a model. The routing *logic* -- prompt assembly, label validation, the
out-of-scope fallback, the two signals -- is fully exercised by them; what a real model actually
replies to those ten is the live run at the end of Day 2, and nothing here pretends to have
measured it.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from agent import graph as ag
from agent.nodes import exits
from agent.nodes.route_intent import check_forced, parse_intent, route_intent_node
from agent.prompts import ROUTER_EXAMPLES, ROUTER_SYSTEM_PROMPT
from agent.state import INTENTS, Evidence, check_update, initial_state
from config import Config

TERM = "2026-sp"
LLM = {"provider": "openai", "model": "gpt-4o-mini", "temperature": 0, "max_tokens": 800}
#: What #36's dispatcher validates when the graph builds it; nothing here retrieves.
AGENT = {"retrieval_mode": "bm25"}
#: What #37's gate validates when the graph builds it; nothing here reaches it either.
EVIDENCE = {"method": "top_score", "thresholds": {"bm25": 10.0, "dense": 0.4, "hybrid": 0.025}}


@pytest.fixture
def cfg() -> Config:
    return Config({"data": {"term": TERM}, "llm": LLM, "agent": AGENT, "evidence": EVIDENCE})


@pytest.fixture(autouse=True)
def no_api_key(monkeypatch) -> None:
    """Nothing in this file may need a key; the router has to be testable without one."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


@pytest.fixture(autouse=True)
def fresh_cache():
    ag.clear_cache()
    yield
    ag.clear_cache()


class CannedChat:
    """One reply, whatever it is asked -- and a record of what it was asked."""

    model = "canned"

    def __init__(self, reply: str) -> None:
        self.reply, self.calls = reply, []

    def complete(self, prompt: str, *, system: str | None = None) -> str:
        self.calls.append((system, prompt))
        return self.reply


class KeyedChat:
    """A reply per query, looked up out of the question the prompt carries.

    An unknown query is an :class:`AssertionError` rather than a default label: a reworded test
    query should fail loudly, not quietly classify as whatever the fake happened to fall back to.
    """

    model = "keyed"

    def __init__(self, replies: dict[str, str]) -> None:
        self.replies, self.calls = replies, []

    def complete(self, prompt: str, *, system: str | None = None) -> str:
        self.calls.append((system, prompt))
        for query, reply in self.replies.items():
            if query in prompt:
                return reply
        raise AssertionError(f"KeyedChat was asked about an unregistered query: {prompt!r}")


class ExplodingChat:
    """What the live account does today: 429 insufficient_quota, from the first call on."""

    model = "exploding"

    def complete(self, prompt: str, *, system: str | None = None) -> str:
        raise RuntimeError("Error code: 429 - insufficient_quota")


def router(cfg: Config, **kwargs):
    """The node under test, with a fake client unless the test asked for something else."""
    kwargs.setdefault("chat", CannedChat("course_fact"))
    return route_intent_node(cfg, **kwargs)


def classify(cfg: Config, query: str, reply: str) -> dict:
    """One query through the real node against a canned completion; the update it returned."""
    node = route_intent_node(cfg, chat=CannedChat(reply))
    return dict(node(initial_state(query)))


# --- the ten handwritten queries ------------------------------------------------------------------

#: #39's handwritten set, with the label each query must get. The queries are copied into
#: ``eval/datasets/day2_queries.jsonl`` by #39, which adds the expected *exit* for each -- that
#: column depends on #36's dispatch and #37's gate and is not this node's business. Five intents
#: are covered, and #39's three exit groups are: five answerable, four that should reach a person,
#: one plainly out of scope.
HANDWRITTEN: tuple[tuple[str, str], ...] = (
    # answerable
    ("How many credits is MATH 241?", "course_fact"),
    ("What do I need before CS 374?", "prereq_chain"),
    ("Which Gen Eds do I still need if I have taken RHET 105 and MATH 241?", "degree_requirement"),
    ("Which intro machine learning course should I take?", "compare_recommend"),
    ("What does the algorithms course cover?", "course_fact"),
    # in scope, but the run should end up handing it to a person
    ("What is left for the Statistics major if I have taken STAT 200?", "degree_requirement"),
    ("What are the prerequisites?", "prereq_chain"),
    ("How many credits is CS 999?", "course_fact"),
    ("Is there a course about quantum origami?", "compare_recommend"),
    # out of scope
    ("Who is the head football coach at Illinois?", "out_of_scope"),
)


def test_the_handwritten_set_covers_all_five_labels_and_is_ten_queries():
    assert len(HANDWRITTEN) == 10
    assert {label for _, label in HANDWRITTEN} == set(INTENTS)


def test_no_handwritten_query_is_one_of_the_prompts_own_examples():
    # A test set drawn from the few-shot examples measures copying, not classification.
    examples = {example for pair in ROUTER_EXAMPLES.values() for example in pair}
    assert {query for query, _ in HANDWRITTEN}.isdisjoint(examples)


@pytest.mark.parametrize(("query", "expected"), HANDWRITTEN)
def test_each_handwritten_query_routes_to_its_label(cfg, query, expected):
    chat = KeyedChat({query: label for query, label in HANDWRITTEN})
    update = route_intent_node(cfg, chat=chat)(initial_state(query))
    assert update["intent"] == expected
    assert update["signals"] == {"intent_raw": expected, "intent_unparsed": False}


def test_the_whole_set_runs_through_one_node_and_one_client(cfg):
    # #39 runs ten queries in one process: one node, one client, ten calls and no state carried
    # between them.
    chat = KeyedChat({query: label for query, label in HANDWRITTEN})
    node = route_intent_node(cfg, chat=chat)
    labels = [node(initial_state(query))["intent"] for query, _ in HANDWRITTEN]
    assert labels == [label for _, label in HANDWRITTEN]
    assert len(chat.calls) == 10


# --- the prompt the node sends --------------------------------------------------------------------


def test_the_node_sends_the_module_prompt_as_the_system_message(cfg):
    chat = CannedChat("course_fact")
    route_intent_node(cfg, chat=chat)(initial_state("How many credits is CS 374?"))
    system, prompt = chat.calls[0]
    assert system == ROUTER_SYSTEM_PROMPT
    assert prompt == "Question: How many credits is CS 374?\nLabel:"


def test_the_query_reaches_the_model_verbatim_and_is_not_rewritten(cfg):
    query = "  How many credits is CS 374??  "
    chat = CannedChat("course_fact")
    state = initial_state(query)
    route_intent_node(cfg, chat=chat)(state)
    assert "How many credits is CS 374??" in chat.calls[0][1]
    assert state["query"] == query


# --- validation: anything but the five labels is out of scope -------------------------------------


@pytest.mark.parametrize("label", INTENTS)
def test_a_clean_label_is_taken_as_it_is(cfg, label):
    update = classify(cfg, "q", label)
    assert update["intent"] == label
    assert update["signals"] == {"intent_raw": label, "intent_unparsed": False}


def test_a_real_out_of_scope_classification_is_not_marked_unparsed(cfg):
    # The two out-of-scope rows Day 5 has to be able to tell apart: this one is a decision.
    update = classify(cfg, "Who is the head football coach at Illinois?", "out_of_scope")
    assert update["intent"] == "out_of_scope"
    assert update["signals"]["intent_unparsed"] is False


def test_a_garbage_completion_lands_on_out_of_scope_and_keeps_what_came_back(cfg):
    # #35's acceptance, verbatim.
    raw = "I think this is about prerequisites!"
    update = classify(cfg, "What do I need before CS 374?", raw)
    assert update["intent"] == "out_of_scope"
    assert update["signals"] == {"intent_raw": raw, "intent_unparsed": True}


@pytest.mark.parametrize(
    "raw",
    [
        "course_fact.",  # the format ignored by one character
        "Course_fact",  # the format ignored by one bit
        "`course_fact`",
        '"course_fact"',
        "Label: course_fact",
        "course_fact (the question asks for one attribute of one course)",
        "prereq_chain or course_fact",
        "waitlist_status",  # an invented sixth category
        "",
        "   ",
    ],
)
def test_every_near_miss_is_refused_and_recorded(cfg, raw):
    # Strict on purpose: each of these is an in-scope query refused, and the cost is paid to keep
    # `intent_unparsed` a count of how often the model ignored the format. Loosening the parser is
    # a Day 5 decision with this number in hand, not a guess made now.
    update = classify(cfg, "How many credits is CS 374?", raw)
    assert update["intent"] == "out_of_scope"
    assert update["signals"] == {"intent_raw": raw, "intent_unparsed": True}


def test_a_client_that_returns_no_string_still_leaves_intent_raw_a_string(cfg):
    # The invariant the forced path depends on: intent_raw is None only when no model was asked.
    update = classify(cfg, "How many credits is CS 374?", None)
    assert update["intent"] == "out_of_scope"
    assert update["signals"] == {"intent_raw": "None", "intent_unparsed": True}


def test_parse_intent_accepts_the_five_and_nothing_else():
    assert [parse_intent(label) for label in INTENTS] == list(INTENTS)
    assert parse_intent(" course_fact\n") == "course_fact"
    for other in ("course_fact.", "COURSE_FACT", "sixth", "", None, 3, ["course_fact"]):
        assert parse_intent(other) is None


# --- an API failure is not a classification -------------------------------------------------------


def test_a_failing_call_propagates_rather_than_becoming_a_refusal(cfg):
    # A 429 turned into out_of_scope would reach #39's report as a routing decision, and would be
    # counted in intent_unparsed alongside real format violations.
    node = route_intent_node(cfg, chat=ExplodingChat())
    with pytest.raises(RuntimeError, match="insufficient_quota"):
        node(initial_state("How many credits is CS 374?"))


# --- the forced label (#39's --intent) ------------------------------------------------------------


@pytest.mark.parametrize("label", INTENTS)
def test_a_forced_label_asks_no_model_at_all(cfg, label):
    node = route_intent_node(cfg, forced=label)
    update = node(initial_state("How many credits is CS 374?"))
    assert update["intent"] == label
    # intent_raw is None exactly when no model was asked, which is what keeps a --intent run
    # legible in #39's report: an empty raw column is a run with no model behind its label.
    assert update["signals"] == {"intent_raw": None, "intent_unparsed": False}


def test_a_forced_run_needs_neither_a_key_nor_an_llm_section():
    node = route_intent_node(Config({"data": {"term": TERM}}), forced="prereq_chain")
    assert node(initial_state("What do I need before CS 374?"))["intent"] == "prereq_chain"


def test_forcing_a_label_that_is_not_one_of_the_five_is_refused(cfg):
    with pytest.raises(ValueError, match="cannot force intent 'waitlist_status'"):
        route_intent_node(cfg, forced="waitlist_status")


def test_check_forced_names_the_five(cfg):
    assert check_forced("course_fact") == "course_fact"
    with pytest.raises(ValueError, match="the five labels are course_fact, prereq_chain"):
        check_forced("Course_fact")


def test_a_chat_client_and_a_forced_label_together_are_refused(cfg):
    # One of the two would have been ignored, and which one is not obvious from the call.
    with pytest.raises(ValueError, match="both `chat` and `forced`"):
        route_intent_node(cfg, chat=CannedChat("course_fact"), forced="course_fact")


# --- building the node ----------------------------------------------------------------------------


def test_building_the_node_reads_no_key_and_opens_no_client(cfg, monkeypatch):
    import llm

    def explode(*args, **kwargs):
        raise AssertionError("a chat client was built before a query reached the node")

    monkeypatch.setattr(llm, "make_chat", explode)
    assert route_intent_node(cfg).__name__ == "route_intent"


def test_the_client_is_built_on_the_first_query_and_reused_after_it(cfg, monkeypatch):
    import llm

    built = []

    def make_chat(config, client=None):
        built.append(config)
        return CannedChat("course_fact")

    monkeypatch.setattr(llm, "make_chat", make_chat)
    node = route_intent_node(cfg)
    assert built == []
    for _ in range(3):
        assert node(initial_state("How many credits is CS 374?"))["intent"] == "course_fact"
    assert len(built) == 1


def test_a_config_that_names_a_provider_with_no_client_fails_at_build_time(cfg):
    # Before the query rather than after it: #39 runs ten of them.
    broken = Config({"data": {"term": TERM}, "llm": {**LLM, "provider": "cohere"}})
    with pytest.raises(ValueError, match="unknown llm.provider"):
        route_intent_node(broken)


def test_a_missing_llm_section_is_a_sentence_not_a_key_error(cfg):
    with pytest.raises(ValueError, match="no `llm` section"):
        route_intent_node(Config({"data": {"term": TERM}}))


def test_an_injected_client_needs_no_llm_section(cfg):
    # The section is what make_chat reads; a fake replaces make_chat, so demanding it would make
    # every test in this repo carry a config it never uses.
    node = route_intent_node(Config({"data": {"term": TERM}}), chat=CannedChat("course_fact"))
    assert node(initial_state("q"))["intent"] == "course_fact"


def test_importing_the_router_pulls_in_neither_openai_nor_chromadb():
    # The #22 rule: `llm` imports `embedding`, which imports `openai` and reads OPENAI_API_KEY, so
    # this module defers it -- a forced --intent run must not need either.
    code = (
        "import agent.nodes.route_intent, sys; "
        "print('openai' in sys.modules, 'chromadb' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False False"


# --- the node inside the graph --------------------------------------------------------------------


def test_the_node_writes_only_its_own_key_and_the_two_router_signals(cfg):
    update = classify(cfg, "How many credits is CS 374?", "course_fact")
    assert set(update) == {"intent", "signals"}
    assert check_update("route_intent", update) == update


def test_an_out_of_scope_classification_ends_the_run_at_its_exit(cfg):
    chat = CannedChat("out_of_scope")
    state = ag.run(
        "Who is the head football coach at Illinois?",
        cfg=cfg,
        nodes={"route_intent": route_intent_node(cfg, chat=chat)},
    )
    assert state["exit"] == "out_of_scope"
    assert state["answer"] == exits.OUT_OF_SCOPE_MESSAGE.format(term=TERM)
    assert state["signals"]["intent_unparsed"] is False
    assert state["docs"] == []


def test_an_unparseable_classification_ends_the_run_at_the_same_exit(cfg):
    # The refusal a garbage completion costs, end to end -- and the run can still say it happened.
    state = ag.run(
        "What do I need before CS 374?",
        cfg=cfg,
        nodes={"route_intent": route_intent_node(cfg, chat=CannedChat("prerequisites, I think"))},
    )
    assert state["exit"] == "out_of_scope"
    assert state["signals"] == {
        "intent_raw": "prerequisites, I think",
        "intent_unparsed": True,
        "retrieval_mode": None,
        "tool_error": None,
        "retrieval_top_score": None,
        "evidence_threshold": None,
        "citations_unknown": None,
    }


def test_an_in_scope_label_reaches_the_dispatch_node(cfg):
    doc = Evidence(
        doc_id="CS 374", text="CS 374 - Algorithms. 4 hours.", score=None, source="get_course"
    )
    state = ag.run(
        "How many credits is CS 374?",
        cfg=cfg,
        nodes={
            "route_intent": route_intent_node(cfg, chat=CannedChat("course_fact")),
            "dispatch": lambda s: {"docs": [doc]},
            "check_evidence": lambda s: {"evidence_ok": True},
            "answer": lambda s: {"answer": "CS 374 is 4 hours [CS 374].", "exit": "answer"},
        },
    )
    assert (state["intent"], state["exit"]) == ("course_fact", "answer")
    assert state["signals"]["intent_raw"] == "course_fact"


def test_a_forced_label_drives_the_whole_chain_with_no_model(cfg):
    # What #39's `--intent prereq_chain` does: the router is real, the model is never asked.
    state = ag.run(
        "What do I need before CS 374?",
        cfg=cfg,
        nodes={
            "route_intent": route_intent_node(cfg, forced="prereq_chain"),
            "dispatch": lambda s: {"docs": []},
            "check_evidence": lambda s: {"evidence_ok": False},
        },
    )
    assert state["intent"] == "prereq_chain"
    assert state["exit"] == "refuse_handoff"
    assert state["signals"]["intent_raw"] is None


def test_the_graph_builds_the_real_router_for_the_live_config():
    # The registry entry, not a stub: the one place here that reads the shipped config.yaml.
    built = ag.build_nodes(ag.load_config())
    assert built["route_intent"].__name__ == "route_intent"
