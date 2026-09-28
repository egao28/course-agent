"""Tests for agent/nodes/answer.py and its prompt: the evidence in, and what the check throws away.

No network, no API key and no credits, the same as ``tests/test_route_intent.py``: the client is
injected, so the prompt assembly, the citation parse and both downgrades are exercised for real
while the account has none. What a *real* model does with the prompt -- constraint 1, that it stays
inside the evidence -- is the one thing a fake cannot show, and it is #39's live run; nothing here
pretends to have measured it.

What is worth testing here is the check, not the wording. Three drafts have to be told apart -- one
that cites a block this run retrieved, one that cites something it never had, and one that cites
nothing at all -- and the last two must both leave through the hand-off exit with a signal saying
which happened. The last section runs the real dispatch node over the live catalog, so the blocks
in the prompt are the ones a person's question would actually produce.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from agent import graph as ag
from agent import prompts
from agent.nodes import exits
from agent.nodes.answer import (
    Citations,
    answer_node,
    citation_spans,
    normalize_id,
    split_citation,
    verify_citations,
)
from agent.nodes.dispatch import dispatch_node
from agent.prompts import ANSWER_SYSTEM_PROMPT, answer_user_prompt, render_evidence
from agent.state import Evidence, check_update, initial_state
from config import Config, load_config

TERM = "2026-sp"
LLM = {"provider": "openai", "model": "gpt-4o-mini", "temperature": 0, "max_tokens": 800}
AGENT = {"retrieval_mode": "bm25"}
EVIDENCE = {"method": "top_score", "thresholds": {"bm25": 10.0, "dense": 0.4, "hybrid": 0.025}}

#: A tool row's block: an exact lookup, no score, the id a person would recognize.
LOOKUP = Evidence(
    doc_id="CS 374",
    text="CS 374 — Algorithms\nCredits: 4 hours\nTerm: 2026-sp\nSections: 12",
    score=None,
    source="get_course",
)
#: A retrieval row's block: the chunk id, and a score that must not reach the prompt.
CHUNK = Evidence(
    doc_id="2026-sp:CS-374",
    text="CS 374 — Algorithms\nCredits: 4 hours",
    score=29.0337,
    source="bm25",
)
#: The requirement row's block. Its ``doc_id`` has a comma in it, which is why a bracket is
#: matched whole before it is split.
PROGRAM = Evidence(
    doc_id="Computer Science, BS",
    text="Computer Science, BS — requirement audit\nStill needed (9):",
    score=None,
    source="check_requirement",
)


@pytest.fixture
def cfg() -> Config:
    return Config({"data": {"term": TERM}, "llm": LLM, "agent": AGENT, "evidence": EVIDENCE})


@pytest.fixture(autouse=True)
def no_api_key(monkeypatch) -> None:
    """Nothing in this file may need a key; the answer node has to be testable without one."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


@pytest.fixture(autouse=True)
def fresh_cache():
    ag.clear_cache()
    yield
    ag.clear_cache()


# --- fakes ----------------------------------------------------------------------------------------


class CannedChat:
    """One draft, whatever it is asked -- and a record of what it was asked."""

    model = "canned"

    def __init__(self, reply) -> None:
        self.reply, self.calls = reply, []

    def complete(self, prompt: str, *, system: str | None = None) -> str:
        self.calls.append((system, prompt))
        return self.reply


class ExplodingChat:
    """What the live account does today: 429 insufficient_quota, from the first call on."""

    model = "exploding"

    def complete(self, prompt: str, *, system: str | None = None) -> str:
        raise RuntimeError("Error code: 429 - insufficient_quota")


def write(cfg: Config, query: str, docs: list[Evidence], reply) -> dict:
    """One query through the real node against a canned draft; the update it returned."""
    state = initial_state(query)
    state["docs"] = list(docs)
    return dict(answer_node(cfg, chat=CannedChat(reply))(state))


def answered(text: str):
    """A fake answer node for the *other* tests' graphs -- used here only via the real one."""
    return lambda state: {"answer": text, "exit": "answer"}


# --- the prompt the node sends --------------------------------------------------------------------


def test_the_node_sends_the_module_prompt_as_the_system_message(cfg):
    chat = CannedChat("CS 374 is 4 hours [CS 374].")
    state = initial_state("How many credits is CS 374?")
    state["docs"] = [LOOKUP]
    answer_node(cfg, chat=chat)(state)
    system, prompt = chat.calls[0]
    assert system == ANSWER_SYSTEM_PROMPT
    assert prompt == answer_user_prompt("How many credits is CS 374?", [LOOKUP])


def test_the_user_message_carries_the_question_and_every_block(cfg):
    prompt = answer_user_prompt("How many credits is CS 374?", [LOOKUP, CHUNK])
    assert prompt.startswith("Question: How many credits is CS 374?\n\nEvidence:\n\n")
    assert prompt.endswith("\n\nAnswer:")
    for block in (LOOKUP, CHUNK):
        assert f"doc_id: {block.doc_id}" in prompt
        assert block.text in prompt


def test_each_block_is_numbered_and_headed_by_its_own_doc_id():
    rendered = render_evidence([LOOKUP, CHUNK, PROGRAM])
    assert rendered.startswith("Evidence 1 of 3\ndoc_id: CS 374\n---\n")
    assert "Evidence 2 of 3\ndoc_id: 2026-sp:CS-374\n---\n" in rendered
    assert "Evidence 3 of 3\ndoc_id: Computer Science, BS\n---\n" in rendered


def test_neither_the_score_nor_the_source_reaches_the_prompt():
    # A score is on a scale that means nothing outside its retriever, and a model shown one will
    # weigh the blocks by it. Whether there is enough here was already decided, by #37.
    rendered = render_evidence([CHUNK])
    assert "29.0337" not in rendered and "29.03" not in rendered
    assert "bm25" not in rendered


def test_the_query_reaches_the_model_verbatim_and_is_not_rewritten(cfg):
    query = "  How many credits is CS 374??  "
    chat = CannedChat("CS 374 is 4 hours [CS 374].")
    state = initial_state(query)
    state["docs"] = [LOOKUP]
    answer_node(cfg, chat=chat)(state)
    assert "How many credits is CS 374??" in chat.calls[0][1]
    assert state["query"] == query


def test_a_query_full_of_braces_is_not_a_format_string():
    assert "{evidence}" in answer_user_prompt("what is {evidence} {{0}}", [LOOKUP])


@pytest.mark.parametrize("query", ["", "   ", "\n"])
def test_an_empty_query_is_refused_rather_than_sent(query):
    with pytest.raises(ValueError, match="query is empty"):
        answer_user_prompt(query, [LOOKUP])


def test_rendering_no_evidence_at_all_is_refused():
    # The gate returns False for an empty `docs`, so this prompt cannot legitimately be built.
    with pytest.raises(ValueError, match="no evidence to answer from"):
        render_evidence([])


def test_the_prompts_example_block_is_rendered_by_the_same_function():
    example = render_evidence(
        [
            Evidence(
                doc_id=prompts.ANSWER_EXAMPLE_ID,
                text=prompts.ANSWER_EXAMPLE_TEXT,
                score=None,
                source="example",
            )
        ]
    )
    assert example in ANSWER_SYSTEM_PROMPT


def test_the_prompts_own_example_answers_pass_the_citation_check():
    # The strongest version of prompts.check_answer_examples: an example that would be discarded
    # teaches the model to write discardable answers, and the bug would surface as a hand-off rate.
    block = Evidence(
        doc_id=prompts.ANSWER_EXAMPLE_ID, text=prompts.ANSWER_EXAMPLE_TEXT, score=None, source="x"
    )
    for _question, answer in prompts.ANSWER_EXAMPLES:
        citations = verify_citations(answer, [block])
        assert citations.verified == [prompts.ANSWER_EXAMPLE_ID]
        assert citations.unknown == []


def test_an_example_answer_that_cites_nothing_is_refused_at_import(monkeypatch):
    monkeypatch.setattr(prompts, "ANSWER_EXAMPLES", (("How many credits?", "Three hours."),))
    with pytest.raises(ValueError, match="cite no block"):
        prompts.check_answer_examples()


def test_the_answer_prompt_is_a_constant_and_rendering_it_again_reproduces_it():
    assert prompts.render_answer_prompt() == ANSWER_SYSTEM_PROMPT


# --- reading the citations out of a draft ---------------------------------------------------------


def test_a_citation_of_a_block_this_run_retrieved_verifies():
    assert verify_citations("CS 374 is 4 hours [CS 374].", [LOOKUP]) == Citations(["CS 374"], [])


@pytest.mark.parametrize("cited", ["CS 374", "cs 374", "CS  374", " CS 374 ", "cS\t374"])
def test_case_and_spacing_are_not_fabrications(cited):
    # A doc_id is an identifier: `[cs 374]` names the block headed CS 374 and there is no
    # ambiguity about which one. What the id is *made of* is left alone -- see the next test.
    citations = verify_citations(f"It is 4 hours [{cited}].", [LOOKUP])
    assert citations == Citations(["CS 374"], [])


@pytest.mark.parametrize("cited", ["CS374", "CS-374", "374", "2026-sp:CS 374"])
def test_a_respelled_id_is_a_miss_and_is_kept_as_it_was_written(cited):
    # Not repaired: this module is not where a course code is respelled, and one rule that holds
    # for course codes, chunk ids and program labels alike is worth more than three that each fit
    # one. The miss is recorded so Day 5 can count it.
    citations = verify_citations(f"It is 4 hours [{cited}].", [LOOKUP])
    assert citations == Citations([], [cited])


def test_an_invented_id_is_recorded_verbatim():
    citations = verify_citations("CS 999 is 3 hours [CS 999].", [LOOKUP])
    assert citations == Citations([], ["CS 999"])


def test_a_chunk_id_cited_as_the_course_code_is_not_resolved_by_containment():
    # The failure this strictness is for: an answer about a course the chunk is not about would
    # pass any "the id appears somewhere in the evidence" rule, with a reassuring bracket after it.
    citations = verify_citations("CS 374 is 4 hours [CS 374].", [CHUNK])
    assert citations == Citations([], ["CS 374"])
    assert verify_citations("[2026-sp:CS-374]", [CHUNK]).verified == ["2026-sp:CS-374"]


def test_a_block_number_is_not_a_citation():
    # The prompt says to cite the id and not the number. Resolving `[1]` to the first block would
    # accept a format the prompt refused and hide how often the model used it.
    assert verify_citations("It is 4 hours [1].", [LOOKUP]) == Citations([], ["1"])


def test_two_ids_in_one_bracket_are_split():
    citations = verify_citations("Both [CS 374, 2026-sp:CS-374] say so.", [LOOKUP, CHUNK])
    assert citations == Citations(["CS 374", "2026-sp:CS-374"], [])


def test_a_comma_shaped_doc_id_stays_one_citation():
    # `Computer Science, BS` is a real doc_id from the requirement row: splitting first would turn
    # the one citation that row can carry into two fabrications.
    citations = verify_citations("Nine groups are left [Computer Science, BS].", [PROGRAM])
    assert citations == Citations(["Computer Science, BS"], [])


def test_a_bracket_holding_one_real_id_and_one_invented_one_contributes_exactly_one_fabrication():
    citations = verify_citations("Both [CS 374; CS 999].", [LOOKUP])
    assert citations == Citations(["CS 374"], ["CS 999"])


def test_a_citation_repeated_four_times_is_one_id():
    draft = "A [CS 374]. B [CS 374]. C [cs 374]. D [CS 374]."
    assert verify_citations(draft, [LOOKUP]) == Citations(["CS 374"], [])


def test_the_ids_come_back_in_the_order_the_answer_cited_them():
    draft = "First [2026-sp:CS-374], then [CS 374], then [CS 999], then [CS 998]."
    citations = verify_citations(draft, [LOOKUP, CHUNK])
    assert citations.verified == ["2026-sp:CS-374", "CS 374"]
    assert citations.unknown == ["CS 999", "CS 998"]


def test_empty_brackets_name_no_id():
    assert citation_spans("It is 4 hours []. [  ]") == []
    assert verify_citations("It is 4 hours [].", [LOOKUP]) == Citations([], [])


def test_a_draft_with_no_brackets_cites_nothing():
    assert verify_citations("CS 374 is 4 hours.", [LOOKUP]) == Citations([], [])


def test_two_citations_in_one_sentence_are_two_spans():
    assert citation_spans("A [CS 374] and B [CS 225].") == ["CS 374", "CS 225"]


def test_a_doubled_bracket_still_reads_as_its_inner_id():
    assert verify_citations("It is 4 hours [[CS 374]].", [LOOKUP]).verified == ["CS 374"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [("CS 374", "cs 374"), ("  CS   374 ", "cs 374"), ("2026-sp:CS-374", "2026-sp:cs-374")],
)
def test_normalize_id_collapses_space_and_case_and_nothing_else(text, expected):
    assert normalize_id(text) == expected


@pytest.mark.parametrize(
    ("span", "expected"),
    [
        ("CS 374, CS 225", ["CS 374", "CS 225"]),
        ("CS 374; CS 225", ["CS 374", "CS 225"]),
        ("CS 374 and CS 225", ["CS 374 and CS 225"]),
        (",,", []),
    ],
)
def test_split_citation_splits_on_commas_and_semicolons_only(span, expected):
    assert split_citation(span) == expected


# --- the three things the node does with a draft --------------------------------------------------


def test_a_grounded_draft_is_kept_and_both_signals_are_written(cfg):
    update = write(cfg, "How many credits is CS 374?", [LOOKUP], "CS 374 is 4 hours [CS 374].")
    assert update == {
        "answer": "CS 374 is 4 hours [CS 374].",
        "exit": "answer",
        "signals": {"citations_verified": ["CS 374"], "citations_unknown": None},
    }


def test_a_fabricated_citation_discards_the_whole_draft_and_claims_the_hand_off_exit(cfg):
    draft = "CS 374 is 4 hours [CS 374]. It is taught by Prof. Nobody [CS 999]."
    update = write(cfg, "Who teaches CS 374?", [LOOKUP], draft)
    assert update["exit"] == "refuse_handoff"
    assert "answer" not in update
    assert update["signals"] == {
        "citations_verified": ["CS 374"],
        "citations_unknown": ["CS 999"],
    }


def test_the_discarded_draft_does_not_reach_the_state_in_any_field(cfg):
    draft = "Prof. Nobody teaches it [CS 999]."
    update = write(cfg, "Who teaches CS 374?", [LOOKUP], draft)
    assert draft not in repr(update)


def test_a_draft_that_cites_nothing_at_all_is_discarded_too(cfg):
    # Rule 4 of the prompt is what makes this fair: even "the catalog does not say" cites the
    # block it looked in, so an uncited draft is not a modest answer -- it is an untraceable one.
    update = write(cfg, "Who teaches CS 374?", [LOOKUP], "The catalog does not say who teaches it.")
    assert update["exit"] == "refuse_handoff"
    assert "answer" not in update
    assert update["signals"] == {"citations_verified": [], "citations_unknown": None}


def test_the_two_discards_are_distinguishable_afterwards(cfg):
    fabricated = write(cfg, "q", [LOOKUP], "It is 4 hours [CS 999].")["signals"]
    uncited = write(cfg, "q", [LOOKUP], "It is 4 hours.")["signals"]
    assert fabricated["citations_unknown"] == ["CS 999"]
    assert uncited["citations_unknown"] is None
    assert uncited["citations_verified"] == []


def test_an_empty_draft_is_an_uncited_one(cfg):
    # OpenAIChat.complete raises on an empty completion (#28), so this is a fake's shape -- but
    # the node must not write an empty answer either way: check_update refuses one.
    update = write(cfg, "q", [LOOKUP], "   ")
    assert update["exit"] == "refuse_handoff"
    assert "answer" not in update


def test_the_node_writes_only_the_keys_it_owns(cfg):
    update = write(cfg, "q", [LOOKUP], "It is 4 hours [CS 374].")
    assert check_update("answer", update) == update


def test_a_draft_over_several_blocks_records_every_id_it_used(cfg):
    draft = "IS 557 is the closest [2026-sp:CS-374], and CS 374 is 4 hours [CS 374]."
    update = write(cfg, "Which ML course?", [CHUNK, LOOKUP], draft)
    assert update["exit"] == "answer"
    assert update["signals"]["citations_verified"] == ["2026-sp:CS-374", "CS 374"]


def test_the_node_refuses_to_run_with_no_evidence(cfg):
    # The gate never passes an empty `docs`, so reaching this is a wiring bug and says so.
    with pytest.raises(ValueError, match="reached with no evidence"):
        answer_node(cfg, chat=CannedChat("[CS 374]"))(initial_state("q"))


def test_a_client_that_returns_something_other_than_a_string_is_an_error(cfg):
    # Raised rather than repr()'d: a repr carries brackets of its own, and the run would report a
    # client bug as a fabricated citation.
    with pytest.raises(TypeError, match="llm.Chat.complete returns the completion as a str"):
        write(cfg, "q", [LOOKUP], ["CS 374"])


def test_an_api_failure_propagates_rather_than_becoming_a_refusal(cfg):
    # #35's rule, one node later: a billing problem must not reach #39's report as a hand-off.
    state = initial_state("q")
    state["docs"] = [LOOKUP]
    with pytest.raises(RuntimeError, match="429"):
        answer_node(cfg, chat=ExplodingChat())(state)


# --- through the graph ----------------------------------------------------------------------------


def test_a_grounded_answer_exits_by_answer_end_to_end(cfg):
    state = ag.run(
        "How many credits is CS 374?",
        cfg=cfg,
        nodes={
            "route_intent": lambda s: {"intent": "course_fact"},
            "dispatch": lambda s: {"docs": [LOOKUP], "signals": {"retrieval_mode": None}},
            "answer": answer_node(cfg, chat=CannedChat("CS 374 is 4 hours [CS 374].")),
        },
    )
    assert state["exit"] == "answer"
    assert state["answer"] == "CS 374 is 4 hours [CS 374]."
    assert state["signals"]["citations_verified"] == ["CS 374"]


def test_a_fabricated_citation_reaches_the_person_as_the_hand_off_message(cfg):
    state = ag.run(
        "Who teaches CS 374?",
        cfg=cfg,
        nodes={
            "route_intent": lambda s: {"intent": "course_fact"},
            "dispatch": lambda s: {"docs": [LOOKUP]},
            "answer": answer_node(cfg, chat=CannedChat("Prof. Nobody [CS 999].")),
        },
    )
    assert state["exit"] == "refuse_handoff"
    assert state["signals"]["citations_unknown"] == ["CS 999"]
    assert "What you asked: Who teaches CS 374?" in state["answer"]
    assert "cited CS 999" in state["answer"]
    assert "advisor" in state["answer"]


def test_an_uncited_draft_reaches_the_person_with_its_own_sentence(cfg):
    state = ag.run(
        "Who teaches CS 374?",
        cfg=cfg,
        nodes={
            "route_intent": lambda s: {"intent": "course_fact"},
            "dispatch": lambda s: {"docs": [LOOKUP]},
            "answer": answer_node(cfg, chat=CannedChat("The catalog does not say.")),
        },
    )
    assert state["exit"] == "refuse_handoff"
    assert exits.UNCITED_REASON in state["answer"]
    assert exits.UNCITED_REMEDY in state["answer"]
    # Not the fabrication sentence: nothing was invented here, and the two must not read alike.
    assert exits.CITATION_REMEDY not in state["answer"]
    assert state["signals"]["citations_unknown"] is None


def test_a_fabrication_outranks_the_score_the_gate_recorded(cfg):
    # Both are in `signals` by the time the hand-off runs: the citation is the more specific thing
    # the run can say about itself, and handoff_reason is ordered that way.
    state = ag.run(
        "Which intro machine learning course should I take?",
        cfg=cfg,
        nodes={
            "route_intent": lambda s: {"intent": "compare_recommend"},
            "dispatch": lambda s: {
                "docs": [CHUNK],
                "signals": {"retrieval_mode": "bm25", "tool_error": None},
            },
            "answer": answer_node(cfg, chat=CannedChat("IS 557 [IS 557].")),
        },
    )
    assert state["exit"] == "refuse_handoff"
    assert state["signals"]["retrieval_top_score"] == 29.0337
    assert "cited IS 557" in state["answer"]
    assert "29.0337" not in state["answer"]


def test_every_signal_of_an_answered_run_survives_to_the_end(cfg):
    state = ag.run(
        "Which intro machine learning course should I take?",
        cfg=cfg,
        nodes={
            "route_intent": lambda s: {
                "intent": "compare_recommend",
                "signals": {"intent_raw": "compare_recommend", "intent_unparsed": False},
            },
            "dispatch": lambda s: {
                "docs": [CHUNK],
                "signals": {"retrieval_mode": "bm25", "tool_error": None},
            },
            "answer": answer_node(cfg, chat=CannedChat("It is 4 hours [2026-sp:CS-374].")),
        },
    )
    assert state["signals"] == {
        "intent_raw": "compare_recommend",
        "intent_unparsed": False,
        "retrieval_mode": "bm25",
        "tool_error": None,
        "retrieval_top_score": 29.0337,
        "evidence_threshold": 10.0,
        "evidence_method": "top_score",
        "conformal": None,
        "citations_verified": ["2026-sp:CS-374"],
        "citations_unknown": None,
        "confidence": None,
        "confidence_raw": None,
    }


def test_a_run_the_gate_refused_never_asks_the_model(cfg):
    chat = CannedChat("[CS 374]")
    state = ag.run(
        "What are the prerequisites?",
        cfg=cfg,
        nodes={
            "route_intent": lambda s: {"intent": "prereq_chain"},
            "dispatch": lambda s: {"docs": [], "signals": {"tool_error": "no_course_code"}},
            "answer": answer_node(cfg, chat=chat),
        },
    )
    assert state["exit"] == "refuse_handoff"
    assert chat.calls == []
    assert state["signals"]["citations_verified"] is None


# --- the client and the config --------------------------------------------------------------------


def test_the_llm_section_is_validated_when_the_node_is_built():
    bad = Config({"data": {"term": TERM}, "llm": {**LLM, "provider": "anthropic"}})
    with pytest.raises(ValueError, match="unknown llm.provider"):
        answer_node(bad)


def test_a_config_with_no_llm_section_is_refused_by_name():
    with pytest.raises(ValueError, match="config has no `llm` section"):
        answer_node(Config({"data": {"term": TERM}}))


def test_an_injected_client_skips_the_config_check_entirely():
    # The reason it can: `make_chat` is what reads the section, and a fake replaces it.
    node = answer_node(Config({"data": {"term": TERM}}), chat=CannedChat("[CS 374]"))
    state = initial_state("q")
    state["docs"] = [LOOKUP]
    assert node(state)["exit"] == "answer"


def test_building_the_node_makes_no_client(cfg, monkeypatch):
    import llm

    monkeypatch.setattr(
        llm, "make_chat", lambda *a, **k: pytest.fail("a client was made before any query")
    )
    assert answer_node(cfg) is not None


def test_the_client_is_made_on_the_first_query_and_reused(cfg, monkeypatch):
    import llm

    made = []
    chat = CannedChat("It is 4 hours [CS 374].")

    def make_chat(config, client=None):
        made.append(config)
        return chat

    monkeypatch.setattr(llm, "make_chat", make_chat)
    node = answer_node(cfg)
    assert made == []
    for _ in range(3):
        state = initial_state("How many credits is CS 374?")
        state["docs"] = [LOOKUP]
        assert node(state)["exit"] == "answer"
    assert len(made) == 1
    assert len(chat.calls) == 3


def test_nothing_here_needs_an_api_key(cfg):
    # The autouse fixture deletes it; the whole file passing is the assertion.
    assert answer_node(cfg, chat=CannedChat("[CS 374]")) is not None


def test_importing_the_node_pulls_in_neither_openai_nor_chromadb():
    code = (
        "import agent.nodes.answer, sys; "
        "print('openai' in sys.modules, 'chromadb' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False False"


def test_the_graph_builds_the_real_answer_node_for_the_live_config():
    built = ag.build_nodes(load_config())
    assert built["answer"].__name__ == "answer"
    assert sorted(built) == [
        "answer",
        "check_evidence",
        "dispatch",
        "out_of_scope",
        "refuse_handoff",
        "route_intent",
    ]


# --- the live catalog -----------------------------------------------------------------------------


def test_the_example_id_in_the_prompt_is_not_a_course_this_catalog_has():
    from agent.nodes.dispatch import catalog_subjects

    assert prompts.ANSWER_EXAMPLE_ID.split()[0] not in catalog_subjects()


def live_docs(query: str, intent: str) -> list[Evidence]:
    """The real dispatch node's evidence for ``query`` -- the live catalog, no index needed."""
    cfg = Config({"data": {"term": TERM}, "agent": {"retrieval_mode": "bm25"}})
    state = initial_state(query)
    state["intent"] = intent  # type: ignore[typeddict-item]
    return list(dispatch_node(cfg)(state)["docs"])


def test_a_real_lookup_block_reaches_the_prompt_with_its_code_as_the_id():
    docs = live_docs("How many credits is MATH 241?", "course_fact")
    prompt = answer_user_prompt("How many credits is MATH 241?", docs)
    assert "Evidence 1 of 1\ndoc_id: MATH 241\n---\nMATH 241 — Calculus III" in prompt
    assert "Credits: 4\nTerm: 2026-sp\n" in prompt
    assert "Prerequisites: MATH 231." in prompt


def test_a_citation_of_that_block_verifies_and_the_answer_stands(cfg):
    docs = live_docs("How many credits is MATH 241?", "course_fact")
    draft = "MATH 241 is 4 hours [MATH 241]."
    update = write(cfg, "How many credits is MATH 241?", docs, draft)
    assert update["exit"] == "answer"
    assert update["signals"]["citations_verified"] == ["MATH 241"]


def test_the_live_requirement_rows_comma_shaped_id_survives_one_bracket(cfg):
    query = "What is left for the Computer Science major if I have taken CS 124?"
    docs = live_docs(query, "degree_requirement")
    assert [block.doc_id for block in docs] == ["Computer Science, BS"]
    draft = "Nine groups are still open [Computer Science, BS]."
    update = write(cfg, query, docs, draft)
    assert update["exit"] == "answer"
    assert update["signals"]["citations_verified"] == ["Computer Science, BS"]


def test_a_real_prereq_chain_block_is_cited_by_the_course_code(cfg):
    query = "What do I need before CS 374?"
    docs = live_docs(query, "prereq_chain")
    prompt = answer_user_prompt(query, docs)
    assert "doc_id: CS 374" in prompt
    assert "Prerequisite statement: One of CS 173, MATH 213; CS 225." in prompt
    draft = 'The catalog asks for "One of CS 173, MATH 213; CS 225." [CS 374].'
    assert write(cfg, query, docs, draft)["exit"] == "answer"
