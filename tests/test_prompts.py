"""Tests for agent/prompts.py: what each prompt must say, and that neither drifts from its version.

Two things are worth a test here and neither is the wording. The first is that the router prompt
teaches exactly the labels the state validates against -- a mismatch means every query the model
labels with the sixth one is scored as unparseable -- and that the answer prompt's own examples
obey the citation rule the answer node enforces (the stronger form of that is in
``tests/test_answer.py``, which runs the real extractor over them). The second is that neither text
can change without its version changing with it: Day 4 ablates prompts and Day 5 reads reports, and
both need "v1" to mean one string rather than whichever string was in the file that week. The
digests below are the mechanism -- edit a prompt, bump its version, update its digest, in one
commit.
"""

from __future__ import annotations

import hashlib

import pytest

from agent import prompts
from agent.state import INTENTS

#: sha256 of ROUTER_SYSTEM_PROMPT, first 16 hex chars, for version v1. Change the prompt and this
#: test fails: bump ROUTER_PROMPT_VERSION and this literal in the same commit, so that a report
#: naming a version names one text.
V2_DIGEST = "3314cfc1f5e97a27"

#: The same, for ANSWER_SYSTEM_PROMPT at ANSWER_PROMPT_VERSION v1.
ANSWER_V1_DIGEST = "6de696d3d0873e19"


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


# --- the prompt is pinned to its version ----------------------------------------------------------


def test_the_router_prompt_is_the_text_its_version_names():
    assert prompts.ROUTER_PROMPT_VERSION == "v2"
    assert digest(prompts.ROUTER_SYSTEM_PROMPT) == V2_DIGEST


def test_the_prompt_is_a_constant_and_rendering_it_again_reproduces_it():
    assert prompts.render_router_prompt() == prompts.ROUTER_SYSTEM_PROMPT


# --- it teaches the five labels, and only those ---------------------------------------------------


def test_every_label_has_a_gloss_and_at_least_two_distinct_examples():
    assert set(prompts.ROUTER_GLOSSES) == set(INTENTS)
    assert set(prompts.ROUTER_EXAMPLES) == set(INTENTS)
    for label, examples in prompts.ROUTER_EXAMPLES.items():
        assert len(set(examples)) >= 2, label


def test_v2_teaches_the_two_shapes_the_model_sent_out_of_scope():
    # Rules 2 and 3 already covered both, and gpt-4o-mini labelled them out_of_scope anyway
    # (2026-09-21): a prerequisite question naming no course, and an existence question about a
    # topic the catalog probably does not carry. Each needed an example of its shape.
    assert "What do I need to take first?" in prompts.ROUTER_EXAMPLES["prereq_chain"]
    assert "Is there a course about beekeeping?" in prompts.ROUTER_EXAMPLES["compare_recommend"]


def test_the_examples_are_not_the_questions_the_router_is_evaluated_on():
    # eval/datasets/day2_queries.jsonl carries both of these. A prompt that quotes its own
    # evaluation set turns the smoke run into a lookup, so v2 teaches their shapes on other
    # questions.
    evaluated = {"What are the prerequisites?", "Is there a course about quantum origami?"}
    taught = {example for examples in prompts.ROUTER_EXAMPLES.values() for example in examples}
    assert not (taught & evaluated)


def test_the_prompt_names_each_label_with_both_of_its_examples():
    for label in INTENTS:
        assert f"- {label}: " in prompts.ROUTER_SYSTEM_PROMPT
        for example in prompts.ROUTER_EXAMPLES[label]:
            assert f"Question: {example}\nLabel: {label}" in prompts.ROUTER_SYSTEM_PROMPT


def test_a_label_added_to_the_state_with_no_examples_fails_at_import(monkeypatch):
    # The real failure this guards: a sixth label reaches INTENTS, the prompt never teaches it,
    # and the model keeps producing five -- which looks like a model problem for a while.
    monkeypatch.setattr(prompts, "INTENTS", (*INTENTS, "waitlist_status"))
    with pytest.raises(ValueError, match="missing \\['waitlist_status'\\]"):
        prompts.check_router_labels()


def test_an_example_for_a_label_the_state_does_not_know_is_refused(monkeypatch):
    monkeypatch.setitem(prompts.ROUTER_GLOSSES, "vibes", "whatever")
    with pytest.raises(ValueError, match="unknown \\['vibes'\\]"):
        prompts.check_router_labels()


def test_one_example_per_label_is_refused(monkeypatch):
    # Two, because the first example of a pair teaches the label's easiest case: one course code
    # and one question word. The second is what carries the boundary.
    monkeypatch.setitem(prompts.ROUTER_EXAMPLES, "course_fact", ("q", "q"))
    with pytest.raises(ValueError, match="two distinct examples for course_fact"):
        prompts.check_router_labels()


# --- what the prompt says, and what it deliberately does not --------------------------------------


def test_the_prompt_asks_for_a_bare_label():
    assert "exactly one label" in prompts.ROUTER_SYSTEM_PROMPT
    assert "No punctuation, no explanation" in prompts.ROUTER_SYSTEM_PROMPT


def test_the_prompt_tells_the_model_not_to_decide_what_the_catalog_has():
    # #39's set has a query about a course the 2026-sp catalog does not contain, and its expected
    # exit is the hand-off, not out-of-scope: "I looked and could not find it" is a different
    # sentence from "I do not do that", and only the node that looked can say the first.
    assert "not whether the catalog can answer it" in prompts.ROUTER_SYSTEM_PROMPT
    assert "refusals made later" in prompts.ROUTER_SYSTEM_PROMPT


def test_the_prompt_names_no_term_and_no_catalog_year():
    # Deliberate: the term belongs to the exit messages, which is where a person reads it. A
    # router that knew the term would start refusing questions about courses outside it.
    for stamp in ("2026", "sp", "spring", "fall"):
        assert stamp not in prompts.ROUTER_SYSTEM_PROMPT.lower().split()


# --- the user half --------------------------------------------------------------------------------


def test_the_user_prompt_puts_the_query_in_a_field_and_cues_the_label():
    assert prompts.router_user_prompt("How many credits is CS 374?") == (
        "Question: How many credits is CS 374?\nLabel:"
    )


def test_a_query_full_of_braces_is_not_a_format_string():
    # The query is a substituted value, never a template, so `{intent}` in a question stays text.
    assert "{intent}" in prompts.router_user_prompt("what is {intent} {{0}}")


@pytest.mark.parametrize("query", ["", "   ", "\n"])
def test_an_empty_query_is_refused_rather_than_sent(query):
    with pytest.raises(ValueError, match="query is empty"):
        prompts.router_user_prompt(query)


# --- the answer prompt: pinned, and it demonstrates its own rules ---------------------------------


def test_the_answer_prompt_is_the_text_its_version_names():
    assert prompts.ANSWER_PROMPT_VERSION == "v1"
    assert digest(prompts.ANSWER_SYSTEM_PROMPT) == ANSWER_V1_DIGEST


def test_the_two_prompts_are_versioned_separately():
    # One version per prompt, not one per module: rewording the router must not make a report that
    # names the answer prompt's version describe a text that did not change.
    assert prompts.ROUTER_SYSTEM_PROMPT != prompts.ANSWER_SYSTEM_PROMPT
    assert digest(prompts.ROUTER_SYSTEM_PROMPT) != digest(prompts.ANSWER_SYSTEM_PROMPT)


def test_the_answer_prompt_states_all_three_constraints():
    # #38's three, in the model's own reading order: only the evidence, say so when it is not
    # there, cite the block.
    assert "Answer only from the evidence blocks." in prompts.ANSWER_SYSTEM_PROMPT
    assert "If the evidence does not contain what was asked, say so" in prompts.ANSWER_SYSTEM_PROMPT
    assert "copy the id from that block's `doc_id:` line" in prompts.ANSWER_SYSTEM_PROMPT


def test_the_answer_prompt_says_an_invented_citation_costs_the_whole_answer():
    # The model is told what the check does, because a rule with no stated consequence reads as a
    # preference -- and the consequence is real: verify_citations discards the draft.
    assert "discarded whole" in prompts.ANSWER_SYSTEM_PROMPT


def test_the_answer_prompt_asks_for_a_citation_on_the_i_do_not_know_answer_too():
    # Rule 4, which is what makes "an answer that cites nothing is discarded" a fair rule.
    assert "including a sentence saying the evidence does not carry something" in (
        prompts.ANSWER_SYSTEM_PROMPT
    )


def test_the_answer_prompt_leaves_the_advisor_hand_off_to_the_exit_that_owns_it():
    assert "no advice to consult an advisor" in prompts.ANSWER_SYSTEM_PROMPT


def test_both_example_answers_cite_the_example_block():
    for question, answer in prompts.ANSWER_EXAMPLES:
        assert f"[{prompts.ANSWER_EXAMPLE_ID}]" in answer, question
    prompts.check_answer_examples()


def test_one_example_shows_the_fact_being_there_and_one_shows_it_missing():
    answers = [answer for _question, answer in prompts.ANSWER_EXAMPLES]
    assert any("does not say" in answer for answer in answers)
    assert any("3 hours" in answer for answer in answers)


def test_no_example_is_refused_when_there_are_none(monkeypatch):
    monkeypatch.setattr(prompts, "ANSWER_EXAMPLES", ())
    with pytest.raises(ValueError, match="ANSWER_EXAMPLES is empty"):
        prompts.check_answer_examples()


def test_the_answer_prompt_names_no_term_and_no_catalog_year():
    # Same rule as the router's, for a different reason: the term is in the evidence blocks
    # themselves, and a term in the prompt would be a config value copied into a module constant.
    for stamp in ("2026", "sp", "spring", "fall", "2026-sp"):
        assert stamp not in prompts.ANSWER_SYSTEM_PROMPT.lower().split()


def test_the_blocks_in_the_prompt_are_numbered_and_carry_the_id_on_its_own_line():
    assert "Evidence 1 of 1\ndoc_id: SUBJ 100\n---\n" in prompts.ANSWER_SYSTEM_PROMPT


def test_the_user_prompt_puts_the_question_and_the_evidence_in_labelled_fields():
    block = prompts.Evidence(doc_id="CS 374", text="CS 374 — Algorithms", score=None, source="t")
    assert prompts.answer_user_prompt("How many credits is CS 374?", [block]) == (
        "Question: How many credits is CS 374?\n\n"
        "Evidence:\n\n"
        "Evidence 1 of 1\ndoc_id: CS 374\n---\nCS 374 — Algorithms\n\n"
        "Answer:"
    )
