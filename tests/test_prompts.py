"""Tests for agent/prompts.py: the router prompt covers the five labels, and it is pinned.

Two things are worth a test here and neither is the wording. The first is that the prompt teaches
exactly the labels the state validates against -- a mismatch means every query the model labels
with the sixth one is scored as unparseable. The second is that the text cannot change without
:data:`~agent.prompts.ROUTER_PROMPT_VERSION` changing with it: Day 4 ablates prompts and Day 5
reads reports, and both need "v1" to mean one string rather than whichever string was in the file
that week. The digest below is the mechanism -- edit the prompt, bump the version, update the
digest, in one commit.
"""

from __future__ import annotations

import hashlib

import pytest

from agent import prompts
from agent.state import INTENTS

#: sha256 of ROUTER_SYSTEM_PROMPT, first 16 hex chars, for version v1. Change the prompt and this
#: test fails: bump ROUTER_PROMPT_VERSION and this literal in the same commit, so that a report
#: naming a version names one text.
V1_DIGEST = "84658d6a8d14d8cd"


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


# --- the prompt is pinned to its version ----------------------------------------------------------


def test_the_router_prompt_is_the_text_its_version_names():
    assert prompts.ROUTER_PROMPT_VERSION == "v1"
    assert digest(prompts.ROUTER_SYSTEM_PROMPT) == V1_DIGEST


def test_the_prompt_is_a_constant_and_rendering_it_again_reproduces_it():
    assert prompts.render_router_prompt() == prompts.ROUTER_SYSTEM_PROMPT


# --- it teaches the five labels, and only those ---------------------------------------------------


def test_every_label_has_a_gloss_and_two_distinct_examples():
    assert set(prompts.ROUTER_GLOSSES) == set(INTENTS)
    assert set(prompts.ROUTER_EXAMPLES) == set(INTENTS)
    for label, examples in prompts.ROUTER_EXAMPLES.items():
        assert len(set(examples)) == 2, label


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
