"""Tests for agent/__main__.py: the flags, the overrides they build, and one whole run.

The run at the end is a real one -- the live ``config.yaml``, the real dispatch node, the real
tools, the real gate -- with the two model nodes replaced, which is the only way it can be one
while the account has no credits. It asks about a course, so nothing here opens the BM25 pickle
or Chroma: a fresh clone has ``data/courses.json`` and no index.
"""

from __future__ import annotations

import pytest

from agent import __main__ as cli
from agent.state import initial_state
from config import load_config


@pytest.fixture(scope="module")
def cfg():
    return load_config()


# --- the flags ------------------------------------------------------------------------------------


def test_a_stand_in_with_no_forced_label_is_refused_before_anything_runs(capsys):
    # A stand-in cannot classify, so --fake-model alone would either crash in the router or, worse,
    # report a label nobody chose. The message has to name the flag that fixes it.
    with pytest.raises(SystemExit) as raised:
        cli.main(["What do I need before CS 374?", "--fake-model"])
    assert raised.value.code == 2
    assert "--intent" in capsys.readouterr().err


def test_a_label_outside_the_five_is_refused_by_the_parser(capsys):
    with pytest.raises(SystemExit) as raised:
        cli.main(["CS 374", "--intent", "course_facts"])
    assert raised.value.code == 2
    assert "course_facts" in capsys.readouterr().err


# --- what the flags replace -----------------------------------------------------------------------


def test_no_flags_replaces_no_nodes(cfg):
    # The real agent: the router classifies and the answer node answers, both against a model.
    assert cli.build_overrides(cfg, None, False) == {}


def test_a_forced_label_reaches_the_state_with_the_raw_signal_empty(cfg):
    node = cli.build_overrides(cfg, "prereq_chain", False)["route_intent"]
    update = node(initial_state("What do I need before CS 374?"))
    assert update["intent"] == "prereq_chain"
    # `intent_raw is None` is what the trace and #39's report read as "no model was asked".
    assert update["signals"] == {"intent_raw": None, "intent_unparsed": False}


def test_the_stand_in_goes_behind_the_answer_node_and_nothing_else(cfg):
    overrides = cli.build_overrides(cfg, "course_fact", True)
    assert sorted(overrides) == ["answer", "route_intent"]


# --- one whole run --------------------------------------------------------------------------------


def test_a_lookup_runs_end_to_end_and_prints_the_whole_trace(capsys):
    cli.main(["How many credits is CS 374?", "--intent", "course_fact", "--fake-model"])
    out = capsys.readouterr().out
    for part in (
        "query     How many credits is CS 374?",
        "intent    course_fact",
        "forced -- no model was asked",
        "evidence  get_course, 1 block",
        "CS 374",
        "exit      answer",
        "cited CS 374",
        "canned text, no model was called",
    ):
        assert part in out, part
    # The footer names what produced the run, which is what a pasted trace has to carry.
    assert f"config {load_config().hash}" in out


def test_a_course_the_catalog_does_not_have_prints_the_hand_off_and_does_not_raise(capsys):
    cli.main(["How many credits is CS 999?", "--intent", "course_fact", "--fake-model"])
    out = capsys.readouterr().out
    assert "exit      refuse_handoff" in out
    assert "course_not_found" in out
    assert "fixed text -- neither refusal exit calls a model" in out


def test_an_out_of_scope_label_ends_the_run_at_the_router(capsys):
    # The real dispatch node raises if it is reached with this label, so the assertion is that
    # the trace prints at all.
    cli.main(["Who is the head football coach?", "--intent", "out_of_scope", "--fake-model"])
    out = capsys.readouterr().out
    assert "exit      out_of_scope" in out
    assert "router's own exit" in out
