"""Tests for #92: ``evidence.method: conformal``, the post-answer gate, and the pinned fit.

Most tests run against the *live* config with the method switched, because the fit's key chain is
the live config's: a synthetic config would fail the check that the gate calibrates this agent.
A test that needs a finite threshold writes a copy of the real fit with its ``t`` changed and pins
that copy's sha -- the key chain it records is untouched, so it still describes this agent.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

import agent.__main__ as cli
from agent import graph as ag
from agent.confidence import settings
from agent.nodes.check_evidence import check_evidence_node
from agent.nodes.confidence_gate import (
    confidence_gate_node,
    conformal_gate,
    post_answer,
)
from agent.nodes.exits import NOTHING_REASON, UNREAD_REASON
from agent.prompts import ANSWER_SYSTEM_PROMPT, answer_user_prompt, render_evidence
from agent.state import Evidence, check_update, initial_state
from agent.trace import format_trace
from config import REPO_ROOT, Config, file_sha256, load_config
from eval.scripts import day2_smoke as smoke

LIVE = load_config()
FIT = REPO_ROOT / LIVE["evidence"]["conformal"]["source"]

LOOKUP = Evidence("CS 374", "CS 374 — Algorithms\n4 hours.", None, "get_course")
STRONG = Evidence("2026-sp:CS-374", "CS 374 — Algorithms", 29.0337, "bm25")
WEAK = Evidence("2026-sp:CS-173", "CS 173 — Discrete Structures", 8.4812, "bm25")


@pytest.fixture(autouse=True)
def fresh_cache():
    ag.clear_cache()
    yield
    ag.clear_cache()


@pytest.fixture(autouse=True)
def no_api_key(monkeypatch):
    # Every chat client here is a fake; a real one would need this and must never be built.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def switched(signal: str | None = None, threshold=..., path: Path | None = None, **gate) -> Config:
    """The live config under ``method: conformal``, optionally pointed at another fit."""
    data = LIVE.to_dict()
    block = dict(data["evidence"]["conformal"])
    if signal is not None:
        block["signal"] = signal
    if threshold is not ...:
        block["threshold"] = threshold
    if path is not None:
        block["source"] = str(path)
        block["source_sha256"] = file_sha256(path)
    block.update(gate)
    data["evidence"] = {**data["evidence"], "method": "conformal", "conformal": block}
    return Config(data)


def refit(tmp_path: Path, **thresholds: float | None) -> Path:
    """A copy of the real fit with the named signals' ``t`` replaced. The key chain is untouched."""
    fit = json.loads(FIT.read_text(encoding="utf-8"))
    for name, t in thresholds.items():
        fit["signals"][name]["t"] = t
        fit["signals"][name]["answers_nothing"] = t is None
    path = tmp_path / "conformal_fit.json"
    path.write_text(json.dumps(fit, indent=2), encoding="utf-8")
    return path


def fitted(tmp_path: Path, signal: str, t: float) -> Config:
    return switched(signal, t, refit(tmp_path, **{signal: t}))


class Scripted:
    """A chat client that replies from a list in order, and keeps every prompt it was sent."""

    model = "scripted"

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.calls: list[tuple[str, str | None, float | None]] = []

    def complete(self, prompt, *, system=None, temperature=None):
        self.calls.append((prompt, system, temperature))
        return self.replies.pop(0)


def router(state):
    return {"intent": "course_fact"}


def dispatcher(*docs: Evidence, tool_error: str | None = None):
    def node(state):
        return {"docs": list(docs), "signals": {"tool_error": tool_error, "retrieval_mode": None}}

    return node


def writer(state):
    return {
        "answer": "CS 374 is 4 hours [CS 374].",
        "exit": "answer",
        "signals": {"citations_verified": ["CS 374"], "citations_unknown": None},
    }


def run(cfg: Config, chat: Scripted, *docs: Evidence) -> dict:
    return ag.run(
        "How many credits is CS 374?",
        cfg=cfg,
        nodes={
            "route_intent": router,
            "dispatch": dispatcher(*(docs or (LOOKUP,))),
            "answer": writer,
            "confidence_gate": confidence_gate_node(cfg, chat=chat),
        },
    )


# --- the hash -------------------------------------------------------------------------------------


def test_the_live_config_keeps_its_hash_with_the_dormant_gate_block_in_it():
    assert LIVE["evidence"]["method"] == "top_score"
    assert "conformal" in LIVE["evidence"]
    assert LIVE.hash == "c73b197d"


def test_a_dormant_gate_block_can_change_without_moving_the_hash():
    data = LIVE.to_dict()
    data["evidence"]["conformal"] = {**data["evidence"]["conformal"], "threshold": 80}
    assert Config(data).hash == LIVE.hash
    assert Config(data).full_hash != LIVE.full_hash


def test_switching_the_method_moves_the_hash_and_so_does_the_threshold_under_it(tmp_path):
    on = switched()
    assert on.hash != LIVE.hash
    assert fitted(tmp_path, "verbalized", 80.0).hash not in {on.hash, LIVE.hash}


def test_top_score_stays_selectable_and_builds_no_confidence_gate():
    assert ag.node_names(LIVE) == (
        "route_intent",
        "dispatch",
        "check_evidence",
        "answer",
        "out_of_scope",
        "refuse_handoff",
    )
    check_evidence_node(LIVE)


# --- the pinned fit -------------------------------------------------------------------------------


def test_the_live_block_reads_the_real_fit():
    gate = conformal_gate(switched())
    assert gate.signal == "verbalized" and gate.chosen
    assert math.isinf(gate.threshold)
    assert (gate.alpha, gate.fit_on, gate.n) == (0.1, "calib", 46)


def test_a_mismatched_sha_makes_the_build_fail(tmp_path):
    path = refit(tmp_path)
    cfg = switched(path=path)
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="sha256"):
        check_evidence_node(cfg)
    with pytest.raises(ValueError, match="sha256"):
        ag.build_graph(cfg)


def test_a_threshold_that_is_not_the_fits_is_refused(tmp_path):
    with pytest.raises(ValueError, match=r"threshold to 80.*t = null"):
        conformal_gate(switched(threshold=80))
    path = refit(tmp_path, verbalized=80.0)
    with pytest.raises(ValueError, match=r"threshold to null.*t = 80.0"):
        conformal_gate(switched(path=path))


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"alpha": 0.2}, "fitted at alpha = 0.1"),
        ({"fit_on": "test"}, "fitted on 'calib'"),
        ({"signal": "dense_top1"}, "#90 fits"),
    ],
)
def test_a_value_the_fit_does_not_hold_is_refused(change, match):
    with pytest.raises(ValueError, match=match):
        conformal_gate(switched(**change))


def test_a_missing_or_unknown_field_is_named():
    data = switched().to_dict()
    del data["evidence"]["conformal"]["source_sha256"]
    data["evidence"]["conformal"]["t"] = 1
    with pytest.raises(ValueError, match=r"missing source_sha256 and has 't'"):
        conformal_gate(Config(data))


@pytest.mark.parametrize(
    ("section", "key", "value", "link"),
    [
        ("agent", "retrieval_mode", "dense", "config_hash"),
        ("signals", "samples", 5, "signals_key"),
        ("conformal", "alpha", 0.2, "conformal_key"),
    ],
)
def test_an_agent_that_is_not_the_one_the_fit_calibrated_is_refused(section, key, value, link):
    data = switched().to_dict()
    data[section] = {**data[section], key: value}
    with pytest.raises(ValueError, match=f"fitted under {link}"):
        conformal_gate(Config(data))


# --- retrieval_score: compared before the answer --------------------------------------------------


def test_retrieval_score_passes_above_t_and_refuses_below_it(tmp_path):
    node = check_evidence_node(fitted(tmp_path, "retrieval_score", 15.0))
    passed = node({**initial_state("q"), "docs": [STRONG]})
    refused = node({**initial_state("q"), "docs": [WEAK]})
    assert passed["evidence_ok"] is True and refused["evidence_ok"] is False
    assert refused["signals"]["retrieval_top_score"] == 8.4812
    assert refused["signals"]["evidence_threshold"] == 15.0
    assert refused["signals"]["evidence_method"] == "conformal"
    assert refused["signals"]["conformal"]["signal"] == "retrieval_score"
    assert check_update("check_evidence", refused) == refused


def test_retrieval_score_passes_a_lookup_it_was_never_fitted_on_and_fails_a_tool_error(tmp_path):
    node = check_evidence_node(fitted(tmp_path, "retrieval_score", 15.0))
    assert node({**initial_state("q"), "docs": [LOOKUP]})["evidence_ok"] is True
    errored = {**initial_state("q"), "signals": {"tool_error": "course_not_found"}}
    assert node(errored)["evidence_ok"] is False


def test_retrieval_score_at_the_real_fit_refuses_every_scored_row(tmp_path):
    cfg = switched("retrieval_score", path=refit(tmp_path))
    assert (
        check_evidence_node(cfg)({**initial_state("q"), "docs": [STRONG]})["evidence_ok"] is False
    )
    assert "confidence_gate" not in ag.node_names(cfg)


# --- verbalized and self_consistency: compared after the answer -----------------------------------


def test_a_post_answer_signal_adds_the_confidence_gate_between_answer_and_end(tmp_path):
    cfg = fitted(tmp_path, "verbalized", 80.0)
    assert post_answer(cfg)
    edges = {(edge.source, edge.target) for edge in ag.build_graph(cfg).get_graph().edges}
    assert ("answer", "confidence_gate") in edges
    assert ("answer", "__end__") not in edges
    assert {("confidence_gate", "__end__"), ("confidence_gate", "refuse_handoff")} <= edges


def test_the_pre_answer_gate_keeps_only_the_hard_failures(tmp_path):
    node = check_evidence_node(fitted(tmp_path, "verbalized", 80.0))
    weak = node({**initial_state("q"), "docs": [WEAK]})
    assert weak["evidence_ok"] is True
    assert weak["signals"]["retrieval_top_score"] == 8.4812
    assert weak["signals"]["evidence_threshold"] is None
    assert node(initial_state("q"))["evidence_ok"] is False


def test_verbalized_passes_at_or_above_t(tmp_path):
    state = run(fitted(tmp_path, "verbalized", 80.0), Scripted("Supported.\nConfidence: 80"))
    assert state["exit"] == "answer" and state["confident"] is True
    assert state["signals"]["confidence"] == 80
    assert state["answer"] == "CS 374 is 4 hours [CS 374]."


def test_verbalized_refuses_below_t_and_the_hand_off_says_against_what(tmp_path):
    state = run(fitted(tmp_path, "verbalized", 80.0), Scripted("Unsure.\nConfidence: 79"))
    assert state["exit"] == "refuse_handoff" and state["confident"] is False
    assert "its verbalized was 79, against a threshold of 80" in state["answer"]
    assert state["signals"]["confidence_raw"] == "Unsure.\nConfidence: 79"


def test_an_unreadable_confidence_refuses_rather_than_defaulting(tmp_path):
    state = run(fitted(tmp_path, "verbalized", 0.0), Scripted("Looks right to me."))
    assert state["exit"] == "refuse_handoff"
    assert state["signals"]["confidence"] is None
    assert UNREAD_REASON.format(signal="verbalized") in state["answer"]


def test_the_real_fit_answers_nothing_and_says_so(tmp_path):
    state = run(switched(), Scripted("Fully supported.\nConfidence: 100"))
    assert state["exit"] == "refuse_handoff"
    gate = state["signals"]["conformal"]
    assert gate["threshold"] is None
    assert NOTHING_REASON.format(**gate) in state["answer"]


def test_the_verbalized_call_is_the_one_the_signals_run_made(tmp_path):
    # #87's collect() and this gate must send the same prompt, or the threshold was fitted on a
    # different signal from the one compared here.
    cfg = fitted(tmp_path, "verbalized", 80.0)
    chat = Scripted("Confidence: 90")
    run(cfg, chat)
    expected = settings(cfg).verbalized_prompt.format(
        question="How many credits is CS 374?",
        evidence=render_evidence([LOOKUP]),
        answer="CS 374 is 4 hours [CS 374].",
    )
    assert chat.calls == [(expected, None, None)]


def test_self_consistency_passes_when_every_pair_agrees(tmp_path):
    cfg = fitted(tmp_path, "self_consistency", 2 / 3)
    chat = Scripted("a [CS 374]", "b [CS 374]", "c [CS 374]", "yes", "yes", "yes")
    state = run(cfg, chat)
    assert state["exit"] == "answer" and state["signals"]["confidence"] == 1.0
    # 3 samples + 3 pairs: the 6 extra calls per answered query the PR quotes.
    assert len(chat.calls) == 6
    prompt = answer_user_prompt("How many credits is CS 374?", [LOOKUP])
    assert chat.calls[0] == (prompt, ANSWER_SYSTEM_PROMPT, 0.7)


def test_self_consistency_refuses_when_too_few_pairs_agree(tmp_path):
    cfg = fitted(tmp_path, "self_consistency", 2 / 3)
    state = run(cfg, Scripted("a", "b", "c", "yes", "no", "no"))
    assert state["exit"] == "refuse_handoff"
    assert state["signals"]["confidence"] == pytest.approx(1 / 3)
    assert state["signals"]["confidence_raw"]["pair_agree"] == [True, False, False]


def test_a_discarded_draft_never_reaches_the_confidence_gate(tmp_path):
    cfg = fitted(tmp_path, "verbalized", 80.0)
    chat = Scripted()
    state = ag.run(
        "q",
        cfg=cfg,
        nodes={
            "route_intent": router,
            "dispatch": dispatcher(LOOKUP),
            "answer": lambda s: {
                "exit": "refuse_handoff",
                "signals": {"citations_verified": [], "citations_unknown": ["CS 999"]},
            },
            "confidence_gate": confidence_gate_node(cfg, chat=chat),
        },
    )
    assert state["exit"] == "refuse_handoff" and state["confident"] is None
    assert chat.calls == []


def test_the_confidence_gate_cannot_be_overridden_into_a_graph_that_has_none():
    with pytest.raises(KeyError, match="built only under evidence.method: conformal"):
        ag.build_nodes(LIVE, {"confidence_gate": lambda s: {"confident": True}})


def test_the_gate_writes_only_its_own_key_and_a_real_bool():
    with pytest.raises(TypeError, match="confident to None"):
        check_update("confidence_gate", {"confident": None})
    with pytest.raises(ValueError, match="belongs to"):
        check_update("confidence_gate", {"exit": "answer"})


# --- the trace ------------------------------------------------------------------------------------


def test_the_trace_prints_the_comparison_and_what_it_was_fitted_on(tmp_path):
    passed = format_trace(run(fitted(tmp_path, "verbalized", 80.0), Scripted("Confidence: 85")))
    assert (
        "gate      passed -- evidence came back; verbalized is checked after the answer" in passed
    )
    assert "check     passed -- verbalized 85 >= 80 (conformal, α=0.10, calib n=46)" in passed
    refused = format_trace(run(switched(), Scripted("Confidence: 100")))
    assert "check     refused -- verbalized 100 < +inf (conformal, α=0.10, calib n=46: " in refused
    assert "draft dropped -- it did not clear the confidence check" in refused


def test_a_top_score_trace_has_no_check_line():
    state = ag.run(
        "q",
        cfg=LIVE,
        nodes={"route_intent": router, "dispatch": dispatcher(LOOKUP), "answer": writer},
    )
    assert "\ncheck " not in format_trace(state)


# --- the CLI and the Day 2 smoke run under the new method -----------------------------------------


def test_the_stand_in_behind_a_verbalized_gate_hands_over_rather_than_rating_itself(tmp_path):
    cfg = fitted(tmp_path, "verbalized", 0.0)
    overrides = cli.build_overrides(cfg, "course_fact", True)
    assert set(overrides) == {"route_intent", "answer", "confidence_gate"}


def test_the_stand_in_cannot_run_a_self_consistency_gate(tmp_path):
    with pytest.raises(ValueError, match="cannot run a self_consistency"):
        cli.build_overrides(fitted(tmp_path, "self_consistency", 0.5), "course_fact", True)


SMOKE_ROWS = [
    smoke.Query("How many credits is CS 374?", "course_fact", "answer", "the lookup row"),
    smoke.Query("How many credits is CS 999?", "course_fact", "refuse_handoff", "not in the term"),
    smoke.Query("Who is the coach?", "out_of_scope", "out_of_scope", "not about courses"),
]


def smoke_dispatcher(state):
    if "999" in state["query"]:
        return {"docs": [], "signals": {"tool_error": "course_not_found", "retrieval_mode": None}}
    return {"docs": [LOOKUP], "signals": {"tool_error": None, "retrieval_mode": None}}


def test_the_smoke_run_still_reaches_all_three_exits_under_conformal(tmp_path):
    # #92's acceptance, with the fake client: --fake-model's stand-in answers, and a scripted
    # client behind the verbalized gate clears it. self_consistency cannot run under a stand-in
    # (above), and under the real fit (t = +inf) no row can take the answer exit at all.
    cfg = fitted(tmp_path, "verbalized", 50.0)
    runs = smoke.run_queries(
        cfg,
        SMOKE_ROWS,
        fake_model=True,
        nodes={
            "dispatch": smoke_dispatcher,
            "confidence_gate": confidence_gate_node(cfg, chat=Scripted("Confidence: 90")),
        },
    )
    assert [r.state["exit"] for r in runs] == ["answer", "refuse_handoff", "out_of_scope"]
    assert all(r.exit_ok for r in runs)


def test_the_smoke_run_under_the_real_fit_hands_over_the_row_that_would_have_answered():
    runs = smoke.run_queries(
        switched(), SMOKE_ROWS, fake_model=True, nodes={"dispatch": smoke_dispatcher}
    )
    assert [r.state["exit"] for r in runs] == ["refuse_handoff", "refuse_handoff", "out_of_scope"]
    # The stand-in's reply to the verbalized prompt carries no Confidence line.
    assert runs[0].state["signals"]["confidence"] is None
