"""Node 5 under ``evidence.method: conformal``: is this written answer confident enough to show?

    check_evidence ──► answer ──► confidence_gate ──► END
         │               │               └──────────► refuse_handoff
         └───────────────┴──────────────────────────► refuse_handoff

#90 fitted one answering threshold per confidence signal on the calib half and chose one signal.
Two of the three signals -- ``verbalized`` and ``self_consistency`` -- are read off a *written
answer*, and ``check_evidence`` runs before there is one. So under those two the graph gains this
node after ``answer``. ``check_evidence`` keeps only the hard failures (a named ``tool_error``,
an empty ``docs``), and the calibrated comparison happens here. Under ``retrieval_score`` the
signal exists before generation, so ``check_evidence`` makes the comparison itself and this node
is not in the graph at all. Under ``top_score`` neither is.

**The fit is pinned, and the pin is checked when the node is built.** ``evidence.conformal`` names
the fit's file and its sha256, and copies the four values the gate acts on: ``signal``,
``threshold``, ``alpha`` and ``fit_on``. :func:`conformal_gate` refuses to build unless:

* the file's sha256 is the pinned one;
* ``alpha`` and ``fit_on`` are the file's, and ``threshold`` is that signal's ``t`` there
  (``null`` on both sides is ``+inf``, "answer nothing");
* the agent is still the one the fit calibrated. The file records the key chain it was produced
  under -- ``config_hash``, ``signals_key``, ``correctness_key``, ``conformal_key`` -- and the live
  config, read as it was before the method was switched, must reproduce each one. A changed answer
  prompt setting, a changed verbalized prompt or a relabelled calib CSV each break one link, and
  the message names which. The thresholds calibrate *that* agent's answers and nothing else.

**A null signal refuses.** A verbalized reply with no readable ``Confidence:`` line, or a
self-consistency judge that said neither yes nor no, has no number to compare. #90 left those rows
out of the fit, so the threshold says nothing about them, and a gate cannot pass what it cannot
measure. They hand over, and ``signals['confidence_raw']`` keeps the reply.

**t = +inf is a real setting, not an error.** It is what #90 produced for every signal at
alpha = 0.10 on 46 calib rows, and under it every answer that reaches this node hands over. The
live config keeps ``method: top_score`` for that reason; the gate is here so the switch is one word
and the comparison is the calibrated one when a fit accepts a threshold.

**The cost.** ``verbalized`` adds 1 chat call to every query that reaches it, and
``self_consistency`` adds ``samples`` + pairs of them -- 6 with the shipped 3 samples. Both are on
top of the router and answer calls.

The chat client is injectable and built on the first query, the way :mod:`agent.nodes.answer`
does it. The hand-off text is not written here: a refusal sets ``confident = False`` and
:func:`agent.nodes.exits.refuse_handoff_node` mints the message, as it does for every other route
to that exit.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

from agent.confidence import Settings, self_consistency, settings, verbalized
from agent.nodes import Node
from agent.prompts import ANSWER_SYSTEM_PROMPT, answer_user_prompt
from agent.state import AgentState
from config import (
    REPO_ROOT,
    Config,
    config_hash,
    conformal_key,
    correctness_key,
    file_sha256,
    signals_key,
)

if TYPE_CHECKING:  # `llm` pulls in openai; see agent/nodes/answer.py
    from llm import Chat

#: The method name that turns this gate on.
METHOD = "conformal"

#: Signals read off the evidence, before an answer exists: compared in ``check_evidence``.
PRE_ANSWER: tuple[str, ...] = ("retrieval_score",)

#: Signals read off a written answer: compared here, after ``answer``.
POST_ANSWER: tuple[str, ...] = ("verbalized", "self_consistency")

#: Every signal #90 fits, in its order (``eval.scripts.risk_coverage.SIGNALS``).
SIGNALS: tuple[str, ...] = PRE_ANSWER + POST_ANSWER

#: What ``evidence.conformal`` must hold.
FIELDS: tuple[str, ...] = ("signal", "threshold", "alpha", "fit_on", "source", "source_sha256")

#: The key chain a fit records, in the order a change would break it.
KEY_CHAIN = (
    ("config_hash", config_hash, "the agent's own settings"),
    ("signals_key", signals_key, "the signals section (the verbalized and judge prompts)"),
    ("correctness_key", correctness_key, "the correctness section (the labels the fit read)"),
    ("conformal_key", conformal_key, "the conformal section (its alpha)"),
)


class Conformal(NamedTuple):
    """The calibrated gate: one signal, its threshold, and where both came from."""

    signal: str
    #: ``math.inf`` when the fit accepted no threshold: answer nothing.
    threshold: float
    alpha: float
    fit_on: str
    #: How many calib rows the signal was fitted on (its non-null rows).
    n: int
    source: str
    #: Whether this is the signal the fit itself chose (highest calib AUROC).
    chosen: bool

    def record(self) -> dict[str, Any]:
        """What ``signals['conformal']`` carries, so a trace can say what it compared against."""
        return {
            "signal": self.signal,
            "threshold": None if math.isinf(self.threshold) else self.threshold,
            "alpha": self.alpha,
            "fit_on": self.fit_on,
            "n": self.n,
            "source": self.source,
        }


def method_of(cfg: Mapping[str, Any]) -> str | None:
    """``evidence.method``, unvalidated -- ``check_evidence`` validates it when it is built."""
    section = cfg.get("evidence")
    return section.get("method") if isinstance(section, Mapping) else None


def signal_of(cfg: Mapping[str, Any]) -> str | None:
    """``evidence.conformal.signal`` when the method is conformal, else ``None``. Unvalidated."""
    if method_of(cfg) != METHOD:
        return None
    block = cfg["evidence"].get("conformal")
    return block.get("signal") if isinstance(block, Mapping) else None


def post_answer(cfg: Mapping[str, Any]) -> bool:
    """Does this config put a confidence gate after ``answer``?"""
    return signal_of(cfg) in POST_ANSWER


def calibrated_config(cfg: Mapping[str, Any]) -> Config:
    """``cfg`` as the fit saw it: the method switched back, so the gate block is dormant again.

    The fit was produced under ``method: top_score``, and :data:`config.DORMANT` leaves
    ``evidence.conformal`` out of the hash under that method, so this config's key chain is the one
    the file recorded -- unless something else changed.
    """
    plain = cfg.to_dict() if isinstance(cfg, Config) else Config(cfg).to_dict()
    plain["evidence"] = {**plain["evidence"], "method": "top_score"}
    return Config(plain)


def _number(value: Any, gloss: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{gloss} is {value!r}, which is not a number")
    return float(value)


def _source_path(source: str) -> Path:
    path = Path(source)
    return path if path.is_absolute() else REPO_ROOT / path


def conformal_gate(cfg: Mapping[str, Any]) -> Conformal:
    """The ``evidence.conformal`` block, checked against the fit it names. See the module docstring.

    Called when a node is built, never per query: it reads one small JSON file.
    """
    section = cfg.get("evidence")
    block = section.get("conformal") if isinstance(section, Mapping) else None
    if block is None:
        raise ValueError(
            "config sets evidence.method to 'conformal' but has no evidence.conformal block; it "
            f"names the fit and copies its values: {', '.join(FIELDS)} (see config.yaml)"
        )
    if not isinstance(block, Mapping):
        raise TypeError(
            f"config's evidence.conformal is {type(block).__name__}; it is a mapping of "
            f"{', '.join(FIELDS)} (see config.yaml)"
        )
    missing = [key for key in FIELDS if key not in block]
    unknown = [key for key in block if key not in FIELDS]
    if missing or unknown:
        parts = []
        if missing:
            parts.append(f"is missing {', '.join(missing)}")
        if unknown:
            parts.append(f"has {', '.join(map(repr, unknown))}, which the gate does not read")
        raise ValueError(f"config's evidence.conformal {' and '.join(parts)} (see config.yaml)")

    signal = block["signal"]
    if signal not in SIGNALS:
        raise ValueError(
            f"config sets evidence.conformal.signal to {signal!r}; #90 fits {', '.join(SIGNALS)}"
        )

    path = _source_path(str(block["source"]))
    if not path.is_file():
        raise FileNotFoundError(
            f"evidence.conformal.source {block['source']!r} does not exist; it is the fit #90 "
            "wrote (eval/runs/conformal_{conformal_key}.json)"
        )
    actual = file_sha256(path)
    if actual != block["source_sha256"]:
        raise ValueError(
            f"{block['source']} has sha256 {actual}, but evidence.conformal.source_sha256 pins "
            f"{block['source_sha256']}; the fit changed since the gate was configured from it. "
            "Re-copy its values into config.yaml and re-pin the sha, or restore the file"
        )
    fit = json.loads(path.read_text(encoding="utf-8"))

    alpha = _number(block["alpha"], "config's evidence.conformal.alpha")
    if alpha != float(fit["alpha"]):
        raise ValueError(
            f"config sets evidence.conformal.alpha to {alpha}, but {block['source']} was fitted "
            f"at alpha = {fit['alpha']}"
        )
    if block["fit_on"] != fit["split"]:
        raise ValueError(
            f"config sets evidence.conformal.fit_on to {block['fit_on']!r}, but "
            f"{block['source']} was fitted on {fit['split']!r}"
        )
    if signal not in fit["signals"]:
        raise ValueError(f"{block['source']} has no fit for {signal!r}")
    fitted = fit["signals"][signal]
    wanted = block["threshold"]
    if wanted is None or fitted["t"] is None:
        agree = wanted is None and fitted["t"] is None
    else:
        agree = _number(wanted, "config's evidence.conformal.threshold") == float(fitted["t"])
    if not agree:

        def spell(t: Any) -> str:
            return "null (+inf, answer nothing)" if t is None else repr(t)

        raise ValueError(
            f"config sets evidence.conformal.threshold to {spell(wanted)}, but {block['source']} "
            f"fitted {signal} at t = {spell(fitted['t'])}"
        )

    calibrated = calibrated_config(cfg)
    for field, key_of, what in KEY_CHAIN:
        live = key_of(calibrated)
        if live != fit[field]:
            raise ValueError(
                f"{block['source']} was fitted under {field} {fit[field]}, but this config gives "
                f"{live}: {what} changed since the fit, so its threshold calibrates a different "
                "agent. Refit (#90) under this config, or restore the setting"
            )

    return Conformal(
        signal=signal,
        threshold=math.inf if fitted["t"] is None else float(fitted["t"]),
        alpha=alpha,
        fit_on=block["fit_on"],
        n=int(fitted["n"]),
        source=str(block["source"]),
        chosen=fit["chosen"]["signal"] == signal,
    )


def confidence_gate_node(cfg: Config, chat: Chat | None = None) -> Node:
    """Build the post-answer gate: the chosen signal on the written answer, against its threshold.

    ``chat`` replaces the client, as in :func:`agent.nodes.answer.answer_node`. It makes the
    signal's own calls -- the verbalized follow-up, or the samples and the judge -- so a test puts a
    scripted fake here and the rest of the graph is unchanged.
    """
    gate = conformal_gate(cfg)
    if gate.signal not in POST_ANSWER:
        raise ValueError(
            f"the confidence gate reads a written answer, and {gate.signal!r} is read before one "
            f"exists; check_evidence compares it. The graph adds this node only for "
            f"{', '.join(POST_ANSWER)}"
        )
    config: Settings = settings(cfg)
    if chat is None:
        from llm import check_llm_config

        check_llm_config(cfg)
    built: list[Chat] = []

    def chat_client() -> Chat:
        if chat is not None:
            return chat
        if not built:
            from llm import make_chat

            built.append(make_chat(cfg))
        return built[0]

    def node(state: AgentState) -> dict[str, Any]:
        answer = state.get("answer")
        docs = state.get("docs") or []
        if not answer or not docs:
            raise ValueError(
                "the confidence gate was reached with no answer or no evidence; the answer node "
                "routes a discarded draft to the hand-off exit, so this is a wiring bug in "
                "agent/graph.py rather than a query"
            )
        query = state["query"]
        raw: Any
        if gate.signal == "verbalized":
            rated = verbalized(chat_client(), config, query, docs, answer)
            value: float | None = rated.value
            raw = rated.raw
        else:
            sampled = self_consistency(
                chat_client(), config, query, answer_user_prompt(query, docs), ANSWER_SYSTEM_PROMPT
            )
            value = sampled.value
            raw = {
                "samples": sampled.samples,
                "pair_agree": sampled.pair_agree,
                "judge_raw": sampled.judge_raw,
            }
        return {
            "confident": value is not None and value >= gate.threshold,
            "signals": {"confidence": value, "confidence_raw": raw},
        }

    node.__name__ = "confidence_gate"
    return node
