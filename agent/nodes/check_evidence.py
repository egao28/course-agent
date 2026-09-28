"""Node 3: is there enough here to answer at all? One threshold per source, and both numbers kept.

    dispatch ──► check_evidence ──► answer
                       └──────────► refuse_handoff

Two methods, named by ``evidence.method``. ``top_score`` is Day 2's threshold, and most of this
docstring is about why the naive version of it does not work. ``conformal`` (#92) is the gate #90
fitted on the calib half: under it this node keeps the hard failures below, compares the
``retrieval_score`` signal itself when that is the one chosen, and otherwise passes every row with
evidence on to the answer and to :mod:`agent.nodes.confidence_gate`, which reads a signal off the
written answer. See **Under conformal** below.

**One fixed number cannot work, because the scales are unrelated.** ``Evidence.score`` is whatever
the retriever that produced it uses, and :mod:`retrieval.types` already says what that means:
comparable within one result list, meaningless across two.

| source | scale | observed on this corpus |
|---|---|---|
| ``bm25`` | Okapi BM25, unbounded | 7.2-30.9 top-1 over the queries below |
| ``dense`` | ``1 - cosine_distance``, higher better | 0.4529-0.6781 top-1 over the queries below |
| ``hybrid`` | RRF, ``sum 1/(rrf_k + rank)`` | 0.0296-0.0328 top-1 -- not gateable, see below |

So the threshold is per source, in ``config.yaml``'s ``evidence`` section, and this module holds no
number of its own -- not even a default. A float here would be a tuning decision outside the config
hash, and the hash is what ties a refusal rate in a report to the threshold that produced it.

**The BM25 number is a cutoff on this corpus, not a confidence, and the measurements say so.**
Top-1 scores on the live index (bm25 key ``877a6610``), against ``bm25: 15.0``:

| query | top-1 | what came first | verdict |
|---|---|---|---|
| ``How many credits is MATH 241?`` | 30.91 | MATH 241 | passes, right |
| ``CS 374`` | 29.03 | CS 374 | passes, right |
| ``Which course covers organic chemistry?`` | 18.65 | CHEM 360 (Environment) | passes, wrong |
| ``Which intro machine learning course should I take?`` | 18.27 | IS 557 | passes |
| ``I want a course on Shakespeare`` | 16.22 | ENGL 218 (Intro to Shakespeare) | passes, right |
| ``linear algebra`` | 14.97 | MATH 227 (Linear Algebra for Data Science) | refused, right |
| ``What does the data structures course cover?`` | 13.20 | ERAM 557 (Semiotics) | refused, wrong |
| ``machine learning`` | 11.91 | IS 557 | refused |
| ``What does the algorithms course cover?`` | 11.79 | ECON 300 | refused, wrong |
| ``Is there a course about quantum origami?`` | 11.06 | ECE 305 | refused, wrong |
| ``introduction to statistics`` | 9.00 | STAT 400 | refused, right |
| ``algorithms`` | 8.48 | CS 374 | refused, right |
| ``asdfghjkl qwertyuiop`` | -- | no hits at all | refused before any threshold |

Two rows of that table are the whole argument. ``algorithms`` puts **CS 374 first at 8.48** -- the
exactly right answer -- and scores *below* ``Is there a course about quantum origami?``, which puts
ECE 305 first at 11.06 and is a course that does not exist. And ``machine learning`` (11.91) and
``Which intro machine learning course should I take?`` (18.27) return the **same document**: BM25
sums a contribution per query term, so the longer sentence scores 53% higher for the same match,
which means part of what this threshold measures is how many words the question had. The value
separates the handwritten set correctly, and it is chosen from that set; it is not a probability,
and nothing here should be read as one. That is the gap Day 5's calibration closes, and it is why
both numbers are recorded on every run rather than compared and dropped.

**The dense number is the same kind of cutoff, on thinner evidence.** Top-1 cosine similarities
over those same twelve queries (collection ``courses_094143d8``, measured 2026-09-21) run from
**0.4529 to 0.6781**, so the ``0.35`` this section carried while the index was unbuilt passed all
twelve -- it refused nothing. ``dense: 0.46`` refuses exactly one row: ``Is there a course about
quantum origami?`` at 0.4529, the one question with no answer, and the next row up (``algorithms``
at 0.4619) clears it by 2%. One negative example and a 2% margin is a floor fitted to a single
query, not a tuned threshold; it is written down here so a report naming it says what it rests on,
and Day 4's negative set is what replaces it. Dense top-1 is the right course only 6 times in 12
(``CS 374`` returns **CS 473** first, at 0.6200), so it is no more a confidence than BM25's is.

**What the RRF threshold turned out to mean: nothing that can be gated on.** With ``rrf_k: 60``
and ``top_k_each: 20``, a document one retriever ranked first scores ``1/61 = 0.0164`` and one both
retrievers returned at their worst reported rank scores ``2/80 = 0.025`` -- so ``0.025`` is exactly
"both retrievers returned it", which is what ``config.yaml`` claimed for it. Measured on
2026-09-21, once the dense index existed, that is not a blunt gate but **no gate at all**: top-1
over the same twelve queries spans **0.0296 to 0.0328**, and ``Is there a course about quantum
origami?`` scores **0.0325**, fourth highest of the twelve. The top document of a fused list is
nearly always one that both retrievers returned near the top, whatever it is, so the score is
almost constant across queries and no cutoff admits the answerable rows while refusing that one.

So ``top_score`` **refuses** a hybrid block rather than comparing it (:data:`UNGATEABLE`). Passing
everything is worse than handing over: it reads in a report as a gate that looked and agreed.
:func:`check_retrieval_mode` raises when the node is built rather than on the first scored query,
so flipping ``agent.retrieval_mode`` to ``hybrid`` fails immediately and visibly instead of
quietly un-gating every answer the agent gives.

**A tool result bypasses the threshold.** ``score is None`` means #36 rendered an exact catalog
lookup, and ``get_course("CS 374")`` returning the record *is* the evidence -- there is no ranking
to be unsure about, and comparing it against a BM25 number would compare it against nothing. It
passes, and both signals stay ``None``, which is also how #39's table tells a tool row from a
retrieval one.

**A negative tool outcome fails without a threshold either.** ``signals["tool_error"]`` set, or
``docs`` empty, is ``evidence_ok = False``: that is what sends #36's missing-code and not-found
cases, and an audit of a program this corpus cannot read, to the hand-off exit. The gate does not
need to know which of the three happened -- :func:`agent.nodes.exits.handoff_reason` reads that off
the same signals and writes the sentence.

**One scored source per run, and a mixed list is refused rather than resolved.** Thresholds are
per source *because* the scales are unrelated, so "the top score" is only defined inside one
source, and the two numbers in ``signals`` can only describe one comparison. #36 produces exactly
one: a retrieval row stamps the mode on every hit it converts, and a tool row produces one block
with no score. A ``docs`` list carrying two retrievers' raw scores is a new dispatch row whose gate
has not been designed yet, and the design belongs in that issue, not in a ``max()`` here.

**Under conformal.** The pinned fit is loaded and checked when the node is built
(:func:`agent.nodes.confidence_gate.conformal_gate`). A named ``tool_error`` and an empty ``docs``
still fail. Then:

* signal ``retrieval_score``: the best score against the fitted ``t``, as ``top_score`` does it but
  with one threshold, fitted for the mode the signals run used. A tool row passes uncompared: #87
  recorded ``retrieval_score = null`` there and #90 fitted on the non-null rows only, so the
  threshold says nothing about lookups, and an exact lookup is still the strongest evidence this
  agent gets. ``t = +inf`` refuses every scored row.
* signal ``verbalized`` or ``self_consistency``: pass. The comparison needs a written answer and
  happens after ``answer``. The best score is still recorded when there is one, with no threshold,
  so the retrieval column stays filled for Day 5.

Both methods write ``signals['evidence_method']``, and ``conformal`` also writes the gate it
applied to ``signals['conformal']``.

**The source is already in the record, so it is not written twice.**
``signals["retrieval_mode"]`` is #36's, and for a scored block it *is* ``Evidence.source`` -- the
dispatcher stamps the mode on every hit -- so a third column here would agree with it on every row
by construction. The refusal above is what keeps that identity safe to rely on.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple

import retrieval
from agent.nodes import Node
from agent.nodes.confidence_gate import Conformal, conformal_gate
from agent.state import AgentState, Evidence
from config import Config

#: How the gate decides, name -> what the name means. One method today; Day 5 adds ``conformal``,
#: which is why ``evidence.method`` is a named choice in the config rather than an implied one --
#: the report of a run gated by a threshold and one gated by a calibrated interval must not be
#: distinguishable only by their date.
METHODS: dict[str, str] = {
    "top_score": "the best-scoring block, against the threshold for the source that produced it",
    "conformal": (
        "the signal and threshold #90 fitted on the calib half (evidence.conformal), compared "
        "here for retrieval_score and after the answer for verbalized and self_consistency"
    ),
}

#: Which sources a threshold may be declared for: :data:`retrieval.MODES`, and nothing else. Those
#: are the only sources a scored block can come from -- every other source is a tool name, and a
#: tool result has no score to compare -- so an extra key here is a typo (``bm_25``) or a threshold
#: nothing will ever read, and both are better as an error than as a line in a tracked config.
SOURCES: tuple[str, ...] = retrieval.MODES

#: Sources whose scores ``top_score`` refuses to compare against a threshold at all, with the
#: measurement that says why. This is not a tuning choice that a config could overrule: a number
#: that is nearly the same for a right answer and for a question with no answer cannot be made
#: into a gate by moving the cutoff, only by changing the method (Day 5's ``conformal``).
UNGATEABLE: dict[str, str] = {
    "hybrid": (
        "an RRF score is a sum of 1/(rrf_k + rank) over the retrievers that returned the "
        "document, so the top document of a fused list scores almost the same whatever it is -- "
        "measured over the twelve queries in tests/test_check_evidence.py the whole top-1 range "
        "is 0.0296 to 0.0328, and the one question with no answer ('Is there a course about "
        "quantum origami?') scores 0.0325, fourth highest of the twelve, so no cutoff separates "
        "that set"
    ),
}


class Verdict(NamedTuple):
    """The gate's decision and the two numbers behind it, or ``None`` where there was no comparison.

    Returned as a record rather than written straight into the state so that :func:`judge` is a
    function of its inputs and the node stays four lines: the same split #36 uses for
    :class:`~agent.nodes.dispatch.Gathered`.
    """

    #: Is there enough to answer from? What the graph routes on.
    ok: bool
    #: The score compared, or ``None`` when nothing was compared (a lookup, a failure, no docs).
    score: float | None
    #: The threshold it was compared against, on the same terms.
    threshold: float | None


# --- the config section ---------------------------------------------------------------------------


def evidence_section(cfg: Mapping[str, Any]) -> Mapping[str, Any]:
    """The ``evidence`` section, or a sentence saying what it is supposed to hold."""
    if "evidence" not in cfg:
        raise ValueError(
            "config has no `evidence` section; it must set method and a threshold per source "
            f"({', '.join(SOURCES)}) -- how the gate decides there is enough to answer from "
            "(see config.yaml)"
        )
    section = cfg["evidence"]
    if not isinstance(section, Mapping):
        raise TypeError(
            f"config's `evidence` section is {type(section).__name__}; it is a mapping with "
            "`method` and `thresholds` (see config.yaml)"
        )
    return section


def check_method(cfg: Mapping[str, Any]) -> str:
    """The gate's method, refused unless this module implements it."""
    section = evidence_section(cfg)
    if "method" not in section:
        raise ValueError(
            "config's `evidence` section has no method; it names how the gate decides, one of "
            f"{', '.join(METHODS)} (see config.yaml)"
        )
    method = section["method"]
    if method not in METHODS:
        known = "; ".join(f"{name} -- {gloss}" for name, gloss in METHODS.items())
        raise ValueError(
            f"config sets evidence.method to {method!r}, which this repo does not implement. "
            f"Methods: {known}"
        )
    return method


def check_number(value: Any, gloss: str) -> float:
    """One threshold or one score as a plain ``float``, or a sentence naming what it was.

    ``bool`` is refused explicitly -- it is an ``int`` in Python, and ``bm25: true`` in YAML would
    otherwise become a threshold of 1.0. The conversion is what keeps ``signals`` a uniform column
    for Day 5 (a YAML ``15`` and a YAML ``15.0`` record identically) and what turns a non-number
    into this sentence instead of a ``TypeError`` from the comparison operator.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{gloss} is {value!r}, which is not a number")
    return float(value)


def thresholds(cfg: Mapping[str, Any]) -> dict[str, float]:
    """The per-source thresholds, validated: exactly :data:`SOURCES`, each a number and none of
    them negative.

    Checked when the node is built rather than when a query arrives -- it reads no index and costs
    nothing, and a config missing the threshold for the one mode a report is about should not
    survive until that report's first query. All three are demanded even though a given run uses
    one: which mode runs is ``agent.retrieval_mode``, a value an ablation moves, and a gate that
    only failed on the mode it was switched to would fail in the middle of a sweep.

    Zero is allowed and negative is not: ``0`` is the no-gate ablation (BM25 drops score-0
    documents, so everything that came back passes), while a negative threshold passes everything
    including a document that matched nothing, which is not a gate with a different setting -- it
    is a gate that is lying about being one.
    """
    section = evidence_section(cfg)
    if "thresholds" not in section:
        raise ValueError(
            "config's `evidence` section has no thresholds; it needs one per source "
            f"({', '.join(SOURCES)}), because their scores are not comparable (see config.yaml "
            "and retrieval/types.py)"
        )
    table = section["thresholds"]
    if not isinstance(table, Mapping):
        raise TypeError(
            f"config's evidence.thresholds is {type(table).__name__}; it is a mapping of source "
            f"-> number, one for each of {', '.join(SOURCES)} (see config.yaml)"
        )
    missing = [source for source in SOURCES if source not in table]
    unknown = [source for source in table if source not in SOURCES]
    if missing or unknown:
        parts = []
        if missing:
            parts.append(f"is missing {', '.join(missing)}")
        if unknown:
            parts.append(f"names {', '.join(map(repr, unknown))}, which retrieve() does not run")
        raise ValueError(
            f"config's evidence.thresholds {' and '.join(parts)}; it holds exactly one threshold "
            f"per retrieval mode ({', '.join(SOURCES)}) -- see config.yaml"
        )
    checked = {}
    for source, value in table.items():
        number = check_number(value, f"config's evidence.thresholds.{source}")
        if number < 0:
            raise ValueError(
                f"config sets evidence.thresholds.{source} to {value!r}; a negative threshold "
                "passes every document including one that matched nothing. 0 is the no-gate "
                "setting if that is what was wanted"
            )
        checked[source] = number
    return checked


def threshold_for(source: str, table: Mapping[str, float]) -> float:
    """The threshold for ``source``, or a sentence about a scored block from somewhere unexpected.

    Reached only by a block that *has* a score, so the source is a retrieval mode or #36 has
    invented a scored tool -- in which case the number it would be compared against is the
    question, and it has to be answered in the config before this node can do anything with it.
    """
    if source not in table:
        raise ValueError(
            f"evidence from {source!r} carries a score, but no threshold is declared for it; "
            f"thresholds exist for {', '.join(table)} -- a tool result carries score=None and is "
            "never compared, so a scored block from anywhere else needs a threshold in config.yaml"
        )
    return table[source]


def gateable() -> str:
    """The sources a ``top_score`` gate can still judge, for the messages that have to name them."""
    return ", ".join(source for source in SOURCES if source not in UNGATEABLE)


def check_gateable(source: str) -> str:
    """Refuse a scored block from a source whose scale this method cannot judge.

    Raised rather than passed or failed, because either verdict would be a claim this gate cannot
    support: passing reads as "the evidence was good enough" and failing as "it was not", and the
    measurement in :data:`UNGATEABLE` says the number distinguishes neither. A run that got here
    was configured to gate something ungateable, and that is a question for ``config.yaml``.
    """
    if source in UNGATEABLE:
        raise ValueError(
            f"evidence from {source!r} cannot be gated by top_score: {UNGATEABLE[source]}. "
            f"Gate a mode whose score means something per document (agent.retrieval_mode: "
            f"{gateable()}), or gate it with evidence.method: conformal (#92)"
        )
    return source


def check_retrieval_mode(cfg: Mapping[str, Any]) -> None:
    """Refuse, when the node is built, a config that would run the gate on an ungateable mode.

    ``agent.retrieval_mode`` belongs to #36 and is validated there; this reads it only to fail
    early, because the alternative is a graph that builds, answers the first four questions and
    raises on the fifth -- the first one whose evidence carried a score. A mode this cannot gate
    is a decision about the whole run, and a run should not start by half-making it.

    Only an exact ungateable name is refused. A missing section, or a mode that is not a string,
    is dispatch's error to report in its own words rather than this node's to guess at.
    """
    section = cfg.get("agent") or {}
    mode = section.get("retrieval_mode") if isinstance(section, Mapping) else None
    if isinstance(mode, str) and mode in UNGATEABLE:
        raise ValueError(
            f"config sets agent.retrieval_mode to {mode!r} and evidence.method to 'top_score', "
            f"which cannot gate it: {UNGATEABLE[mode]}. Every run would pass the gate, which in a "
            f"report is indistinguishable from evidence that was checked and found good. Set "
            f"agent.retrieval_mode to one of {gateable()}, or gate it with evidence.method: "
            "conformal (#92)"
        )


# --- the decision ---------------------------------------------------------------------------------


def one_source(docs: Sequence[Evidence]) -> tuple[str, float]:
    """The source every block in ``docs`` came from, and the best score in the list.

    ``max`` rather than ``docs[0].score``: the retrievers do return their hits in rank order, but
    the gate does not need that to be true, and reading position 0 would make a reordered list a
    silently different verdict.

    More than one source is refused -- see the module docstring. The message names them, because
    the fix is a decision about how two scales are combined and not an adjustment here.
    """
    sources = sorted({block.source for block in docs})
    if len(sources) > 1:
        raise ValueError(
            f"docs carries scored evidence from {len(sources)} sources ({', '.join(sources)}); "
            "their scores are not comparable, so there is no 'top score' to threshold. #36 "
            "produces one source per run -- a row that mixes two needs its own gate decision"
        )
    best = max(
        check_number(block.score, f"a {sources[0]} evidence block's score") for block in docs
    )
    return sources[0], best


def judge(docs: Sequence[Evidence], tool_error: Any, table: Mapping[str, float]) -> Verdict:
    """Is there enough here to answer from? The verdict, and the comparison that produced it.

    Four outcomes, in the order they are checked -- and only the last one consults a threshold:

    * a named ``tool_error``: the tool this question needed produced nothing, so there is nothing
      to be confident *about*. The label is more specific than any number and the hand-off message
      uses it instead.
    * no ``docs``: a retrieval that came back empty, which is the honest bottom of the scale.
    * any block with ``score is None``: an exact catalog lookup is in the evidence, and it passes
      without a threshold. Any, not all -- one certain block is enough, whatever came with it.
    * otherwise: the best score in the list against its own source's threshold, ``>=`` passing --
      unless the source is one :data:`UNGATEABLE` names, which raises instead of returning a
      verdict the number cannot support.
    """
    if tool_error:
        return Verdict(False, None, None)
    if not docs:
        return Verdict(False, None, None)
    if any(block.score is None for block in docs):
        return Verdict(True, None, None)
    source, score = one_source(docs)
    check_gateable(source)
    threshold = threshold_for(source, table)
    return Verdict(score >= threshold, score, threshold)


def judge_conformal(docs: Sequence[Evidence], tool_error: Any, gate: Conformal) -> Verdict:
    """The pre-answer half of the calibrated gate. See "Under conformal" in the module docstring.

    The same hard failures as :func:`judge`, in the same order. After them only a
    ``retrieval_score`` gate compares anything; a post-answer signal passes, with the best score
    kept and no threshold beside it.
    """
    if tool_error or not docs:
        return Verdict(False, None, None)
    if any(block.score is None for block in docs):
        return Verdict(True, None, None)
    _, score = one_source(docs)
    if gate.signal != "retrieval_score":
        return Verdict(True, score, None)
    return Verdict(score >= gate.threshold, score, gate.threshold)


def check_evidence_node(cfg: Config) -> Node:
    """Build the gate: the config's method, and a verdict plus its two numbers on every query.

    Nothing is injectable. The thresholds are the whole behaviour of this node and they are read
    from the config on purpose: they are hashed, the hash names the report, and a threshold handed
    in through a keyword would be a tuning decision that no report could be traced back to. An
    ablation builds a variant :class:`~config.Config` -- the idiom :mod:`config` documents -- and
    gets a new hash and a new report name for free.
    """
    method = check_method(cfg)
    # Validated under either method: `top_score` stays selectable, so a config that switches to
    # conformal must still carry a Day 2 gate that would build.
    table = thresholds(cfg)
    gate = conformal_gate(cfg) if method == "conformal" else None
    if gate is None:
        check_retrieval_mode(cfg)

    def node(state: AgentState) -> dict[str, Any]:
        signals: Mapping[str, Any] = state.get("signals") or {}
        docs = state.get("docs") or []
        if gate is None:
            verdict = judge(docs, signals.get("tool_error"), table)
        else:
            verdict = judge_conformal(docs, signals.get("tool_error"), gate)
        return {
            "evidence_ok": verdict.ok,
            "signals": {
                "retrieval_top_score": verdict.score,
                "evidence_threshold": verdict.threshold,
                "evidence_method": method,
                "conformal": None if gate is None else gate.record(),
            },
        }

    node.__name__ = "check_evidence"
    return node
