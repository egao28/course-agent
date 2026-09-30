"""One run, printed: what was asked, what was decided, and what the decision was made from.

    query     How many credits is MATH 241?
    intent    course_fact       forced -- no model was asked
    evidence  get_course, 1 block

      #  score  doc_id    first line
      1      —  MATH 241  MATH 241 — Calculus III

    gate      passed -- an exact catalog lookup carries no score, so no threshold was consulted
    exit      answer            cited MATH 241
    answer    from a stand-in -- canned text, nothing was sent to a model

    MATH 241 — Calculus III [MATH 241]

``python -m agent`` prints this, and #39's report renders the same facts per query in markdown,
so the two read off one set of functions here rather than two formatters that drift. A bare answer
is not a trace: which label was chosen, which row of the dispatch table ran, what the gate
compared and which exit was taken are what a person debugs the graph with, and on a refusal the
answer text is the *only* part that is fixed in advance.

**Everything printed is read off the final state.** :func:`agent.graph.run` returns it whole --
``intent``, ``docs``, ``evidence_ok``, ``signals``, ``exit``, ``answer`` -- so nothing here needs
the graph to report on itself as it goes, and a trace can be rendered from a state a test built by
hand. The one fact the state does not carry is which chat client the caller put behind the answer
node, because that is the caller's own choice; it is the ``model`` argument of
:func:`format_trace`.

**"No model was asked" is read off ``intent_raw``, not off a flag.** #35 leaves that signal
``None`` when and only when the label was forced, so a trace cannot claim a classification that
did not happen -- the same invariant the report's intent column rests on.

**The dispatch table is not repeated here.** Which row ran is said by what came back: a retrieval
row stamps ``signals["retrieval_mode"]``, a tool row names itself in ``Evidence.source``, and a
row that produced nothing leaves ``signals["tool_error"]``. A mapping from intent to tool name in
this module would be a second copy of :mod:`agent.nodes.dispatch`'s table, and the copy would be
the one that went stale.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from agent.state import AgentState, Evidence

#: A number that does not exist for this row: a tool result's score, a threshold never consulted.
#: The same glyph :data:`retrieval.MISSING` uses, so the two tables read the same way.
MISSING = "—"

#: How much of an evidence block's first line the terminal table shows. The block's whole text is
#: what the answer node was given; this column is for recognizing the course, not for reading it.
LINE_WIDTH = 52

#: The label column of the trace. Wide enough for ``evidence`` with two spaces after it.
LABEL_WIDTH = 10

#: What :class:`llm.StandInChat` reports as its model name. Matched as a string rather than
#: imported: ``llm`` reaches ``openai`` through ``embedding``, and a trace must render in a
#: checkout with no key. ``tests/test_trace.py`` asserts the two still agree.
STANDIN = "stand-in"

#: The evidence table's columns.
EVIDENCE_HEADERS = ("#", "score", "doc_id", "first line")

#: Which of them are numbers, and so right-aligned.
NUMERIC = {0, 1}


def signals(state: Mapping[str, Any]) -> dict[str, Any]:
    """The run's signals, as a plain dict -- ``{}`` for a state that never got any."""
    return dict(state.get("signals") or {})


def docs_of(state: Mapping[str, Any]) -> list[Evidence]:
    """The evidence this run gathered."""
    return list(state.get("docs") or [])


def answer_node_ran(state: Mapping[str, Any]) -> bool:
    """Did #38 run? ``citations_verified is None`` is its "never ran" marker (see agent/state.py).

    An empty list is not the same answer: that is a draft that cited nothing and was discarded,
    which is a model call that happened.
    """
    return signals(state).get("citations_verified") is not None


# --- the lines ------------------------------------------------------------------------------------


def intent_note(state: Mapping[str, Any]) -> str:
    """What the router did to arrive at the label: nothing, a clean parse, or a refused string.

    ``intent_raw is None`` is #35's invariant for a forced label -- no other path through the node
    leaves it empty -- so this says "no model was asked" only when none was.
    """
    marks = signals(state)
    raw = marks.get("intent_raw")
    if raw is None:
        return "forced -- no model was asked"
    if marks.get("intent_unparsed"):
        return f"unparsed -- the model said {raw!r}, which is not one of the five labels"
    if raw != state.get("intent"):
        return f"the model said {raw!r}"
    return "classified"


def gathered(state: Mapping[str, Any]) -> str:
    """Where the evidence came from, in the words of whatever produced it.

    Read off the run rather than off the intent: ``retrieval_mode`` is set by a retrieval row,
    ``Evidence.source`` names the tool on a tool row, and ``tool_error`` is what is left when a row
    produced nothing. See the module docstring on why the dispatch table is not repeated here.
    """
    marks = signals(state)
    docs = docs_of(state)
    mode = marks.get("retrieval_mode")
    if mode:
        plural = "hit" if len(docs) == 1 else "hits"
        return f"retrieve({mode}), {len(docs)} {plural}"
    if docs:
        sources = ", ".join(dict.fromkeys(block.source for block in docs))
        plural = "block" if len(docs) == 1 else "blocks"
        return f"{sources}, {len(docs)} {plural}"
    tool_error = marks.get("tool_error")
    if tool_error:
        return f"none -- the tool it needed reported {tool_error}"
    if state.get("evidence_ok") is None and state.get("exit") == "out_of_scope":
        return "not gathered -- the router's own exit ended the run"
    return "none -- nothing came back"


def number(value: float) -> str:
    """A threshold or signal value as the trace prints it: ``+inf`` for "answer nothing"."""
    return "+inf" if math.isinf(value) else f"{value:g}"


def calibration(gate: Mapping[str, Any]) -> str:
    """``conformal, α=0.10, calib n=46`` -- what a conformal comparison was fitted on."""
    note = f"conformal, α={gate['alpha']:.2f}, {gate['fit_on']} n={gate['n']}"
    if gate.get("threshold") is None:
        note += ": no threshold passed, so it answers nothing"
    return note


def threshold_of(gate: Mapping[str, Any]) -> float:
    """The gate's threshold as a float, with "no threshold" read as ``inf`` -- answer nothing."""
    return math.inf if gate.get("threshold") is None else float(gate["threshold"])


def gate_note(state: Mapping[str, Any]) -> str:
    """The gate's verdict and the comparison behind it, both read off ``signals`` (#37).

    The four outcomes are the gate's own, in its order: a tool that produced nothing, an empty
    ``docs``, a lookup that carries no score, and a real comparison. The numbers are printed
    whichever way the comparison went -- a reader who can see both can tell "nothing matched" from
    "the threshold is too high", which is the whole reason #37 records the pair.
    """
    ok = state.get("evidence_ok")
    if ok is None:
        return "not reached -- no evidence was gathered for this run"
    marks = signals(state)
    score = marks.get("retrieval_top_score")
    threshold = marks.get("evidence_threshold")
    source = marks.get("retrieval_mode") or "retrieval"
    gate = marks.get("conformal")
    if gate and score is not None and threshold is not None:
        comparison = f"{score:.4f} {'>=' if ok else '<'} {number(threshold)}"
        return f"{'passed' if ok else 'refused'} -- {comparison} ({source}; {calibration(gate)})"
    if gate and ok and gate["signal"] != "retrieval_score":
        return f"passed -- evidence came back; {gate['signal']} is checked after the answer"
    if score is not None and threshold is not None:
        comparison = f"{score:.4f} {'>=' if ok else '<'} {threshold} ({source})"
        return f"{'passed' if ok else 'refused'} -- {comparison}"
    if ok:
        return "passed -- an exact catalog lookup carries no score, so no threshold was consulted"
    tool_error = marks.get("tool_error")
    if tool_error:
        return f"refused -- {tool_error}: there is nothing to be confident about"
    return "refused -- no evidence came back to compare"


def check_note(state: Mapping[str, Any]) -> str | None:
    """#92's post-answer check, or ``None`` when this run's graph had no such node."""
    marks = signals(state)
    gate = marks.get("conformal")
    if not gate or gate["signal"] == "retrieval_score":
        return None
    confident = state.get("confident")
    if confident is None:
        return "not reached -- no answer was kept to check"
    value = marks.get("confidence")
    if value is None:
        unread = f"no {gate['signal']} score could be read off the reply"
        return f"refused -- {unread} ({calibration(gate)})"
    comparison = f"{gate['signal']} {number(value)} {'>=' if confident else '<'} "
    comparison += number(threshold_of(gate))
    return f"{'passed' if confident else 'refused'} -- {comparison} ({calibration(gate)})"


def exit_note(state: Mapping[str, Any]) -> str:
    """What the exit node had to say for itself: the citations, or why the draft was dropped."""
    marks = signals(state)
    if state.get("confident") is False:
        return "draft dropped -- it did not clear the confidence check"
    unknown = marks.get("citations_unknown")
    if unknown:
        return f"draft dropped -- it cited {', '.join(unknown)}, which this run never retrieved"
    verified = marks.get("citations_verified")
    if verified:
        return f"cited {', '.join(verified)}"
    if verified is not None:
        return "draft dropped -- it cited nothing at all"
    return ""


def answer_note(state: Mapping[str, Any], model: str | None) -> str:
    """Who wrote the text below the trace: a model, a stand-in for one, or neither.

    The two refusals are fixed text (:mod:`agent.nodes.exits`), and saying so on every refusing run
    is the point -- a reader who cannot tell a generated refusal from a minted one cannot tell what
    this repo actually refuses.
    """
    if not answer_node_ran(state):
        return "fixed text -- neither refusal exit calls a model"
    written_by = "a stand-in" if model == STANDIN else model
    if state.get("exit") != "answer":
        drafter = written_by or "the model"
        return f"drafted by {drafter} and discarded -- what follows is the hand-off"
    if model == STANDIN:
        return "stand-in -- canned text, no model was called (the citation check is real)"
    if written_by is None:
        return "generated from the evidence above"
    return f"generated by {written_by} from the evidence above"


# --- the table ------------------------------------------------------------------------------------


def evidence_rows(
    state: Mapping[str, Any], width: int = LINE_WIDTH
) -> list[tuple[str, str, str, str]]:
    """One row per evidence block: its position, its score, its ``doc_id``, its first line.

    ``score`` is ``—`` for a tool result rather than ``0`` or blank: that block carries no ranking,
    which is exactly what let it past the gate without a threshold (#37).
    """
    return [
        (
            str(n),
            MISSING if block.score is None else f"{block.score:.4f}",
            block.doc_id,
            elide(first_line(block.text), width),
        )
        for n, block in enumerate(docs_of(state), start=1)
    ]


def first_line(text: str) -> str:
    """The first non-empty line of an evidence block -- what names the course in a table."""
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return ""


def elide(text: str, width: int) -> str:
    """``text`` cut to ``width`` characters, ending in ``…`` when anything was dropped."""
    return text if len(text) <= width else text[: width - 1].rstrip() + "…"


def format_evidence_table(state: Mapping[str, Any], indent: str = "  ") -> str:
    """The evidence blocks as an aligned table, or nothing at all when there are none."""
    rows = evidence_rows(state)
    if not rows:
        return ""
    columns = zip(EVIDENCE_HEADERS, *rows, strict=True)
    widths = [max(len(cell) for cell in column) for column in columns]
    return "\n".join(indent + row_line(row, widths) for row in (EVIDENCE_HEADERS, *rows))


def row_line(row: Sequence[str], widths: Sequence[int]) -> str:
    """One table row, numbers right-aligned and text left-aligned, with no trailing spaces."""
    cells = [
        cell.rjust(width) if i in NUMERIC else cell.ljust(width)
        for i, (cell, width) in enumerate(zip(row, widths, strict=True))
    ]
    return "  ".join(cells).rstrip()


# --- the trace ------------------------------------------------------------------------------------


def labelled(label: str, value: str, note: str = "") -> str:
    """One ``label   value   note`` line, aligned down the page and never trailing spaces."""
    line = f"{label.ljust(LABEL_WIDTH)}{value}"
    return f"{line.ljust(LABEL_WIDTH + 19)}{note}".rstrip() if note else line.rstrip()


def format_trace(state: AgentState, model: str | None = None) -> str:
    """The whole run as a person reads it: the six lines, the evidence table, then the answer.

    ``model`` names the chat client the caller put behind the answer node -- ``"gpt-4o-mini"``,
    ``"stand-in"`` -- and is the one thing the state cannot say for itself. ``None`` prints the
    answer without attributing it, which is right for a caller that did not choose.
    """
    lines = [
        labelled("query", state.get("query", "")),
        labelled("intent", str(state.get("intent")), intent_note(state)),
        labelled("evidence", gathered(state)),
    ]
    table = format_evidence_table(state)
    if table:
        lines += ["", table, ""]
    lines.append(labelled("gate", gate_note(state)))
    check = check_note(state)
    if check is not None:
        lines.append(labelled("check", check))
    lines += [
        labelled("exit", str(state.get("exit")), exit_note(state)),
        labelled("answer", answer_note(state, model)),
        "",
        state.get("answer") or "",
    ]
    return "\n".join(lines).rstrip() + "\n"
