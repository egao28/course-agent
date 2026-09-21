"""The two exits that do not call a model: out of scope, and hand it to a person.

Both messages are fixed text with the run's own facts filled in, and no completion is requested
for either. Nothing grounds a generated refusal -- there is no evidence behind it by definition --
and a model asked to write one will negotiate with itself and produce a partial answer with a
hedge in front of it, which is the failure the evidence gate exists to prevent. #38's answer node
is the only node that calls a model, and it is only reached when there is something to ground an
answer in.

The hand-off message is the "转人工" half of Day 2, so it is written as something a person would
actually read: what was asked, where the run stopped, and what would get the question answered.
The middle line comes from ``signals`` -- the gate's threshold and score, #36's three
``tool_error`` labels, the citation check's rejects -- which is the second reason those signals
are recorded rather than compared and dropped. A refusal that cannot say why it refused teaches
the reader nothing, and teaches Day 5 less.

``{term}`` comes from the config rather than from the text, because "the catalog does not say"
means a particular catalog: ``data/courses.json`` holds one term, and a course that is simply not
offered this spring is indistinguishable here from one that never existed.
"""

from __future__ import annotations

from typing import Any

from agent.nodes import Node
from agent.state import AgentState
from config import Config

OUT_OF_SCOPE_MESSAGE = (
    "That is outside what I can answer. I work from one source -- the University of Illinois "
    "course catalog for {term} -- which lets me answer questions about course descriptions and "
    "credits, prerequisite chains, and progress against the Gen Ed requirements. I have not tried "
    "to answer this one from general knowledge."
)

HANDOFF_MESSAGE = (
    "I could not answer this from the {term} course catalog, so I am handing it over rather than "
    "guessing.\n\n"
    "What you asked: {query}\n"
    "Where I stopped: {reason}\n"
    "What would get it answered: {remedy}\n\n"
    "An academic advisor can also see what this catalog export cannot -- your transcript, "
    "program requirements outside the Gen Eds, and sections as they actually fill."
)

#: Reason and remedy for each ``signals['tool_error']`` a node can set -- #36's three, declared in
#: :data:`agent.nodes.dispatch.TOOL_ERRORS` and cross-checked by a test. Anything else is quoted
#: verbatim by :func:`handoff_reason` rather than guessed at, so a new tool error reaches the
#: reader as itself and shows up here afterwards.
REASONS: dict[str, tuple[str, str]] = {
    "no_course_code": (
        (
            "the question does not name a course, and which one was meant is not something to "
            "guess at -- answering confidently about the wrong course is worse than not answering"
        ),
        "name it by code: 'CS 374', 'cs 374' and 'CS374' all reach the same catalog record",
    ),
    "course_not_found": (
        (
            "I looked the course up and this catalog does not have it -- either there is no such "
            "course, or it is not offered in the one term this export covers"
        ),
        (
            "check the subject and the number, or ask for the code it is cross-listed under -- "
            "the catalog lists 'CS 374' and 'ECE 374' as separate records"
        ),
    ),
    "no_program": (
        (
            "I could not tell which program to audit -- this export carries the campus General "
            "Education rules and a small number of hand-transcribed degrees, and auditing you "
            "against a program you are not in is wrong in the direction that costs a semester"
        ),
        (
            "name it: 'gened' for the campus General Education requirements, or one of the degrees "
            "this export carries, the way the catalog spells it -- for anything else, the "
            "program's catalog page and your department advisor are the source"
        ),
    ),
}

#: When the gate refused on a score. Both numbers are in the sentence because the threshold is a
#: config value (#37) and a reader comparing the two can tell "no match" from "threshold too high".
SCORE_REASON = (
    "nothing matched closely enough -- the best {source} match scored {score:.4f}, against the "
    "{source} threshold of {threshold}"
)
SCORE_REMEDY = (
    "try the words the catalog itself uses (a subject code, a course title), or ask for one course "
    "at a time"
)

#: When #38's citation check rejected the draft. This one is not the reader's fault and the
#: message says so: the answer existed and was discarded on purpose.
CITATION_REASON = (
    "I drafted an answer, but it cited {ids}, which is not in the evidence I retrieved -- a "
    "citation I cannot check is not one I will pass on, so I dropped the answer"
)
CITATION_REMEDY = "ask again, more narrowly; the evidence I did retrieve is listed in the trace"

EMPTY_REASON = "nothing in the catalog came back for it at all"
EMPTY_REMEDY = (
    "the course may not be offered in {term}, or may be listed under a different subject code"
)

DEFAULT_REASON = "the evidence I gathered was not enough to answer from"
DEFAULT_REMEDY = "rephrase the question, or ask about one specific course"


def handoff_reason(state: AgentState, term: str) -> tuple[str, str]:
    """Why this run stopped, and what would unblock it -- both read off ``signals`` and ``docs``.

    Ordered by how specific the evidence is, not by which node ran last. A named ``tool_error`` is
    the most specific thing a run can say about itself, a rejected citation names the exact
    fabrication, a score pair names the comparison that failed, and an empty ``docs`` is what is
    left when a retrieval simply returned nothing.
    """
    signals: dict[str, Any] = dict(state.get("signals") or {})

    tool_error = signals.get("tool_error")
    if tool_error:
        if tool_error in REASONS:
            return REASONS[tool_error]
        return (
            f"the tool I needed reported {tool_error!r} and produced nothing to answer from",
            DEFAULT_REMEDY,
        )

    unknown = signals.get("citations_unknown")
    if unknown:
        ids = unknown if isinstance(unknown, str) else ", ".join(unknown)
        return CITATION_REASON.format(ids=ids), CITATION_REMEDY

    score = signals.get("retrieval_top_score")
    threshold = signals.get("evidence_threshold")
    if score is not None and threshold is not None:
        source = signals.get("retrieval_mode") or "retrieval"
        return (
            SCORE_REASON.format(source=source, score=score, threshold=threshold),
            SCORE_REMEDY,
        )

    if not state.get("docs"):
        return EMPTY_REASON, EMPTY_REMEDY.format(term=term)

    return DEFAULT_REASON, DEFAULT_REMEDY


def out_of_scope_node(cfg: Config) -> Node:
    """Exit 3: the question is not about this catalog, and the run says so and stops."""
    term = cfg["data"]["term"]
    message = OUT_OF_SCOPE_MESSAGE.format(term=term)

    def node(state: AgentState) -> dict[str, Any]:
        return {"answer": message, "exit": "out_of_scope"}

    node.__name__ = "out_of_scope"
    return node


def refuse_handoff_node(cfg: Config) -> Node:
    """Exit 2: in scope, but not answerable from the evidence -- so it goes to a person."""
    term = cfg["data"]["term"]

    def node(state: AgentState) -> dict[str, Any]:
        reason, remedy = handoff_reason(state, term)
        return {
            "answer": HANDOFF_MESSAGE.format(
                term=term, query=state["query"], reason=reason, remedy=remedy
            ),
            "exit": "refuse_handoff",
        }

    node.__name__ = "refuse_handoff"
    return node
