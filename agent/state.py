"""What flows through the graph, and which node is allowed to write each part of it.

One ``TypedDict`` and one record shape, in one module, because six nodes (#35-#38 plus the two
fixed exits) read them and a state whose shape is decided by whichever node got there first is not
a state -- it is six private dicts sharing a variable name.

**``docs`` is one record shape whatever produced it.** A retrieval hit and a tool result both
become :class:`Evidence`, so the answer node (#38) has a single input format and its citation
check works the same for both. Normalizing here rather than in the answer prompt is what stops
"tool answers" and "retrieved answers" from becoming two code paths that drift apart.

**``signals`` has a declared key set, not a growing one.** :data:`SIGNAL_KEYS` is the whole list,
prefilled with ``None`` by :func:`initial_signals`, and :func:`check_signals` refuses anything
else. Day 5 calibrates on these numbers across many runs; a dict that grows keys ad hoc over five
days gives it a ragged table, and adding a signal should be a one-line declaration here rather
than a keyword typed once in a node and never read.

**Who writes what is data, in :data:`NODE_WRITES`.** LangGraph does not police this: a node that
returns a key the state schema does not have has its update **silently dropped** (probed on
langgraph 1.2.11 -- no exception, no warning), and a node that writes another node's key wins by
running later. Both are the same bug -- a state that is nobody's responsibility -- so
:func:`check_update` turns them into a sentence naming the node, the key and its owner.
``agent.graph`` wraps every node in it.

The three exits are the one thing a run must always be able to say about itself, so ``exit`` is a
field rather than something inferred from which of ``answer`` / ``evidence_ok`` / ``docs`` happen
to be set. #39's report counts exits; inference would make that count a guess.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Annotated, Any, Literal, NamedTuple, TypedDict

#: The five buckets #35's router puts a query in. ``Literal`` so a typo in a dispatch table is a
#: type error before it is a run that silently retrieved instead of looking up.
Intent = Literal[
    "course_fact", "prereq_chain", "degree_requirement", "compare_recommend", "out_of_scope"
]

#: The same five, as data -- for the router's prompt, #36's dispatch table, and the CLI's
#: ``--intent`` choices.
INTENTS: tuple[Intent, ...] = (
    "course_fact",
    "prereq_chain",
    "degree_requirement",
    "compare_recommend",
    "out_of_scope",
)

#: How a run ended. Day 2's whole acceptance is that all three of these happen.
Exit = Literal["answer", "refuse_handoff", "out_of_scope"]

#: The same three, as data -- #39's report counts one row per name.
EXITS: tuple[Exit, ...] = ("answer", "refuse_handoff", "out_of_scope")


class Evidence(NamedTuple):
    """One block an answer may be built from: a retrieved chunk, or a tool's result.

    A ``NamedTuple`` for the reason :class:`retrieval.types.Hit` is one -- it is an internal record
    passed between modules, not a schema published to anybody. The Pydantic models in
    ``agent/tools/`` are the MCP contract; this shape never leaves the process.

    ``score`` is ``None`` for a tool result, and that is load-bearing rather than tidy: the
    evidence gate (#37) reads ``None`` as "an exact catalog lookup, there is no ranking to be
    unsure about" and lets it past the threshold. A tool result scored 1.0 would be compared
    against a BM25 number, which means nothing (see :mod:`retrieval.types` on why scores are not
    comparable across sources).
    """

    #: What a citation in the answer names: a chunk's ``{term}:{SUBJ}-{NUM}`` id, or a course code.
    doc_id: str
    #: The text the answer must be traceable to -- the indexed chunk, or the rendered tool result.
    text: str
    #: The producing retriever's own scale, or ``None`` for a tool result. Never cross-comparable.
    score: float | None
    #: Where it came from: a retrieval mode (``"bm25"``, ``"hybrid"``) or a tool name
    #: (``"get_course"``). #37 picks its threshold by this, so it names the *source*, not the node.
    source: str


#: Every key ``signals`` may hold, and the whole of what Day 5 gets to calibrate on. Adding one is
#: a line here plus the node that fills it; a node that invents a key is refused by
#: :func:`check_signals` rather than quietly logged into a column nothing reads.
SIGNAL_KEYS: tuple[str, ...] = (
    # #35, the router: what the model actually emitted, and whether it was one of the five labels.
    "intent_raw",
    "intent_unparsed",
    # #36, dispatch: which retrieval mode ran, and why a tool produced nothing.
    "retrieval_mode",
    "tool_error",
    # #37, the gate: the number it compared and the number it compared against.
    "retrieval_top_score",
    "evidence_threshold",
    # #38, the answer: the doc_ids the answer cited and this run can show, and the ones it cited
    # that were never in `docs`. Both, because the pair is what tells the three outcomes apart: a
    # verified list with no unknowns is an answer that stood, an unknown id is a fabrication that
    # discarded it, and an empty verified list with no unknowns is a draft that cited nothing at
    # all -- which is also discarded. `citations_verified is None` means this node never ran.
    "citations_verified",
    "citations_unknown",
)


def initial_signals() -> dict[str, Any]:
    """Every signal key, set to ``None``.

    Prefilled rather than empty so that every run produces the same-shaped record: Day 5 reads a
    rectangular table where a missing measurement is a ``None`` cell, not an absent key that could
    equally mean "the node never ran" or "the node forgot".
    """
    return dict.fromkeys(SIGNAL_KEYS)


def check_signals(signals: Any, node: str = "a node") -> dict[str, Any]:
    """Refuse a signal key :data:`SIGNAL_KEYS` does not declare; return the update.

    The message names the key and the file, because the fix is a declaration and not a rename:
    if a node has something worth recording, the list above is where it becomes something Day 5
    can find.
    """
    if not isinstance(signals, Mapping):
        raise TypeError(f"{node} set signals to {type(signals).__name__}, which is not a mapping")
    unknown = [key for key in signals if key not in SIGNAL_KEYS]
    if unknown:
        known = ", ".join(SIGNAL_KEYS)
        raise ValueError(
            f"{node} wrote undeclared signal(s) {', '.join(map(repr, unknown))}; declare them in "
            f"SIGNAL_KEYS in agent/state.py first -- known signals: {known}"
        )
    return dict(signals)


def merge_signals(
    left: Mapping[str, Any] | None, right: Mapping[str, Any] | None
) -> dict[str, Any]:
    """Reducer for ``signals``: later keys win, earlier keys survive.

    Without a reducer LangGraph replaces a state key with each node's update, so the router's
    ``intent_raw`` would be erased the moment the gate recorded a threshold -- four nodes each
    writing one key, and only the last one's surviving. Every other key in :class:`AgentState` is
    written by exactly one node, which is why this is the only one that needs one.

    Right-hand keys win per key, so a node can also correct a signal it wrote earlier. The merge
    is idempotent, which matters: langgraph 1.2.11 applies the reducer to the same update twice on
    the final superstep (probed), so an accumulating reducer -- ``operator.add`` over a list --
    would double-count there.
    """
    merged = dict(left or {})
    merged.update(right or {})
    return merged


class AgentState(TypedDict):
    """One message in, one of three exits out -- and everything the run learned on the way.

    Written by exactly one node each, except ``signals`` (every node, merged) and ``query`` (the
    caller). :data:`NODE_WRITES` is that table, and :func:`check_update` enforces it.
    """

    #: The question as asked. Nothing in the graph rewrites it; #38's prompt quotes it verbatim.
    query: str
    #: One of :data:`INTENTS`, set by #35. ``None`` only before the router has run.
    intent: Intent | None
    #: The evidence, normalized, set by #36. Empty is a real outcome -- it is what #37 refuses on.
    docs: list[Evidence]
    #: #37's verdict: is there enough here to answer at all?
    evidence_ok: bool | None
    #: Everything a later day wants to calibrate on. See :data:`SIGNAL_KEYS`.
    signals: Annotated[dict[str, Any], merge_signals]
    #: The answer, or one of the two fixed exit messages. ``None`` until an exit node writes it.
    answer: str | None
    #: Which of the three exits this run took. Never inferred -- see the module docstring.
    exit: Exit | None


#: Which state key each node may write. ``signals`` is shared and is not listed; everything else
#: has exactly one owner, and ``query`` has none -- a node that rewrites the question is a bug
#: this table makes loud. The two exit nodes write the same pair as #38's answer node: an exit is
#: an exit however it was reached.
NODE_WRITES: dict[str, frozenset[str]] = {
    "route_intent": frozenset({"intent"}),  # 35
    "dispatch": frozenset({"docs"}),  # 36
    "check_evidence": frozenset({"evidence_ok"}),  # 37
    "answer": frozenset({"answer", "exit"}),  # 38
    "out_of_scope": frozenset({"answer", "exit"}),
    "refuse_handoff": frozenset({"answer", "exit"}),
}

#: The key every node may write, on top of its own.
SHARED_WRITES = frozenset({"signals"})

#: The keys :class:`AgentState` actually has, so a typo is caught here rather than dropped by
#: LangGraph. Read off the TypedDict so the two can never disagree.
STATE_KEYS = frozenset(AgentState.__annotations__)


def initial_state(query: str) -> AgentState:
    """The state a run starts from: the query, the signal skeleton, and nothing decided yet.

    Every key is present from the start -- ``None`` for the undecided ones -- so a node reading a
    key before its owner ran gets ``None`` and not a ``KeyError``, and the routers can tell
    "not set" from "set to nothing".
    """
    if not query.strip():
        raise ValueError("query is empty; there is nothing to route, retrieve or answer")
    return AgentState(
        query=query,
        intent=None,
        docs=[],
        evidence_ok=None,
        signals=initial_signals(),
        answer=None,
        exit=None,
    )


def owners_of(key: str) -> str:
    """Which node(s) own ``key``, for an error message that says where the write belongs."""
    owners = [node for node, keys in NODE_WRITES.items() if key in keys]
    return ", ".join(owners) if owners else "the caller of run(), nobody in the graph"


def check_intent(value: Any, node: str) -> Intent:
    """Refuse an ``intent`` that is not one of the five labels.

    #35 validates the *model's* string and falls back to ``out_of_scope`` for anything unparseable,
    which is its job; by the time an intent reaches the state it is a label this repo knows, or a
    bug in the node. A free-text label must never reach #36's dispatch table.
    """
    if value not in INTENTS:
        raise ValueError(
            f"{node} set intent to {value!r}, which is not one of the five labels "
            f"({', '.join(INTENTS)}); an unparseable classification routes to 'out_of_scope' "
            "with signals['intent_unparsed'] set, it does not become a sixth label"
        )
    return value  # type: ignore[return-value]


def check_exit(value: Any, node: str) -> Exit:
    """Refuse an ``exit`` that is not one of the three."""
    if value not in EXITS:
        raise ValueError(f"{node} set exit to {value!r}; the exits are {', '.join(EXITS)}")
    return value  # type: ignore[return-value]


def check_docs(value: Any, node: str) -> list[Evidence]:
    """Refuse anything in ``docs`` that is not :class:`Evidence`.

    This is what makes "one record shape whatever produced it" true rather than intended. A raw
    :class:`retrieval.types.Hit` is the tempting mistake -- it has ``doc_id``, ``text`` and
    ``score``, so #38's prompt would render it and only the missing ``source`` would go wrong,
    in #37, one node later.
    """
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise TypeError(f"{node} set docs to {type(value).__name__}; it must be a list of Evidence")
    for item in value:
        if not isinstance(item, Evidence):
            raise TypeError(
                f"{node} put {type(item).__name__} in docs; every block must be an "
                "agent.state.Evidence -- a retrieval Hit and a tool result are converted, not "
                "passed through (see #36)"
            )
    return list(value)


def check_update(node: str, update: Any) -> dict[str, Any]:
    """Validate one node's state update: the node, the keys, and the values it claims.

    Four failures, all of which are silent without this: a key the state schema does not have
    (LangGraph drops it), a key belonging to another node (last write wins), an undeclared signal
    (Day 5 reads a column nothing filled), and a value outside the field's domain (an invented
    intent label, a ``docs`` list of raw hits). Each is reported as a sentence naming the node --
    the node is what has to change, and by the time the graph has finished, the update it came
    from is gone.
    """
    if node not in NODE_WRITES:
        known = ", ".join(NODE_WRITES)
        raise KeyError(f"no node named {node!r} in NODE_WRITES; the graph's nodes are: {known}")
    if not isinstance(update, Mapping):
        raise TypeError(
            f"node {node!r} returned {type(update).__name__}; a node returns a mapping of the "
            "state keys it wrote (an empty one if it wrote nothing)"
        )
    allowed = NODE_WRITES[node] | SHARED_WRITES
    for key in update:
        if key not in STATE_KEYS:
            raise ValueError(
                f"{node} returned {key!r}, which is not a key of AgentState "
                f"({', '.join(sorted(STATE_KEYS))}); LangGraph would drop it silently"
            )
        if key not in allowed:
            raise ValueError(
                f"{node} wrote {key!r}, which belongs to {owners_of(key)}; a node writes its own "
                f"key and signals, nothing else (it may write: {', '.join(sorted(allowed))})"
            )
    if "signals" in update:
        check_signals(update["signals"], node)
    if "intent" in update:
        check_intent(update["intent"], node)
    if "exit" in update:
        check_exit(update["exit"], node)
    if "docs" in update:
        check_docs(update["docs"], node)
    if "evidence_ok" in update and not isinstance(update["evidence_ok"], bool):
        raise TypeError(
            f"{node} set evidence_ok to {update['evidence_ok']!r}; the gate returns True or "
            "False -- None is 'the gate has not run', which it has"
        )
    if "answer" in update and (
        not isinstance(update["answer"], str) or not update["answer"].strip()
    ):
        raise ValueError(
            f"{node} set answer to {update['answer']!r}; an exit writes the text a person reads, "
            "and an empty answer reaches them as an empty reply rather than as an error"
        )
    return dict(update)
